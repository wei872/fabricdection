# DINOv2-S 共享特征开放集检测

## 结构与预算

```text
原图 → 640×640 RGB [0,1]
   ↓ 224×224 + ImageNet 标准化
官方预训练 DINOv2 ViT-S/14（默认冻结，一次前向）
   ├─ 中间层 2/5/8/11 → 适配层 → PAN → P3/P5 已有类别检测
   │                                      ↓ top-k64 / 两层阈值 / 同类 NMS
   └─ 第 5 层 patch 特征 → L2 归一化 → 32 个正常原型余弦距离
                                          ↓ 校准阈值 / 连通区域
输出已有类别框、未知纹理偏离区域（均映射回原图）
```

原始 DINOv2 四层均为同分辨率 patch 网格，不是原生多分辨率金字塔。
检测器不再计算／插值不被 PAN 使用的 stride-4 detail 分支；其已注册投影仍保留在权重结构中。
官方源码随机 ViT-S 接口检查得到 encoder 22,056,576 参数；不是 ViT-B，也不是蒸馏 CNN。

默认检测 640、ViT224、width0.5、DWConv、top-k64。NMS 用 OpenCV 原生 greedy 实现，
将小候选集合一次性传回 CPU，避免旧 GPU 上每候选一次 Python/CUDA 同步循环。
这只是减少潜在开销，并非已经证明 1660 上 30 FPS。

## 初始化与训练控制

`--dino-weights` 严格加载官方 raw backbone state_dict，Hub 仅构造架构（`pretrained=False`），
不再下载预训练参数；本地源码可用 `--dino-repo` 指定。完整检测器 checkpoint 则直接恢复 encoder、
细节层、适配层和检测头，不叠加官方文件。原始文件只预训练 encoder，不代表检测器已经学会布匹缺陷。

新训练以独立 `lr_decay_epochs` 控制 cosine 衰减（默认 50），预算设置 500 轮也不会将其拉长；
衰减后固定 5% LR 下限，验证早停可选。完整续训保留原始周期和早停状态；
需要新周期时用完整检测器 `--pretrained` 配新输出目录。
CIoU／像素 L1 和梯度裁剪日志仅用于诊断，不改变既有损失公式，也不证明实际收敛轮数改善。
详见 [官方文件加载与训练控制](TRAINING.md#21-直接加载官方权重文件推荐显式指定)。

## 正常与异常如何学习

- 已有异常：使用 YOLO 标注训练类别、定位和 objectness。
- 正常负样本：检测目标为空，训练背景抑制，不伪造未知类别标签。
- 正常纹理：仅显式正常样本的共享 ViT 特征更新有界在线原型；缺陷图不更新原型。
- 原型：最远点初始化，后续按余弦最近分配和 EMA 更新，不保存全部训练 patch。
- 未知分数：`1 - max_cosine(normal_prototypes)`，范围 `[0,2]`，不是概率。
- 阈值：独立正常集的每图最大分数顺序统计量；对每个 patch 逐个阈值化，再提取连通区域。
- 相同 ViT 前向产生两路特征；无额外 teacher、重建网络或全量记忆库查找。

正常原型与阈值保持 FP32，FP16 仅是可选的 encoder／head 推理精度。
校准 checkpoint 保存原型、阈值、校准输入大小、精度标志；重训更新原型使阈值失效。
完整推理入口拒绝未校准模型、尺寸或精度不匹配，不静默当作只有已知检测的系统。

## 能做什么，不能证明什么

1. **未知是偏离正常的提示，不是可靠的新类别识别。** 已知缺陷也可能出现在未知区域中；两路不是互斥分类。
2. 未见布种／经纬密度／光照也可能偏离正常而误报。需真实正常数据覆盖工况，并按新域复测、必要时再校准。
3. 少数正常原型可能欠拟合复杂正常纹理；可实验增加原型数（≤256），但不能以此承诺泛化。
4. ViT224 只有16×16 patch 网格，640输入对应约40像素的patch间距。早已缩掉的微小疵点不能靠插值或检测图640恢复。
5. 升到ViT280/336、切片或增加多尺度通常增加计算；既要小疵点召回又要30FPS可能不可同时满足。
6. 校准目标基于正常分布可交换性，不是任意新布匹上的误报保证；合成增强无法替代独立跨布卷测试。
7. 六张量 ONNX 仅包含已有类别预测，不包含原型／未知区域；全流程 TensorRT/INT8 未验证。

## 验收与现有证据

训练和校准命令见 [TRAINING.md](TRAINING.md)。

速度验收要求真实GTX1660、batch1、真实代表帧、完整校准未知路径，至少10帧、10次预热、100次采样；
平均≥30FPS且P95≤33.3ms。报告包含预处理/H2D/ViT/NMS/未知区域/坐标/D2H，
不含相机等待、磁盘解码、显示和写盘。速度验收与质量验收严格分开。

已做的是 tiny hub 的训练→校准→预测→评估→测速失败门禁回归，以及官方源码随机 ViT-S 的前后向接口检查。
**没有官方预训练权重的真实训练、跨布匹泛化、未知召回或1660速度实测结果。**

## Native-detail extension

Current defaults retain the 224 ViT pass but add a native RGB-to-P3 detail path
(PixelUnshuffle, narrow depthwise-separable convolutions, gated residual).
Unknown scoring also uses a fixed, native stride-8 grayscale texture descriptor
bank alongside the frozen semantic prototypes. Joint image-max calibration is
required again; the six-tensor ONNX artifact still excludes unknown processing.
See TRAINING section 11 for migration, supervision changes and explicit limits.
Interpolated DINO maps alone still cannot recover lost detail; these new routes
supply additional original-pixel information, not a proof of recall or 1660 FPS.

## 1280lite profile (optional; 640 default retained)

Native input/detail CNN 1280; ViT224; PAN on a capped640-equivalent grid.
Pool native detail for semantic PAN fusion, then restore its native high-frequency
residual after PAN. The fine head is160² at stride8; the coarse head remains20²
at stride64. Training, decoding, export metadata and geometry diagnostics all
share these strides. No extra learned parameters or second ViT pass are added.
Native unknown descriptors run on160² and require new normal statistics/calibration.

CNN Conv-BN folding is eval-only and does not rewrite training checkpoints.
The profile reduces CNN compute compared with naive full-pyramid1280, not with
640, and does not reduce the official ViT parameter count. See TRAINING section12
for measured Conv MAC scope, migration and the outstanding real GTX1660/quality acceptance.
