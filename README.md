# JingWei · DINOv2-S 布匹缺陷与未知疵点检测

单一主干：**官方预训练 DINOv2 ViT-S/14 → 轻量 PAN／检测头 + 共享特征正常原型**。
已有类别用框标注学习，正常图片作为背景负样本并更新正常纹理原型。
未知疵点由偏离正常原型的 patch 区域提示，不需要第二次 ViT 推理。

> 目标是 GTX 1660、batch=1、30 FPS。**目前没有 1660 实测，不是已保证达标。**
> 未知检测是新颖性启发式，不保证发现所有未知缺陷；跨经纬密度／光照泛化需独立实物数据验证。

## 默认预算

| 项目 | 默认值 |
|---|---|
| 主干 | `dinov2_vits14`，官方标准最小档 ViT-S/14，encoder 约 22.1M 参数 |
| 主干策略 | 官方预训练、冻结；不蒸馏、不替换 CNN |
| 检测输入 / ViT 输入 | 640×640 / 224×224 |
| neck 宽度 / 候选数 | 0.5 / top-k 64 |
| 正常原型 | 32 个，FP32 存储约 48 KiB；不保存整套 patch 记忆库 |
| 训练正常图占比 | 50%，正常图检测标签为空 |
| 精度 | 默认 FP32；1660 不支持 BF16，FP16 是否更快必须实测 |

224 的 ViT 输入只有 16×16 patch 网格，640 图上的未知区域较粗，微小疵点可能丢失。
调大 ViT 输入（如 280/336）可能改善细节，但会更贵；不能只通过缩图刷 FPS 而不测召回率。

## 安装与数据

Python ≥ 3.10。先安装适合显卡的 PyTorch，再运行：

```bat
pip install -r requirements.txt
```

编辑 `configs/fabric19.yaml`，配置相互独立的：

- `train` / `val`：已有类别的 YOLO 缺陷图和框标签。
- `normaltrain`：确认无缺陷的正常图，参与背景训练和正常纹理学习。
- `normalcal`：独立正常图，**只在训练后校准未知阈值**，默认 1% 目标至少需要 99 张；推荐更多独立原图。
- `normalval`：独立正常测试图，测实际误报率。
- `unknownval`：可选，训练中留出的新缺陷类型图片和 YOLO 框；类别值忽略，只测未知区域定位。

按原始布卷／布种／经纬密度／光照条件分组划分，不能把同一原图的增广副本分到不同集合。
增强不能替代覆盖不同工况的真实正常图和缺陷图。

## 训练、校准、验收

所有工具默认使用唯一模型配置 `configs/jingwei_dinov2.yaml`。

### 1. 从官方预训练 ViT-S 开始

```bat
python tools/train_detector.py --data configs/fabric19.yaml --epochs 50 --batch 2 --val-batch 1 --amp off --out runs/dinov2s
```

首次无需 `--pretrained`。官方权重下载失败会报错，不静默回退随机主干。
离线可把模型配置的 `dino_repo` 改为官方源码目录，`dino_weights` 改为匹配的
`dinov2_vits14_pretrain.pth`。原始 ViT 权重不要传给 CLI 的 `--pretrained`。

### 2. 校准独立正常集

```bat
python tools/calibrate_anomaly.py --data configs/fabric19.yaml --ckpt runs/dinov2s/best.pt --out runs/dinov2s/calibrated.pt --target-fpr 0.01
```

校准不更新原型，也不接触未知测试集。1% 是同分布假设下的**未知分支图像级**目标，
不是全流程误报率或跨工况保证。校准后必须在 `normalval`、`unknownval` 验证。

### 3. 评估与预测

```bat
python tools/evaluate.py --data configs/fabric19.yaml --ckpt runs/dinov2s/calibrated.pt --batch 1 --diag
python tools/predict.py --ckpt runs/dinov2s/calibrated.pt --source D:/fabric_test/ --out runs/predict
```

JSON 分别输出 `detections`（已有类别）和 `unknown_regions`（未知纹理偏离提示），框均为原图坐标。
未知得分不是类别概率，已有缺陷也可能出现在未知区域内；不要把两者当成互斥分类。
未校准的 checkpoint 默认拒绝完整预测／测速；调试仅已有类别时显式传 `--known-only`。

### 4. 在真正的 1660 上验收 30 FPS

```bat
python tools/bench_device.py --ckpt runs/dinov2s/calibrated.pt --source D:/fabric_acceptance/ --device cuda --warmup 30 --iters 300 --target-fps 30 --require-target --out runs/1660_acceptance.json
```

包含 CPU 预处理、H2D、同次 ViT、检测解码／NMS、未知区域、原图坐标及输出 D2H；
不含相机等待、磁盘解码、显示、写盘。验收检查 **平均 ≥30 FPS 且 P95≤33.3 ms**，
必须是真实 1660、真实帧、开启已校准未知分支；CPU／合成输入不能通过。
这仍只是速度验收，必须另外确认已有类别召回、未知定位召回和正常误报率。

## 低召回／零出框诊断

若 loss 下降但 P/R 很差，先检查过滤前分数，避免直接改学习率或把低出框当成低误报优势：

```bat
python tools/evaluate.py --data configs/fabric19.yaml --ckpt runs/dinov2s/best.pt --known-only --candidate-conf 0.001 --val-conf 0.15 --score-stats --diag --batch 1 --workers 0
```

此命令只改变当次评估的候选过滤，主要 P/R 仍按 0.15 报告，不改变训练／部署配置。
它能区分过滤因素与错类／定位问题，不能单凭低阈值 mAP 提升声称模型更好。
详细判断见训练手册第 8 节。

定位误报的训练匹配已改为几何可行匹配，避免把目标分配到无法用非负距离表示它的框外网格。
旧 ViT-S 权重仍可推理／warm start，但旧匹配目标的 `--resume` 不再默默复用优化器。
继续训练请用 `--pretrained runs/dinov2s/best.pt` 并另开输出目录；见手册第 9 节。

## 恢复、文档与测试

`--resume runs/dinov2s/last.pt` 恢复完整训练状态；`--pretrained` 是完整检测权重的显式 warm start。
当前 ViT-S＋新尺寸／未知原型的结构不兼容此前 ViT-B 检测 checkpoint，需要重新训练。
数据和旧权重文件不删除。

- [数据、训练、校准与泛化验收](docs/TRAINING.md)
- [结构、预算与限制](docs/DINOV2_DETECTOR.md)

```bat
python -m unittest discover -s tests -v
```

回归测试使用 tiny 本地 hub 验证流程，不证明真实模型准确率或显卡速度。
ONNX 目前只实验性导出已有类别的六张量；未知模块仍为 PyTorch/OpenCV，不能宣称全流程 TensorRT/INT8 已支持。
