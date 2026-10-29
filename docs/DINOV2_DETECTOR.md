# 直接使用官方预训练 DINOv2 做缺陷检测

## 与原方案的区别

原 `pretrain_dino.py` 已经使用了官方预训练 DINOv2，但它只把特征蒸馏到一个
轻量 CNN，阶段 A 的主干仍是 CNN。**新路径是直接把已预训练的 DINOv2
ViT-B/14 留在检测网络中**，不进行这一步学生蒸馏，也不从零训练 ViT。

```text
RGB / ImageNet normalization
    → 官方预训练 DINOv2 ViT-B/14（默认冻结，保持 eval）
    → 4 个中间层（ViT-B 为 2/5/8/11，0-based）
    → 可训练 1×1 投影 + 插值形成 stride 4/8/16/32 特征金字塔
    → PAN + 双尺度检测头
    → 可选同类 NMS
```

注意：这些尺度由同一个 patch 网格的不同层插值得到，不是真正的原生多分辨率
ViT 特征。插值不能恢复编码器输入缩小时丢失的细小缺陷信息。
原始 RGB 按 ImageNet mean/std 归一化，不使用 CNN 的 LocalContrastNorm。

这是**直接使用预训练视觉特征进行检测迁移学习**，不是已预训练的缺陷检测器：
适配器、PAN 和检测头仍需使用真实框标注训练。不能承诺几轮达到某个精确率。

## Windows CMD：直接开始阶段 A

先更新代码，把 `configs/fabric19.yaml` 中路径改成真实数据位置。不要运行阶段 0：

```bat
python tools/train_detector.py --cfg configs/jingwei_dinov2.yaml --data configs/fabric19.yaml --epochs 50 --batch 2 --val-batch 2 --num-workers 4 --amp bf16 --log-interval 20 --val-interval 1 --out runs/detector_dinov2
```

首次运行时 **不传 `--resume`，也不传旧的 `--pretrained dino_init.pt`**。
配置里 `model.dino_pretrained: true` 会调用官方 hub 的
`dinov2_vitb14(pretrained=True)`，下载/读取官方权重；失败会报错，不会悄悄用随机权重。

应出现：

```text
DIRECT pretrained DINOv2 backbone=dinov2_vitb14 frozen=True encoder_input=448
initialization=official pretrained DINOv2; pyramid adapters / detector head=new
best_metric=best_f1  warmup_epochs=1
```

`xFormers is not available` 表示使用标准 PyTorch attention，并非未加载权重。
默认 batch=2 是较保守的起点，不保证所有 GPU 都够；OOM 时先改 batch=1 和
val-batch=1，不要直接解冻整个 ViT。不要把新配置与旧 CNN checkpoint 混用。

## 离线 / 已经下载好大模型权重

在 `configs/jingwei_dinov2.yaml` 的 `model` 下修改已有字段：

```yaml
dino_repo: C:/models/dinov2
dino_weights: C:/models/dinov2_vitb14_pretrain.pth
dino_pretrained: true
```

- `dino_repo` 是官方 DINOv2 源码目录（含 `hubconf.py`）。只有缓存了 S 模型，
  不代表已经缓存 B 模型的权重。
- `.pth` 必须是与 `dino_name` 一致的标准官方 backbone-only state_dict；严格检查
  所有键和形状，不接受不匹配的 teacher/student checkpoint。
- 也可改成 `dinov2_vits14` 直接用已有的 S 预训练主干，或资源允许时使用
  `dinov2_vitl14`；对应的本地权重也需一起换。
- 不要把原始 ViT `.pth` 传给 `--pretrained`。该 CLI 参数用于完整的本项目检测模型；
  **原始 DINOv2 主干权重由 `model.dino_weights` 加载**。
- 标准官方 DINOv2 backbone 的许可证与专用医疗等变体不同；此路径仅针对标准模型。

## 恢复、评估与推理

使用同一配置恢复新检测模型：

```bat
python tools/train_detector.py --cfg configs/jingwei_dinov2.yaml --data configs/fabric19.yaml --resume runs/detector_dinov2/last.pt --epochs 50 --batch 2 --val-batch 2 --amp bf16 --out runs/detector_dinov2
python tools/evaluate.py --cfg configs/jingwei_dinov2.yaml --data configs/fabric19.yaml --ckpt runs/detector_dinov2/best.pt --detector-only --diag --batch 2
python tools/predict.py --cfg configs/jingwei_dinov2.yaml --ckpt runs/detector_dinov2/best.pt --source D:/fabric_test/
```

检测 checkpoint 包含 ViT 权重。恢复/推理会以 `pretrained=False` 构建架构，再严格
加载检测 checkpoint，不重复下载官方权重，但仍需本地源码或可用的 hub 源码缓存。
缺失任何 encoder 权重都会报错。使用相同配置恢复；不要在完整 resume 时改
冻结策略、模型结构、warmup 长度或模型选择指标。

若要从新检测 checkpoint 开一组新的学习率实验，使用
`--pretrained runs/detector_dinov2/last.pt --lr 0.0001 --out runs/detector_dinov2_lr1e4`，
它只加载权重，不恢复旧 optimizer/scheduler。旧 CNN 的 `last.pt` 不能迁移到这个新架构。

## 新配置中的训练稳定性 / 精确率调整

- **冻结官方 ViT**：不反传大主干，仅更新适配器、PAN 和头；`model.train()` 不会
  误把 frozen encoder 切到 train 模式。
- **AdamW lr=5e-4 + 1 epoch warmup + cosine 衰减**：避免新头刚开始时使用过激的更新。
- **Softplus 距离输出**：负回归 logit 也有梯度，避免 ReLU 负区间梯度为零。
  仅新配置启用，旧 CNN 的 ReLU 行为不变。
- **同类别 NMS IoU=0.5**：抑制同类别重叠重复框，不抑制不同类别。
  它不是训练收敛技巧，只是可选后处理；不能解决错分类、漏检或不准确定位。
  设置 `model.detector_nms_iou: 0` 可关闭，原 n/s/p 默认仍不启用 NMS。
- **按 `best_f1` 选 best.pt**：不再因为高 large-recall 就选误检较多的模型。
  `best_f1` 对应的阈值仍需验证后显式应用到 `fusion.final_score_thresh`；
  保存 best 并不会自动修改部署阈值，也不构成独立测试集结果。
- **TP / FP / FN 日志**：直接观察误检和漏检，不把训练 loss 当成准确率。
- **`--no-augment`**：支持小样本过拟合排查；正常完整训练默认保留增广。
- 默认关闭尚未训练的 normality / foreground 分支，先集中验证监督检测。
  此配置是检测专用，不能直接照抄原阶段 B/C/D 的异常检测流程。

如果冻结后检测头已稳定学习，但指标长时间不再提升，可另开实验尝试
`dino_freeze: false`、`dino_lr_mult: 0.1`（默认）。训练脚本把 encoder 单独分组，
其学习率为检测头的 1/10；显存开销会明显增加。使用 `--pretrained` 而不是全状态
`--resume` 开始这组实验，因为优化器参数组已经改变。不要将解冻当成默认必需步骤。

## 分辨率、部署与验证边界

- 检测输入维持 1280；默认 DINO 实际输入为 448×448（patch grid 32×32）。
  若小缺陷漏检多，优先对比 `dino_input_size: 672`（必须为 14 的倍数），
  并减小 batch。更大的 patch 网格会增加 attention 显存/计算，未保证笔记本显卡可用。
- DINOv2 **保留在推理图**，不再是原 pico 模型，不承诺原来的实时 FPS。
  ViT-B/14 encoder 约 86.6M 参数，完整 checkpoint 大小和每轮保存开销明显增大。
- 当前验证范围为 PyTorch 训练/恢复/推理。此新路径没有完成 ONNX/TensorRT 部署验证，
  不应沿用原 CNN 的实时性、INT8 或导出支持承诺。
- 做固定划分 A/B 时，要分别记录 backbone 改动、NMS 改动与阈值，避免把后处理
  提升当成主干收敛提升。先确认训练集上能拟合，再用独立验证/测试集检查泛化。
- 自动测试覆盖 fake encoder 的预训练请求、梯度隔离、金字塔形状、NMS、完整 CLI
  训练/恢复、严格加载。官方 ViT-B 架构接口另用随机权重验证过，但**没有在此环境
  下载官方预训练权重或使用用户真实数据训练**，不据此声称真实精确率已提升。

```bat
python -m unittest discover -s tests -p test_dinov2_backbone.py -v
```
