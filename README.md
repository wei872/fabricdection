# JingWei · DINOv2 布匹缺陷检测

只保留一条训练路径：**官方预训练 DINOv2 ViT → 特征适配金字塔 → PAN → 缺陷检测头**。
默认冻结 ViT，使用真实 YOLO 框标注训练适配器和检测头，支持 19 类布匹缺陷。
不再提供 CNN 主干、离线 teacher 蒸馏、记忆库或异常重建训练流程。

> DINOv2 是预训练视觉主干，不是已经训练好的缺陷检测器。检测头仍需学习；
> 代码测试通过不代表在真实数据上已达到目标精确率。大模型保留在推理端，不承诺实时 FPS。

## 安装

Python ≥ 3.10。先按设备安装合适的 PyTorch，再安装其余依赖：

```bat
pip install -r requirements.txt
```

首次训练需要访问官方 DINOv2 hub 源码和权重地址；也支持本地源码及官方 `.pth`。
`xFormers is not available` 通常表示回退标准 PyTorch attention，不是权重加载失败。

## 训练：一个配置、一个入口

把 `configs/fabric19.yaml` 中的 `train`、`val` 改成实际图片目录。
YOLO 标签目录按 `images/train → labels/train`、`images/val → labels/val` 推断。
`normalval` 仅用于正常图误报率评估，可省略或设置为 `null`。

```bat
python tools/train_detector.py --data configs/fabric19.yaml --epochs 50 --batch 2 --val-batch 2 --amp bf16 --out runs/detector_dinov2
```

所有工具默认使用 **`configs/jingwei_dinov2.yaml`**。首次运行不传 checkpoint，
自动加载 `dinov2_vitb14(pretrained=True)`。失败会报错，不静默回退随机主干。

已下载官方权重时，在该配置的 `model` 中修改已有项：

```yaml
dino_repo: C:/models/dinov2                  # 官方源码目录，包含 hubconf.py
dino_weights: C:/models/dinov2_vitb14_pretrain.pth
dino_pretrained: true
```

### 恢复训练

```bat
python tools/train_detector.py --resume runs/detector_dinov2/last.pt --epochs 50 --batch 2 --val-batch 2 --out runs/detector_dinov2
```

`--resume` 恢复模型、优化器、scheduler 和随机状态。
若要重置学习率和训练计划，改用 `--pretrained CHECKPOINT --lr 0.0001` 并另开输出目录。
**`--pretrained` 仅接收完整 DINOv2 检测权重；原始 ViT `.pth` 应配置在 `model.dino_weights`。**

## 评估、预测与测速

```bat
python tools/evaluate.py --ckpt runs/detector_dinov2/best.pt --data configs/fabric19.yaml --diag --batch 2
python tools/predict.py --ckpt runs/detector_dinov2/best.pt --source D:/fabric_test/ --out runs/predict
python tools/bench_device.py --ckpt runs/detector_dinov2/best.pt --iters 20
```

预测结果为原图坐标，中文类别名保存在 `results.json`，图片绘制使用 ASCII 类别标签。
图片、文件夹、视频和中文路径均支持；不包含未知异常类或未标注异常兜底。
测速包含检测解码与 NMS，不包括读图、缩放和输入传输。

## 默认训练策略

- 官方 ViT-B/14 主干冻结，适配层与检测头可训练。
- AdamW，学习率 `5e-4`，1 epoch warmup 后 cosine 衰减。
- Softplus 距离回归，保留已修复的逐网格 box loss。
- 双分配监督，推理使用一对一 objectness 和可选同类 NMS。
- 按验证集 `best_f1` 保存 `best.pt`；对应阈值不会自动写入推理配置。
- 检测图为 1280×1280，DINO 输入默认 448×448。微小缺陷可对比 672，代价是更多显存与计算。

## 代码目录

| 路径 | 职责 |
|---|---|
| `configs/jingwei_dinov2.yaml` | 唯一模型配置 |
| `configs/fabric19.yaml` | 数据路径、类别元数据 |
| `jingwei/dinov2_backbone.py` | 官方预训练 ViT、中间层输出和可训练适配器 |
| `jingwei/neck.py`, `head.py`, `losses.py` | PAN、检测头和监督损失 |
| `jingwei/model.py` | 训练／推理及六张量导出接口 |
| `jingwei/data.py`, `image_io.py` | 标注读取、增广及 Unicode I/O |
| `tools/train_detector.py` | 唯一训练入口 |
| `tools/evaluate.py`, `predict.py` | 指标与实际预测 |
| `tools/bench_device.py`, `export.py` | 实测延迟、实验性 ONNX 导出 |
| `tests/` | 无网络回归测试，使用 tiny hub fixture 而非实际大模型权重 |

## 文档与测试

- [训练、数据、调参与 checkpoint 说明](docs/TRAINING.md)
- [模型结构、计算与验证边界](docs/DINOV2_DETECTOR.md)

```bat
python -m unittest discover -s tests -v
```

本次整理移除了旧方案源码、配置、脚本和对应文档／测试，未删除已有图片或训练产物。
旧 CNN／蒸馏 checkpoint 不受支持，旧配置会明确报错。PyTorch 是当前验证路径；
ONNX 是六预测张量的实验导出接口，不宣称已经验证真实 ViT 的 TensorRT/INT8 部署。
