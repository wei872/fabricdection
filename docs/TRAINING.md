# DINOv2 检测训练手册

所有命令在仓库根目录执行。模型配置只有 `configs/jingwei_dinov2.yaml`，所有工具默认使用它。
只有一个训练入口 `tools/train_detector.py`，不再需要多阶段预训练、建库或异常融合标定。

## 1. 数据准备

编辑 `configs/fabric19.yaml`：

```yaml
train: D:/fabric/images/train
val: D:/fabric/images/val
normalval: D:/fabric/normal/images/val  # 可省略或 null；仅评估 FPR
```

路径可以是绝对路径或相对仓库根目录的路径。不自动拼接数据 YAML 的 `path` 字段。
训练标签必须位于对应 `labels/train`，验证标签位于 `labels/val`，与图片同名：

```text
D:/fabric/
  images/train/布面1.jpg
  labels/train/布面1.txt
  images/val/布面2.jpg
  labels/val/布面2.txt
```

每行 `class_id cx cy width height`，坐标归一化到 `[0,1]`，类别编号为 `0..18`。
模型配置的 `data.nc` 和 `data.names` 决定检测输出和显示名称；数据 YAML 中类别定义应保持一致。
空标签表示背景图；当前 reader 也把缺失标签视为背景，因此必须自行排查路径误配、漏标。
训练集和验证集按原始布面图划分，不要把同一图的增广版本分到两边。
非方形原图会连同标注框一起缩放为配置的方形检测输入。

正常图不参与额外分支训练。若无 `normalval`，日志中的 `normal_fpr=nan` / 评估的 `N/A`
表示没有测量，不代表误报率为零；不存在 fallback 到另一个训练目录的行为。

## 2. 官方预训练权重

默认配置：

```yaml
model:
  backbone: dinov2
  dino_name: dinov2_vitb14
  dino_repo: facebookresearch/dinov2
  dino_weights: null
  dino_pretrained: true
  dino_freeze: true
  dino_input_size: 448
```

第一次训练加载官方已预训练的 ViT，检测适配器和头随机初始化。
没有独立 teacher 或学生蒸馏过程。网络失败会报错，不会静默使用随机 ViT。
已缓存 ViT-S 不代表已缓存 ViT-B；所需权重和 `dino_name` 必须匹配。

离线机器将已有字段改为：

```yaml
dino_repo: C:/models/dinov2
dino_weights: C:/models/dinov2_vitb14_pretrain.pth
```

源码目录须包含 `hubconf.py`，`.pth` 必须是对应标准官方 backbone-only state_dict，严格检查键和形状。
不要把原始 ViT `.pth` 当成完整检测 checkpoint 传给 CLI 的 `--pretrained`。
可换 `dinov2_vits14` 或 `dinov2_vitl14`，但也要使用对应权重并评估显存。

## 3. 开始训练

Windows CMD：

```bat
python tools/train_detector.py --data configs/fabric19.yaml --epochs 50 --batch 2 --val-batch 2 --num-workers 4 --amp bf16 --log-interval 20 --val-interval 1 --out runs/detector_dinov2
```

首次不传 `--resume` 或 `--pretrained`。应看到：

```text
DIRECT pretrained DINOv2 backbone=dinov2_vitb14 frozen=True encoder_input=448
initialization=official pretrained DINOv2; pyramid adapters / detector head=new
```

- `--batch`、`--val-batch` 都从 2 起步；OOM 时降为 1。
- `--num-workers 0` 用于排查读图／Windows worker 问题，正常可用 4。
- RTX 4070 可尝试 BF16；其他设备若不支持，使用 `--amp off` 或 FP16。
- `--log-interval 20` 控制 batch 日志，0 关闭。`--val-interval 1` 每轮验证，0 关闭。
- `--save-interval 5` 额外保留轮次快照；默认每轮覆盖 `last.pt`，改善时保存 `best.pt`。
- `--val-max` / `--normal-max` 用固定随机种子抽取评估子集，正式指标使用全部验证集。
- `--no-augment` 关闭训练增广，用于 8–32 张真实图的小样本过拟合排查。
- `--smoke N` 只替换成合成输入，**仍使用配置的真实 DINOv2 主干**，不是准确率实验，也不会绕过权重下载。

训练日志显示分项损失、mAP50、P/R/F1、TP/FP/FN 和可选正常图 FPR。
回归损失包含像素 L1，绝对值受缺陷大小影响，不应与其他模型的 loss 数字直接比较。

## 4. 恢复与重新开始优化

```bat
python tools/train_detector.py --resume runs/detector_dinov2/last.pt --epochs 50 --batch 2 --val-batch 2 --out runs/detector_dinov2
```

完整 `--resume` 恢复模型、optimizer、scheduler、随机状态、epoch 和 iteration；新 checkpoint
还保存 FP16 scaler 状态和 NumPy 全局 RNG。旧 checkpoint 若缺少 scaler 会明确警告并重置 loss scale。
`best.pt` 是权重文件，用 `--pretrained` 初始化；只有 `last.pt`／轮次快照可以完整恢复，
不完整的 `--resume` 会报错，不会静默从第 1 轮重训。
多 worker 的预取／worker 内部状态不保存在 checkpoint，恢复不保证逐 batch 位级一致。
保持同一结构、冻结设置、warmup、总训练计划。恢复旧 optimizer 后 `--lr` 不覆盖其学习率；
不要通过改 `--epochs` 假设 cosine 会自动重启。

如果要保留检测权重、重新设置优化计划：

```bat
python tools/train_detector.py --pretrained runs/detector_dinov2/last.pt --lr 0.0001 --epochs 30 --batch 2 --val-batch 2 --out runs/detector_dinov2_lr1e4
```

`--pretrained` 与 `--resume` 互斥。两者都仅接受结构匹配的完整 DINOv2 检测权重，严格加载；
恢复时不重复下载官方预训练权重，但仍需 hub 源码或本地源码目录。
若使用此前的直接 DINOv2 检测权重，需要改用当前清理后的配置、保持模块尺寸一致；
旧 CNN／学生蒸馏权重不能使用。六张量导出接口不兼容旧的异常图导出接口。

## 5. 精确率、阈值与模型选择

默认 AdamW `5e-4`，1 epoch warmup + cosine；ViT 冻结。
若要解冻，另开实验把 `dino_freeze: false`，使用 `--pretrained` 重新初始化 optimizer；
encoder 学习率是检测头学习率乘 `dino_lr_mult`（默认 0.1），显存会明显增加。

`train.best_metric` 支持 `best_f1`（默认）、`f1`、`precision` 或 `mAP50`。
只追求 precision 可能让召回过低，应同时检查 P/R。`best_f1` 的阈值不会自动成为推理阈值，
需要在验证后设置 `inference.final_score_thresh`。

```bat
python tools/evaluate.py --data configs/fabric19.yaml --ckpt runs/detector_dinov2/best.pt --diag --batch 2
python tools/predict.py --ckpt runs/detector_dinov2/best.pt --source D:/fabric_test/
```

推理有两层得分过滤：`model.score_thresh` 和 `inference.final_score_thresh`。
评估 CLI 的 `--val-conf` 仅调整 P/R 统计阈值，不会恢复已被过滤的框，也不改变 FPR 的出框阈值。
若需低阈值诊断，应在独立配置同时降低两层阈值；当前 mAP 也受模型前置过滤影响，
不是未经阈值截断的 COCO 全量评测。

`model.detector_nms_iou: 0.5` 抑制同类别重复框，0 关闭。
NMS 不能修复错分类、背景误报或定位不准，也不等于训练收敛改进。
同一 checkpoint 对比 NMS 和阈值时，应分开记录，避免把后处理收益归因于预训练主干。

## 6. 验证与导出边界

```bat
python -m unittest discover -s tests -v
python tools/bench_device.py --ckpt runs/detector_dinov2/best.pt --warmup 3 --iters 20
```

测试使用本地 tiny hub fixture，无需网络，覆盖反向传播、冻结、严格权重加载、恢复、
数据和中文路径、评估／预测／测速 CLI、同类 NMS、预测框原图坐标和六张量接口。
这些测试不证明真实数据精确率。官方大模型下载与真实训练需在用户机器执行。

实验性导出需另装 `onnx`：

```bat
python tools/export.py --ckpt runs/detector_dinov2/best.pt --out runs/export/dinov2_detector.onnx --check
```

输出仅 `cls3/reg3/obj3/cls5/reg5/obj5`，需外部解码、阈值与 NMS。
当前验证的是 PyTorch 和六张量接口；未完成真实 ViT ONNX/TensorRT/INT8 部署验证，
不宣称导出一定成功或达到任何实时速度。
