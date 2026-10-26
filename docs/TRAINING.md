# 训练与数据准备手册

> 配套设计说明见 [DESIGN.md](DESIGN.md)。所有命令在仓库根目录执行。

## 1. 环境

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# 可选
pip install onnx tensorrt ultralytics
# 阶段 0 用 DINOv2 teacher：需要 torchvision + 可访问 torch.hub；
# 内网机器请 clone https://github.com/facebookresearch/dinov2 并下载 backbone-only 权重。
```

最低验证环境（本仓库实测）：Python 3.11 + torch 2.2.2 (CPU 即可) +
numpy 1.26 + opencv-python-headless + PyYAML + scipy + tqdm。
跑 `python -m unittest discover -s tests -v` 应输出 `OK`（23 项）。
阶段 0 的 `--smoke` 模式用 synthetic teacher，不需要下载 DINOv2 权重。

## 2. 数据准备

### 2.1 目录约定

你的 `dataset/dataset.yaml` 里是 Windows 绝对路径：

```yaml
train: D:/tyhn/yolo/dataset/source5_augmented/images/train/
val:   D:/tyhn/yolo/dataset/source5_augmented/images/val/
normaltrain: D:/tyhn/yolo/dataset/source5_augmented/normal/images/train/
normalval:   D:/tyhn/yolo/dataset/source5_augmented/normal/images/val/
```

推荐整理成（相对路径，跨平台）：

```
dataset/
  images/train/   images/val/              # 缺陷图
  labels/train/   labels/val/              # YOLO .txt，每行: cls cx cy w h (归一化 0~1)
  normal/images/train/  normal/images/val/ # 纯正常图，无标签
```

然后修改 `configs/fabric19.yaml` 的四个路径。类别名已与 `dataset/dataset.yaml` 完全一致（19 类）。
**原始 `dataset/dataset.yaml` 不要改**，它作为类别真源保留。

### 2.2 上线前的数据检查

```bash
# 1) 图片与标签一一对应；2) 每类样本数；3) bbox 面积分布（决定"大缺陷"阈值）
python - <<'PY'
from pathlib import Path
from collections import Counter
import numpy as np
root = Path('dataset')
for split in ('train','val'):
    imgs = list((root/'images'/split).glob('*.jpg'))
    labs = list((root/'labels'/split).glob('*.txt'))
    print(split, 'images', len(imgs), 'labels', len(labs),
          'missing', len({p.stem for p in imgs} - {p.stem for p in labs}))
    c, areas = Counter(), []
    for f in labs:
        for line in f.read_text().splitlines():
            p = line.split()
            if len(p) >= 5:
                c[int(float(p[0]))] += 1
                areas.append(float(p[3]) * float(p[4]))     # 归一化面积
    a = np.array(areas)
    print('  per-class:', dict(sorted(c.items())))
    print('  area ratio quantiles:', np.round(np.quantile(a, [.5,.9,.95,.99]), 4))
PY
```

若 `area ratio` 的中位数远小于 `0.001`，说明大缺陷很少，
需要在 `configs/fabric19.yaml` 里调 `large_area_ratio`（同步改两个
`jingwei_*.yaml` 的 `fusion.large_area_ratio`）。

### 2.3 normaltrain 的关键要求

- **必须只含真正无缺陷的图**，包含所有布种、所有光照/反光状态；
- **必须包含布幅变化和边缘正常折入样本**（真实图或 `data.py` 合成增广），
  否则前景/折入上下文学不到正常流形，产线上容易把折入误判成疵点；
- 建议每个布种 ≥ 200 张；新布种上线只需追加图片并重跑阶段 C；
- `normalval` 与 `normaltrain` 不重叠，用于阈值标定（目标 FPR = 0.5%，
  折入子集单独看 `FPR@fold`）。

## 3. 阶段 0（推荐）+ 五阶段训练

**从零训练效果通常较差**，建议先跑阶段 0：用冻结的 DINOv2 当离线 teacher，
把它的密集 patch 特征蒸馏进我们的轻量 CNN。teacher 只在训练时存在，
推理图 / MACs / 1660S 实时性完全不变。完整说明见
[DINO_PRETRAIN.md](DINO_PRETRAIN.md)。

```bash
# 阶段 0：DINOv2 离线蒸馏（只用 normaltrain，无标签）
python tools/pretrain_dino.py \
    --cfg configs/jingwei_n.yaml --data configs/fabric19.yaml \
    --teacher dinov2_vits14 --iters 5000 --out runs/dino
# 无数据 / 无网络自检：python tools/pretrain_dino.py --smoke 4 --iters 2

# 阶段 A：有监督检测（train + labels），从阶段 0 权重热启动
python tools/train_detector.py \
    --cfg configs/jingwei_n.yaml --data configs/fabric19.yaml \
    --resume runs/dino/dino_init.pt \
    --epochs 150 --batch 8 --out runs/detector

# 阶段 A 同时训练 defect_fg 前景头（YOLO 框级弱监督）；data.py 也会以 25%
# 概率合成布幅边缘正常折入增广，让 normaltrain / detector 都把它见成背景。
# 详见 docs/FOREGROUND.md。
# 阶段 B：无监督正常性（normaltrain，主干冻结）
python tools/train_normality.py \
    --cfg configs/jingwei_n.yaml --data configs/fabric19.yaml \
    --ckpt runs/detector/last.pt --iters 10000 --out runs/normality

# 阶段 C：构建布种条件记忆库 + 阈值标定（normaltrain + normalval）
python tools/build_memory_bank.py \
    --cfg configs/jingwei_n.yaml --data configs/fabric19.yaml \
    --ckpt runs/normality/normality.pt --out runs/bank

# 阶段 D：融合 / 安全网阈值网格搜索（val + normalval）
python tools/calibrate_fusion.py \
    --cfg configs/jingwei_n.yaml --data configs/fabric19.yaml \
    --ckpt runs/bank/memory_bank.pt --max-val 500 \
    --out configs/fusion_calibrated.yaml

# 阶段 E：评估 + 导出
python tools/evaluate.py  --cfg configs/jingwei_n.yaml --ckpt runs/bank/memory_bank.pt
python tools/export.py    --cfg configs/jingwei_n.yaml --ckpt runs/bank/memory_bank.pt \
    --proto 0 --check --out runs/export/jingwei_n.onnx
```

把 `configs/fusion_calibrated.yaml` 里的 `fusion.anomaly_thresh` /
`fusion.final_score_thresh` 回填到 `configs/jingwei_n.yaml` 后再导出。

### 3.1 阶段 A 细节

- `model.set_stage("detector")` 跳过正常性分支，训练更快；
- 损失 = o2m 密集 + o2o 匈牙利 + 前景对齐（见 DESIGN.md §7）；
- 优化器 AdamW + cosine，梯度裁剪 5.0；
- 若某类 AP 特别低，在 `train.class_weights` 里填 19 个权重（如按 `1/sqrt(样本数)` 归一化）；
- **建议**：先用 `normaltrain` 里的图做"纯负样本"加入训练（无框），可显著降低 FPR。

### 3.2 阶段 B 细节（最容易出错）

- 主干 / neck / 头 / ROI **全部冻结**，BN 置 eval；
- 只训练 `PatchFeatureHead`、`NoisyBottleneck`、4 个 `VitBlock`、输出投影、融合门；
- 损失 `HardMiningCosineLoss` 的 `p` 在前 1000 iter 从 0 升到 0.9；
- 如果 `rec_loss` 不下降或降到 ~0：检查是否有参数漏冻结（塌缩解）；
- 可选合成异常增广：在正常图上随机 CutPaste / 画细线（模拟断纬）/ 局部抹平（模拟稀缝），
  但**不要用真实缺陷图**训练 B 分支，否则会失去"发现新异常"的能力。

### 3.3 阶段 C 输出

```
runs/bank/
  fabric_prototypes.npz     # K 个布种原型中心 + 描述符归一化统计量
  memory_bank.pt            # centroids / mu / sigma / tau / 重建残差标定 + 完整权重
```

日志示例（实测 smoke）：

```
prototype  1:    6400 patch features -> 256 centroids
prototype  1: mu=0.18120 sigma=0.06895 tau=2.755
prototype  0: mu=0.18852 sigma=0.21816 tau=3.703 (conservative fallback)
reconstruction: tau=1.03675 slope=40.59
```

`(conservative fallback)` 表示 `normalval` 里没有该布种，用最保守阈值兜底，属正常现象。

### 3.4 阶段 D 细节

网络对 val / normalval **只前向一次**并缓存特征，之后 15 组阈值只在缓存上跑
decode + 安全网，秒级完成。目标函数：

```
maximise   Recall@large - 2·max(0, FPR - target_fpr)
subject to FPR ≤ max(3·target_fpr, 0.02)
```

若没有一组满足约束，说明 `target_fpr` 设得太严，或 `normaltrain` 污染 /
布种覆盖不足（回到 §2.3）。

## 4. 快速自检（无真实数据）

```bash
python -m unittest discover -s tests -v                 # 23 项，约 41 s
python tools/benchmark.py --all                         # p/n/s 参数量 / MACs / 延迟估算
python tools/bench_device.py --cfg configs/jingwei_p.yaml --precision both  # 在你的 GPU 上实测
python tools/train_detector.py   --smoke 2 --epochs 1 --batch 1 --device cpu
python tools/train_normality.py  --smoke 4 --iters 2 --batch 2 --device cpu
python tools/build_memory_bank.py --smoke 4 --device cpu
python tools/calibrate_fusion.py --smoke --device cpu
python tools/evaluate.py --smoke --device cpu
python tools/export.py --check --out runs/export/jingwei_n.onnx
```

`--smoke` 用随机权重 + 合成织纹图，只验证"代码能跑通"，
其 mAP/recall 为 0 是预期的。

## 5. 常见故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `no images under ...` | 路径没改 / 图片扩展名不在列表 | 检查 `configs/fabric19.yaml`；支持 jpg/jpeg/png/bmp/tif |
| 标签全空 | `labels` 目录推断失败 | 显式传 `--labels`（工具会自动把 `/images/` 换成 `/labels/`） |
| `rec_loss` → 0 | 主干没冻结，出现塌缩解 | 检查阶段 B 只把 `normality_parameters()` 加入优化器 |
| 全图误报 | 记忆库没标定 / 用了未训练权重 | 必须先跑阶段 C；`tau` 由 `normalval` 标定 |
| 大缺陷仍漏 | `anomaly_thresh` 太高 / `large_area_ratio` 太大 | 跑阶段 D；降低 `large_area_ratio` |
| OOM | 1280² 训练 batch 太大 | 减小 `detector_batch`；开 AMP（`--amp`，可自行加） |
| ONNX 导出失败 | 缺 `onnx` | `pip install onnx`；注意保持 `numpy<2`（torch 2.2 要求） |
| `AttributeError: config has no key ...` | yaml 缺键 | 对照 `configs/jingwei_n.yaml` 补齐 |
| 阶段 0 `torch.hub.load` 失败 | 内网无法访问 GitHub / 权重 URL | clone DINOv2 到本地，用 `--repo /path/to/dinov2 --weights /path/to/xxx.pth`；见 [DINO_PRETRAIN.md](DINO_PRETRAIN.md) |
| 阶段 0 OOM | teacher 太大 / crop / batch 太大 | 换 `dinov2_vits14`、`--batch 2`、`--crop 224`；推理图不受影响 |
| 阶段 0 loss 不降 | iters 太少 / normaltrain 布种单一 | 先跑 5000 iters；确认 normaltrain 覆盖全部布种与光照 |
| 布边正常折入仍误报 | `edge_band` 太窄 / `defect_fg` 把折入当前景 | 增大 `edge_band_px` / `edge_attenuation`；补充 normaltrain 折入图；见 [FOREGROUND.md](FOREGROUND.md) |
| 真实边缘缺陷被压掉 | 抑制过强 | 降低 `edge_attenuation`、提高 `edge_peak_override`，或增加带框边缘缺陷样本训练 `defect_fg` |
| 1660 SUPER 上 FP16 比 FP32 还慢 | TU116 能用 TensorRT，但没有 Tensor Core；TRT FP16 可能落到慢 kernel | 用 `--precision both` 实测，取快的；先构建 FP32 engine 再 A/B FP16，见 [DEPLOYMENT.md §4.1](DEPLOYMENT.md) |
| 1660 SUPER 上不到 40 FPS | n/s 变体算力过大 | 换 `configs/jingwei_p.yaml`（8.42 G MACs） |
| 后处理占了 3–4 ms | decode + 连通域在 CPU 上 | 用 uint8 输入、`top_k` 降到 100、`out_grid` 降到 96 |
