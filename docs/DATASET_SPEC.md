# FADNet-RT 训练数据集规范（v1）

双分支模型需要两类数据：**正常样本**（两分支都要用）与**带标注异常样本**（仅 S 分支监督 + 标定/评估用）。本规范给出落盘格式，与 `scripts/prepare_dataset.py` 的输出完全一致。

## 1. 目录结构

```
dataset_root/
├── fabric_meta.yaml                  # 布种登记（下详）
├── normal/<fabric_id>/images/*.png   # 正常 1280×1280 tiles（灰度存 3 通道亦可）
├── abnormal/<fabric_id>/images/*.png # 异常 tiles
├── abnormal/<fabric_id>/labels/*.txt # YOLO 框标注：cls xc yc w h（0~1 归一化）
├── abnormal/<fabric_id>/masks/*.png  # 可选：大缺陷像素掩膜（阈值标定/像素评估用）
├── ignore_masks/<fabric_id>/*.png    # 可选：笔迹/水渍等"忽略区"掩膜（Q1）
│                                     #   与 abnormal/images 同名；训练时该区域不计损失、不入记忆库
├── synth/det/images|labels/          # prepare_dataset.py --inject 生成的合成缺陷池
├── splits/
│   ├── train_normal.txt              # 相对路径清单（normal PNG 的 85%）
│   ├── val_normal.txt                # 剩余 15%（U 分支标定与阈值估计用！）
│   ├── train_det.txt / val_det.txt   # 含框标注的异常样本清单
│   └── calib_large.txt               # 大缺陷掩膜清单（τ_large 标定输入）
├── thresholds: 由 train_ustream.py 标定生成 → configs/thresholds.yaml（全局唯一，z 空间）
├── class_calibration: 由 eval_metrics 记录拟合 → configs/class_calibration.yaml（逐类置信度）
└── data_ultralytics.yaml             # ultralytics 训练入口（path/train/val/nc/names）
```

## 2. 布种台账 `fabric_meta.yaml`（**可选**，仅追溯用；模型与阈值不依赖它）

> 2026-10 修订：产线无法做布种登记。模型走免登记自适应（design §3.5）：
> 单一全局记忆库 + 全局阈值（自归一化 z 空间）。以下字段能填则填，填不了不影响训练/推理。

```yaml
KBA2605:            # fabric_id（从文件名前缀自动提取：KBA26050009_... → KBA2605）
  fabric_code: KBA2605            # MES 布种代码（可选）
  warp_density_per_cm: 38         # 经密（可选）
  weft_density_per_cm: 32         # 纬密（可选）
  yarn_thickness: fine            # coarse|medium|fine（可选）
  gloss: matte                    # matte|semi|glossy（可选，增强强度参考）
```

## 3. 标注规则（Q2 类别终版前的默认）

- 类别 id 与 `configs/data_fabric.yaml` 的 `names` 编号一致（0–13，其中 13=unknown_anomaly **不人工标注**）。
- 框贴缺陷外接矩形，可旋转细长缺陷允许分段多个框（不强制单框全景）。
- 大缺陷（当量 ≥ A_large，默认 50×50 px `[Q6]`）额外提供像素掩膜 → 入 `masks/`。
- 翻转增强时 `broken_warp⇄broken_weft` 类名联动（`sync_direction_class: true`）。
- 疑似笔迹（Q1）：统一画 `ignore_masks`，不要当缺陷标注，也不要留在正常样本里。

## 4. 各分支使用方式

| 分支 | 训练输入 | 验证/标定输入 |
|---|---|---|
| U 重构（Dinomaly） | `splits/train_normal.txt`（仅正常，含 ignore 掩膜剔除） | `splits/val_normal.txt` → 全局 τ_u（z 空间 p99） |
| U 记忆库（PatchCore） | 同上（style_center 后入**全局** Coreset，无梯度） | 同上；异常样本只用于 τ_large 标定 |
| S 检测（YOLO27s 风格） | `abnormal + synth + train_normal(背景)` | `splits/val_det.txt` |

## 5. 数量建议（起步门槛）

- 正常：每一布种 ≥ 500 帧（建议 ≥ 2000，覆盖一卷布的不同区段与照明漂移；
  tiles 由 2440 条带切块，1 帧条带 ≈ 2 tile，量很快能凑齐）。
- 异常：每一缺陷类 ≥ 50 个真实框（达不到的类由合成引擎补足，合成:真实 ≤ 2:1）。
- 验收集：含大缺陷的代表性帧 ≥ 200 帧（覆盖每一处"大缺陷"类型 ≥ 20 例），单独保留不参与训练。

## 6. 从相机原始数据到规范格式

```bash
# data/ 中的 BMP 条带 → dataset_root/（ROI+平场+切块+划分+可选合成缺陷注入）
python scripts/prepare_dataset.py --src data --dst dataset_fabric \
    --tile 1280 --overlap 0.15 --inject 2 --flatfield
```
