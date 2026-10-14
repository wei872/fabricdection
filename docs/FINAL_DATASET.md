# FADNet-RT 最终数据集规格（唯一权威版 v1.0）

> 本文定义**训练/测试脚本直接消费的最终数据集**长什么样：目录、文件、格式、
> 数量、质量门。由 `scripts/prepare_dataset.py` 从你的 YOLO 源（source5）生成；
> 人工需要保证的是"源数据 + 数量 + 标注质量"，其余机器生成。
> 相关：接入流程见 `DATASET_SPEC.md`；模型设计见 `FADNet-RT_design.md`。

---

## 1. 目录树（最终形态）

```
dataset_fabric/                              # ← 训练根目录（--root / data 配置指向这里）
│
├── normal/global/images/                    # 【U 分支】正常样本，仅图片
│   ├── src5_train_003_t00_0_0.png           #   1280×1280，灰度存 3 通道 PNG
│   ├── src5_train_014_t01_1088_0.png        #   命名：<源图stem>_t<块号>_<x>_<y>
│   └── ...                                  #   ≥ 1000 张（见 §4）
│
├── abnormal/global/images/                  # 【S 分支监督 + 评估】缺陷 tiles
│   ├── src5_train_000_t00_0_0.png
│   └── ...
├── abnormal/global/labels/                  #   与 images 同名 .txt，YOLO 归一化框
│   ├── src5_train_000_t00_0_0.txt           #   例：0 0.512500 0.335000 0.045000 0.090000
│   └── ...
│
├── synth/global/images/                     # 【可选】合成缺陷补足池（缺量类才生成）
│   └── synth_hole_00042.png
├── synth/global/labels/
│   └── synth_hole_00042.txt
│
├── ignore_masks/global/                     # 【可选】笔迹/跨块碎片忽略区，uint8 0|255
│   └── src5_train_014_t01_1088_0.png        #   与 normal/abnormal 图片同名同尺寸
│
├── splits/                                  # 清单（绝对路径，一行一个）
│   ├── train_normal.txt                     #   U 分支训练：normal 的 train 部分
│   ├── val_normal.txt                       #   U 分支标定 τ_u：normal 的 val 部分
│   ├── train_det.txt                        #   S 分支训练：abnormal(train)+synth
│   ├── val_det.txt                          #   S 分支/验收评估：仅真实 abnormal(val)
│   └── calib_large.txt                      #   τ_large 标定：含大缺陷的 val tiles
│
├── data_ultralytics.yaml                    # S 分支训练入口（txt 清单式）
├── fabric_meta.yaml                         # 免登记：仅来源追溯信息
├── dataset_report.json                      # 质量报告（§6 门禁依据）
│
└── （标定产物，训练后回写 configs/，随数据集版本管理）
    configs/thresholds.yaml                  #   全局 τ_u / τ_large（z 空间，免登记）
    configs/class_calibration.yaml           #   逐类置信度温度 + 严重度权重
```

**谁读什么**：

| 脚本 | 读 |
|---|---|
| `train_ustream.py --root dataset_fabric` | splits/train_normal（训练）、splits/val_normal（标定）、ignore_masks |
| `train_sstream.py --data dataset_fabric/data_ultralytics.yaml` | splits/train_det / val_det（经 yaml） |
| `test_e2e.py` | configs/thresholds.yaml、class_calibration.yaml、记忆库/解码器权重 |
| `eval_metrics.py --root dataset_fabric` | abnormal/*/labels（GT）+ verdicts.jsonl |

---

## 2. 文件格式细则

### 2.1 图片（normal / abnormal / synth）
- 尺寸：**1280×1280**（源大图切块、小图反射补齐）；
- 通道：灰度内容存 3 通道 PNG（线扫单色相机；彩色源自动转灰度）；
- 已做：ROI 裁切可选、**不做平场**（平场在推理管线在线做，数据集保留原始光照分布以供增强学习）；
- 同源图增强副本不得跨 train/val。

### 2.2 标签 txt（abnormal、synth）
- 每行 `cls xc yc w h`，中心式、按 1280 归一化 6 位小数；空文件不允许出现在 abnormal；
- `cls` 取值 = 下表 id；**13 号 unknown_anomaly 永不人工标注**（推理期 F-Head 产出）；
- 一疵点一框；细长可旋转缺陷允许沿走向分段多框；跨块裁剪框按 §3 规则保留。

| id | 类名 | 中文 | 标注要点 |
|---|---|---|---|
| 0 | broken_warp | 断经 | 经向断头线段 |
| 1 | broken_weft | 断纬 | 纬向断头线段 |
| 2 | thin_thick_weft | 稀/密纬（档） | 纬密异常带 |
| 3 | double_pick | 双纬 | |
| 4 | slub | 竹节 | |
| 5 | skip_draw | 跳花/错花 | |
| 6 | hole | 破洞 | 含破边洞 |
| 7 | knot | 结头 | |
| 8 | stain_oil | 油污渍 | 边界取可见污染外沿 |
| 9 | foreign_fiber | 异纤/飞花 | 线状异物贴线宽 |
| 10 | streak_bar | 条影/经柳纬档 | |
| 11 | edge_defect | 边疵 | |
| 12 | other_known | 其它已知 | 评审可细分时拆类（Q2） |
| 13 | unknown_anomaly | 未知缺陷（兜底） | **不标注** |

### 2.3 ignore_masks
- uint8 灰度 PNG，255=忽略；来源：笔迹圈划（Q1）、跨块碎片框（r∈[0.02,0.15)）；
- U 分支训练/记忆库入库时掩膜替换；S 分支清单不含带 ignore 的正常图。

### 2.4 splits 清单
- 绝对路径、LF 换行、无空行；train/val 互斥；val_det 只含真实标注。

### 2.5 data_ultralytics.yaml（生成示例）

```yaml
path: /srv/dataset_fabric
train: [/srv/dataset_fabric/splits/train_det.txt]
val:   [/srv/dataset_fabric/splits/val_det.txt]
nc: 14
names: {0: broken_warp, 1: broken_weft, 2: thin_thick_weft, 3: double_pick,
        4: slub, 5: skip_draw, 6: hole, 7: knot, 8: stain_oil, 9: foreign_fiber,
        10: streak_bar, 11: edge_defect, 12: other_known, 13: unknown_anomaly}
```

### 2.6 dataset_report.json（生成示例·截选）

```json
{"source": "D:/tyhn/yolo/dataset/source5_augmented",
 "unmapped_classes": {},
 "images": {"train": 4200, "val": 780},
 "tiles": {"tiles_normal": 5100, "tiles_abnormal": 3900, "normal_with_ignore": 120},
 "classes": {"broken_warp": {"boxes": 610, "large": 42, "area_p50": 3800, "images": 585},
             "hole":        {"boxes": 240, "large": 96, "area_p50": 9000, "images": 231}}}
```

---

## 3. 源 → 最终的转换规则（机器执行，人工知悉）

- 切块 1280²、重叠 15%；跨块框 r=框∩tile/框：≥0.6 保留；0.15–0.6 保留截断框；
  0.02–0.15 → ignore 掩膜；<0.02 丢弃；框距块边 <32px 的块**不入 normal**；
- 无标签图 → normal；源 train/val 划分**原样尊重**；正常图无 val 时自动留置 15%；
- 大缺陷（面积 ≥ `large_area`，默认 2500px² ≈50×50）所在 val tile → calib_large.txt。

---

## 4. 数量门槛（最终数据集验收线）

| 集合 | 门槛 | 说明 |
|---|---|---|
| normal 总量 | ≥1000 tiles（每布种 ≥500 帧源图） | 覆盖在产全部布种与光照漂移；建议 2000+ |
| val_normal | ≥50 tiles 且覆盖全部在产布种 | τ_u 标定用 |
| train_det 每类真实框 | ≥50 | 不足 → 补采优先，合成补足（synth:真实 ≤2:1） |
| val_det | ≥200 tiles，每类 ≥ max(1, 15%×该类 train 框数) | 只含真实标注 |
| calib_large | 大缺陷 ≥200 例、每类大缺陷 ≥20 | τ_large 与"大缺陷召回=100%"验收用 |
| 每类 large 字段 | 报告中非零（常见类） | 零 large 的类说明标注或采样有偏 |

---

## 5. 标注质量规则（人工）

1. 框贴缺陷**可见外沿**，不松不紧（松框 >2px 边距会拉低 IoU 验收）；
2. 大缺陷**整框标全**，不许只标最重局部（召回红线标定依赖）；
3. 笔迹/粉笔圈划 ≠ 缺陷：给 ignore 掩膜或 `--pen`，**不得当缺陷标**（Q1）；
4. 同一物理缺陷在重叠块中各自成框（机器处理），人工不要手工复制框；
5. 类名统一走 `configs/class_map.yaml` 别名表；新增类先评审（Q2）再进 CLASSES；
6. 增强集（source5_augmented）：同源图副本限同一 split。

---

## 6. 开训前门禁（全过才进 M1）

- [ ] `dataset_report.json`：unmapped_classes 为空；每类 boxes ≥50（或已批准合成补足）；
- [ ] val_det 每类 ≥1 框；calib_large ≥200；
- [ ] ignore 掩膜抽检 50 张无误（笔迹 vs 真缺陷）；
- [ ] 面积 p50 <32² 的类已复核（碎框误标）；
- [ ] 人工抽查 train/val 各 100 tiles 框质量 ≥98% 合格；
- [ ] `train_ustream` 标定产出 thresholds.yaml（tau_u 非 provisional）；
- [ ] 阴影模式基线：val_det 上大缺陷召回 = 100%（IoU0.3）、已知类 Acc ≥96%。

---

## 7. 从 source5 生成（一条命令）

```bash
# configs/dataset.yaml 已指向 D:/tyhn/yolo/dataset/source5_augmented
python scripts/prepare_dataset.py
# 检查 dataset_fabric/dataset_report.json → 过 §6 门禁 → 开训
```
