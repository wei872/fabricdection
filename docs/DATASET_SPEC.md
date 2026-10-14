# FADNet-RT 训练数据集规范（v2：以已标注 YOLO 数据集为主入口）

> v2 变更（2026-10）：真实数据源 = **用户已有的 YOLO 格式疵点标注数据集**；
> `data/` BMP 条带降为备选/演示源（模式 B）。免布种登记不变（design §3.5）。

## 0. 一句话流程

```
你的 YOLO 数据集 ──prepare_dataset.py --yolo──▶ 内部规范库 dataset_root ──▶ 三个训练/测试脚本
```
内部规范库结构与 v1 相同，**训练/测试脚本零改动**。

## 1. 对接入数据的要求（YOLO 源）

| 项 | 要求 |
|---|---|
| 布局 | 任选：`root/images + root/labels`；`root/{train,val}/images + labels`；有 `root/data.yaml` 或 `root/classes.txt` 更好 |
| 标签 | YOLO txt：`cls xc yc w h`（0~1 归一化、中心式）；**空 txt / 缺 txt = 正常图** |
| 类别名 | 通过 `classes.txt`/`data.yaml` 给出；与本项目 14 类的对应关系写 `configs/class_map.yaml`（别名支持中英文）；未映射类别会告警入报告 |
| 图像 | 尺寸任意（大图自动重叠切块）；灰度/彩色均可（内部转灰度三通道）；jpg/png/bmp/webp |
| 框 | 贴缺陷外接矩形；细长可旋转缺陷允许分段多框；**大缺陷（≥A_large，默认 50×50px）务必整框标全**（召回红线标定依赖） |
| 一图多疵 | 支持；每疵点独立框 |
| 笔迹/水印 | 若图上有人工圈划笔迹：提供同名 ignore 掩膜，或加 `--pen` 启发式检测（Q1 待确认前建议人工复核） |
| 数量门槛 | 每类真实框 ≥50（不足自动告警，可 `--inject-topup` 合成补足，合成:真实 ≤2:1）；正常图 ≥500 帧/布种（无标注图即正常图）；含大缺陷的图 ≥200（calib 用，从 val 自动挑） |

## 2. 内部规范库结构（prepare 输出，勿手改）

```
dataset_root/
├── {normal,abnormal,synth}/global/images/*.png     # 1280×1280 tiles
├── {abnormal,synth}/global/labels/*.txt            # tile 内归一化 YOLO 框
├── ignore_masks/global/*.png                       # 笔迹/跨块碎片忽略区（可选）
├── splits/{train,val}_normal.txt                   # U 分支（仅正常）
├── splits/{train,val}_det.txt                      # S 分支（abnormal+synth / 仅真实 abnormal）
├── splits/calib_large.txt                          # 含大缺陷的 val tiles（τ_large 标定）
├── fabric_meta.yaml                                # 免登记：仅来源追溯
├── data_ultralytics.yaml                           # ultralytics 入口（txt 清单式）
└── dataset_report.json                             # 质量报告：逐类框数/大缺陷数/面积p50/未映射类
```

## 3. 大图切块与跨块框转换（1280²、重叠 15%）

r = (框 ∩ tile) / 框面积：

| r | 处理 |
|---|---|
| ≥ 0.6（`--keep-ratio`） | 保留裁剪框 |
| 0.15–0.6 且交集 ≥32² | 保留裁剪框（partial，让模型见截断形态） |
| 0.02–0.15 | 进 ignore 掩膜（不学不报）；tile 无其他框时仍作正常样本（U 分支掩膜该区域） |
| < 0.02 | 丢弃 |
| 框在 tile 外但距边 <32px | 该 tile **不入正常集**（防缺陷近邻污染），直接跳过 |

重叠区的框在相邻 tile 各出现一次：训练有利；推理端 F-Head 跨块归并去重（IoU≥0.5）。

## 4. 划分与标定清单

- **按源图分层**划分 train/val（同一源图的 tiles 不跨 split，防泄漏）；贪心保证每类 val 框数 ≥ max(1, 15%×该类总数)。
- `val_det.txt` 只含**真实**标注（synth 不入 val，验证不掺水）。
- U 分支正常集 = 无框 tiles（含 ignore 掩膜者仅 U 可用，S 背景清单不含）。
- τ_u：val_normal 自归一化 z 的 p99（全局唯一，免登记）；τ_large：calib_large 上大缺陷最小 z −5% 裕量。
- 逐类置信度温度/严重度权重：val_det 匹配记录拟合（eval_metrics → class_calibration.yaml）。

## 5. 质量门（prepare 报告 + 人工复核）

1. `dataset_report.json` 逐类：boxes / large / area_p50 / 覆盖图数 —— 低于门槛的类补采或合成；
2. 未映射类别名清单 → 补 `class_map.yaml` 或扩 CLASSES（Q2 终版）；
3. 面积 p50 异常小的类（<32²）复核是否碎框误标；
4. ignore 掩膜抽检（笔迹 vs 真缺陷，Q1）；
5. val 集每类 ≥1 框，否则该类指标不可信（报告标红）。

## 6. 命令

```bash
# 主路径：你的 YOLO 数据集
python scripts/prepare_dataset.py --yolo /path/to/labeled --class-map configs/class_map.yaml \
    --min-per-class 50 --inject-topup --pen
# 备选：相机条带（无标注）
python scripts/prepare_dataset.py --src data --dst dataset_fabric --inject 1 --flatfield
```
