# 代码审查与优化建议（v0.2.4）

> 审查对象：`jingwei/`（推理库）+ `tools/`（训练/评估/部署 CLI）+ `tests/` + `configs/`。
> 审查方式：逐文件静态审查 + 全部单元测试实跑 + 针对性探针脚本（下面每条结论都附复现方式或实测数字）。
> 结论：**架构与工程质量已经不错，但存在 1 个会让"未知异常兜底"整体失效的 P0 级标定缺陷**，
> 以及若干 P1 级坐标/平台/多进程/依赖兼容问题。本轮已修复 24 项（含 5 个 P0），剩余 13 项
> （其中 2.13 汇总了 8 个低风险小项）按优先级列在第 2 节。
> **第 0.5 / 0.6 节专门回答「precision 为什么涨不上来」**（v0.1.9 第一轮 + v0.2.0 第二轮），
> 并给出定位工具 `tools/diagnose_precision.py`。

---

## 0. 环境与基线

| 项目 | 值 |
|---|---|
| 沙箱 | 2 vCPU / 3 GB RAM / 无 GPU（只为复现正确性与相对性能，不代表产线速度） |
| Python / Torch | 3.11.2 / 2.4.1+cu121（CPU 执行） |
| 基线（改动前） | `python -m unittest discover -s tests -v` → **Ran 35 tests in 123.6 s — OK (skipped=1)** |
| 改动后 | **Ran 51 tests in 119 s — OK**（新增 16 项回归测试；装上 `onnx` 后 ONNX 导出测试不再 skip） |
| 规模 | `jingwei_n` 6.36 M 参数 / 16.62 G MACs @1280；`jingwei_p` 2.93 M / 8.09 G MACs |

只装 `requirements.txt` 时 `test_19_pico_onnx_export` 会被 skip（`onnx` 只是可选依赖），也就是说**导出路径在默认环境里其实从未被验证**，见 2.12。

---

## 0.5 专题：为什么 precision 涨不上来（v0.1.9，P0）

用户实测（stage A，`--val-conf` 默认 = `final_score_thresh` = 0.15）：

```
epoch 1  mAP50=0.1336  P=0.0927  R=0.2956  large_recall=0.3876  normal_fpr=0.0375
epoch 2  mAP50=0.2457  P=0.0828  R=0.6101  large_recall=0.6873  normal_fpr=0.3125
epoch 3  mAP50=0.3306  P=0.0712  R=0.7327  large_recall=0.8078  normal_fpr=0.0250
```

recall / mAP50 / large_recall 都在稳步上涨，**只有 precision 在掉**。这不是模型没学好，
而是三处"只关心召回、不关心精度"的机制叠加，其中第 1 条是决定性的。

### 原因 1（决定性）：decode 没有做峰值抑制 → 一个缺陷变成上百个框

`jingwei/head.py::NMSFreeDetect.decode` 原来只有 `top_k + score_thresh`。但 o2o 分支的 objectness
在缺陷周围是**一片高原**，不是尖峰：

* `build_dense_targets` 给 o2m 分支的正样本是 **GT 中心 ±1.5·stride 的整片区域**（1280 输入、stride 8 时约 7×7=49 个 cell）；
* 两个分支共享 `cls/reg` 特征，所以 `cls` 概率同样是一片高原；
* decode 之前把高原上**每一个** cell 都解码成一个框。

实测（`tests/test_model.py::test_44`，一个 12×12 的自信缺陷区域）：

```
boxes for ONE defect blob: 144        # 分数全是 0.935
```

于是 `precision = TP/(TP+FP) = 1/144 ≈ 0.7%`——**即使定位完全正确，precision 也不可能超过 1%**。
这也正好解释了"recall 涨、precision 不涨"：多放框只会给 TP 增加机会，同时按同样速率堆积 FP。

修复：`model.peak_filter`（默认开）在 decode 里做 **3×3 局部极大值过滤**：

```python
ranked = obj_2d - 1e-6 * flat_index      # 平局用索引破，饱和/常量高原只留一个峰
pooled = F.max_pool2d(ranked, k, 1, k // 2)
score  = score.masked_fill(ranked < pooled, -1.0)
```

* 同一个缺陷：**144 → 1 个框**（`test_44`），两个分开的缺陷仍然是 2 个框（`test_45`）；
* 仍然 **NMS-free**：没有数据相关的循环、没有 IoU 矩阵、延迟仍然是静态上界；
* 在**图外**执行（`decode` 本来就在 TensorRT 引擎之外），导出图不变；
* `peak_filter: false` / `peak_kernel: 1` 可一键回退做 A/B。

### 原因 2：模型选择指标本身不含 precision

`tools/train_detector.py` 原来：

```python
score = 0.5 * val_metrics["mAP50"] + 0.5 * val_metrics["large_recall"]   # 两项都被"多出框"单调改善
```

`large_recall` 是纯召回；`mAP50` 对每帧框数基本免疫（排序不变、AP 不变），却同样不会因为
precision 变差而下降。于是 `best.pt` 永远是"最激进"的那个 epoch，`large_recall=0.81` 就这么被选中了，
而它的 precision 只有 0.07。

修复：

```python
score = best_f1 - 2 * max(0, normal_fpr - target_fpr)     # best_f1 是阈值无关的 P/R 联合指标
```

启动日志会打印这条公式；`large_recall` / `mAP50` 仍然原样打印，只是不再单独决定 `best.pt`。

### 原因 3：阶段 D 的标定目标会把 precision 再卖一次

`tools/calibrate_fusion.py` 原来 `score = large_recall - 2*max(0, fpr - target_fpr)`，
约束只有"整图 FPR ≤ 0.5%（放宽到 2%）"。这个目标对 `final_score_thresh` 单调 →
网格搜索必然选**最低阈值 0.10**，实测 precision ≈ 0.1，等于把刚修好的精度又换回去了。

修复：每个候选都实测 val precision，默认按 **F1** 排序（`--objective f1`），
新增 `--min-precision` 硬约束，输出 yaml 里写入 `val_precision / val_f1`。

### 原因 4（新增诊断能力）：以前根本看不出 FP 的来源

每个 epoch 现在多打一行：

```
val   epoch   3/150  mAP50=... precision=... recall=... best_f1=...@0.28 large_recall=...
      boxes/img= 12.4  FP: duplicate=812 wrong_class=301 localization=88 background=120  (best operating point: conf>=0.28)
```

读法（这直接决定下一步该改什么）：

| 主导项 | 含义 | 对策 |
|---|---|---|
| `duplicate` 高 | 同一缺陷多个框（峰值抑制/阈值） | `model.peak_filter: true`、提高 `final_score_thresh` 到 `best operating point` |
| `wrong_class` 高 | 框对了但类别错 | 打开 `train.class_weights`（19 类长尾），或用阶段 D 的 ROI 头重分类 |
| `localization` 高 | IoU 落在 0.1–0.5 | 回归/DFI 侧：提高 `box_*` 权重、检查 `reg_scale` |
| `background` 高 | 纯误报（布面正常纹理） | 2.3 节的"正常图入训练"（hard negative）+ 阶段 B/C 的异常图门控 |

### 复现与验证

```bash
# 一个缺陷 -> 1 个框（修复后）；144 个框（peak_filter: false）
python -m unittest tests.test_model.TestModel.test_44_peak_filter_prevents_duplicate_boxes -v
python -m unittest tests.test_model.TestModel.test_45_peak_filter_keeps_separate_defects -v
python -m unittest tests.test_model.TestModel.test_46_selection_metric_is_precision_aware -v
python -m unittest tests.test_model.TestModel.test_47_fusion_calibration_ranks_by_f1 -v
```

### 重训后应该看到什么

* `boxes/img` 从几十降到个位数（每条产线帧的真实缺陷数），`duplicate` 应接近 0；
* 在相同 `--val-conf` 下 precision 应从 ~0.1 跳到 **0.4–0.8**（取决于类别是否可分）；
* `best_f1` 与 `best operating point` 会给出真正的上线阈值；把 `--val-conf` 设成它再复看 precision。

如果 precision 仍低，按上表看 `wrong_class` / `background` 哪一项主导——那才是下一步要改的东西
（分类权重 or 正常图 hard negative），而不是继续调阈值。

> 顺带解释你日志里的 `unknown_boxes=0`：stage A 用 `set_stage('detector')` 显式关掉了 normality 分支
> （记忆库和重建器此时都未训练，其异常图近似常量），所以安全网在 stage A 不产生框——这是设计如此，
> 不是 bug。安全网要等阶段 B/C 之后才会出现。

---

## 0.6 专题：precision 第二轮（v0.2.0）——「还是上不去」的其余三条通路

v0.1.9 修掉了"一个缺陷 144 个框"之后，如果 precision 仍上不去，剩下的只可能是下面三条之一。
这一版把三条都堵上，并且**给了一个一条命令定位到底是哪一条**的工具。

### 通路 A：同一缺陷被 P3 和 P5 各报一次（跨尺度重复）

峰值过滤是**逐尺度**的。一个真实缺陷在 P3 上是一个小框、在 P5 上是一个大框——
两者是**嵌套**关系，IoU 只有 ~0.05（小框面积 / 大框面积），所以任何 IoU-based NMS 都不会删。
结果：一个缺陷 = 2 个框 = 1 TP + 1 FP。

实测（`tests/test_model.py::test_49`）：

```
dedup=True   -> 1 box     [476,476,492,492]
dedup=False  -> 2 boxes   [476,476,492,492] + [400,400,656,656]      # 同一缺陷，两级尺度
两个相距很远的缺陷 -> 2 boxes（不会被误删）
```

修复：`decode` 末尾加跨尺度去重（`model.dedup`，默认开）：
复制判定 = `IoU > dedup_iou(0.70)` **或** `交/较小面积 > dedup_contain(0.85)`——
后者专门抓"小框套在大框里"的嵌套重复。贪心按分数保留，框数 ≤ `top_k`，仍是固定上界、
仍在导出图之外，**依然没有 NMS**。

### 通路 B：92% 的像素从未学过「这是背景」

`train.obj_o2o` 原来是 OHEM：只对最难的 `o2o_neg_ratio(2%)`，且有 `o2o_min_neg=2048` 兜底，
即 P3 的 25600 个 cell 里只有 **2048 个（8%）** 拿到过梯度。实测（`test_50`）：

```
ohem   obj_o2o=0.0564  grad 非零 cell =  2049/25600   grad@远处背景 = +0.000e+00
focal  obj_o2o=3.0260  grad 非零 cell = 25600/25600   grad@远处背景 = +1.301e-05
```

也就是说：一块**简单背景**（p≈0）的 cell 在 OHEM 下**永远拿不到"你是背景"的梯度**。
如果它因为某次更新漂到 objectness ≈ 0.9，就再也没有东西把它拉回来 —— 这类误报**无法用阈值消除**
（它和真缺陷的分数一样高），正是 `background` 类 FP 的来源之一。

修复：`train.obj_o2o_mode: focal`（默认）—— 对**全部** cell 做 sigmoid focal loss（γ=2, α=0.25），
按正样本数归一化，这就是 RetinaNet 的标准做法。`ohem` 保留用于 A/B。
⚠️ focal 的绝对量级约为 ohem 的 ~50 倍，所以三个配置里的 `loss_weights.obj_o2o` 已从 **1.5 → 0.5**。

### 通路 C：损失被 cls_o2m 吞掉，回归/objectness 几乎没在学

新加的 `loss share` 行（每个 epoch 打印一次）在 smoke 上就直接暴露了问题：

```
epoch   1/1  cls_o2m=68.215 | box_o2m=17.290 | ... | obj_o2o=1.062 | ...
      loss share: cls_o2m=50%  box_o2m=25%  box_o2o=23%
```

`obj_o2m` / `obj_o2o` 两项加起来不到 **1%**。objectness 就是部署时真正决定"这帧有没有东西"的量，
它几乎没有梯度 —— 这与"precision 上不去、recall 却能涨"的现象完全吻合（框能出来，但置信度/背景区分没有在学）。

**先看这一行**：如果 `obj_o2o` 占比 < 2%，就把 `train.loss_weights.obj_o2o` 抬高
（focal 模式下建议 1.5–3.0），或者把 `cls_o2m` 从 1.0 降到 0.3–0.5 再看一轮。
这一条是**超参问题，不需要改代码**，但以前没有任何地方能看见它。

### 一条命令定位：`tools/diagnose_precision.py`

```bash
python tools/diagnose_precision.py --cfg configs/jingwei_p.yaml \
    --ckpt runs/detector/best.pt --max-val 300
```

对给定 checkpoint 直接回答六个问题（每一节末尾给出"下一步改什么"）：

0. **门控存活**：全 val 集上 decode 出来的最大分数与候选数 —— 若为 0，说明模型根本不会出框，
   下面的阈值扫描没有意义（对应 `train_detector` 的 `gate:` 行，见 §0.7）；

1. **阈值扫描**：conf 0.05→0.60 的 boxes/img、precision、recall、F1、normal_FPR，
   并给出 F1 最优阈值（这个值就该填进 `fusion.final_score_thresh`）；
2. **FP 解剖**：duplicate / wrong_class / localization / background 各占多少；
3. **按尺度**：P3(0) / P5(1) / 安全网(-1) 各自的 TP/FP —— 判断是不是某一级在刷框；
4. **按类别**：每类 GT 数、TP、FP、precision、recall（19 类长尾里谁在拖后腿）；
5. **混淆对**：`GT 类 -> 预测类` top-N（决定要不要上 `class_weights` 或 ROI 头）。

工具打印的 verdict 直接对应动作：

| 主导项 | verdict 给出的动作 |
|---|---|
| `duplicate` | 保持 peak_filter/dedup，把 `final_score_thresh` 提到 F1 最优点 |
| `wrong_class` | 设 `train.class_weights`（按训练集逆频率）或打开 `fusion.roi_head` |
| `localization` | 提高 `box_o2o/box_o2m` 权重，检查 `reg_scale` 与 GT 框质量 |
| `background` | 阶段 A 混入正常图做 hard negative，保持 `obj_o2o_mode: focal` |

实现细节：它先把 `fusion.final_score_thresh` 临时置 0 收集**未过滤**的候选框，再离线扫阈值；
normal 集只前向一次（缓存后逐阈值统计 FPR），所以 300 张 val + 100 张 normal 在单卡上约几十秒。

---

## 0.7 专题：precision 第三轮（v0.2.1）——objectness 门在冷启动时是关的

第三轮来自一条真实训练日志（RTX 4070 laptop，440 张训练图，batch 8，bf16，
`--resume runs/dino/dino_init.pt`，lr 2e-3）：

```
epoch   1/150  ... | val: mAP50=0.0000 precision=0.0000 recall=0.0000 best_f1=0.0000@0.00 normal_fpr=0.0000
      boxes/img=  0.0  FP: duplicate=0 wrong_class=0 localization=0 background=0
[!] no usable detections this epoch
      loss  box_o2m=32.03 cls_o2m=46.91 box_o2o=20.24 obj_o2o=2.352 obj_o2m=0.028 total=154.3
      loss share: box_o2m=42%  cls_o2m=30%  box_o2o=26%
epoch   2/150  ... 同上（precision = recall = 0）
      loss  box_o2m=28.75 cls_o2m= 4.11 box_o2o=12.03 obj_o2o=1.218 obj_o2m=0.025 total= 87.7
      loss share: box_o2m=66%  box_o2o=27%  cls_o2m=5%
```

`boxes/img = 0.0` 意味着模型**一个框都没输出**。这不是"precision 低"，而是"precision 的分母是 0"——
所以这一轮先修"门根本没开"，否则后面所有阈值/权重调参都是在真空里做。

部署分数是 `score = sigmoid(obj_o2o) · sigmoid(cls)`，decode 门槛 `model.score_thresh=0.05`，
融合后还有 `fusion.final_score_thresh=0.15`。初始时 `obj_o2o.bias=-4`（p=0.018）、
`cls_pred.bias=-2`（p=0.119），初始分数 ≈ **0.002** —— 比两个门槛小 1~2 个数量级，
所以**最初几十步不出框是正常的**。问题在于它一直不出，而且下面四条缺陷把"学会"这条路也堵住了。

### 先排除一个错误猜想：bias 初值不是瓶颈

一开始我把原因算成"Adam 单步最多挪 lr，从 -4 走到 0 要 ~1850 步 ≈ 34 个 epoch"。
**这个推理是错的**：objectness 是 1×1 conv，它靠**权重**（96 个通道 × 特征幅值）就能把 logit
拉起来，不需要 bias 走完全程。实测（head-only 冷启动，冻结的随机特征，Adam lr=2e-3）：

| 探针 | 首次出框 | score 越过 0.15 | 第 0 步框数 |
|---|---|---|---|
| bias -4（旧代码 645041e） | it 5 | it 8 | 0 |
| bias -4（本轮修复后） | it 3 | it 7 | 0 |
| bias -2（本轮修复后） | it 0 | it 1 | **99**（`top_k` 拉满） |

所以 `model.obj_bias_init` 保持 -4.0，只作为调试开关暴露出来；把它预置成 0.12 只会让第 0 步吐出
99 个框、首轮 precision ≈0、`normal_fpr` 爆表。

顺带说明为什么这个探针**复现不出**用户的现象：合成任务只有一个 GT、特征可以立刻背下来，
回归几步就拟合好了，于是 `qmax > 0`、质量目标有意义。用户真实数据上回归远没有拟合，
质量目标才塌陷成 0 —— 日志里 `obj_o2m` 恒为 0.025 就是它的指纹。

### 缺陷 1（决定性）：objectness / 分类的**正样本目标本身是 0**

`build_dense_targets` 里 objectness 目标取的是"当前预测框与 GT 的 IoU"：

```python
qual = box_ciou(pb, boxes[g]).clamp(0, 1)     # pb 来自 pred["reg"]
t_qual[b, 0, idx] = qual                      # 冷启动时 ≈ 0
...
target = t_cls * t_qual                       # 分类目标也被乘上 ≈0
```

新模型的框是一团退化的点，CIoU ≤ 0 → `clamp` 后全是 0。于是：

* `obj_o2m` 被训练成 **0**（日志里恒为 0.025 ≈ "全图预测 0，已经很准了"）；
* 部署用的 `cls` 被训练成 **0.02**（一个真实缺陷的分类目标），而 `cls_pred` 是 o2m/o2o 共用的卷积；
* 分数 = 0.018 × 0.02 = 3.6e-4，**永远**过不了门。

**修复**：按该尺度内最大质量归一化并设下限（TAL/YOLOv8 的做法），冷启动时退化为全 1.0，
回归学好之后再按质量排序；分类目标改回 one-hot（`train.cls_quality_target: false`）。

### 缺陷 2：`obj_o2o` 用 `sum / n_pos` 归一化 → 正样本项是舍入误差

160×160 的 P3 图上有 25600 个 cell，其中正样本 2–8 个。`foc.sum() / n_pos`
把 25k 个背景 cell 的和除以 8，于是这项在总 loss 里只占 **0.7%**（日志：`obj_o2o=0.61/87.7`）。
换句话说：唯一决定"要不要出框"的分支，梯度贡献不到 1%。

**修复**：平衡归一化（`正样本均值 + 负样本均值`），正负各占该项 50%，与图大小无关。
`obj_o2m` / `cls_o2m` 同样处理（`cls_o2m` 原来是 `sum/(n_pos)`，实测膨胀到 46.9，占 30%）。

### 缺陷 3：回归的 L1 项没有归一化 → `box_o2m` 吞掉 66%

```python
losses["box_o2m"] += (1 - box_ciou(...)).mean() + 0.1 * F.l1_loss(pred_d, tgt_d)
```

`pred_d / tgt_d` 是**像素**距离（1280² 输入下可达 1280），而 `1 - CIoU ≤ 2`，
两个尺度加起来最多 4 —— 日志里 `box_o2m = 28.75`，也就是**至少 25 来自这个像素 L1**（L1 ≈ 250 px）。
于是"回归"这项的值 90% 以上来自一个未归一化的 L1，且它的量级**随输入尺寸线性变化**
（320² 与 1280² 训练，梯度比例完全不同，同一份超参换个 `--size` 就变味）。
这就是 `loss share: box_o2m=66%` 的来源，也间接造成缺陷 1：回归被这个巨项拖着走，
`obj_o2m` 的正样本目标永远是 0。

**修复**：以图像边长归一化（`pred_d / img_ref`），所有 loss 项都回到 O(1)，
`loss share` 才能跨分辨率、跨 epoch 比较。

### 缺陷 4：focal `alpha=0.25` + `obj_o2o` 权重 0.5（上一轮改错的方向）

正样本比例是 0.1%–1%，`alpha=0.25` 是**在惩罚前景**（背景拿 0.75）。
上一轮把 `loss_weights.obj_o2o` 从 1.5 降到 0.5，理由是"focal 绝对量级约为 ohem 的 50 倍"；
但量级下降后 0.7% 的占比让这个分支实际上被关掉了。
**修复**：`alpha=0.5`（平衡）、权重回到 1.5、新增 `obj_o2o_neg_weight` 单独控制背景项。

### 现在怎么判断"门有没有开"

val 块新增一行（数值来自 `head.decode` 的 `last_stats`，是**加阈值之前**的统计）：

```
      gate: max_score=0.0812 (conf=0.15)  obj_max=0.412  cls_max=0.204  cand>0.15=0 -> out=0
```

* `max_score < conf` ⇒ **模型根本没出框**（objectness/分类冷启动问题），
  这时告警会直接写 "the deployed gate is SHUT"，并提示去看
  `train.obj_quality_floor` / `model.obj_bias_init` / `train.obj_o2o_alpha`；
* `max_score ≫ conf` 而 precision 仍低 ⇒ 才是"框错了"，去看 FP 分解（0.5/0.6 节）。

同时 `model.obj_bias_init` / `model.cls_bias_init`（logit，默认 -2.0 = 12% 先验）
让门只需挪 0.4 个 logit（≈200 步 ≈ 4 个 epoch，配合分类头学到 0.9）就能开。

### 重训后应该看到的（同一条命令，不加任何新参数）

| 指标 | 修复前 | 修复后（预期） |
|---|---|---|
| `epoch 1–2 boxes/img` | 0.0（且之后很久仍为 0） | 前若干个 epoch 可能仍为 0（分数 ≈0.02 < 门槛），但 `gate: max_score` 会逐 epoch 上升 |
| `gate: max_score` | 未打印 | 可见；越过 `conf` 后开始出框 |
| `loss share: obj_o2o` | 0.7% | 冷启动 58%（正样本还在 0.02），学开后回落到个位数——低位是"分支已解掉"，不是没在学（smoke 实测） |
| `loss: box_o2m` | 32 / 28.8 | **~2.1**（smoke 实测；像素 L1 归一化，与输入尺寸无关） |
| `loss: obj_o2m` | 0.025（恒定 = 死的） | 起始 ~8.0（目标=1.0，预测 0.018，两个尺度各 ~4），随训练下降 |
| `loss: cls_o2m` | 46.9（`sum/n_pos` 膨胀） | 0.2 → 0.005 |

smoke 实测（320²，4 张合成图，2 epoch）的 `loss share` 从
`box_o2m=66% / obj_o2o=0.7%` 变成 `obj_o2m=26~28% obj_o2o=17~24% box_o2m=16~19% box_o2o=13~15% cls_o2m=12%`。

回归测试（`test_53`–`test_56`）把四条缺陷都钉住了：质量目标不得为 0、
分类在缺陷上必须被"推高"、平衡 focal 的量级必须是 O(1)、
head 在 300 步 Adam 内必须把分数推过 `score_thresh`、验证集必须能区分"没出框"和"框错了"。

---

## 0.8 专题：precision 第四轮（v0.2.2）——门是开了，但**整块布面都在门里**

第四轮来自用户重跑 stage A 后贴回的日志。对比 §0.7 的旧日志，`boxes/img` 已经不再是 0
（第三轮"门打不开"确实修好了），但出现了**下一个失败模式**：

```
val   epoch   1/150  mAP50=0.0000 precision=0.0001 recall=0.0086 best_f1=0.0002@0.06
                     normal_fpr=1.0000  boxes/img= 85.0
      FP: background=8893 localization=454
      gate: max_score=0.9997 (conf=0.15) obj_max=1.000 cls_max=1.000 cand>0.15=24607 -> out=9348
...
val   epoch   9/150  mAP50=0.1060 precision=0.0033 recall=0.1900 best_f1=0.0867@0.74
      boxes/img= 55.0   FP: background=4026 localization=555
loss  cls_o2m=0.371 box_o2m=1.264 obj_o2m=1.568 cls_o2o=0.286 box_o2o=0.429 obj_o2o=0.086 total=6.485
      loss share: box_o2m=39%  obj_o2m=24%  box_o2o=13%  fg_context=9%  cls_o2o=7%  cls_o2m=6%
```

`normal_fpr = 1.0000`（每张正常布都出框）、`boxes/img = 42~97`、`precision = 0.003`、
`best_f1` 九个 epoch 只从 0.0083 涨到 0.0867。`mAP50` 在涨（**定位**在学会），
但"这个位置到底是不是缺陷"的判别没有收敛。

### 诊断：用日志反推背景 objectness

`obj_o2o` 是平衡 focal（`alpha=0.5, gamma=2`），负样本项的解析值与预测概率一一对应：

| 背景 obj 预测 p | 单格负样本项 |
|---|---|
| 0.02 | 0.000 |
| 0.10 | 0.001 |
| 0.20 | 0.004 |
| 0.50 | **0.087** |
| 0.70 | 0.295 |

日志 `obj_o2o = 0.086` ⇒ **背景 cell 的平均 objectness ≈ 0.5**（正样本项在 p≈0.9 时只有 1e-3 量级）。
`obj_o2m = 1.568` 同理 ⇒ 训练分支的背景 objectness ≈ 0.78。

再看部署门（`score = obj · cls ≥ 0.15`）：

| obj | cls | score | 是否出框 |
|---|---|---|---|
| 0.02 | 0.40 | 0.008 | 否（正确） |
| **0.50** | **0.40** | **0.200** | **是（误报）** |
| 0.78 | 0.60 | 0.468 | 是（误报） |

⇒ `precision ≈ 0.003`、`normal_fpr = 1.0`、`boxes/img = 55`、`FP: background = 4026`
**全是同一个数**：背景电平 0.5。不是检测能力问题，是 objectness 场的**校准/选择性**问题。

### 为什么背景停在 0.5：四条结构性原因（+2 条观测盲区）

**原因 1（决定性）：objectness 是 neck 上的一个裸 1×1 卷积。**

`cls` / `reg` 各有两层 3×3 DWConv 的 stem，objectness 却直接在 neck 特征上做 1×1。
"这里有没有缺陷"本质是**局部对比**判断（这块和周围不一样），1×1 卷积没有邻域可比，
只能输出一张随特征平滑变化的图；而 o2o 分支要求"每个 GT 只有一个正 cell"（单峰），
1×1 卷积在结构上做不到，于是它收敛到最省力的平坦解 0.5。

受控 A/B（head-only，冻结**真实 backbone** 的特征，320²，batch 4 = 2 缺陷 + 2 空图，
AdamW 2e-3，300 步；`def`/`norm` 分别是缺陷图/正常图上 decode 出的框数）：

| 配置 | 100 步 smax / p99 | 300 步 smax / p99 | def / **norm**（300 步） |
|---|---|---|---|
| `neck` 1×1 + 旧权重 + neg1.0 | 0.012 / 0.011 | **0.033 / 0.028** | 0 / 0（**门根本没开**） |
| `reg` stem + 旧权重 + neg1.0 | 0.573 / 0.571 | **0.947 / 0.535** | 4 / **2** |
| `neck` 1×1 + 新权重 + neg2.5 | 0.010 / 0.010 | **0.034 / 0.031** | 0 / 0（**门根本没开**） |
| `reg` stem + 新权重 + neg2.5（**v0.2.2 发布值**） | 0.523 / 0.520 | **0.525 / 0.521** | 2 / **2** |

也就是说：**调权重救不了 1×1 的 objectness**（换权重后仍然 0.033 = 门永不开），
换成回归 stem 才把 `score` 从 0.03 抬到 0.5 以上。这是本轮唯一"非它不可"的改动。

**原因 2：唯一决定出框的分支只分到 1.3% 的 loss。**
epoch 9 的 `loss share` 里 `box_o2m + obj_o2m = 63%` —— 这两项属于 o2m 训练分支，
**推理时完全不跑**；而 `obj_o2o` 连 1% 都不到（`1.5 × 0.086 / 6.485 = 2%`，share 表里四舍五入为 0）。
梯度绝大部分花在不会部署的机器上。（注意：这一项是**必要条件而非充分条件**——
单独改权重（上表第 3 行）并没有把门打开，但它决定了背景电平能不能被压下去。）

**原因 3：平衡归一化 = 50/50 份额，没有"关闭背景"的压力。**
`_balanced_bce` 给正负各 50% 份额，这与图尺寸无关（这是第三轮的正确修复），
但它同时也意味着：一个只会输出 0.5 的分支在损失上**不亏**。`obj_o2o_neg_weight = 1.0`
时背景项的极小值就在 p≈0.5 附近；要给"压低背景"付钱，这个权重必须 > 1。
（`obj_o2m` / `cls_o2m` 同理；`score = obj·cls` 是乘积，两个因子都要低。）

**原因 4：质量目标的 `qmax` 归一化在冷启动时是噪声放大。**
第三轮把目标改成 `qual / qmax`（下限 0.5）以避免"目标恒为 0"。但当所有 IoU 都是 ~0.02 时，
它们之间的差别**只是噪声**：最大的一格被映射成 1.0、邻居被 clamp 到 0.5，谁大谁小每个
batch 都翻转。现在改成 `w·(qual/qmax) + (1-w)`，`w = qmax`：回归还没学时目标恒为 1.0
（"这里确实有目标"），回归学好之后质量排序再平滑淡入。

**观测盲区 1：`normal_fpr` 是 0/1 指标。** 55 框/图和 1 框/图都报 `1.0000`——
上一轮"以为修好了"就因为这一行。

**观测盲区 2：`gate: max_score` 会被单个走运的 cell 骗到 0.9997。**
logits 无界，25600 个 cell 里总有一个大值；真正要看的是 **p99（背景电平）**。
本轮新增 `gate: p99=… obj_mean=…` 与 `normal: boxes/img=… max_score=…`。

**已排除的错误猜想：`obj_bias_init` 不是瓶颈（实测）。**
§0.7 的探针说"bias=-4 时第 7 步就开"，本轮再测同一个问题（冻结真实特征、320²、AdamW 2e-3）：

| `obj_bias_init` | 首次越过 0.15 | 第 0 步框数 | 100 步 p99 | 300 步 p99 |
|---|---|---|---|---|
| -4.0（**保持不动**） | it 7 | 0 | 0.011 | 0.028 |
| -2.0 | it 0 | **99**（`top_k` 拉满） | 0.52 | 0.51 |
| 0.0 | it 0 | 150（`top_k` 拉满） | 0.5 | 0.5 |

bias 高只会让**第 0 步就淹没在 0.5 的背景里**，而这正是用户看到的现象。
所以 `obj_bias_init` 保持 **-4.0**（继续作为调试开关暴露），不要为了"快点出框"调高它。

### 修复（v0.2.2）

| 位置 | v0.2.1 | v0.2.2 | 理由 |
|---|---|---|---|
| `model.obj_head_from` | 无（写死 neck 1×1） | **`reg`**（+3×3→5×5 DWConv stem） | 原因 1，唯一"非它不可"的改动 |
| `train.loss_weights` | `obj_o2o 1.5, cls_o2o 1.5, box_o2o 2.0, box_o2m 2.0, obj_o2m 1.0, cls_o2m 1.0` | `obj_o2o 6.0, cls_o2o 3.0, box_o2o 2.5, box_o2m 1.2, obj_o2m 0.7, cls_o2m 0.9` | 部署分支从 ~2% 提到 ~35-40%，o2m 让位（原因 2） |
| `train.obj_o2o_neg_weight` | 1.0 | **2.5** | 给"压低背景"付钱（原因 3） |
| `train.obj_o2m_neg_weight` | 无（写死 1.0） | **2.0** | o2m 的背景电平（0.78）会通过共享特征/e2e 传给部署分支 |
| `train.cls_o2m_neg_weight` | 无（写死 1.0） | **2.0** | `score = obj·cls` 是乘积，两个因子都要低 |
| `train.obj_quality_floor` | 0.5 | **0.0** | 0.5 等于把"定位很差的匹配"教成 objectness 0.5，比背景还像目标 |
| `build_dense_targets` | `qual/qmax` + 下限 | `w·(qual/qmax) + (1-w)`，`w = qmax` | 原因 4 |
| `model.obj_bias_init` | -4.0 | **-4.0（不变）** | 见上表：调高只会制造 99~150 个初始框 |
| `model.score_thresh` | 0.05 | **0.15（= `fusion.final_score_thresh`）** | 只保留一个绝对阈值（§2.13） |
| 日志 | `gate: max_score obj_max cls_max` | 增加 `p99 / obj_mean / normal: boxes-img, max_score` | 观测盲区 1、2 |
| val 日志 | `precision=…` | 增加 `-> P=… R=…`（F1 最优点） | 下发阈值处的 precision 会长期偏低，而"模型自己的最优点"才是真实水平（实测 0.0026 vs 5.5%） |

代价（1280²，pico）：**8.086 → 8.257 G MACs（+2.1%）**，参数 2.928 → 2.979 M，仍低于 9 G 预算。

### 兼容性：旧 checkpoint 怎么用

`obj_pred` / `obj_o2o` 的输入通道从 `c`（neck）改成 `h`（reg stem），并新增 `obj_stem`
（2 个尺度 × 2 个头 = **4 个张量**换了形状，另有 28 个 `obj_stem.*` 是新张量）。

* `load_state_dict(strict=False)` **只容忍缺失/多余 key，不容忍形状变化**，所以
  `--resume runs/dino/dino_init.pt` 一开始会直接 `RuntimeError: size mismatch ...` 退出。
  已修（`tools/_common.py::split_shape_mismatch`）：形状不符的张量被**跳过并逐条打印**
  （`[!] dino_init: 4 tensor(s) skipped because their shape changed ...`），
  主干 / neck / cls / reg / 质量分支的权重照常加载 → **weight-only warm start 可用**。
* **完整 checkpoint（`last.pt`，带 optimizer）不能跨架构续训**：AdamW 的 state 是按参数
  *位置*索引的，多了一个 `obj_stem` 就会 `param group size mismatch`（或者更糟：静默错位）。
  现在检测到形状变化 / param group 不匹配时会把 optimizer + scheduler 状态一起丢掉，降级成
  weight-only warm start 并明确告知（epoch 计数与 LR schedule 从 0 重新开始）。
* 想严格复现旧结构做 A/B：`model.obj_head_from: neck`（但那样门就是关的，见原因 1）。

### 重训后应该看到的

| 指标 | v0.2.1 实测 | v0.2.2 预期 |
|---|---|---|
| `normal: boxes/img`（新增） | 55（老日志只有 `normal_fpr=1.0` 一个 bit） | 首 epoch 可能仍偏高，但应**逐 epoch 单调下降** |
| `gate: p99`（新增） | 未打印（`max_score`=0.9997 会骗人） | 从 ~0.5 降到 < 0.1；`max_score` 与 p99 应**拉开** |
| `gate: obj_mean`（新增） | 未打印 | 0.3 → 0.05 以下 |
| `loss share: obj_o2o` | 1.3% | 随 `p99` 下降而变小（分支被解掉后损失自然小），不要反过来当目标 |
| `-> P=… R=…`（新增） | — | F1 最优点的 precision 是真实水平，应逐 epoch 上升 |
| `precision`（下发阈值 0.15 处） | 0.003 | 与 `normal: boxes/img` 反向变化；仍低则说明阈值需要重标定（`calibrate_fusion.py`） |

一句话：**先看 `p99` 和 `normal: boxes/img`，再看 precision。** p99 不降，precision 不会涨。

### 回归测试

`test_58`（`neg_weight` 真的能压低背景项）、`test_59`（objectness 读 reg stem；`neck` 可回退）、
`test_60`（`score_p99`/`obj_mean` 语义：平坦场 p99 = score_max）、
`test_61`（用**发布的 config** 端到端：120 步内门必须开，且两个阈值必须相等）、
`test_62`（`_best_f1` 返回最优点 P/R；`normal_fpr` 返回 boxes/img 与 max_score）、
`test_63`（旧布局 checkpoint 的 warm start：4 个形状不符的张量被跳过并报告，
完整 checkpoint 则自动降级为 weight-only，不崩）。

---

## 0.9 专题：precision 第五轮（v0.2.3）——门开了、框有了，但**布面上到处都算目标**

v0.2.2 上线后用户重跑（同一份数据，440 张缺陷图，val ≈110 张）：

```
val   epoch   1/150  mAP50=0.0080 precision=0.0001 recall=0.0086 best_f1=0.0002@0.06
                     normal_fpr=1.0000  boxes/img= 85.0  FP: background=8893 localization=454
      gate: max_score=0.9997 obj_max=1.000 cls_max=1.000 cand>0.15=24607 -> out=9348
      loss share: box_o2m=37%  obj_o2m=24%  cls_o2m=20%  box_o2o=12%  cls_o2o=3%
...
val   epoch   9/150  mAP50=0.0450 precision=0.0033 recall=0.1897 best_f1=0.0867@0.74
      boxes/img= 55.0
      loss share: box_o2m=38%  obj_o2m=24%  fg_context=10%  box_o2o=9%  cls_o2m=7%  cls_o2o=5%
```

好消息：**门从第 1 个 epoch 就开着**（`max_score=0.9997`，不再是 0.0025），
`mAP50` 从 0 涨到 0.106，`recall` 从 0.0086 涨到 0.19 —— §0.7/§0.8 的两处冷启动修复确实生效了。

坏消息：**`boxes/img = 85`、`normal_fpr = 1.0000`、`precision ≈ 0.003`**。
检测器在 85% 的布面上都认为"这里有东西"，真正的缺陷（19% 召回）被 8893 个背景框埋掉。

### 先看一个容易被忽略的事实：这次运行的**权重是旧的**

日志里 `loss share` 的反解（`份额 × total ÷ 原始值`）：

| 项 | 日志份额 | 反解出的权重 | v0.2.2 发布的权重 |
|---|---|---|---|
| `box_o2m` | 38% | ≈2.1 | 1.2 |
| `obj_o2m` | 24% | ≈1.1 | 0.7 |
| `box_o2o` | 9% | ≈1.5 | 2.5 |
| `cls_o2o` | 5% | ≈1.2 | 3.0 |
| `obj_o2o` | **2%** | ≈1.5 | **6.0** |

也就是说：这次跑的是 **v0.2.1 的 `loss_weights`**（旧配置 2.0/1.0/1.5/1.5），
而 head 是新的（否则门打不开）。**改代码后一定要确认 `configs/*.yaml` 也被拉下来了**：
`git status` 有本地改动、或者只更新了 `jingwei/` 目录，就会得到"新头 + 旧权重"的组合，
而这一组合恰好复现了"obj_o2o 只占 2%"的老毛病。

### 这一轮的真正问题：部署分支仍然被饿着

即便用 v0.2.2 的权重，按日志里的原始值算 `obj_o2o` 也只占 ~10%；
而 o2m 两项（`box_o2m + obj_o2m`）合计 62%，**推理时一行都不跑**。
`obj_o2o` 是唯一决定"这个 cell 要不要出框"的项，它拿不到梯度，就只能停在
"布面上到处都是 0.2~0.5"这种平坦解——于是 224 个候选/图（`cand>0.15=24607`）里
有 85 个最终落进输出。

同时 `obj_o2o` 的**背景压力**（`obj_o2o_neg_weight`）和"整帧都是正常布"的样本
这两个最直接的杠杆在 v0.2.2 里还不到位：

* 平衡归一化把背景项的限制在 50% 份额，`neg_weight=1.5` 不足以把背景压到 0.02；
* 阶段 A **一张正常布都没见过**（数据集里 `normaltrain` 一直没被用），
  所以"整帧无目标"是一个分布外的输入——模型从没学过这种图应该全 0。

### 修复（v0.2.3）

| 位置 | v0.2.2 | v0.2.3 | 理由 |
|---|---|---|---|
| `loss_weights` | `box_o2m 1.2, obj_o2m 0.7, box_o2o 2.5, cls_o2o 3.0, obj_o2o 6.0` | `box_o2m 1.0, obj_o2m 0.3, box_o2o 2.5, cls_o2o 3.0, obj_o2o **10.0**` | 部署分支从 ~24% 提到 ~45%（按实测原始值），`obj_o2o` 2%→13-19% |
| `train.normal_image_frac` | 无（阶段 A 只看缺陷图） | **0.25** | 每 4 张里混 1 张 `normaltrain` 的**无框**图：让"整帧正常"成为被监督的样本（`NegativeMixDataset`） |
| `--normal-frac` | 无 | 新增 | 命令行临时覆盖 |
| `obj_o2o_neg_weight` | 2.5 | 2.5（不变） | 已够；下面的权重才是瓶颈 |
| `loss share` 行 | 只显示 ≥2% 的项 | **`obj_o2o` 恒定显示** + 部署分支 <15% 时告警 | 2% 的项被过滤掉了，这正是"分支饿着"没被发现的原因 |
| `train_detector --smoke` | 合成 GT 固定写在 1280 尺度 | 按 `--size` 缩放 | 320² 冒烟时框全在图外，`obj_o2m` 等数值失真 |

按实测原始值（`box_o2m 1.31, obj_o2m 1.66, cls_o2m 0.46, cls_o2o 0.30, box_o2o 0.45, obj_o2o 0.10`）
算出的新份额（不含 fg 项）：`box_o2m 25% / box_o2o 21% / obj_o2o 19% / cls_o2o 17% / obj_o2m 9% / cls_o2m 8%`
—— **部署分支 58%**，`obj_o2o` 19%，o2m 只剩 34%。`test_65` 把这条不变量钉住。

### 重训后应该看到的

| 指标 | v0.2.2 实测 | v0.2.3 预期 |
|---|---|---|
| `loss share: obj_o2o` | 2%（旧权重）/ ~10%（新权重） | 13–20%，且**稳定后不再掉回个位数** |
| `normal: boxes/img` | 55–85 | 逐 epoch 单调下降（先降到 10 量级，再降到个位数） |
| `gate: p99` | 与 `max_score` 几乎一样（都是 1.0 附近） | 与 `max_score` 拉开：p99 < 0.1 |
| `precision`（下发阈值） | 0.003 | 与 `boxes/img` 反向变化；先看 `boxes/img`，它先动 |
| `best_f1` | 0.0867@0.74 | 上升；若 `-> P=… R=…` 明显高于下发阈值处的 precision，说明该重标定阈值 |

> ⚠️ 一定要确认 `configs/jingwei_p.yaml` 里的 `loss_weights` 是
> `obj_o2o: 10.0, obj_o2m: 0.3`（`git status` 看一下有没有本地未拉取的改动）。
> 用旧权重训出来的模型，`obj_o2o` 依然只有 2% 的梯度，precision 不会动。

回归测试：`test_64`（正常帧作为纯负样本混合、epoch 长度不变、`collate` 契约）、
`test_65`（发布的权重必须让部署分支拿到 ≥45%、`obj_o2o` ≥15%，且 `normal_image_frac > 0`）。

---

## 0.10 专题：precision 第六轮（v0.2.4）——正样本被匹配到了**离缺陷三百多像素**的地方

v0.2.3 上线后的重跑（第 1 个 epoch，权重确认是新表）：`obj_o2o` 份额从 2% 涨到 **29%**，
部署分支合计 ~54% —— 上一轮的权重确实生效了。但仍然：

```
epoch 1/150  fg_defect=1.125 ... total=17.712 (37.9s)
      loss share: obj_o2o=29%  box_o2o=21%  cls_o2m=14%  obj_o2m=12%  box_o2m=12%
                  fg_context=6%  cls_o2o=4%  fg_align=2%
val   epoch   1/150  mAP50=0.0000 precision=0.0000 recall=0.0000
      boxes/img= 82.6  FP: background=8827 localization=261
      gate: max_score=0.6514 p99=0.5251  obj_max=0.995 obj_mean=0.148 cls_max=0.982
            cand>0.15=9286 -> out=9090  normal: boxes/img=61.9 max_score=0.651
```

三个数字就够了：**`obj_mean=0.148`**、阈值 **`0.15`**、**61.9 框/图（正常布面）**。
objectness 场的平均值正好停在阈值上 —— 这不是"阈值没调好"，是模型的输出
**在整张布面上就是一个偏置**。

### 根因 1：`obj_mean ≈ 1 / obj_o2o_neg_weight`

`obj_o2o` 是平衡 focal：正样本均值（每图 ~8 个 cell）+ `w_neg × 负样本均值`（~25 600 个 cell）。
只要正样本那 8 个 cell 无法用特征和背景分开，唯一的解就是**抬高整张图的偏置**——
两者在权值空间里达到平衡时 `p_eq ≈ K / w_neg`。用 v0.2.3 唯一的实测点标定：
`w_neg = 2.5 → obj_mean = 0.148`，即 `K ≈ 0.37`；代入阈值 0.15 得 `w_neg ≥ 4.9` 才安全。

**但为什么那 8 个 cell 无法和背景分开？这才是真问题。**

### 根因 2：一对一匹配的正样本**根本不在缺陷上**

`hungarian_match` 的候选池以前是**当前分数图的 top-k**（每个尺度 800 个 cell / 25 600 个）。
冷启动时这张分数图没有信息量，于是候选池 = 图像左上角的 3%。用仓库里的复现脚本
（`reg=1.0` 均匀、全图 logit 相同，即"分数图完全无信息"）：

```
OLD match: scale 0 cell (23,109) -> (188,876) px, GT centre (520,520) => 44.5 cells away
NEW match: 偏差 ≤ 1.5 cells（o2o_radius）
```

44.5 个 cell 意味着回归头要从一块普通布面的 cell 输出 **~356 px** 的四个距离。这在特征上
是不可学的（那块布面不含"缺陷在哪里"的任何信息），最小二乘的最优解就是**数据集平均偏移**，
于是每个亮着的 cell 都吐出一个"平均尺寸"的框 —— 正是 `background=8827`、
`localization` 只有 261、precision 0.0000 的来源。同时 `t_obj=1` 被写在普通布面上，
objectness 分支被正面教成"这里有目标"。

更糟的是它**自我强化**：被错误匹配的 cell 变成高分 cell，于是下一轮仍在候选池里，
换任何 loss 权重都救不回来（这解释了为什么 v0.2.1→v0.2.3 三轮调权都只改变了
"框有多少"，没改变"框落在哪"）。

### 根因 3：objectness stem 是 5×5 低通，表示不了单 cell 峰

`obj_stem = DWConv(3×3) → DWConv(5×5)`（v0.2.2 为提高感受野加的）。对"这块布和周围不像"
是对的，但对**单 cell 目标**是错的算子：7×7 平滑之后，`t_obj=1` 只落在一格上的约束无法满足，
网络只能用"整体抬高"来折中 —— 根因 1 的偏置解因此成为**唯一可行解**。

### 修复（v0.2.4）

| 改动 | 之前 | 现在 | 理由 |
|---|---|---|---|
| **一对一匹配** | 候选池 = 当前分数 top-k（自指） | 每个 GT 只在自己的 **FCOS 中心区域**内选（`train.o2o_radius: 1.5`，与 o2m 同一规则；无可用 cell 时退回最近 cell） | 正样本必须在目标上：回归目标才落在头的动态范围内，`t_obj` 才写在缺陷上。YOLOv10 的 TAL 用同一条规则过滤候选 |
| **objectness 残差路径** | `obj = 1×1( obj_stem(f_reg) )` | `obj = 1×1( f_reg + obj_stem(f_reg) )` | 保留 9-cell 上下文 **和** 一条不衰减的局部通路，单 cell 峰才表示得出来。0 参数、0 MACs |
| **背景压力** | `obj_o2o_neg_weight: 2.5`（`p_eq ≈ 0.148`，正压在阈值上） | `6.0`（`p_eq ≈ 0.062`） | 只有单侧压力能降 "背景水位"，不能靠调阈值 |
| **`obj_o2o` / `box_o2o`** | 10.0 / 2.5 | 8.0 / 3.0 | 背景压力改由 neg_weight 提供后，把份额还给**回归**（框要落在缺陷上靠它） |
| **`match_cell` 诊断** | 无 | 每个 epoch 打印 `match_cell=`（正样本离 GT 中心的格数，应为 ≤2，>3 告警） | 这个数就是根因 2 的探针；以前它不存在，所以三轮都没发现 |
| **`best operating point`** | `conf>=0.65 -> P=0.000 R=0.000` | 无任何 TP 时直接写 "no true positive at any threshold" | `best_f1_thr` 在 F1 恒为 0 时退化成"全验证集最高分那一个框"，读起来像阈值 bug |

### 探针证据（head-only，冻结真实 backbone/neck 特征，320²，AdamW 2e-3，300 步）

输入 4 张图：2 张带 3 个缺陷、2 张**无缺陷**；`boxes@…` 是 `decode` 在阈值 0.15 下真正吐出的框数
（每 50 步采样，同一个种子、同一份特征）。

| 配置 | `score_max` | `p99`（背景电平） | `obj_mean` | boxes@缺陷(3 个) | **boxes@正常(2 张)** |
|---|---|---|---|---|---|
| v0.2.1 发布值 | 0.033 | 0.028 | 0.027 | **0** | 0 |
| bias -1 / neg 2.5 | 0.933 | 0.264 | 0.043 | 4 | 2 |
| bias -1 / neg 6.0 | 0.954 | 0.375 | 0.042 | 5 | 2 |
| bias -4 / neg 2.5（v0.2.3 发布值） | 0.992 | 0.102 | 0.018 | 3 | 1 |
| bias -4 / neg 4.0 | 0.988 | 0.099 | 0.016 | 3 | 0 |
| **bias -4 / neg 6.0（v0.2.4 发布值）** | 0.985 | **0.057** | 0.016 | **3** | **0** |
| bias -4 / neg 6.0，**关掉残差路径** | 0.988 | 0.105 | 0.020 | 3 | 0 |

读法：

1. **v0.2.1 那行复现了用户的原始日志**：300 步后 `score_max` 只有 0.033，一个框也吐不出来
   （正是 `boxes/img=0.0`、precision 全 0 的那次）。
2. 一旦 objectness 读 reg stem（v0.2.2 的正解），门就开到 0.98+，而且
   **3 个缺陷 = 3 个框**（NMS-free 的性质成立）。
3. 背景电平只被两件事影响：`obj_o2o_neg_weight`（2.5 → 0.102、4.0 → 0.099、6.0 → 0.057，单调）
   和**残差路径**（同权重下 0.105 → 0.057，约 2×）。这两个就是本节的两处主改。
4. 注意"p99 低"本身不是健康：v0.2.1 的 p99 最低（0.028），因为门根本没开。
   要看 `score_max` 与 `p99` 的**差距**，以及正常图上的框数。
5. 表里 `obj_mean` 只有 0.016–0.02（真实训练里是 0.148）：这是冻结的**随机**特征，
   量级不可直接比，可比的是相对排序；真跑时 `obj_mean` 必须明显低于 0.15。
6. `obj_bias_init` 仍是 **-4.0**（§0.8 曾根据 neck-1×1 的旧探针把它改成 -1.0，
   在那个 head 上"能开"确实是 -1 的门槛；换成 reg stem 之后 -4 反而场更尖：
   p99 0.102 vs 0.264）。**不要**为了让门早开去抬 bias。

### 重训时该看的三行

| 日志 | 之前 | 期望 |
|---|---|---|
| `match_cell=` | 无（旧匹配在 40+ cells） | ≤ 2，且稳定 |
| `gate: obj_mean` | 0.148（= 阈值） | 0.02–0.06；`p99` 与 `max_score` 拉开 |
| `normal: boxes/img` | 61.9 | 逐 epoch 下降；它先动，precision 后动 |

> ⚠️ 旧 checkpoint 不兼容（`obj_o2o` 的输入通道 `c → h`、新增 `obj_stem`），
> stage A 必须从头训；`--resume` 会打印 shape mismatch 清单。

回归测试：`test_66`（匹配的正样本必须落在 `o2o_radius` 内，且 `match_cell` 把它报出来）、
`test_67`（残差路径让 1×1 在 5×5 被清零后仍非恒定）、
`test_68`（`K/neg_weight < 0.5 × 阈值`、`obj_head_residual` 为真、`o2o_radius ≤ 2`、两个阈值相等）。

---

## 1. 本轮已修复（按严重程度；0.5 / 0.6 / 0.7 三节专列的 precision 修复也在此列）

> v0.2.4 的第六轮（**一对一匹配只在 GT 中心区域内选正样本**、objectness 残差路径、
> 背景压力 2.5→6.0、`match_cell` 诊断）见 **§0.10**，回归测试 `test_66`–`test_68`。
>
> v0.2.3 的第五轮（部署分支的 loss 份额、正常帧作为纯负样本、`loss share` 告警）见 **§0.9**，
> 回归测试 `test_64`–`test_65`。
>
> v0.2.2 的第四轮（objectness 读 reg stem、部署分支 loss 份额、背景压力、质量目标混合、单一阈值）见 **§0.8**，
> 改动落在 `jingwei/losses.py` / `jingwei/head.py` / `tools/evaluate.py` / `tools/train_detector.py` / 三个 config，
> 回归测试 `test_58`–`test_62`。
>
> v0.2.1 的四项冷启动修复（质量目标塌陷、平衡归一化、像素 L1 未归一化、objectness 分支权重）见 **§0.7**，改动落在 `jingwei/losses.py` / `jingwei/head.py` / `tools/evaluate.py` / `tools/train_detector.py` / 三个 config，回归测试 `test_53`–`test_56`。

### P0-1 `AnomalyFusion(mode="gate")` 对概率做了第二次 sigmoid → 标定阈值失效

`jingwei/normality.py::AnomalyFusion`。

`a_nn` / `a_rec` 本身已经是**标定过的概率**（记忆库 `tau`、重建 `rec_tau` 都定义在"该点输出 = 0.5"），
部署链路又拿融合结果去和 `fusion.anomaly_thresh = 0.5` 比较。旧的 `gate` 把概率再喂进一个 1×1 conv + sigmoid，
等于**二次压缩**——而 `sigmoid(0) = 0.5`，于是任何非零输入都会越过阈值：

| 输入 a_nn = a_rec | 旧 gate 输出 | 是否 > anomaly_thresh 0.5 |
|---|---|---|
| 0.00 | 0.5000 | 临界 |
| 0.01 | 0.5025 | **是** |
| 0.05 | 0.5125 | **是** |
| 0.10 | 0.5250 | **是** |
| 0.50（标定点） | 0.6225 | **是** |

复现：`python -c "from jingwei.normality import AnomalyFusion; ..."`（见提交说明）。
后果：整幅 `anomaly_map` 恒在 0.5 以上 → 连通域变成一个覆盖全图的巨块 → 安全网在**每一帧**都吐一个
全幅 `未知异常` 框（叠加 P0-2）。目前只被前景门控 `× fabric_mask(≈0.88) × fold 衰减` 偶然掩盖，属于"靠运气成立"。

修复：`gate` 改为在 **logit 空间**做可学习混合
`sigmoid(w1·logit(a_nn) + w2·logit(a_rec) + b)`，初始 `w1+w2 = α+β = 1, b = 0`，
于是 `a_nn == a_rec == t` 时输出**恰好等于 t**，0.5 这个标定工作点、以及每个布种原型各自的 `tau` 全部保持有效；
同时仍是可学习、可导、可 ONNX 导出的普通 conv。回归测试 `test_38_anomaly_fusion_preserves_calibration`。

### P0-2 整幅异常 → 输出一个全屏 `未知异常` 框

`jingwei/fusion.py::safety_net_numpy`。

安全性原则是"任何未被覆盖的大连通域都要出框"，但当异常图被**全局事件**点亮（光源闪烁 / 整幅布面偏暗 /
阈值失配 / 上一条的 gate 缺陷）时，连通域就是整幅图，规则忠实地给出 `[0, 0, 1280, 1280]`。
实测：`a = 0.55` 常数图、`fabric_mask = 0.95` → `n_safety_boxes = 1, boxes = [[0,0,1280,1280]]`。
这种框对操作员没有任何信息量，且必然是误报。

修复：新增 `fusion.max_component_area_ratio`（默认 0.90）——超过该面积占比的连通域被视为**全局事件**，
不出框、不 boost，并通过 `n_global_components` 上报（便于监控"标定漂移/光源异常"）。
局部大缺陷（47% 画面）仍然照常出框，不破坏"大缺陷不漏"。回归测试 `test_39`。

### P1-1 `tools/predict.py` 的框坐标没有回投到原图分辨率

网络输入被 `cv2.resize` 压成 `1280×1280`，解码出的框是**网络像素**；旧代码直接把这套坐标画在原图 `bgr` 上。
对 640×360 的产线图，框会落在画面外（负坐标/越界），`results.json` 里的 `box` 也不是原图坐标。
1280² 的图看不出来，非方形布幅一上手就是错的。

修复：新增 `boxes_to_image_pixels()`（按 `w0/size, h0/size` 缩放 + 裁剪），绘图与 JSON 都用原图坐标，
并在 JSON 里补 `image_size / network_size` 两个字段（下游不必再猜）。回归测试 `test_40`。

### P1-2 `LocalContrastNorm` 的 box filter 没有分离 → 占了整网约 1/4 的算子量

`jingwei/layers.py::LocalContrastNorm._box`。

`avg_pool2d(k=15, stride=1)` 的代价是 `k² = 225` MAC/像素/通道，而 LCN 一次 forward 跑 **4 次**：

* 旧：`4 × 225 = 900` MAC/px/ch × 3ch × 1280² ≈ **4.4 G MAC**（占整网 16.62 G 的 26%），
  而 `utils.profile_macs` 只统计 Conv/Linear，**完全看不到池化**，所以 MACs 表里一直是隐形的；
* 新：box 核严格可分离，两次一维池化 `2k = 30` MAC/px/ch → 约 **0.59 G MAC**（-87%）。

`count_include_pad=False` 时二维 box 除的是"有效格数 = n_rows × n_cols"，两次一维池化乘积完全一致，
因此数值上**完全等价**（float32 下 max |diff| ≈ 2.2e-7）。同机实测 LCN 模块 1280²：
**≈1.80 s → 0.85 s（2.1×）**（CPU 内存带宽受限，GPU 上收益主要来自 3.75× 的算术量下降）。
回归测试 `test_37`（含边角逐点比对）。

### P1-3 `list_images()` 把整棵树遍历 14 次

`jingwei/data.py::list_images`。`IMG_EXTS` 有 14 个 pattern，每个 `rglob` 都是一次完整目录遍历；
20k 张的训练集在**每个** DataLoader worker 启动时都要付 14 次。改成一次 `rglob("*")` + 后缀判断。
顺带修掉两个隐藏问题：

* `rglob("*.jpg")` 是**大小写敏感**的，`IMG_EXTS` 只枚举了全小写/全大写 → `.WebP / .Jpg / .Jpeg` 这类文件被静默漏掉；
* `rglob` 会返回**目录**，一个名叫 `xxx.jpg` 的文件夹会被当成图片塞进数据集。

回归测试 `test_41`。

### P1-4 `tools/calibrate_fusion.py` 在 Windows 上永远找不到标签

```python
ldir = str(vdir).replace("/images/", "/labels/")   # 旧
```
Windows 上 `Path` 渲染成 `D:\tyhn\yolo\...\images\train`，`"/images/"` **匹配不到**，
标签目录退化成图片目录 → `_load_label` 全部找不到 → GT 恒为 0 → 所有预测算 FP → 阶段 D 的
`large_recall` 恒为 0，网格搜索选出的是一组纯噪音参数。改用已有的 `infer_labels_dir()`（它同时处理
结尾就是 `/images` 的情况和反斜杠）。同时给 `normalval` 加了 `normaltrain` 回退（与其它工具一致）。

### P1-5 阶段 0 的视图增广用标准库 `random`，但 worker 没有 seed

`jingwei/dino.py::make_views / strong_photometric / _random_crop` 用的是全局 `random`；
`jingwei/data.py::_seed_worker` 只 seed 了 `np.random` 和数据集自己的 `Generator`。
Linux 默认 `fork` 起 worker，所有子进程**继承同一份** `random` 状态 → 各 worker 抽到的裁切/翻转/擦除序列完全相同，
阶段 0 的增广多样性打了折扣（`np` 那条已经被修过，`random` 这条被漏了）。
修复：`_seed_worker` 内 `random.seed(seed)`。回归测试 `test_43`。

### P1-6 `tools/evaluate.py::benchmark` 在 `--device cuda:0` 时不同步

`if device == "cuda"` 与字符串比较，`cuda:0` 判否 → 全程不 `torch.cuda.synchronize()`，
量到的是**内核入队时间**而不是执行时间（会报出一个漂亮但错误、小得多的延迟）。改为
`str(device).startswith("cuda")`。

### P1-7 NumPy 2 下所有验证指标直接崩溃

`tools/evaluate.py::average_precision` 用了 `np.trapz`，而 NumPy 2.0 把它改名 `np.trapezoid` 并从命名空间移除。
`requirements.txt` 虽然钉了 `numpy<2`（理由写的是 torch wheel 依赖），但现实里 numpy 2 已经是默认版本
（torch ≥ 2.5 的 wheel 就支持它），一旦装上就是：

```
AttributeError: module 'numpy' has no attribute 'trapz'. Did you mean: 'trace'?
```

而且它挂在 **每个 epoch 的 validation** 上（`train_detector.py` → `validate_detector` → `evaluate_map`），
训练第一个 epoch 结束就会崩，`tests 26/27/30` 三项同时红。修复：`_TRAPZ = getattr(np, "trapezoid", None) or getattr(np, "trapz")`。
本机用 **numpy 2.4.6 + onnx 1.23** 重跑：42 项全绿（含 ONNX 导出）。

### P2 级修正（顺手做掉）

| 项 | 位置 | 说明 |
|---|---|---|
| 导出打印的输出形状是硬编码 | `tools/export.py` | 原来写死 `(1,19,160,160)` / `n3=(1,96,160,160)`；P 变体是 128 网格、n3=72ch，属于"文档性错误"。改为按 `cfg` 计算 `nc / G / size/8 / size/32 / p3_ch·width`。 |
| 不完整 checkpoint 静默加载 | `tools/_common.py` | `load_state_dict(strict=False)` 的返回值被丢弃，换宽度/换变体时会静默留下随机初始化的层。新增 `load_state_dict_report()`，打印 missing/unexpected 键数与示例。回归测试 `test_42`。 |
| 视频推理先把每一帧写盘再读回 | `tools/predict.py` | 双倍 I/O、长视频能吃满磁盘。改为 `iter_sources()` 流式解码；旧行为用 `--dump-frames` 保留。 |
| `detector-only` 推理的返回 dict 少 3 个 key | `jingwei/model.py` | `evaluate/predict` 之外的用户代码要做分支判断。现在统一返回 `n_safety_boxes / n_unknown_boxes / n_global_components`（值 0）。 |
| 测试数量与实际不符 | `README.md`、`docs/*` | 文档写"31 项"，实际 35 项（本轮后 42 项）；运行时长同步更新。 |
| `normality.coreset_ratio` 是**无效配置** | `configs/*.yaml` | 三个配置都写了这个键，但 `build_memory_bank.py` 只用 `n_centroids`，改它不会有任何效果——典型的调参陷阱。已在 yaml 中标注（真正生效的是 `n_centroids`）。 |

---

## 2. 仍然建议优化的地方

> 严重度：**P0** 会让产品结论失效；**P1** 明显影响精度/误报/可复现性；**P2** 性能或健壮性；**P3** 可维护性。

### 2.1 P0 · 标定链路：`tau` 在"原始信号"上标定，部署却在"门控后信号"上判定

`tools/build_memory_bank.py` 用 `bank.raw_distance()` 和重建残差的**裸概率**分位数定 `tau / rec_tau`，
目标是"0.5% 的 normalval 像素越过 0.5"。但部署时异常图还要乘
`fabric_mask(≈0.88) × (1 − fold_attenuation·fold_band·(1−defect_fg))(≈0.9)`，
即整体被压到 **≈0.8 倍**；`fusion.anomaly_thresh = 0.5` 对应的分位点因此整体漂移，
`normality.target_fpr = 0.005` 在部署图上并不成立（目前靠 `calibrate_fusion.py` 再网格搜 `anomaly_thresh` 补救，属于二次打补丁）。

建议（改动集中在一个脚本里，收益明确）：在 `build_memory_bank` 里缓存每个 normalval 图像的
`(a_nn, residual, fabric_mask, fold_band, defect_fg)`，然后对 `rec_tau` 做**一维二分**：
对候选 `tau` 重算 `a_rec = σ((res−tau)·slope)` → 融合 → 前景门控 → 取
`quantile(deployed_map, 1 − target_fpr)`，令其等于 `cfg.fusion.anomaly_thresh`。
整条链都在 CPU/GPU 上的缓存张量上算，几十次迭代即可，成本可忽略，但标定从此和部署严格一致。

### 2.2 P0-adjacent · 阶段 B 对"异常图/融合门控"没有任何监督信号

`tools/train_normality.py` 的 loss 只有 `HardMiningCosineLoss(target, reconstructed)`，
`normality_parameters()` 里却把 `fusion.gate` 也列为可训练（它永远拿到 `grad = None`，`AdamW` 直接跳过）。
也就是说 `gate` 的"可学习"名不副实。两条路选一条：

* **简单**：`gate` 从 `normality_parameters()` 里去掉，配置默认改用 `fusion: fixed`（线性、保标定）；
* **有用**：给阶段 B 加一个 normal-only 的抑制项，例如
  `L_anom = mean(relu(a_deployed − q))`，`q` 为当前 batch 的 `1−target_fpr` 分位，
  这样门控/融合真的会朝"正常图不打分"的方向学习。

### 2.3 P1 · 阶段 A **没有用过一张正常图**（与文档描述不符）

`jingwei/data.py::fabric_photometric` 的注释写"让 normaltrain 与 detector 的 hard-negative 都见到这种
'像缺陷但不是缺陷'的模式"，但 `tools/train_detector.py` 只读 `paths["train"]`（全是有缺陷的图），
`normaltrain` 只在阶段 0 / B / 建库时用到。于是：

* 检测器从未见过"整幅正常 + 布边折入"的负样本，`defect_fg / fabric_mask / fold_band` 的负样本只有
  缺陷图里的背景像素；
* `normal_fpr` 只能靠后处理阈值压，而不是靠模型学会"这不是缺陷"。

建议：`--normal-frac 0.2`，每批按比例从 `normaltrain` 取图、`boxes=[] labels=[]` 走一次检测损失
（`build_dense_targets` 已支持空 GT，`o2o` 分支就是纯负样本，正是 hard-negative 的用法）。
这是当前**性价比最高**的误报优化，工作量约 40 行 + 一个 CLI 参数。

### 2.4 P1 · 视频推理没有跨帧关联，同一缺陷会被重复计数

`tools/predict.py` 逐帧独立推理，`results.json` 每帧一条。产线上一个缺陷会在几十帧里各报一次，
下游计数/报警会被放大。建议加轻量的帧间关联（IoU 或匈牙利匹配 + 生命期管理），
并输出稳定 `track_id`；顺带可做时间维平滑（`anomaly_map` 的 EMA）进一步压误报。

### 2.5 P2 · `build_dense_targets` 的 `(HW, G, 4)` 中间张量是阶段 A 的显存热点

`jingwei/losses.py`。P3 上 `HW = 25600`，`G` 个 GT：`G=20` 时单个 `d/l/t/r` 组合就是
`25600×20×4×4B ≈ 8 MB`（`cost`、`valid`、`centre` 各再一份）。batch 8 + AMP 下这是最主要的临时分配。
建议先按 GT 的外接范围过滤候选中心（缺陷通常只覆盖画面一小块），或按 GT 分块求 `min`，
预期能把这一步的峰值显存降一个量级。

### 2.6 P2 · `safety_net_numpy` 每个连通域都新建一次 torch 张量做 IoU

`jingwei/fusion.py`：循环内 `torch.as_tensor(boxes)` + `box_iou`，噪声大时是"数百次小张量分配"。
建议在循环外把候选框统一转成一次 `np.ndarray`，连通域框收集完再**一次性** `box_iou(comps, boxes)` 求覆盖关系。
既快也更好读（现在的 `hit.any()` 语义依赖每轮的闭包）。

### 2.7 P2 · `PatchFeatureHead` 在 pico 上用 bilinear 把 160→64 降采样

`jingwei/normality.py`：`160 % 64 != 0` 时走 `F.interpolate(..., mode="bilinear")`，
2.5 倍直接双线性下采样没有低通，织纹高频会混叠回记忆库特征（PatchCore 原本是 avg-pool）。
建议 A/B：`avg_pool2d(f3, 2)`（160→80）再 bilinear 到 64，或干脆把 pico 改成 `grid=80 / out_grid=160`
（代价是 +36% 的记忆库与后处理算力）。改完需要重跑 `test_19` 的 ONNX 静态形状检查。

### 2.8 P2 · `FabricPrototypeBank` 在 `forward` 里改自己的权重

`raw_distance()` 发现 `k != self._active` 就 `set_active(k)`（`scorer.weight.copy_`）。
这违反"前向是纯函数"的假设，会打断 `torch.compile` / CUDA Graph / 多流并行，也让 batch 内多布种时的
行为依赖遍历顺序。产线做法：导出/上线前 `set_active(k)` 固定，`raw_distance` 只做断言；
需要一帧内切布种时，用 `F.conv2d` 的权重 gather（`centroids[ids]` → `(B·K, D, 1, 1)`）显式实现。

### 2.9 P2 · 度量口径

* `utils.profile_macs` 只统计 Conv/Linear（以及带 `macs()` 的注意力），**池化完全不计**。LCN 修复前
  这里藏了 4.4 G MAC。建议把 `nn.AvgPool2d/MaxPool2d` 也算进去（`out.numel()×C×k²`，stride>1 时要小心），
  或者至少在 `benchmark.py` 输出里显式标注"不含池化"。注意：改了口径就要同步更新 README/DESIGN 里的
  16.62 G / 8.09 G。
* `evaluate.average_precision` 是 VOC2007 的 101 点插值（偏乐观），不是 COCO 的累积 AP；
  建议在报表里写清楚，或直接用 101 点之外的面积法。
* `calibrate_fusion.py` 的约束写的是"FPR ≤ target_fpr"，代码里却是
  `ok = fpr <= max(target_fpr*3, 0.02)`（放宽到 3 倍 / 2%）。至少把实际约束值写进输出的
  `measured` 字段，否则复现时会以为搜的是 0.5%。

### 2.10 P2 · `YOLODefectDataset` 用文件 stem 匹配标签

`_label_path()` 只取 `Path(img).stem`。`list_images` 是**递归**的，如果以后出现
`images/a/0.jpg` 与 `images/b/0.jpg`，两张图会共用 `labels/0.txt`（静默错标）。
建议优先按相对路径镜像查找 `labels/.../0.txt`，找不到再退回 stem（保持向后兼容）。

### 2.11 P2 · 阶段 B 的超参与配置漂移

`tools/train_normality.py` 里 `weight_decay=1e-4`、`clip_grad_norm_(max_norm=0.1)` 都是硬编码，
而 `cfg.train.weight_decay` 写着 5e-4；调参时改了 yaml 却不生效。建议读配置并写进 checkpoint 的 `train` 元数据。

### 2.12 P2 · 导出/ONNX 路径在默认环境里没有被验证

`test_19_pico_onnx_export` 在没装 `onnx` 时 skip，而 `requirements.txt` 把 onnx 列为可选。
已新增 **`requirements-dev.txt`**（`-r requirements.txt` + `onnx / onnxsim / pytest`），装上它
`test_19_pico_onnx_export` 就会真正执行（本机已验证：onnx 1.23 + numpy 2.4 下 pico 的
`anomaly=(1,1,128,128)`、`context=(1,3,128,128)` 全为静态形状，`onnx.checker` 通过）。
建议把「`pip install -r requirements-dev.txt` + 全量测试 + `tools/export.py --check`」作为 CI 的最小门禁。

### 2.13 ~~P1~~ 已修（v0.2.2）· 两个绝对阈值耦合在同一条分数上

`decode` 用 `model.score_thresh`（0.05）截断候选，融合路径再用 `fusion.final_score_thresh`（0.15）
截断最终框，两者都作用在 `sigmoid(obj)·sigmoid(cls)`（融合后只增不减）上。0.05 因此几乎不起作用，
真正决定"有没有框"的是 0.15 —— 而它同时也是 `--val-conf` 的默认值。
建议：把 decode 的阈值改成"融合阈值的下界 + 安全余量"（例如 `score_thresh = min(score_thresh, 0.5*final)`），
或者干脆让 decode 只做 top-k + 峰值，把唯一的绝对阈值留给融合/标定链路，
这样"调阈值"就只有一个旋钮，`calibrate_fusion.py` 的搜索结果也能直接用于部署。

> **v0.2.2 已处理**：`model.score_thresh` 从 0.05 改成 **0.15（= `fusion.final_score_thresh`）**，
> 两个键在三个 config 里显式相等；`test_61` 断言这一点。日志里的 `cand>conf` 与 `out`
> 因此可以直接比较。

### 2.14 P3 · 其它（低风险、可择机处理）

| 项 | 位置 | 说明 |
|---|---|---|
| 重复的推理后处理 | `jingwei/model.py` | `inference_from_features` 与 `postprocess_from_tensors` 各有一份"decode + 逐图 fuse"循环，已经出现轻微分叉（后者多了 `feats is None` 保护）。建议抽成 `_decode_and_fuse(preds, anomaly, context, feats)`。 |
| `decode` 的 Python 双层循环 | `jingwei/head.py` | batch>1 时可以向量化；`_points_cache` 无上限（分辨率/设备组合多了会累积），可加 `maxsize`。 |
| 死代码 / 保留别名 | `config.merge`、`utils.smooth_anomaly / boxes_to_distances / scale_boxes / human_macs`、`foreground.foreground_loss / context_only`、`DefectForegroundHead` | 都没有调用方（有的是历史 API 别名）。建议要么删，要么在 docstring 里标 `deprecated`。 |
| DINO 每帧新建 ImageNet 均值张量 | `jingwei/dino.py::patch_map` | 每次 forward `torch.tensor(...)`×2，改成 `register_buffer`。 |
| `_random_crop` 逐样本 Python 循环 | `jingwei/dino.py` | `batch=4` 无所谓；teacher 变大/批量变大时可用 `unfold` 或 `gather` 向量化。 |
| 阶段 0 的主干 BN 用 batch=4 的即时统计量 | `tools/pretrain_dino.py` | 蒸馏结束后阶段 B 又切回 `eval()`（用 running stats），两者统计口径不同。建议加 `--bn-eval` 选项（蒸馏期间冻结 BN）或改用 GroupNorm 的主干，做一次 A/B。 |
| `predict.py` 单张推理 | `tools/predict.py` | 产线吞吐可批处理（`batch>1` 时 `decode` 的循环才划算）；顺带把 `fabric_descriptor` 的 FFT 从主线程挪到预取线程。 |

---

## 3. 本轮变更清单

```
jingwei/head.py        decode 增加 3x3 局部极大值过滤（一个缺陷 144 框 → 1 框），
                       NMS-free / 静态延迟 / 可在图外执行；model.peak_filter 可回退
                       跨尺度去重（嵌套小框套大框 containment>0.85）：model.dedup
jingwei/losses.py      o2o objectness 支持 focal（全 cell，默认）/ ohem（旧行为）
jingwei/fusion.py      scale_idx 透传：按 P3/P5/安全网分别统计 TP/FP
jingwei/layers.py      LocalContrastNorm._box → 可分离 box（等价，-87% 算术）
jingwei/data.py        list_images 单次遍历（+ 大小写/目录修正）；_seed_worker 增 random.seed
jingwei/normality.py   AnomalyFusion(gate) → logit 空间混合，保标定
jingwei/fusion.py      max_component_area_ratio：全局异常不再输出全幅框；n_global_components 上报
jingwei/model.py       detector-only 推理返回统一 dict schema
tools/predict.py       框坐标回投原图；视频流式解码（--dump-frames 可选）；boxes_to_image_pixels()
tools/calibrate_fusion.py  infer_labels_dir（Windows）+ normaltrain 回退；
                       候选按 val F1 排序 + --min-precision（不再只有 FPR 约束）
tools/train_detector.py    best.pt 选择指标改为 best_f1（含 FPR 罚项）+ 每 epoch 打印
                       boxes/img 与 FP 四分类；新增 loss share 行（发现 cls 吞掉 50%）
tools/diagnose_precision.py  **新增**：一条命令定位 precision 的五个来源 + verdict
tools/evaluate.py      benchmark 在 cuda:N 上正确同步；average_precision 兼容 numpy 2 (np.trapezoid)
tools/export.py        输出形状按 cfg 计算
tools/_common.py       load_state_dict_report：missing/unexpected key 显式告警
configs/*.yaml         新增 peak_filter / peak_kernel / dedup / obj_o2o_mode 等；
                       obj_o2o 权重 1.5 → 0.5（focal 量级）；coreset_ratio 标注为无效
tests/test_model.py    +16 项回归（37–52），test_24/34 适配新契约
README.md / docs/*     测试数量与时长更正；新增本文件
```

v0.2.1（第 3 轮，§0.7）增量：

```
jingwei/losses.py      新增 _balanced_bce（正/负样本均值，量级与图尺寸无关）
                       quality 目标按尺度最大 IoU 归一化 + 下限 obj_quality_floor（冷启动不再塌成 0）
                       分类目标改回 one-hot（cls_quality_target 可切回旧行为）
                       cls_o2m / obj_o2m 改平衡归一化（曾膨胀到 46.9 / 恒为 0.025）
                       box_o2m / box_o2o 的像素 L1 按图像边长归一化（曾占 66%）
                       obj_o2o focal 改平衡形式 + obj_o2o_neg_weight
jingwei/head.py        obj_bias_init / cls_bias_init 可配；decode 记录 last_stats
                       （score_max / obj_max / cls_max / n_above / n_out，阈值之前）
tools/evaluate.py      evaluate_map 汇总 gate_max / obj_max / cls_max / n_above / n_out
tools/train_detector.py  val 块新增 `gate:` 行；门关时告警指向正确的旋钮
tools/diagnose_precision.py  [0] 门控存活检查（max decoded score）
configs/*.yaml         obj_bias_init / cls_bias_init / obj_quality_floor / cls_quality_target /
                       obj_o2m_pos_weight / cls_o2m_pos_weight / obj_o2o_neg_weight；
                       obj_o2o 权重 0.5 → 1.5，alpha 0.25 → 0.5
tests/test_model.py    +5 项回归（53–57），test_50 的"focal > ohem"改成"错误背景必须被罚"
```

v0.2.2（第 4 轮，§0.8）增量：

```
jingwei/head.py        obj_head_from="reg"：objectness 读回归 stem + 自带 3x3->5x5 DWConv stem
                       （原来是 neck 上的裸 1x1，背景电平永远 0.5；+2.1% MACs / +0.05 M 参数）
                       decode.last_stats 增加 score_p99（背景电平）与 obj_mean
jingwei/losses.py      obj_o2m_neg_weight / cls_o2m_neg_weight（默认 2.0，把背景压低）
                       build_dense_targets 的质量目标改成 w*rel+(1-w)，w=qmax（冷启动=全 1.0）
tools/evaluate.py      best_precision / best_recall（F1 最优点上的 P/R）
                       normal_fpr(return_extra=True)：normal boxes/img + max_score
                       liveness 增加 gate_p99 / obj_mean
tools/train_detector.py gate 行增加 p99 / obj_mean / normal boxes-img；boxes/img 行增加最优点 P/R
tools/diagnose_precision.py  [0] 门控存活检查
configs/*.yaml         obj_head_from / obj_o2m_neg_weight / cls_o2m_neg_weight；
                       loss_weights 重分配（obj_o2o 1.5->6.0，o2m 让位）；obj_o2o_neg_weight 1.0->2.5；
                       obj_quality_floor 0.5->0.0；score_thresh 0.05->0.15
tests/test_model.py    +5 项回归（58–62）
```

v0.2.3（第 5 轮，§0.9）增量：

```
tools/_common.py       NegativeMixDataset：按比例把 normaltrain 的无框图当纯负样本混进 batch
tools/train_detector.py --normal-frac；loss share 恒显示 obj_o2o，部署分支 <15% 时告警
                       --smoke 的合成 GT 按 --size 缩放（原来固定 1280 尺度）
configs/*.yaml         loss_weights 重分配（obj_o2o 6.0 -> 10.0，obj_o2m 0.7 -> 0.3，
                       box_o2m 1.2 -> 1.0）；新增 normal_image_frac: 0.25
tests/test_model.py    +2 项回归（64–65）
```

v0.2.4（第 6 轮，§0.10）增量：

```
jingwei/losses.py      hungarian_match：候选池从"当前分数 top-k"改为"每个 GT 的 FCOS 中心区域"
                       （train.o2o_radius，默认 1.5 stride；无有效 cell 时退回最近 cell）
                       新增 detached 诊断 match_cell（正样本离 GT 中心的格数，>3 时 train 告警）
jingwei/head.py        obj_residual：obj_src = f_reg + obj_stem(f_reg)，单 cell 峰可表示
                       （model.obj_head_residual，0 参数 / 0 MACs，可 A/B）
configs/*.yaml         obj_o2o_neg_weight 2.5 -> 6.0（背景水位 p ≈ K/w，2.5 时实测 0.148 = 阈值）
                       obj_o2o 10 -> 8、box_o2o 2.5 -> 3.0（把份额还给回归）；新增 o2o_radius
tools/train_detector.py 每 epoch 的 `match_cell=`；匹配健康告警；
                       F1 恒为 0 时不再把"全验证集最高分那个框"当成 operating point
tests/test_model.py    +3 项回归（66–68）；test_50 的对照改用实际匹配到的 cell
```

## 4. 复现命令

```bash
# 全量测试（本机 67 项；装 requirements-dev.txt 后 onnx 用例不再 skip）
python -m unittest discover -s tests -v

# P0-1：融合门控是否破坏 0.5 标定点
python - <<'EOF'
import torch
from jingwei.normality import AnomalyFusion
f = AnomalyFusion("gate", 0.6, 0.4).eval()
for v in (0.01, 0.05, 0.5):
    a = torch.full((1, 1, 2, 2), v)
    print(v, "->", float(f(a, a)[0, 0, 0, 0]))
EOF

# P0-2：整幅异常不应产出全幅框（应打印 0 与 1）
python -m unittest tests.test_model.TestModel.test_39_safety_net_ignores_global_anomaly -v

# P1-2：LCN 可分离性 / 数值等价（应打印 ~2e-7 的 max diff）
python -m unittest tests.test_model.TestModel.test_37_local_contrast_norm_separable_box -v

# P1-1：非 1280 图片的框坐标（JSON 里 box 应为原图像素）
python tools/predict.py --cfg configs/jingwei_p.yaml --source 你的图.jpg --out runs/predict
```
