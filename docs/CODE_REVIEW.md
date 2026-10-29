# 代码审查与优化建议（v0.2.0）

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

对给定 checkpoint 直接回答五个问题（每一节末尾给出"下一步改什么"）：

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

## 1. 本轮已修复（按严重程度；0.5 节的三项 precision 修复也在此列）

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

### 2.13 P3 · 其它（低风险、可择机处理）

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

## 4. 复现命令

```bash
# 全量测试（本机 42 项 / 90 s，含 1 个 onnx skip）
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
