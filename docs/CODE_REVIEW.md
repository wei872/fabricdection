# 代码审查与优化建议（v0.1.8）

> 审查对象：`jingwei/`（推理库）+ `tools/`（训练/评估/部署 CLI）+ `tests/` + `configs/`。
> 审查方式：逐文件静态审查 + 全部单元测试实跑 + 针对性探针脚本（下面每条结论都附复现方式或实测数字）。
> 结论：**架构与工程质量已经不错，但存在 1 个会让"未知异常兜底"整体失效的 P0 级标定缺陷**，
> 以及若干 P1 级坐标/平台/多进程/依赖兼容问题。本轮已修复 15 项（含 P0），剩余 13 项（其中 2.13 汇总了 8 个低风险小项）按优先级列在第 2 节。

---

## 0. 环境与基线

| 项目 | 值 |
|---|---|
| 沙箱 | 2 vCPU / 3 GB RAM / 无 GPU（只为复现正确性与相对性能，不代表产线速度） |
| Python / Torch | 3.11.2 / 2.4.1+cu121（CPU 执行） |
| 基线（改动前） | `python -m unittest discover -s tests -v` → **Ran 35 tests in 123.6 s — OK (skipped=1)** |
| 改动后 | **Ran 42 tests in 91.3 s — OK**（新增 7 项回归测试；同机同时长下降 26%；装上 `onnx` 后 ONNX 导出测试不再 skip） |
| 规模 | `jingwei_n` 6.36 M 参数 / 16.62 G MACs @1280；`jingwei_p` 2.93 M / 8.09 G MACs |

只装 `requirements.txt` 时 `test_19_pico_onnx_export` 会被 skip（`onnx` 只是可选依赖），也就是说**导出路径在默认环境里其实从未被验证**，见 2.12。

---

## 1. 本轮已修复（按严重程度）

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
jingwei/layers.py      LocalContrastNorm._box → 可分离 box（等价，-87% 算术）
jingwei/data.py        list_images 单次遍历（+ 大小写/目录修正）；_seed_worker 增 random.seed
jingwei/normality.py   AnomalyFusion(gate) → logit 空间混合，保标定
jingwei/fusion.py      max_component_area_ratio：全局异常不再输出全幅框；n_global_components 上报
jingwei/model.py       detector-only 推理返回统一 dict schema
tools/predict.py       框坐标回投原图；视频流式解码（--dump-frames 可选）；boxes_to_image_pixels()
tools/calibrate_fusion.py  infer_labels_dir（Windows）+ normaltrain 回退
tools/evaluate.py      benchmark 在 cuda:N 上正确同步；average_precision 兼容 numpy 2 (np.trapezoid)
tools/export.py        输出形状按 cfg 计算
tools/_common.py       load_state_dict_report：missing/unexpected key 显式告警
configs/*.yaml         coreset_ratio 标注为无效键
tests/test_model.py    +7 项回归（37–43），test_24/34 适配新契约
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
