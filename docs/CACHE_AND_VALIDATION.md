# 训练缓存、每 epoch 验证指标与 1660S 实时预测

> 针对真实训练中的四个问题：
> 1. “为什么日志只有 loss，没有验证集精确率 / 召回率？”
> 2. “图片读取太慢，能不能像 YOLO 一样先生成 `train.cache`？”
> 3. “未知疵点类型也要生成预测框，同时必须保证 GTX 1660 SUPER 实时。”
> 4. “CUDA device-side assert / `IndexKernel.cu:93` 怎么定位并修复？”（见第 6 节）

---

## 1. 每 epoch 验证指标

`tools/train_detector.py` 现在默认**每个 epoch 验证一次**（`--val-interval 1`），
日志示例：

```
val   epoch   1/200  mAP50=0.7123  precision=0.7450  recall=0.6812  f1=0.7117  large_recall=0.9960  normal_fpr=0.0040  unknown_boxes=3  (12.4s)
best   -> runs/detector/best.pt  (score=0.8542)
```

指标含义：

| 字段 | 含义 |
|---|---|
| `mAP50` | 19 类在 IoU=0.5 的 mAP |
| `precision` | 在 `final_score_thresh` 下的全局 TP/(TP+FP) |
| `recall` | TP/(TP+FN) |
| `f1` | 2PR/(P+R) |
| `large_recall` | 大缺陷召回，主验收指标 |
| `normal_fpr` | `normalval` 正常图误报率 |
| `unknown_boxes` | 标签 19 / `未知异常` 的预测框数量 |

- `best.pt` 按 `0.5*mAP50 + 0.5*large_recall` 保存。
- `last.pt` 现在包含 `metrics` 字段，断点续训时也会记录历史最佳分数。
- `--val-interval 0` 关闭验证；`--val-max 200 --normal-max 200` 只抽样子集加速。
- 独立评估：`python tools/evaluate.py --cfg configs/jingwei_p.yaml --ckpt runs/detector/best.pt --cache disk`。

> 你之前贴的日志里 epoch 汇总只有 `cls_o2m / fg_context ... total`，那是训练 loss，
> 不是验证指标；且该日志没有 `fg_defect/fg_fabric/fg_fold/fg_global` 子损失，
> 说明当时运行的是旧版本。请先拉取最新提交再训练。

---

## 2. YOLO 式 `train.cache`

> **注意**：如果“加了缓存训练反而更慢”或 `--num-workers` 只能为 0，
> 那是数据增强和 worker 序列化的问题，不是缓存的问题，
> 见 **[docs/SPEED.md](SPEED.md)**（本次已修复：反光增强 411 ms → 2.9 ms/张，
> worker 不再 pickle 整个 memmap）。

新增 `tools/cache_dataset.py` 与 `jingwei/cache.py`。支持三种模式：

| 模式 | 数据文件 | 特点 | 适用 |
|---|---|---|---|
| `disk` | `*.npy` 内存映射 | 预先解码并缩放到 1280² 的 RGB uint8，训练时无逐图磁盘 I/O | 推荐，SSD/机械盘都明显加速 |
| `ram` | 同上，加载进内存 | 最快，但占 `N*1280*1280*3` 字节 | 内存充足的小/中数据集 |
| `bytes` | `*.bin` | 只缓存原始 JPEG/PNG 字节，训练时从内存解码，无磁盘寻道 | 磁盘空间有限，仍想减少 I/O |

缓存文件与 YOLO `train.cache` 类似，由一个 `*.json` 元数据文件 + 一个数据文件组成：

```
runs/cache/train_1280_disk_xxxxxxxxxx.json
runs/cache/train_1280_disk_xxxxxxxxxx.npy
```

元数据记录图片路径、原始尺寸、YOLO 标签（归一化）、偏移表和签名。
签名由图片路径 / 大小 / mtime + 标签文件内容哈希生成；增删图片或改标签会**自动重建**。
路径使用 `ensure_ascii=False` 的 UTF-8 JSON，中文 / Unicode 目录名正常工作。

### 2.1 预生成缓存

```bash
# 一次生成 train / val / normaltrain / normalval 四个缓存
python tools/cache_dataset.py --cfg configs/jingwei_p.yaml \
    --splits train,val,normaltrain,normalval --mode disk

# 也可以只对训练集建 RAM 缓存
python tools/cache_dataset.py --cfg configs/jingwei_p.yaml \
    --splits train --mode ram --cache-dir runs/cache
```

### 2.2 训练时加载缓存

```bash
python tools/train_detector.py --cfg configs/jingwei_p.yaml \
    --cache disk --cache-dir runs/cache --epochs 200

python tools/train_normality.py --cfg configs/jingwei_p.yaml \
    --cache bytes --cache-dir runs/cache --iters 10000

python tools/pretrain_dino.py --cfg configs/jingwei_p.yaml \
    --cache disk --cache-dir runs/cache --iters 5000
```

- 阶段 0 / A / B 训练脚本都接受 `--cache / --cache-dir / --rebuild-cache`。
- 训练时若缓存不存在会自动构建，并打印 `building cache` 进度。
- `--rebuild-cache` 强制重建。
- `--cache none` 是旧行为（每次直接读图）。
- 缓存只影响数据读取；模型权重、ONNX 图、断点续训格式不变。

### 2.3 速度预期

你贴的日志：7384 iter / 2h10m ≈ 0.95 it/s。主要瓶颈通常是：
- 每个 epoch 重复打开 / 解码 7384 张图片；
- Windows 上中文路径 + 多 worker 的 `cv2.imread` 开销；
- 机械硬盘随机读。

`disk` 模式把随机读变成顺序内存映射；`ram` 模式完全去掉文件 I/O。
实际提升取决于磁盘和 CPU，**用相同 `--log-interval` 对比 epoch 时间**即可验证。
如果仍慢，把 `--num-workers` 设为 2~4 并确认 `pin_memory=True`（当前已默认）。

> `disk/ram` 缓存大小约 `N * 1280 * 1280 * 3` 字节。
> 7384 张约 36 GB；磁盘紧张时请用 `bytes` 模式，或只缓存 `train`。

---

## 3. 未知疵点类型也生成预测框

`fusion.emit_unknown_boxes: true` 现在是三个配置的默认值。
安全网不再只兜底 `large_area_ratio` 以上的大异常，还会对**更小的未知异常连通域**
输出 `label=19 / 未知异常` 的预测框。

```yaml
fusion:
  emit_unknown_boxes: true
  unknown_min_area_ratio: 0.0002   # 约 5 个 anomaly cell @160²
  unknown_peak_thresh: 0.55        # 小区域需要更高峰值，抑制噪声
  max_unknown_boxes: 20            # 每帧最多额外未知框
  large_area_ratio: 0.001          # 大缺陷兜底阈值不变
```

关键保证：

- 大异常连通域（面积 ≥ `large_area_ratio`）**不受 `max_unknown_boxes` 限制**，永不丢弃；
- 小未知区域按异常峰值排序，最多保留 `max_unknown_boxes` 个；
- 所有未知框仍经过动态 `fabric_mask / fold_band / defect_fg` 检查，背景和正常折入不会误报；
- 已有检测框 IoU 命中时只提分、不重复出框。

关闭方式：配置 `emit_unknown_boxes: false`，或 `predict.py --no-unknown`。
评估脚本会单独统计 `unknown_boxes`，它们不参与 19 类 mAP，但参与大缺陷召回。

---

## 4. GTX 1660 SUPER 实时预测路径

**训练和推理都用 `configs/jingwei_p.yaml`**，不要拿 `n/s` 的 checkpoint 直接跑 1660S。
若 `--ckpt` 是 n/s 权重而 `--realtime-1660` 切到 pico，PyTorch 会以 `strict=False`
加载并留下大量随机初始化参数，日志会给出警告；此时精度不可信，应重新用 pico
配置训练，或先做权重蒸馏/适配。

```bash
# 预生成 1660S 实时配置预测，自动切 pico + FP32 + 实测 P99
python tools/predict.py --cfg configs/jingwei_p.yaml --realtime-1660 \
    --ckpt runs/detector/best.pt --source D:/产线图片/布匹.jpg \
    --out runs/predict
```

`--realtime-1660` 行为：

1. 读取 GPU 名称 / SM 版本，检测到 GTX 16 系列或无 Tensor Core 卡时自动切到 pico；
2. 强制 FP32（1660S 的 FP16 常因无 Tensor Core 反而更慢）；
3. `roi_head=false`、`cudnn.benchmark=true`；
4. 用 `tools/bench_device.py` 的实测逻辑打印 P50 / P99 / FPS；
5. P99 > 25 ms 时提示导出 TensorRT engine 并重新实测。

独立压测：

```bash
python tools/bench_device.py --cfg configs/jingwei_p.yaml --precision both
python tools/export.py --cfg configs/jingwei_p.yaml --out runs/export/jingwei_p.onnx
trtexec --onnx=runs/export/jingwei_p.onnx --saveEngine=jingwei_p.engine --workspace=4096 \
        --minShapes=images:1x3x1280x1280 --optShapes=images:1x3x1280x1280 \
        --maxShapes=images:1x3x1280x1280
```

当前 pico 变体实测计算量：**2.93 M / 8.09 G MACs @1280²**。
在 1660S 上的估算为 **13.2–20.1 ms / 50–76 FPS**，但最终必须以你机器上的
`bench_device.py` / `trtexec --duration=60` P99 为准。

---

## 5. 建议的完整训练命令

```bash
# 0) 先校验 YOLO 标签 / 1-based 类别号，再生成缓存（只做一次）
python tools/validate_labels.py --cfg configs/jingwei_p.yaml \
    --data configs/fabric19.yaml --cache disk \
    --cache-dir runs/cache --splits train,val
# 若报告 1-based：加 --fix-one-based --rebuild-cache，或训练时加 --label-offset 1
python tools/cache_dataset.py --cfg configs/jingwei_p.yaml \
    --splits train,val,normaltrain,normalval --mode disk

# 1) 阶段 0 DINOv2 离线蒸馏
python tools/pretrain_dino.py --cfg configs/jingwei_p.yaml --cache disk \
    --iters 5000 --out runs/dino

# 2) 阶段 A 监督检测（每 epoch 输出验证指标）
python tools/train_detector.py --cfg configs/jingwei_p.yaml --cache disk \
    --resume runs/dino/dino_init.pt --epochs 200 --val-interval 1 \
    --val-max 300 --normal-max 300 --out runs/detector

# 3) 阶段 B  normality / 记忆库
python tools/train_normality.py --cfg configs/jingwei_p.yaml --cache disk \
    --ckpt runs/detector/best.pt --iters 10000 --out runs/normality

# 4) 评估 + 1660S 实时预测
python tools/evaluate.py --cfg configs/jingwei_p.yaml \
    --ckpt runs/normality/last.pt --cache disk
python tools/predict.py --cfg configs/jingwei_p.yaml --realtime-1660 \
    --ckpt runs/detector/best.pt --source D:/产线图片/ --out runs/predict
```

---

## 6. CUDA `device-side assert triggered` / `IndexKernel.cu:93` 排查

你贴的 Windows / RTX 4070 Laptop 日志：

```
set CUDA_LAUNCH_BLOCKING=1
python tools/train_detector.py ... --cache disk --cache-dir D:/tyhn/yolo/dataset/runs/cache ...
cache hit  -> train_1280_disk_*.json  (29533 images, mode=disk)
...
CUDA error: device-side assert triggered
  File "...\jingwei\losses.py", line 101, in build_dense_targets
    g = gi[idx]
IndexKernel.cu:93: index out of bounds
```

`CUDA_LAUNCH_BLOCKING=1` 只是把异步错误定位到第一条出错的 kernel，**不是修复方法**。
这个位置有两类真实根因，当前版本都已修复：

1. **旧代码的兜底分支索引错误。** 极小 / 贴边目标没有任何 centre-sampling
   正样本时，旧代码把「网格展平索引」`nearest` 当成「GT 索引」去索引
   `torch.arange(G)`（长度只有目标数 G），在 CUDA 上直接越界。现在改为
   `g = arange(G)`、`idx = nearest.clamp(0, H*W-1)`；CPU 单元测试
   `test_34_dense_target_fallback_never_indexes_oob` 用 2×2 网格 + 两个贴边
   小框复现这个越界并验证不再发生。
2. **YOLO 标签类别号越界。** `nc=19` 时合法类别号必须是 `0..18`。如果 `.txt`
   实际是 `1..19`（1-based），或某个文件混入 `19 / 99 / -1`，CUDA 的
   `labels[g]` / `cls_all[:, labels]` 会触发 device-side assert。现在
   `sanitize_detection_targets()` 在 CPU 侧先丢弃非法类别 / 非法框并只警告一次；
   `build_dense_targets()` / `hungarian_match()` 还有明确 CPU-side
   `ValueError`，不再把问题藏进 CUDA kernel。

### 6.1 先用校验器看原始标签和缓存

```bat
python tools/validate_labels.py ^
    --cfg configs/jingwei_p.yaml ^
    --data D:/tyhn/yolo/dataset/dataset.yaml ^
    --nc 19 --splits train,val ^
    --cache disk --cache-dir D:/tyhn/yolo/dataset/runs/cache
```

它会检查：类别范围、1-based 证据、非有限 / 非正 / 越界框、格式错误行，
以及 `*.json` 缓存里是否还残留旧的非法类别号。输出示例：

```
raw class ids: count=...  min=1  max=19  inferred_shift=1
errors: bad_class=...  bad_box=...  malformed_line=...
cache: records=...  errors=...
RESULT: INVALID. Fix the reported labels / cache before CUDA training.
```

`--labels-dir "D:/.../labels/{split}"` 可在标签目录不符合
`images/ -> labels/` 约定时覆盖；`{images_dir}` 也可作为占位符。

### 6.2 两种修复方式（任选其一）

**A. 永久改成 YOLO 标准 `0..18`，并重建缓存（推荐）：**

```bat
python tools/validate_labels.py ^
    --cfg configs/jingwei_p.yaml ^
    --data D:/tyhn/yolo/dataset/dataset.yaml ^
    --nc 19 --splits train,val ^
    --fix-one-based --label-offset auto ^
    --cache disk --cache-dir D:/tyhn/yolo/dataset/runs/cache ^
    --rebuild-cache
```

默认会为每个被改写的 `.txt` 生成 `*.txt.bak`；`--no-backup` 可关闭。
`--rebuild-cache` 会重新生成 JSON/BIN/NPY，保证缓存里也是 0-based 标签。
报告里可能仍会列出 `stale_cache`（描述的是**修复前**的旧缓存），带
`[now rebuilt]` 标记且最终 `RESULT: OK` 即表示已修好。
缓存中的标签**始终是 0-based**，所以校验器不会对缓存再减一次偏移；
`label_offset` 只描述原始 `.txt` 的解释方式。
另外，缓存格式版本已升到 `CACHE_VERSION = 3`，旧缓存即使不使用校验器，
第一次训练也会自动重建一次，避免继续读到旧的非法标签。

**B. 不改 `.txt`，训练和建缓存时统一偏移：**

```bat
python tools/cache_dataset.py --cfg configs/jingwei_p.yaml --mode disk ^
    --data D:/tyhn/yolo/dataset/dataset.yaml --label-offset 1 ^
    --cache-dir D:/tyhn/yolo/dataset/runs/cache --rebuild

python tools/train_detector.py --cfg configs/jingwei_p.yaml --cache disk ^
    --cache-dir D:/tyhn/yolo/dataset/runs/cache --label-offset 1 ^
    --resume runs/dino/dino_init.pt --epochs 200 --batch 4 --out runs/detector
```

`--label-offset auto` 也可以：只有出现 `cls >= nc` 且所有类别都在 `1..nc`
时才自动减 1，不会误伤正常的 `0..18` 数据。

> **auto 是按整个 split 解析一次的**，不是逐图 / 逐 batch。否则某个 `.txt`
> 恰好只含 `1..nc-1` 时不会被平移，同一个数据集里 0-based 与 1-based 混用，
> 比直接报错更糟。训练启动时会打印 `label_offset=auto -> 1 (inspected N label
> files, nc=19)`；缓存元数据里 `label_offset` 记录的是**已解析**的值，
> `label_offset_requested` 保留原始请求。

相同参数也适用于 `tools/evaluate.py`、`tools/calibrate_fusion.py` 和
`tools/validate_labels.py`。训练 checkpoint 会记录 `label_offset`，
断点续训时若当前参数不一致会打印 WARN。

### 6.3 修复后的行为

- 训练启动日志会打印 `label_offset=...`；
- 非法 `.txt` / 缓存行会在 CPU 侧被丢弃并给出一次
  `RuntimeWarning` + `tools/validate_labels.py` 提示；
- 极小 / 贴边目标不再触发 CUDA 越界，仍会得到兜底正样本；
- 每 epoch 验证指标、`best.pt`、断点续训、未知疵点框、1660S 实时路径均不变；
- 可以把 `CUDA_LAUNCH_BLOCKING=1` 去掉再跑，正常训练速度更快。

相关单元测试：`test_33_sanitize_detection_targets_and_label_offset`、
`test_34_dense_target_fallback_never_indexes_oob`、
`test_35_validate_labels_cli_fix_and_rebuild_cache`、
`test_36_cache_builder_normalises_one_based_labels`。
