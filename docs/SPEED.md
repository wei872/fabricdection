# 训练速度：为什么“加了缓存反而更慢”，以及怎么真正跑快

> 对应问题：29 533 张 1280×1280 数据，`--cache disk` 命中缓存后仍然
> `0.13 ~ 0.9 it/s`，ETA 上千小时；`--num-workers 4` 直接
> `MemoryError / EOFError: Ran out of input`，所以只能 `--num-workers 0`。

先说结论：**缓存本身没问题，真正的瓶颈是数据增强里的一次高斯模糊**，
以及 `np.memmap` 被 pickle 进 worker 导致只能退化成 0 个 worker。
两处都已在本次提交修复，`--num-workers 4/8` 现在可以正常用。

---

## 1. 三个叠加的根因

### 1.1 反光（reflectance）增强：单张图 0.41 秒

旧代码在 `jingwei/data.py::fabric_photometric` 里：

```python
field = cv2.resize(field, (w, h), interpolation=cv2.INTER_CUBIC)
field = cv2.GaussianBlur(field, (0, 0), sigmaX=max(4, w / 12))   # 1280/12 ≈ 106 px
x = np.clip(x * (1.0 + 0.18 * field), 0, 1)                      # 全图 float32
```

- 1280 宽的图像 `sigmaX ≈ 106`，OpenCV 需要极大核（或 DFT），
  **单次调用就要 ~0.54 s**；
- 而且整条链路是 `uint8 → float32 → 逐元素乘加/clip → uint8`，
  每次都在 4.9 MB×3 通道上做多趟读写。

以 `--batch 8` 计：`8 × 0.42 s ≈ 3.4 s` 才凑出一个 batch，加上模型 step，
就是你看到的 `0.08 ~ 0.14 it/s`。**和磁盘、和缓存都没有关系**，
所以“缓存命中”自然一点都没变快。

### 1.2 每张图都做一次 uint8 → float CHW

`to_chw_float()` 在 **CPU 上逐样本**做 `transpose(2,0,1) → float32 → /255`：
4.9 MB → 18.9 MB 的分配与读写，在 1280² 上约 6.7 ms/张，
再走一遍 `pin_memory` 和 PCIe。这部分本可以放到 GPU 上一次性做。

### 1.3 `--num-workers > 0` 直接 MemoryError

Windows 上是 `spawn`，worker 启动时要把 dataset 对象 pickle 过去。
`np.memmap` 继承 `ndarray.__reduce__`，**pickle 会把整个 145 GB 的缓存内容
序列化给每个 worker**——于是：

```
reduction.dump(process_obj, to_child) -> MemoryError
EOFError: Ran out of input
```

所以只能 `--num-workers 0`：数据增强、解码、collate 全部串行在训练主进程，
GPU 大部分时间在等数据。

---

## 2. 修好之后（本机 2 核 CPU 实测，1280×1280）

| 步骤 | 修复前 | 修复后 | 倍数 |
|---|---|---|---|
| `fabric_photometric` | 411.3 ms | 2.9 ms | **×143** |
| `geometric_augment` | 8.0 ms | 1.6 ms | ×5 |
| 单样本（增强 + uint8 tensor） | ~419 ms | ~4.5 ms | **×94** |
| 每批 uint8→float CHW | CPU 6.7 ms/张 | GPU 上一次 `div_` | — |
| `--num-workers 4` | MemoryError | 正常，4 个 worker 各自 mmap | — |

改法（都在 `jingwei/data.py`）：

1. **反光场不再做全分辨率高斯**：在一个 ~12×12 的控制点网格上采样高斯场，
   用 `cv2.resize(..., INTER_LINEAR)` 升采样到全图（`sigma≈110 px` 的平滑场
   与 12 点双线性插值视觉上等价），耗时从 ~540 ms 降到 ~0.6 ms。
2. **整条增强链路留在 uint8**：曝光/对比度合成成一张 256 项 LUT（`cv2.LUT`），
   反光用 `cv2.multiply(img, gain8, scale=1/64)` 定点乘法，经纬粗细用 3×3 盒滤波，
   布幅折入只改边带切片。
3. **几何增强重排**：先各向异性 resize → 最后一次 resize 到 1280 →
   再做翻转/rot90（`cv2.flip` / `cv2.rotate`），不再在原始大图上反复整图拷贝。
   顺带修正了 rot90 的框变换（原来用 `w` 同时当宽高，非方图会错位）。
4. **dataset 输出 uint8 HWC**，`collate` 只做一次 `torch.stack`，
   `to_model_input()` 在 GPU 上做 `permute → float → /255`，
   PCIe 传输量减到 1/3，CPU 少一次 18.9 MB 的浮点往返。
5. **worker 不再搬运像素**：`_CachedBase.__getstate__` 丢掉 memmap，
   worker 里用文件路径自己重新 `mmap`（只读，OS page cache 共享）。
6. `worker_init_fn` 里 `cv2.setNumThreads(1)`，避免 4 个 worker 各开满 CPU 线程互相抢。
7. **worker 启动方式默认 `spawn`**（`--mp-start spawn`，与 Windows 一致）：
   Linux 上 `fork` 在 torch/OpenMP 线程已初始化后会**死锁在第一个 batch**
   （表现为“卡在 iter 1 不动”）。spawn 慢 1~2 秒启动一次
   （`persistent_workers=True`，整轮只付一次），换来跨平台一致、不死锁。
   训练开始前会打印 `workers ready in 1.3s (4 processes, start=spawn)`，
   第一个 iter 的 `data=` 不再包含这段启动时间。

---

## 3. 现在推荐的训练命令（你的 4070 Laptop + 29 533 张）

```bat
python -u tools/train_detector.py ^
  --cfg configs/jingwei_p.yaml --data configs/fabric19.yaml ^
  --cache disk --cache-dir D:/tyhn/yolo/dataset/runs/cache ^
  --resume runs/dino/dino_init.pt ^
  --epochs 200 --batch 8 --num-workers 4 --prefetch-factor 4 ^
  --amp auto --log-interval 20 --val-interval 1 --val-max 1000 ^
  --out runs/detector
```

- `--amp auto`：4070 支持 bf16，自动用 bf16（**不需要** loss scaling）；
  Turing/16 系（1660 SUPER）没有 bf16，会退回 fp16 + `GradScaler`；
  想完全 fp32 就 `--amp off`。
- `--num-workers`：Windows 默认 2，Linux 默认 4；**先用
  `--bench-data 20` 选最优值**，见下节。`--prefetch-factor` 默认 2
  （每个预取 batch 在 1280²/batch8 时约 39 MB，别调太大）。
  `--mp-start spawn` 是默认值，Windows 上没有选择（也无需改）。
- `--val-max 1000`：7383 张验证集每 epoch 全跑一遍大约多花 1 分钟；
  训练期抽 1000 张看趋势，最终再用 `tools/evaluate.py` 全量评估。
- `--epoch-iters 1000`：想看指标更快时，一个 epoch 只跑 1000 个 iter。

先量数据侧，再决定 worker 数 / 缓存模式：

```bat
python tools/bench_dataloader.py --cfg configs/jingwei_p.yaml --data configs/fabric19.yaml ^
  --cache disk --cache-dir D:/tyhn/yolo/dataset/runs/cache --batch 8 --num-workers 4 --batches 20
:: 也可对比：--cache none / --cache bytes / --num-workers 0 8 12 / --no-aug
```

输出 `img/s` 和“一个 epoch 仅加载耗时”。判读规则：

- loader 的 `img/s ÷ batch` **明显大于**训练日志里的 `it/s`
  → 瓶颈在 GPU，继续加大 `--batch` 或开 `--amp`；
- 小于 `it/s × batch` → 瓶颈在数据，增加 `--num-workers`
  （一般到 `min(CPU 物理核数, 8)` 收益就见顶），或用 `--cache disk`；
- 两者接近 → 已经平衡，别再加 worker（只会抢 CPU）。

训练日志现在每次打印都带数据/计算拆分，可以直接看瓶颈：

```
epoch   1/200  iter    20/3692  loss=...  lr=...  12.4 it/s  data=18ms step=62ms  ...
```

`data=18ms` 是等一个 batch 的时间（含 worker 增强），`step=62ms` 是
forward+backward+optimizer。**`step` 远大于 `data` 就是 GPU 受限**，
反之调 worker / 换缓存模式。

---

## 4. 其它加速项（按收益排序）

| 手段 | 收益（1280², 4070 Laptop 量级） | 备注 |
|---|---|---|
| 修好增强 + 多 worker | **几十倍**（本次提交） | 之前 0.1 it/s，现在受 GPU 限制 |
| `--amp auto`（bf16） | step 约 1.3–1.8× | 与 pico 模型、NMS-free head 兼容；bf16 无需 scaler |
| `--batch 8/16` | 吞吐 +20–40% | 显存够就行；lr 可酌情 ×1.2 |
| `--cache disk` | 去掉 JPEG 解码（20–40 ms/张） | 145 GB；机械盘随机读也可能 5–15 ms/张，此时 `bytes` 更省空间 |
| 每 N 个 epoch 验证 | val 全量 7383 张 ≈ 1 min | `--val-interval 2/5` |
| `--epoch-iters` | 灵活控制每 epoch 时长 | 适合先跑通再放大 |

> AMP 的正确性：`jingwei/losses.py::hungarian_match` 里有
> `.cpu().numpy()`，在 autocast（bf16/fp16）下 numpy 不接受 bf16，
> 会抛 `TypeError: Got unsupported ScalarType BFloat16`。已在本次提交里
> 改成 `.detach().float().cpu().numpy()`，并加了单元测试
> （`test_41_amp_and_loader_flags`：bf16 autocast 下前向+反向，loss 与梯度必须有限）。

---

## 5. 常见问题

**Q: `--num-workers 4` 还是 MemoryError？**
A: 确认拉到了本次提交（`jingwei/cache.py` 里应有 `__getstate__`）。
另外 Windows 每个 worker 都是独立进程，`--batch` 太大 + `--prefetch-factor`
过高也会把内存打满：先 `--prefetch-factor 2`。
`ram` 模式 + 多 worker 会每个 worker 复制一份全量缓存，代码里已自动降级为
`disk`（mmap），务必用 `disk`。

**Q: 缓存还值得留吗？**
A: 值得，但**不是**因为 I/O：它把 JPEG 解码（20–40 ms/张，CPU 受限）
变成一次 mmap 读（0.5 ms）。如果你的盘是机械盘且缓存读只有 5–15 ms/张，
用 `--cache bytes`（只存原始字节，占空间 ~1/8）也能避免随机寻道。
如果暂时不想用缓存，`--cache none` 也是完整支持的路径（解码在 worker 里并行）。

**Q: 之前 0.9 it/s 里 “it/s” 为什么忽高忽低？**
A: 日志里的 `it/s` 是区间平均，前几个 batch 包含 worker 启动、page cache 预热；
稳定后才是真实速度。现在 `--bench-data` 会在训练前给出稳定值。

**Q: 增强变快了，会不会影响精度？**
A: 分布没变：曝光/对比度仍然是 `(x-0.5)*U(0.75,1.35)+0.5*U(0.85,1.15)`（改成
LUT 等价的 8-bit 查表），反光仍是零均值平滑场 × `1±0.18`，
经纬粗细仍是轻度模糊/锐化，布幅折入只改边带亮度、不动标注框。
`tests/test_model.py::test_37_augmentation_speed_budget` 与
`test_39_geometric_augment_box_alignment` 分别守住“速度”和“框对齐”。
