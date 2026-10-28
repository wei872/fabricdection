# 每 epoch 验证指标、大数据集训练速度与 1660S 实时预测

> 针对真实训练中的三个问题：
> 1. “为什么日志只有 loss，没有验证集精确率 / 召回率？”
> 2. “图片读取太慢，能不能像 YOLO 一样先生成 `train.cache`？”（结论：对本数据集无效，已移除）
> 3. “未知疵点类型也要生成预测框，同时必须保证 GTX 1660 SUPER 实时。”

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
- 验证默认 `--val-batch 8 --val-workers 2`，一次遍历同时算 mAP / PR / 大缺陷召回。
- 独立评估：`python tools/evaluate.py --cfg configs/jingwei_p.yaml --ckpt runs/detector/best.pt --batch 8 --workers 2`。

> 你之前贴的日志里 epoch 汇总只有 `cls_o2m / fg_context ... total`，那是训练 loss，
> 不是验证指标；且该日志没有 `fg_defect/fg_fabric/fg_fold/fg_global` 子损失，
> 说明当时运行的是旧版本。请先拉取最新提交再训练。

---

## 2. 大数据集读取：为什么移除了 `train.cache`

早期版本提供 `tools/cache_dataset.py` + `jingwei/cache.py`（disk / ram / bytes）。
在本项目的数据集（29533 张 1012×2448 PNG，共 145 GB）上实测**加缓存反而更慢**，
因此缓存功能已被彻底删除；`tools/train_*.py` / `evaluate.py` 不再接受 `--cache`。

| 方案 | 实测 / 问题 |
|---|---|
| `disk`（npy 内存映射） | train 缓存 145 GB、val 36 GB，D: 盘被占满；Windows `spawn` 会把持有 mmap 的 Dataset 整个 pickle → `MemoryError` / `EOFError` |
| `ram` | 需要上百 GB 内存，16 GB 机器不可行 |
| `bytes` | 只省 I/O，PNG 解码仍是 41 ms/张，无收益 |

真正有效的优化（均已落地）：

1. **uint8 光度增强**：`fabric_photometric` 改为 LUT + 低分辨率光照场 + `cv2.multiply`，
   1280² 下 277 ms → **约 12 ms/张**，数值差 ≤7 LSB；
2. **先 resize 再增强**：`geometric_augment` 先把长条原图缩到 1280²，再做光度处理；
3. **多 worker + 正确播种**：`--num-workers 4`，`make_worker_init_fn` 保证每个 worker
   增强不同（未播种时所有 worker 会得到完全相同的增强）；
4. **AMP + 大 batch**：CUDA 默认 `--amp bf16`，batch 8 显存约 4.3 GiB；
5. **验证批量化**：`--val-batch 8 --val-workers 2`，且 mAP 与大缺陷召回一次遍历完成。

实测（RTX 4070 Laptop，`jingwei_n`，1280²）：

| 项目 | 结果 |
|---|---|
| 单张：解码 | 41 ms（PNG 1012×2448） |
| 单张：geom + photo + to_chw | 约 25 ms |
| 数据管线 4 workers | **39 img/s**（6 workers 48 img/s） |
| GPU 训练 batch 8 + bf16 | **约 21 img/s**（单步约 0.38 s） |
| GPU 训练 batch 4 + bf16 | 约 19 img/s |

即 4 个 worker 的读图速度已是 GPU 消耗速度的约 2 倍，继续加 worker 不会更快。
如果仍然读图慢，先确认 **D: 盘没有满**（缓存曾占掉 183 GB，只剩 17 GB）。

```bash
# 推荐：阶段 A
python tools/train_detector.py --cfg configs/jingwei_p.yaml \
    --num-workers 4 --amp bf16 --batch 8 --epochs 200
```

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
# 1) 阶段 0 DINOv2 离线蒸馏
python tools/pretrain_dino.py --cfg configs/jingwei_p.yaml --num-workers 4 \
    --iters 5000 --out runs/dino

# 2) 阶段 A 监督检测（每 epoch 输出验证指标）
python tools/train_detector.py --cfg configs/jingwei_p.yaml --num-workers 4 --amp bf16 \
    --resume runs/dino/dino_init.pt --epochs 200 --val-interval 1 \
    --val-max 300 --normal-max 300 --out runs/detector

# 3) 阶段 B  normality / 记忆库
python tools/train_normality.py --cfg configs/jingwei_p.yaml --num-workers 4 \
    --ckpt runs/detector/best.pt --iters 10000 --out runs/normality

# 4) 评估 + 1660S 实时预测
python tools/evaluate.py --cfg configs/jingwei_p.yaml \
    --ckpt runs/normality/last.pt --batch 8 --workers 2
python tools/predict.py --cfg configs/jingwei_p.yaml --realtime-1660 \
    --ckpt runs/detector/best.pt --source D:/产线图片/ --out runs/predict
```
