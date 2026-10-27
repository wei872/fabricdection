# 实时部署手册（1280×1280）

> 配套设计说明见 [DESIGN.md](DESIGN.md) §8。

## 1. 部署拓扑

```
相机 (1280², 25~120 FPS)
   │  DMA / GMSL / GigE
   ▼
[GPU] TRT engine  ──► 8 个张量（静态形状）[+n3/n5 if roi_head]
   │  cls3(1,19,160,160)  reg3(1,4,160,160)  obj3(1,1,160,160)
   │  cls5(1,19,40,40)    reg5(1,4,40,40)    obj5(1,1,40,40)
   │  anomaly(1,1,G,G)       context(1,3,G,G)
   ▼
[CPU/GPU 后处理，固定上界]
   1. 布种描述符 → prototype id            O(1280²) FFT，可与上一帧流水
   2. decode: top-k(300) + 阈值            ~27k 候选，无 NMS
   3. 连通域安全网（160² 上） + 局部+全局注意力 context 动态前景/折入抑制
   4. ROI 二级分类（可选）
   ▼
结果（19 类框 / 未知异常框）→ PLC / MES
```

**关键点**：只有张量部分进 TRT；decode 与安全网在图外，全是固定形状、
固定上界的操作，没有数据相关的循环，保证 P99 延迟可控。

## 2. 导出

```bash
python tools/export.py \
    --cfg configs/jingwei_n.yaml \
    --ckpt runs/bank/memory_bank.pt \
    --proto 0 --check \
    --out runs/export/jingwei_n.onnx \
    --opset 17
```

导出前会自动：

- `strip_train_only()`：删除 `obj`（o2m）、训练期 `fg` 对齐分支；**保留** `FabricContext` 局部+全局注意力前景头；
- `bank.set_active(proto)`：把该布种原型的 centroids 固化进 1×1 卷积权重；
- 把高斯平滑核变成常量 buffer，使 `anomaly` 输出也是静态形状。

实测导出结果（`onnx.checker` 通过，约 24 MB）：

```
images  (1,3,1280,1280)
cls3    (1,19,160,160)     reg3  (1,4,160,160)     obj3 (1,1,160,160)
cls5    (1,19,40,40)       reg5  (1,4,40,40)       obj5 (1,1,40,40)
anomaly (1,1,160,160)      context(1,3,160,160) = [defect_fg, fabric_mask, fold_band]
```
`jingwei_p` 的 `anomaly / context` 为 `(1,1,128,128)` / `(1,3,128,128)`，
其余输出形状与通道语义不变。

### 2.1 多布种怎么办

三种方案，按推荐顺序：

1. **产线已知布种**（MES 下发）：为每个布种导出一个 engine（或导出 K 个，
   运行时按 prototype id 选择）。K 个小 engine 切换代价低。
2. **单 engine + 运行时换权重**：`bank.set_active(k)` 只改一个 1×1 卷积的权重
   （`K×D = 256×128 = 32768` 个 float，0.13 MB），切换 < 0.1 ms。
   需要 TRT 支持 refit 或把该权重作为图输入。
3. **K 路并行**：把 K 组 centroids 堆成一个 `1×1 conv(D, K·K)`，一次算完再按
   prototype 选对应通道段。计算量 ×K，仅在 K 很小（≤4）时划算。

## 3. TensorRT 构建

```bash
trtexec --onnx=runs/export/jingwei_n.onnx \
        --saveEngine=runs/export/jingwei_n.engine \
        --fp16 --workspace=4096 \
        --minShapes=images:1x3x1280x1280 \
        --optShapes=images:1x3x1280x1280 \
        --maxShapes=images:1x3x1280x1280
```

INT8 量化（需要每个布种各约 100 张正常图的校准集）：

```bash
trtexec --onnx=runs/export/jingwei_n.onnx --saveEngine=.../jingwei_n_int8.engine \
        --int8 --calib=<calibration_cache> --fp16 ...
```

**INT8 注意**：异常分数是 cosine 距离，量化误差会直接抬高 FPR。
务必在 INT8 引擎上**重新跑阶段 C 的阈值标定**（`build_memory_bank.py --ckpt`
指向 INT8 校准后的权重），不要沿用 FP32 的 `tau`。

## 4. 延迟预算（参考标定，非实测）

| 设备 | jingwei_p | jingwei_n | jingwei_s |
|---|---:|---:|---:|
| RTX PRO 6000 | 1.99 ms / 502 FPS | 2.83 ms / 354 FPS | 4.54 ms / 221 FPS |
| RTX 4090 | 2.39 ms / 419 FPS | 3.64 ms / 275 FPS | 6.20 ms / 161 FPS |
| RTX 3060 | 4.36 ms / 229 FPS | 7.70 ms / 130 FPS | 14.54 ms / 69 FPS |
| Jetson AGX Orin | 7.52 ms / 133 FPS | 14.20 ms / 70 FPS | 27.88 ms / 36 FPS |

估算方法见 DESIGN.md §8.2（锚定 YOLO27n 官方 0.44 ms / 7.2 G MACs）。
该估算**只含计算**，未含相机曝光/传输、PLC 通信与 GPU 频率波动。
上线前必须在目标设备上用真实 1000 帧测 P50/P99：

```bash
python tools/evaluate.py --cfg configs/jingwei_n.yaml --ckpt runs/bank/memory_bank.pt
# 或直接用 trtexec 的 --duration/--iterations 做 kernel 级 benchmark
```

**验收**：`P99 ≤ 25 ms` 且 `P99/P50 ≤ 1.3`。

## 4.1 GTX 1660 SUPER 专项（无 Tensor Core，重点）

### 术语澄清：TensorRT ≠ Tensor Core

**1660 SUPER 可以用 TensorRT，但它没有 Tensor Core。** 这张卡是 Turing TU116、
计算能力 **sm_75 / 7.5**；NVIDIA 官方支持矩阵说明 TensorRT 11.4 支持
SM 7.5 及以上硬件，因此本项目的 ONNX → TensorRT 部署路径在 1660S 上完全可用
（[TensorRT 支持矩阵](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html)、
[Turing 架构说明](https://developer.nvidia.com/blog/nvidia-turing-architecture-in-depth/)、
[1660S 规格](https://www.techpowerup.com/gpu-specs/geforce-gtx-1660-super.c3458)）。

**但“能跑 TensorRT”不等于“有 Tensor Core”。** TU116 物理上没有 Tensor Core /
RT Core；FP16 只是 2:1 的 shader 速率，INT8 只能走 DP4A，不能套用 RTX 显卡上
FP16/INT8 Tensor Core 的加速数据。所以 §4.1 的 A/B 实测建议依然成立：先构建 FP32
engine，再对比 FP16 engine，取 P99 更快的那个。

### 硬件事实（决定了不能直接套用 §4 的表）

GTX 1660 SUPER = Turing **TU116，1408 CUDA 核，1785 MHz，没有 Tensor Core、没有 RT Core**：

| 项 | 数值 |
|---|---|
| FP32 | **5.03 TFLOPS** |
| FP16 | 10.05 TFLOPS（**2:1 的 shader 速率**，不是 Tensor Core 的 8×） |
| 显存 | 6 GB GDDR6 / 192-bit / **336 GB/s** |
| PCIe | 3.0 ×16 |
| 计算能力 | sm_75（**但无 TC**） |

参考：[TechPowerUp 规格](https://www.techpowerup.com/gpu-specs/geforce-gtx-1660-super.c3458)、
[Tom's Hardware 对 TU116 用专用 FP16 核替代 Tensor Core 的说明](https://www.tomshardware.com/reviews/nvidia-geforce-gtx-1660-turing-tu116,6027.html)、
[wccftech 评测确认无 Tensor Core / 336 GB/s](https://wccftech.com/review/nvidia-geforce-gtx-1660-super-review-msi-gaming-x/2/)。

**§4 的表是"以 YOLO27n 在 RTX PRO 6000 上的 FP16 TRT 0.44 ms 为锚点"标定的，
这个锚点依赖 Tensor Core，在 1660S 上无效。** 必须用独立的"峰值 × 实际效率"模型。

### 估算（1280²，FP32 TRT，效率 20–35% + 4 ms 固定开销）

| 变体 | MACs | 1660S 估算延迟 | 估算 FPS | 能否实时（≤25 ms） |
|---|---|---|---|---|
| **jingwei_p** | **8.09 G** | **13.2 – 20.1 ms** | **50 – 76 FPS** | ✅ 稳 |
| jingwei_n | 16.62 G | 22.9 – 37.0 ms | 27 – 44 FPS | ⚠️ 临界，看实测 |
| jingwei_s | 34.11 G | 42.8 – 71.8 ms | 14 – 23 FPS | ❌ 基本不行 |

（FP16 shader 若能达到 ~1.3×：p 11.1–16.4 ms / n 18.5–29.4 ms / s 33.8–56.2 ms。
但见下方风险，**不要假设 FP16 一定更快**。）

### 三个必须先验证的风险

1. **FP16 可能不加速甚至倒退**。1660S 是 sm_75 但无 Tensor Core，TensorRT 的
   FP16 卷积会落到 `IMPLICIT_PRECOMP_GEMM`，而 FP32 能走 Winograd——已有
   [NVIDIA/TensorRT #4714](https://github.com/NVIDIA/tensorrt/issues/4714) 与
   [vLLM #54950](https://github.com/vllm-project/vllm/issues/54950) 报告这类卡上
   FP16 比 FP32 慢。**必须 `--precision both` 做 A/B**。
2. **INT8 不要指望**。没有 INT8 Tensor Core，只能走 DP4A 整数点积，收益不确定；
   而且异常分数是 cosine 距离，量化误差会直接抬高 FPR。建议先跳过 INT8。
3. **前后处理占比变高**。本仓库在 CPU 上实测 decode + 安全网约 **3.5 ms**
   （`tools/bench_device.py` 的 `post_ms_p50`）；再加 H2D（1280²×3 uint8 ≈ 5 MB，
   PCIe 3.0 约 0.5 ms；若传 FP32 张量则 20 MB ≈ 2 ms）。所以 §4 的 1.2 ms 开销
   对 1660S 太乐观，这里按 4 ms 估。**用 uint8 输入、在 GPU 上做 /255。**

### 落地步骤（在 1660S 上 30 分钟内得到真实答案）

```bash
# 1) 用 pico 配置（专为无 TC 卡裁剪，保留全部"大缺陷不漏"机制）
python tools/benchmark.py --cfg configs/jingwei_p.yaml

# 2) 在 1660S 上实测 PyTorch 图（上界），A/B 测 FP32 / FP16
python tools/bench_device.py --cfg configs/jingwei_p.yaml --precision both
python tools/bench_device.py --cfg configs/jingwei_p.yaml --compare   # 同时测 p/n/s
# 预测 CLI 可自动检测 1660/1650/TU116/TU117，强制 pico + FP32 并打印 P99
python tools/predict.py --cfg configs/jingwei_p.yaml --realtime-1660 \
    --ckpt runs/detector/best.pt --source D:/产线图片/ --out runs/predict
# 注意：best.pt 也必须由 jingwei_p.yaml 训练得到；n/s 权重切 pico 会警告且精度不可信

# 3) 导出并构建 TRT engine（先试 FP32，再试 FP16，不要直接上 INT8）
python tools/export.py --cfg configs/jingwei_p.yaml --ckpt runs/bank/memory_bank.pt \
    --out runs/export/jingwei_p.onnx --check
trtexec --onnx=runs/export/jingwei_p.onnx --saveEngine=runs/export/jingwei_p.engine \
        --workspace=2048 --duration=60 \
        --minShapes=images:1x3x1280x1280 --optShapes=images:1x3x1280x1280 \
        --maxShapes=images:1x3x1280x1280
# 再构建一份 --fp16 的，比较两者 P99，取快的那个

# 4) 若 DWConv 在无 TC 卡上成为访存瓶颈，A/B 切换成密集卷积
#    configs/jingwei_p.yaml:  model.dwconv: false
python tools/bench_device.py --cfg configs/jingwei_p.yaml --precision both
```

`tools/bench_device.py` 会打印 GPU 名称、sm 版本、是否疑似无 Tensor Core，
并给出 `net / post / P50 / P95 / P99 / FPS` 以及 `P99 ≤ 25 ms` 的判定，
结果写入 `runs/bench_device.json`。**PyTorch 图是上界，TRT engine 通常再快 1.5–3×。**

### 显存

6 GB 完全够：pico 参数 2.93 M（FP32 约 11 MB），1280² batch=1 的激活峰值
估计 < 1 GB，TRT workspace 1–2 GB。无需担心 OOM；别把 batch 调大即可。

### 结论

- **原版 jingwei_n 在 1660S 上处于临界**（约 27–44 FPS），有 risk，不建议直接上产线；
- **jingwei_s 达不到实时**；
- **换成 `configs/jingwei_p.yaml` 后，1660S 上估算约 50–76 FPS**，
  且保留 LocalContrastNorm、布种原型、双无监督信号、**局部+全局注意力动态前景/折入上下文**与确定性安全网，
  "大型缺陷不漏"的机制没有被削弱，只是砍掉了通道数、2 个重建块和 ROI 二级分类。
- 最终以你自己机器上的 `bench_device.py` + `trtexec` 实测为准。

---

## 5. 流水线并行（把延迟藏起来）

```
时间 ─►
帧 N  : [CPU: 读图+LCN预处理] [GPU: engine N] [CPU: decode+安全网 N]
帧 N+1:                    [CPU: 读图]      [GPU: engine N+1] [CPU: post N+1]
```

- 布种描述符（2D FFT）只依赖当前帧，可与 GPU 推理重叠；
- `context(1,3,G,G)` 与 `anomaly` 一起出图；三通道来自局部+全局注意力，安全网据此动态拒绝背景和正常折入，不依赖固定边缘像素；
- `LocalContrastNorm` 已包含在导出图内，无需 CPU 预处理；
- 用 CUDA streams + pinned memory + 双缓冲，把 H2D/D2H 与计算重叠；
- 后处理的连通域在 160² 上做，通常 < 0.3 ms（CPU）。

## 6. 上线检查清单

- [ ] `configs/fabric19.yaml` 四个路径正确，val 与 train 无泄漏
- [ ] 阶段 A–E 全部跑完，`runs/bank/memory_bank.pt` 存在
- [ ] `configs/fusion_calibrated.yaml` 的阈值已回填到模型配置
- [ ] `Recall@large ≥ 99.5%`、`FPR@normal ≤ 0.5%`、`FPR@fold ≤ 0.5%`（真实 val/normalval）
- [ ] 每个布种都有 normaltrain 覆盖；未覆盖原型走 conservative fallback
- [ ] 正常折入/布幅变化样本已覆盖；局部+全局注意力 `foreground` 阈值按产线 A/B 调好（不再调 `edge_band_px`）
- [ ] 生产图片/视频路径含中文/空格时用 `jingwei/image_io.py` 读写（已默认接入）；回归测试 `test_24_unicode_paths` 通过
- [ ] 非方形长条布幅帧（如 1040×2448）在 `augment=True` 下能正常训练；回归测试 `test_25_non_square_photometric_augment` 通过
- [ ] 阶段 0 / A / B 训练日志可见（启动信息 + batch 进度 / ETA）；回归测试 `test_26_training_progress_logs` 通过
- [ ] 阶段 0 / A / B 可用 `last.pt` 断点续训；回归测试 `test_27_resume_training` 通过
- [ ] 真实 `NormalImageDataset` / `YOLODefectDataset` 的 numpy RNG state 可写入 `weights_only=True` checkpoint 并恢复；回归测试 `test_28_numpy_rng_state_checkpoint` 通过
- [ ] 阶段 A 每 epoch 输出 val 精确率 / 召回率 / F1 / mAP50 / 大缺陷召回，并用 `best.pt` 部署；回归测试 `test_30_validation_metrics_and_best_checkpoint` 通过
- [ ] 训练集已生成 YOLO 式 `train.cache` 并用 `--cache disk` 训练；中文路径回归测试 `test_31_yolo_style_image_cache` 通过
- [ ] 未知疵点类型也生成预测框，`emit_unknown_boxes / unknown_min_area_ratio / max_unknown_boxes` 按产线 FPR A/B 调好；回归测试 `test_32_unknown_defect_boxes` 通过
- [ ] ONNX 8 个输出（含 context）形状静态；TRT engine 在目标设备构建成功
- [ ] INT8 情况下阈值已重新标定
- [ ] 1000 帧 P50/P99 延迟达标，P99/P50 ≤ 1.3
- [ ] 影子模式（只记录不拦截）运行 2 周，回归误报难例
- [ ] 相机端光源/偏振检查：高频反光条纹无法由算法完全消除

## 7. 运行时最小示例（Python，PyTorch 版）

```python
import cv2, torch, numpy as np
from jingwei import build_model, load_config
from jingwei.fabric import FabricPrototypes, fabric_descriptor

cfg = load_config('configs/jingwei_n.yaml')
model = build_model(cfg).eval().cuda()
model.load_state_dict(torch.load('runs/bank/memory_bank.pt')['model'], strict=False)
proto = np.load('runs/bank/fabric_prototypes.npz')
FP = FabricPrototypes(proto['centers'], proto['mean'], proto['std'])

def infer(bgr):
    k = FP.assign(fabric_descriptor(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)))
    rgb = cv2.cvtColor(cv2.resize(bgr, (1280, 1280)), cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb.transpose(2,0,1)).float()[None].cuda() / 255.0
    with torch.no_grad(), torch.cuda.amp.autocast():          # FP16
        d = model(x, proto_id=k)[0]
    return d['boxes'].cpu(), d['scores'].cpu(), d['labels'].cpu(), k
```

`tools/predict.py` 提供等价的 CLI（含画框与 JSON 输出）。加 `--realtime-1660`
会在 1660/1650/TU116/TU117 上自动切 pico + FP32、实测 P99，并在结果 JSON 中
记录 `realtime_1660 / emit_unknown / unknown_boxes`。未知疵点类型默认生成
`label=19 / 未知异常` 预测框；`--no-unknown` 可关闭。
