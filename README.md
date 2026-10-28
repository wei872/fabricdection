# 经纬 · JingWei — 1280×1280 布匹缺陷实时检测

一套面向布匹（19 类缺陷）的**新模型设计 + 可运行的 PyTorch 参考实现**。
融合三篇工作的互补优势：

| 参考 | 借鉴 |
|---|---|
| [Ultralytics YOLO27](https://docs.ultralytics.com/models/yolo27) | 双尺度检测、加宽浅层、NMS-free、前景对齐监督、深层注意力 → 实时骨架 |
| [PatchCore](https://github.com/amazon-science/patchcore-inspection) | 局部聚合块特征、正常特征记忆库、coreset 压缩 → **布种条件记忆库** |
| [Dinomaly (CVPR 2025)](https://github.com/guojiajeremy/Dinomaly) | 冻结基础特征 + 噪声瓶颈 + 线性注意力 + 宽松重建 → **轻量重建残差** |
| [DINOv2 (TMLR 2024)](https://github.com/facebookresearch/dinov2) | 自监督密集 patch 特征 → **阶段 0 离线 teacher 蒸馏**，训练后丢弃，不进推理图 |

**核心思路**：一个共享 CNN 主干同时喂给「有监督 19 类检测头」和「无监督正常性双信号」，
再用一个**确定性安全网**把异常图上任何未被覆盖的大连通域强制输出成框 ——
因此大型缺陷不依赖检测器的置信度，也不会被漏标污染。布种变化
（经纬线粗细 / 经纬密 / 反光）由「局部对比度归一化 + 布种原型 + 每原型独立记忆库」处理。
**布幅变化导致的边缘正常折入**则由逐帧**局部+全局注意力**
`fabric_mask / fold_band / defect_fg` 前景上下文单独处理（动态跟随布边，
不再依赖固定边缘像素），避免被安全网误判成疵点；详见
[docs/FOREGROUND.md](docs/FOREGROUND.md)。

---

## 交付物

| 文件 | 内容 |
|---|---|
| **[docs/DESIGN.md](docs/DESIGN.md)** | **完整设计说明**（需求/架构/公式/训练/实时预算/风险/路线图），主交付物 |
| [docs/TRAINING.md](docs/TRAINING.md) | 数据准备、阶段 0 + 五阶段训练命令、调参与故障排查 |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | ONNX/TensorRT 导出、延迟预算、流水线、上线清单 |
| **[docs/DINO_PRETRAIN.md](docs/DINO_PRETRAIN.md)** | **用 DINOv2 做阶段 0 离线蒸馏预训练：为什么需要、命令、许可证、A/B 验证** |
| **[docs/FOREGROUND.md](docs/FOREGROUND.md)** | **局部+全局注意力前景/背景分离、动态布幅跟随、边缘正常折入误报抑制** |
| **[docs/CACHE_AND_VALIDATION.md](docs/CACHE_AND_VALIDATION.md)** | **每 epoch 精确率/召回率/mAP、未知疵点框、1660S 实时预测命令，以及大数据集训练速度实测** |
| **[docs/CODE_REVIEW.md](docs/CODE_REVIEW.md)** | **代码审查与优化建议：已修复 15 项（含融合门控标定失效、全幅误框、框坐标、LCN 算子量、NumPy 2 兼容）+ 13 项待办与优先级** |
| `jingwei/` | 参考实现：主干 / 双尺度头 / 记忆库 / 线性注意力重建 / 安全网 / 损失 / 数据集 / **Unicode 安全图像 I/O** / **安全 checkpoint 加载** / DINO teacher 工具 |
| `configs/` | `jingwei_n.yaml`（实时优先）、`jingwei_s.yaml`（精度优先）、**`jingwei_p.yaml`（GTX 1660 SUPER 等无 Tensor Core 卡专用）**、`fabric19.yaml`（数据）；均含 `dino:` 阶段 0 配置 |
| `tools/` | **DINOv2 蒸馏预训练**、训练（启动 / batch 级日志 + **完整断点续训** + **每 epoch 验证指标**）、建库、标定、评估、推理、导出、算力剖析、**真实设备延迟实测** CLI |
| `tests/test_model.py` | 42 项单元测试（含中文路径 / 非方形图 / 训练日志 / 断点续训 / numpy RNG / 动态注意力前景 / 验证指标 / 未知框） |

> `dataset/dataset.yaml` 保持原样，作为 19 个类别名的真源。

---

## 快速开始

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v     # 应输出 OK (42 tests，含验证指标 / 未知缺陷框)
python tools/benchmark.py --both            # 参数量 / MACs / 延迟估算
```

接入真实数据：把 `configs/fabric19.yaml` 的 `train / val / normaltrain / normalval`
指向你的目录（原 `dataset/dataset.yaml` 里是 `D:/tyhn/yolo/...`），然后按
[docs/TRAINING.md](docs/TRAINING.md) 跑**阶段 0（推荐）+ 五阶段**。
从零训练效果通常较差，阶段 0 用冻结 DINOv2 做离线蒸馏，teacher 不进推理图，
详见 [docs/DINO_PRETRAIN.md](docs/DINO_PRETRAIN.md)。

```bash
# 多 worker 读图 + uint8 增强（大数据集请务必保留 --num-workers > 0）
python tools/pretrain_dino.py     --cfg configs/jingwei_n.yaml --num-workers 4 --teacher dinov2_vits14 --log-interval 100
# 每 epoch 输出 val 精确率/召回率/mAP50/大缺陷召回，并保存 best.pt
python tools/train_detector.py    --cfg configs/jingwei_n.yaml --resume runs/dino/dino_init.pt --num-workers 4 --amp bf16 --log-interval 20 --val-interval 1
python tools/train_normality.py   --cfg configs/jingwei_n.yaml --num-workers 4 --ckpt runs/detector/best.pt --log-interval 100
python tools/build_memory_bank.py --cfg configs/jingwei_n.yaml --ckpt runs/normality/normality.pt
python tools/calibrate_fusion.py  --cfg configs/jingwei_n.yaml --ckpt runs/bank/memory_bank.pt
python tools/evaluate.py          --cfg configs/jingwei_n.yaml --ckpt runs/bank/memory_bank.pt --batch 8 --workers 2
python tools/export.py            --cfg configs/jingwei_n.yaml --ckpt runs/bank/memory_bank.pt --check
# 1660S 实时预测：自动切 pico + FP32 + 实测 P99
python tools/predict.py           --cfg configs/jingwei_p.yaml --realtime-1660 --ckpt runs/detector/best.pt --source D:/产线图片/
```

训练命令现在会立即打印启动信息，并按 `--log-interval` 打印 batch 级进度 / ETA；
阶段 A 还会按 `--val-interval` 输出验证指标。`loss=0.0000` 是短区间平均，
`epoch_avg=` 才是整轮平均。若控制台仍被缓冲，用 `python -u tools/train_detector.py ...`
或设置 `PYTHONUNBUFFERED=1`。详见
[docs/CACHE_AND_VALIDATION.md](docs/CACHE_AND_VALIDATION.md)。

中断后直接把 `--resume` 指向对应阶段的 `last.pt` 即可继续：

```bash
python tools/pretrain_dino.py    --cfg configs/jingwei_n.yaml --resume runs/dino/last.pt
python tools/train_detector.py   --cfg configs/jingwei_n.yaml --resume runs/detector/last.pt
python tools/train_normality.py  --cfg configs/jingwei_n.yaml --resume runs/normality/last.pt
```

图片 / 目录 / 视频文件名支持中文、空格及大写扩展名（`.JPG/.PNG/.WEBP`）；
`jingwei/image_io.py` 统一用 `np.fromfile+imdecode` / `imencode+tofile` 读写，
`VideoCaptureUnicode` 处理中文视频路径，Windows 下无需重命名数据。
非方形长条布幅图（如 1040×2448）也可直接训练；阶段 0 / A / B 支持
完整 checkpoint 断点续训；详见
[docs/DESIGN.md §12.5–12.8](docs/DESIGN.md)。

阶段 A 会同时训练局部+全局注意力的 `defect_fg / fabric_mask / fold_band`；`data.py` 还会合成布幅边缘正常折入增广，
避免产线把折入误判成疵点，详见 [docs/FOREGROUND.md](docs/FOREGROUND.md)。

无数据也能验证全流程（随机权重 + 合成织纹，只验证可运行）：
给上述命令加 `--smoke`。

---

## 已验证结果（实测，非目标值）

```
python -m unittest discover -s tests -v
Ran 42 tests in ~90 s — OK
  jingwei_n = 6.36 M / 16.62 G MACs @1280
  jingwei_p = 2.93 M /  8.09 G MACs @1280
  stage 0 DINOv2 distillation + local+global attention fg context = smoke OK
  Chinese/Unicode image, folder, video and .npz paths = OK
  non-square 1040x2448-like fabric frame augmentation = OK
  stage 0 / A / B startup + per-iteration progress logs = OK
  stage 0 / A / B full checkpoint resume + weight-only warm start = OK
  numpy BitGenerator RNG state (PCG64/Philox/SFC64/MT19937) = OK
  local+global attention dynamic fold_band (no fixed edge pixels) = OK
  per-epoch val precision/recall/F1/mAP50/large recall = OK
  unknown defect type boxes (label 19) = OK
```

| 变体 | 参数 | 1280² MACs | 640² MACs | RTX 4090 估算 | Jetson AGX Orin 估算 | **GTX 1660 SUPER 估算** |
|---|---|---|---|---|---|---|
| **jingwei_p** | 2.93 M | **8.09 G** | 2.45 G | 2.39 ms / 419 FPS | 7.52 ms / 133 FPS | **13.2–20.1 ms / 50–76 FPS ✅** |
| **jingwei_n** | 6.36 M | **16.62 G** | 5.11 G | 3.64 ms / 275 FPS | 14.20 ms / 70 FPS | 22.9–37.0 ms / 27–44 FPS ⚠️ |
| **jingwei_s** | 11.48 M | 34.11 G | 10.05 G | 6.20 ms / 161 FPS | 27.88 ms / 36 FPS | 42.8–71.8 ms / 14–23 FPS ❌ |

> 1660 SUPER 一列用的是**独立的"FP32 峰值 × 实际效率"模型**，不是右表其它列的
> Tensor-Core 锚点（该卡无 Tensor Core）。详见 [docs/DEPLOYMENT.md §4.1](docs/DEPLOYMENT.md)。
> 上述数字已包含新的局部+全局注意力前景头（注意力部分约 +0.12~0.26 G MACs）。

对照：YOLO27n 官方 640² = 7.2 G FLOPs / 3.0 M 参数（线性外推到 1280² 约 28.8 G）。
JingWei-n 在 1280² 仍只用了约六成的计算量。

其他实测：

- **ONNX 导出成功**，`onnx.checker` 通过，输出全部静态形状：
  pico 为 8 个（`cls3 … anomaly` + `context (1,3,128,128) =
  [defect_fg, fabric_mask, fold_band]`），n/s 的 context 为 `(1,3,160,160)`，
  且因 ROI 二级分类额外输出 `n3/n5` 共 10 个。
- **安全网**：160² 异常图上一个大连通域、零检测框 → 强制输出 1 个
  "未知异常"框；`emit_unknown_boxes=true` 时更小的未知疵点连通域也会出框；
  已有框则提分且不重复出框；正常图 / 正常折入 0 框。
- **布种描述符**：周期 5/9/13/19 px、亮度 100/170、反光 0/0.9 的合成织纹下，
  周期估计误差 < 0.6 px，8 维描述符最大差异 < 0.004，k-means 把每个周期的
  4 张图完整归为同一个布种原型。
- **记忆库卷积等价性**：`1 - max_k cos(q, c_k)` 与 1×1 卷积 + 通道 max 精确一致（1e-4）。
- **阶段 0 DINOv2 蒸馏**：`pretrain_dino.py --smoke` 生成 `dino_init.pt`，
  `train_detector.py --resume` 可热启动，`target_proj` 自动重新冻结；
  单元测试 20/21 覆盖 teacher→student 反向传播与 CLI 端到端。
- **中文 / Unicode 路径**：`jingwei/image_io.py` 收敛了全部图像/视频 I/O；
  单元 24 覆盖中文图片名/目录名/中文 `.npz` 原型/中文视频名，以及
  `predict.py --source/--out` 的端到端读写与 ASCII 控制台输出。
- **非方形长条布幅图**：`fabric_photometric` 的反光场经 `cv2.resize` 后会被压成
  2D，现已补回通道维；1040×2448 这类长条图可直接 `augment=True` 训练，单元 25 覆盖。
- **checkpoint 加载**：统一走 `jingwei.utils.torch_load`，默认 `weights_only=True`，
  消除 PyTorch 2.6+ 的 `FutureWarning`；第三方权重含自定义对象时自动回退并提示。
- **训练日志**：阶段 0 / A / B 启动即打印配置、数据、设备、参数量与输出路径；
  训练中默认每 20 个 batch（阶段 0 / B 默认 100 iters）打印 loss、lr、it/s、ETA。
  全部日志 `flush=True`，可用 `--log-interval N` 调整或 `python -u` 关闭缓冲。
- **断点续训**：`last.pt` 现在保存模型 + 优化器 + scheduler + RNG + epoch/iter
  完整状态；`--resume` 自动识别完整 checkpoint 与 weight-only 热启动。
  `--save-interval` 保留编号快照，写入使用临时文件 + `os.replace` 原子替换。
  numpy `BitGenerator.state` 的 dict / ndarray 形态会递归转成纯 Python 容器，
  兼容 Windows / numpy 2.x 且仍是 `weights_only=True` 友好；不再因
  `'dict' object has no attribute 'tolist'` 在保存 checkpoint 时崩溃。
  详见 [docs/DESIGN.md §12.8](docs/DESIGN.md)。

---

## 在 GTX 1660 SUPER 上能否实时？

**能，但不要用 `jingwei_n/s`，要用专门的 `jingwei_p`。**

1660 SUPER 是 TU116、**没有 Tensor Core**（FP32 5.03 TFLOPS，FP16 只是 2:1 的
shader 速率，336 GB/s）[1](https://www.tomshardware.com/reviews/nvidia-geforce-gtx-1660-turing-tu116,6027.html)。
之前那张延迟表以 RTX PRO 6000 的 FP16 TRT 为锚点，**在这张卡上不成立**。
按独立的峰值×效率模型估算（1280²、FP32 TRT、效率 20–35% + 4 ms 开销）：

| 变体 | 1660S 估算 | 结论 |
|---|---|---|
| **jingwei_p** (8.09 G MACs) | **13.2–20.1 ms / 50–76 FPS** | ✅ 稳定实时 |
| jingwei_n (16.62 G) | 22.9–37.0 ms / 27–44 FPS | ⚠️ 临界，需实测 |
| jingwei_s (34.11 G) | 42.8–71.8 ms / 14–23 FPS | ❌ 达不到 |

预测时直接加 `--realtime-1660`：检测到 1660/1650/TU116/TU117 会自动切 pico + FP32
并实测 P99，例如 `python tools/predict.py --cfg configs/jingwei_p.yaml --realtime-1660 ...`。
注意：checkpoint 必须也是 pico 配置训练出来的；若拿 n/s 权重切 pico，日志会警告且精度不可信。

`jingwei_p` 保留了全部关键机制（去反光、布种原型、双无监督信号、**局部+全局注意力前景/背景与
动态折入抑制**、确定性安全网），只砍了通道数 / 重建块数 / 网格分辨率 /
ROI 二级分类，**"大缺陷不漏"没有被削弱**。

> **术语澄清：TensorRT ≠ Tensor Core。** 1660S 是 sm_75，**可以用 TensorRT**
> （[NVIDIA 支持矩阵](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html)），
> 但 TU116 芯片**没有 Tensor Core**。前面说的“不能用 Tensor-Core 锚点”不是不能用
> TensorRT；只是 FP16/INT8 在这张卡上要走普通 CUDA shader / DP4A，必须 A/B 实测。

三个注意点：① 这类卡上 **FP16 有时比 FP32 更慢**（TRT 的 FP16 卷积落到慢 kernel，
见 [NVIDIA/TensorRT #4714](https://github.com/NVIDIA/tensorrt/issues/4714)），
必须 A/B 实测；② **INT8 不要指望**（无 INT8 Tensor Core，只能 DP4A）；
③ 前后处理约占 3–4 ms，用 uint8 输入省 H2D。

**在你自己的卡上验证（PyTorch 图是上界，TRT engine 通常再快 1.5–3×）：**

```bash
python tools/bench_device.py --cfg configs/jingwei_p.yaml --precision both --compare
python tools/export.py --cfg configs/jingwei_p.yaml --ckpt runs/bank/memory_bank.pt --check
trtexec --onnx=runs/export/jingwei_p.onnx --saveEngine=... --duration=60
```

完整分析见 [docs/DEPLOYMENT.md §4.1](docs/DEPLOYMENT.md)。

---

## 三个最重要的设计决定

1. **共享主干，不跑第二编码器**。Dinomaly 原版在推理时跑冻结 ViT，1280² 下不可能实时；
   本设计把无监督分支全部接到同一份主干特征上，只增加约 4.8 G MACs。
2. **记忆库 = 一个冻结的 1×1 卷积**。PatchCore 的 k-NN 搜索在 L2 归一化后等价于
   `1 - max_k <q, c_k>`，即 1×1 卷积 + 通道 max。彻底去掉 faiss / 动态索引。
3. **大缺陷不漏靠确定性规则，不靠置信度**。只要异常图上出现面积 ≥ 0.1% 图幅的
   连通域且没有被检测框覆盖，就一定输出一个框（标为"未知异常"）。
   分类精度由监督头 + ROI 二级分类保证，二者解耦。

---

## 诚实声明

- 仓库里**没有真实图片**（只有 `dataset/dataset.yaml` 指向你的 Windows 路径）。
  因此所有精度数字（mAP、召回、FPR）都是**设计目标/验收指标**，不是实测结果；
  实测的只有参数量、MACs、张量形状、单元测试、ONNX 导出与工具链可运行性。
  `--smoke` 模式下 mAP/recall = 0 是预期的（随机权重 + 合成图）。
- 延迟数字是**估算**：有 Tensor Core 的卡用官方 YOLO27n 的 0.44 ms / 7.2 G MACs
  标定，无 Tensor Core 的卡（1660S）用 FP32 峰值×效率模型。**都必须在目标设备上
  用 `tools/bench_device.py` + `trtexec` 实测**。本沙箱没有 GPU，无法替你实测 1660S。
- **YOLO27 截至 2026-10 尚未发布权重/配置**（官方文档写明 "not yet available"）。
  本仓库是自包含的等价实现（YOLO26/YOLO11 风格 CNN + 上述创新），
  YOLO27 主干/头是可插拔升级件，迁移路径见 [DESIGN.md §12.4](docs/DESIGN.md)。

---

## 建议的验收标准

| 指标 | 目标 |
|---|---|
| 大缺陷召回 `Recall@large`（面积 ≥ 0.1% 图幅） | **≥ 99.5%** |
| 正常图误报 `FPR@normal` | ≤ 0.5% |
| mAP50 | ≥ 0.70 |
| 端到端 P99 延迟 @1280² | ≤ 25 ms（≥ 40 FPS） |

若用于论文/产品/专利，请按 Apache-2.0 致谢
[Dinomaly](https://github.com/guojiajeremy/Dinomaly) 与
[PatchCore](https://github.com/amazon-science/patchcore-inspection)。
