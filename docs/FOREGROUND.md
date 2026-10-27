# 前景/背景分离与布幅边缘折入抑制

> 对应代码：`jingwei/foreground.py`、`jingwei/fusion.py` 的 `safety_net_numpy`、
> `jingwei/data.py` 的 `fabric_photometric`；配置段 `foreground:`。

## 1. 问题

产线上**布幅会左右移动、宽度也会变化**。布边在运动中会正常折入 / 卷边（折入），
在图像上表现为一条长、暗、高对比的条带。对纯异常检测模型来说它非常像
「大缺陷」：原来的确定性安全网会把这个大连通域强制输出成 `未知异常` 框，
造成误报。

用户要求的核心能力是：

> **一定要区分背景和疵点前景**。
> 背景 = 非布幅区域 + 正常布面（包括正常折入）；疵点前景 = 真实缺陷。

## 2. 解决思路：不是再堆一个检测器，而是给模型“上下文”

直接把折入当成新缺陷类去标，成本很高且无法穷举布幅变化。我们用三层上下文：

| 信号 | 来源 | 含义 |
|---|---|---|
| `fabric_mask` | 无参数、逐帧从图像纹理能量估计 | 当前帧的**有效布幅区域**；背景/机台 = 0 |
| `edge_band` | 由 `fabric_mask` 的左/右边界算出的内侧软带 | 布边正常折入 / 阴影最可能出现的区域 |
| `defect_fg` | 小卷积头，用 YOLO 缺陷框弱监督训练 | **疵点前景**概率；正常折入没有框，因此是负样本 |

`fabric_mask` 不依赖固定 ROI，逐帧从图像算，所以**布幅变化自动跟随**。
`defect_fg` 初始 bias 设为 -3，未训练时也倾向输出 0，避免一开始就把折入当前景。

推理图新增一个 3 通道张量：

```
context (1,3,G,G) = [defect_fg, fabric_mask, edge_band]
G = normality.out_grid   (n/s: 160, pico: 128)
```

## 3. 有效异常图

 normality 分支产生的原始异常图记为 `A`，导出前先做上下文门控：

```
A_eff = A * fabric_mask * (1 - edge_attenuation * edge_band * (1 - defect_fg))
```

含义：

- 背景区域直接乘 `fabric_mask` 抑制；
- 布边折入带内，若 `defect_fg` 低，则按 `edge_attenuation` 衰减；
- 真实边缘缺陷若被 `defect_fg` 判为前景，衰减项接近 0，异常图保留；
- `A_eff` 同时用于检测头 objectness 的异常增益和安全网连通域。

## 4. 安全网规则（保留“大缺陷不漏”）

` safety_net_numpy(..., context=ctx)` 在每个大异常连通域上额外检查：

1. `valid_frac`：区域内 `fabric_mask >= fabric_mask_thresh` 的比例；
   低于 `min_fabric_frac` 视为背景，直接跳过。
2. `edge_frac` / `fg_frac` / 长宽比 `aspect`：若
   - 位于 `edge_band`，
   - 形状细长（`aspect >= fold_aspect_ratio`），
   - `defect_fg` 低，
   - 且异常峰值 `< edge_peak_override`，
   
   则判定为**正常折入**，不输出 `未知异常` 框。
3. 若 `defect_fg` 高、或异常峰值超过 `edge_peak_override`，或组件不在边缘带，
   仍按原规则强制输出框。**大缺陷不漏的机制没有被削弱**。
4. 如果检测头已经在该处有框，只做弱提分；折入状组件提分系数降为 0.25。

单元测试 `test_23` 明确验证：折入条带 0 框、中央大缺陷 1 框、
边缘带内高 `defect_fg` 的真实缺陷 1 框。

## 5. 训练怎么学出来

### 5.1 缺陷前景头（阶段 A 自动训练）

`foreground_loss = BCE(pos_weight=5) + 0.5 * soft Dice`，
正样本来自 YOLO 框内缩 8% 的区域，负样本是框外所有像素。
`train` 里有缺陷框；`normaltrain` 没有框，且阶段 0/阶段 B 会把正常图
（含合成折入）作为负样本。因此学到的是「缺陷前景」而不是「暗条带」。

阶段 A 命令不变，`foreground` 头已自动加入 `parameter_groups("detector")`：

```bash
python tools/train_detector.py --cfg configs/jingwei_n.yaml --resume runs/dino/dino_init.pt
```

### 5.2 正常折入数据增广

`jingwei/data.py::fabric_photometric` 以 25% 概率在左/右布边合成
渐变暗条 + 折痕（只改亮度，**不移动任何标注框**）。这样：

- normality 分支 / 记忆库把折入见过的模式学成正常；
- detector 在缺陷图上也见到折入状背景，但那里没有框，成为 hard negative；
- 不依赖人工标注「折入」掩膜。

若你的真实 `normaltrain` 已经包含大量折入图，可以把概率调低；若误报仍集中在
边缘，可提高概率或增大 `edge_band_px`。

## 6. 配置

三个模型配置都含 `foreground:` 段，常用项：

```yaml
foreground:
  enabled: true
  hidden: 32                  # 缺陷前景头隐层通道
  edge_band_px: 64            # 布边内侧折入带宽度（像素）
  edge_attenuation: 0.80      # 折入带内异常衰减强度
  texture_thresh: 0.20        # 纹理能量→布幅掩膜阈值
  slope: 30.0
  target_shrink: 0.08         # YOLO 框内缩后作前景正样本
  pos_weight: 5.0
  loss_weight: 0.5
  fabric_mask_thresh: 0.30
  edge_band_thresh: 0.08
  defect_fg_thresh: 0.25
  min_fabric_frac: 0.50
  fold_aspect_ratio: 3.0
  edge_peak_override: 0.80
```

调参建议：

- 误报仍多 → 提高 `edge_attenuation` / `edge_band_px` / `min_fabric_frac`；
- 真实边缘小缺陷被压 → 降低 `edge_attenuation`，或提升 `defect_fg` 训练
  （增大 `loss_weight`、加更多带框边缘缺陷样本）；
- 布幅背景与布面纹理接近 → 调 `texture_thresh` / `slope`；
- 极端强反光被误当背景 → 检查 `LocalContrastNorm` 和 `normaltrain` 覆盖。

## 7. 部署与实时性

ONNX / TensorRT 图新增 `context` 输出，但全部是静态形状、静态算子：

```
cls3 (1,19,160,160)   reg3 (1,4,160,160)   obj3 (1,1,160,160)
cls5 (1,19,40,40)     reg5 (1,4,40,40)     obj5 (1,1,40,40)
anomaly (1,1,G,G)     context (1,3,G,G)   # pico: 8 outputs total
[+ n3/n5 when roi_head=true]              # n/s: 10 outputs total
```

`postprocess_from_tensors` 会解析 `context` 并传给融合/安全网。
`tools/calibrate_fusion.py` 在阶段 D 标定阈值时也会缓存并传入 context，
因此阈值搜索与部署行为一致。

新增开销（实测 MAC counter）：

| 变体 | 原 MACs | 含前景上下文 | 增量 |
|---|---:|---:|---:|
| jingwei_p | 7.59 G | **8.42 G** | +0.83 G |
| jingwei_n | 15.65 G | **17.12 G** | +1.47 G |
| jingwei_s | 32.48 G | **35.80 G** | +3.32 G |

1660S 上 `jingwei_p` 估算仍约 **48–74 FPS**（FP32 TRT 模型），
仍在实时范围内；最终以 `tools/bench_device.py` 实测为准。

## 8. 已验证

- `test_22`：合成布幅 + 折入图，`fabric_mask` 正确区分布面/背景，
  `edge_band` 只出现在布边内侧；
- `test_23`：安全网拒绝正常折入，但保留中央大缺陷和高前景的边缘缺陷；
- 24 项单元测试全过（含中文 / Unicode 路径回归）；ONNX 导出 8 个静态输出（pico）通过 `onnx.checker`；
- `train_detector.py` 训练图中新增 `fg_context` 损失，反向传播有限。

## 9. 边界

- `fabric_mask` 是**无参数启发式**，不是像素级分割 GT；极端背景（例如和布纹
  同样高频的网带）需要调阈值或补充真实布幅掩膜训练。
- `defect_fg` 是弱监督（框级）训练，不是精细分割；它的作用只是给安全网一个
  「这里像不像真缺陷」的二验信号，不替代 19 类分类头。
- 若折入本身就伴随真实疵点（折入处断纱等），必须靠 `defect_fg` 或
  `edge_peak_override` 放行；建议在这类样本上专门做 A/B 验证。
