# 前景 / 背景分离：动态局部 + 全局注意力版本

> **2026-10 架构修正**：旧版把“布边正常折入”近似成**距图像边缘固定
> 64 px 的带**。这是错的——布幅会移动、旋转、透视变形，折入在图像上的
> 像素位置**不固定**。当前实现改为类似 Transformer 的**局部 + 全局注意力**
> 前景上下文头：边界由每层特征动态预测，不再用图像边缘像素距离。

---

## 0. 一句话原理

```
P3 局部纹理 ─┐
P4 局部语义 ─┼─ Local conv (stride 8) ─┐
             │                         ├─ 动态FiLM 融合 ─┬─ defect_fg (YOLO框弱监督)
P5 下采样 ───┴─ Global token (stride 32) ─┤                  ├─ fabric_mask (布幅mask)
               + CLS + LinearAttention2 ─┘                  └─ fold_band  (动态折入带)
```

- **局部分支**看 P3/P4 的织纹、暗条、缺陷边缘 → 像素级缺陷前景；
- **全局分支**在 stride-32 token 上做 `LinearAttention2`（Dinomaly 同款
  ELU+1 线性注意力），加可学习 `[CLS]` token → 整幅图的布幅位置/宽度/
  全局亮度先验，并生成逐通道 **FiLM (γ, β)** 调制局部特征；
- 三个头输出 `defect_fg / fabric_mask / fold_band`，打包成
  `context (1,3,G,G)` 导出到 ONNX / TensorRT；
- 安全网同时使用这三张图，但判定标准是**预测的动态边界**，不是图像边框。

---

## 1. 问题：为什么固定边缘像素不行

| 现象 | 旧固定带 | 新动态注意力 |
|---|---|---|
| 布幅整体左移 120 px | 带还在图像右边 → 漏抑制 | attention 全局 token 感知布幅位置，fold_band 跟着移动 |
| 布幅变窄 / 倾斜 / 透视 | 固定宽度覆盖不全 | 局部分支逐像素重建边界 |
| 折入在图像中间（拼接/折叠） | 完全不在带内 → 误报 | 全局上下文可在任意位置生成 fold_band |
| 真实疵点恰好在旧带内 | 被无差别衰减 | `defect_fg` 高 / 峰值 override 强制保留 |

训练代码里**唯一**还出现 `kernel_size=5` 的地方是
`build_fabric_context_targets()` 里对**弱监督伪标签**做形态学膨胀/平滑，
以降低标注噪声；它不参与推理，也不是“边缘多少像素”的超参。

---

## 2. 模块实现（`jingwei/foreground.py`）

### 2.1 `LocalGlobalForegroundAttention`

| 子模块 | 作用 | 关键实现 |
|---|---|---|
| `stem` | 降 P3/P4/P5 到统一 `hidden` 通道 | 1×1 Conv + GN + SiLU |
| `local` / `fuse` | 局部纹理 + 上下文 | 3×3 depthwise + GroupNorm + 1×1 pointwise |
| `token_proj` | 生成全局 token | P3@S/32 + P5 → 1×1 Conv |
| `cls_token` | 全局布幅摘要 | 可学习 `(1,1,attn_dim)` |
| `attention` | 全局交互 | `VitBlock`(LinearAttention2, ELU+1) × `attention_blocks` |
| `global_proj` | 全局调制局部 | CLS → hidden 的 γ（tanh 后 FiLM 广播） |
| `fg_head` | 缺陷前景 | 1×1 Conv → sigmoid，bias 初始 -3 |
| `mask_head` | 有效布面 | 1×1 Conv → sigmoid，bias 初始 +2 |
| `fold_head` | 折入/边界带 | 1×1 Conv → sigmoid，bias 初始 -2 |

所有注意力 token 在 **stride 32**（1280 输入 → 40×40 + 1 CLS = 1601 tokens）
上计算；`LinearAttention2` 是 O(N) 而非 O(N²)，保证实时性。
FiLM 输出在空间上广播，因此全局信息能影响每个像素的局部判断。

`forward_parts(feats)` 返回：

```python
{
  "defect_fg":  (B,1,G,G),
  "fabric_mask":(B,1,G,G),
  "fold_band":  (B,1,G,G),
  "global_logit": (B,1),
  "context":    (B,3,G,G),
}
```

### 2.2 `FabricContext`

部署时保留的包装器：

```python
ctx = FabricContext(cfg.foreground, channels=(...), img_size=1280,
                    out_grid=160)
A_eff, context = ctx(raw_anomaly, feats)
```

- `A_eff = A * fabric_mask * (1 - fold_attenuation * fold_band * (1 - defect_fg))`
- `fold_band` 来自注意力输出，随输入图像动态变化；
- `fold_attenuation` 只控制衰减**强度**，不定义带的位置。

训练时：

```python
parts = ctx.forward_parts(feats)
tgt_fg = build_foreground_target(targets, ...)         # YOLO 框
tgt_fabric, tgt_fold, tgt_global = ctx.build_targets(x, tgt_fg)
loss, parts = ctx.loss(parts, tgt_fg, tgt_fabric, tgt_fold, tgt_global)
```

---

## 3. 弱监督：标签从哪来

没有像素级布幅/折入标注，所以使用**可解释的逐帧伪目标 + 可学习修正**：

1. `build_foreground_target()`：把 YOLO 框内缩 8% 的区域置 1，其余置 0；
   正常图 / normaltrain 全 0。这是 `defect_fg` 的强监督信号。
2. `build_fabric_context_targets(x, ...)`：
   - 在 320×320 缩略图上算亮度归一化高频织纹能量
     `E = (I - AvgPool9(I))² / (AvgPool9(I)² + 0.02)`；
   - `E / (mean + 2·std)` 抑制整体反光/亮度差异；
   - `sigmoid((smooth(E) - texture_thresh) * slope)` → `fabric_mask` 伪目标；
   - 对伪目标做 `max_pool - min_pool` 形态学边界 → `fold_band` 伪目标；
   - `global_target = mean(fabric_target)` → 监督 CLS 头的全局布幅占比。
3. 上述伪目标**只在训练时计算**，且会随每张图的布幅位置/宽度变化。
   推理图没有 `cv2` / numpy / 形态学操作，ONNX 全静态。

> 注意：伪目标不是最终预测。attention 头通过可见光纹理 + 全局 token + YOLO
> 框监督，学会修正伪目标里的噪声（例如高对比背景经编/金属、真实疵点边缘）。

---

## 4. 损失函数

```
L_ctx = BCE(defect_fg, box_target, pos_weight=5)
      + w_fabric * BCE(fabric_mask, pseudo_fabric)
      + w_fold   * BCE(fold_band,   pseudo_fold)
      + w_global * BCE(global_logit, pseudo_global)
```

总损失：`L_det + 0.5·L_ctx`（`foreground.loss_weight`）。
正常折入图没有框 → `defect_fg` 全 0 是显式负样本；
大缺陷在框内 → `defect_fg` 高，折入带衰减被 `(1-defect_fg)` 自动抵消。

---

## 5. 融合门与安全网（`jingwei/fusion.py`）

`context (1,3,G,G)` 在 `fuse_detections()` 中被 resize 到 anomaly 网格：

| 通道 | 新语义 | 安全网用途 |
|---|---|---|
| 0 | `defect_fg` | 高 → 真缺陷，不被折入抑制 |
| 1 | `fabric_mask` | 低于 `fabric_mask_thresh` 的区域占比过高 → 背景，跳过 |
| 2 | `fold_band` | **动态**折入/边界带；与低 `defect_fg` 联合判为正常折入 |

判定逻辑（简化）：

```python
if valid_frac < min_fabric_frac:
    continue                                  # 背景
fold_like = (fold_frac > fold_band_thresh
             and fg_frac < defect_fg_thresh)
if fold_like and peak < fold_peak_override:
    continue                                  # 正常折入，不报
# 否则：大缺陷照常输出 / 提升已有框
```

`use_aspect_cue` 默认 **false**——长宽比只是可选辅助，主判据是学习到的
`fold_band + defect_fg`。`fold_aspect_ratio` 保留为兼容/调试项，不再是核心。

`fusion.emit_unknown_boxes: true` 还会让**较小的未知异常连通域**也输出
`label=19 / 未知异常` 框；大缺陷连通域永远保留，不受
`max_unknown_boxes` 限制。未知框仍要经过上述 `fabric_mask / fold_band /
defect_fg` 检查，因此不会把背景或正常折入当成缺陷。详见
[CACHE_AND_VALIDATION.md](CACHE_AND_VALIDATION.md)。

---

## 6. 配置（三个变体统一）

```yaml
foreground:
  enabled: true
  mode: local_global_attention
  hidden: 24/32/48
  attention_dim: 64
  attention_heads: 4
  attention_blocks: 2
  boundary_kernel: 5        # 仅平滑训练伪标签
  fold_attenuation: 0.80
  texture_thresh: 0.20
  slope: 30.0
  target_shrink: 0.08
  pos_weight: 5.0
  loss_weight: 0.5
  fabric_loss_weight: 1.0
  fold_loss_weight: 0.5
  global_loss_weight: 0.2
  fabric_mask_thresh: 0.30
  fold_band_thresh: 0.08
  defect_fg_thresh: 0.25
  min_fabric_frac: 0.50
  use_aspect_cue: false
  fold_aspect_ratio: 3.0
  fold_peak_override: 0.80
```

已删除/弃用：`edge_band_px`、`edge_attenuation`、`edge_band_thresh`、
`edge_peak_override`。`fusion.py` 仍兼容读取旧键名作为 fallback，但新配置
不应再依赖它们。`fold_band` 的位置由网络预测，与这些像素常量无关。

---

## 7. ONNX / TensorRT

`context` 仍是第 8 个输出（与旧版形状完全一致），所以部署代码不需要改：

| 变体 | context 形状 | 说明 |
|---|---|---|
| `jingwei_p` | `(1,3,128,128)` | pico 网格 |
| `jingwei_n/s` | `(1,3,160,160)` | N/S 网格 |

导出前 `strip_train_only()` 只删除 o2m / fg-align 训练分支，
**保留整个注意力前景头**。全局注意力在 stride-32 token 上用
`LinearAttention2`，无动态 shape、无 `NonZero`、无 ROI 变长循环，
onnx.checker 与 pico/N 导出均通过。

实测计算量（`tools/benchmark.py --all`，1280²）：
`jingwei_p` 2.93 M / 8.09 G MACs（注意力 0.12 G）、
`jingwei_n` 6.36 M / 16.62 G（0.21 G）、
`jingwei_s` 11.48 M / 34.11 G（0.26 G）。
相比旧固定边缘带版本，注意力头没有增加总延迟预算。

---

## 8. 与旧版对比（回归）

| 测试 | 旧版 | 新版 |
|---|---|---|
| 大块未知异常 | ✅ 输出 `未知异常`（面积兜底） | ✅ 不变 |
| 已有框叠加异常 | ✅ 提分不重复框 | ✅ 不变 |
| 正常低异常 | ✅ 不加框 | ✅ 不变 |
| 正常折入长条 | 固定边缘带抑制 | ✅ 动态 `fold_band` 抑制，布幅移位仍有效 |
| 布幅移到图像中部 | ❌ 固定带失效 | ✅ 全局 token 跟随 |
| 真实边缘缺陷 | ✅ `defect_fg`/峰值 override 保留 | ✅ 不变 |
| 背景经编纹理 | 可能被当布面 | ✅ `fabric_mask` + 全局先验联合抑制 |

---

## 9. 验证

```bash
/home/user/.venv/bin/python -m unittest discover -s tests -v
/home/user/.venv/bin/python tools/export.py --cfg configs/jingwei_p.yaml --out runs/export/p.onnx
/home/user/.venv/bin/python tools/export.py --cfg configs/jingwei_n.yaml --out runs/export/n.onnx
```

- `test_22`：验证 `build_fabric_context_targets()` 在布幅平移时输出
  动态 `fabric_mask / fold_band / global_target`，且 `fold_band` 跟随布边
  而非固定图像边缘；
- `test_29`：验证 `LocalGlobalForegroundAttention` 的局部+全局双分支、
  动态 FiLM、context 形状、损失与梯度；
- `test_23`：端到端大缺陷 / 正常折入 / 边缘真缺陷 / 大兜底回归；
- `test_31`：未知疵点类型（含小面积异常连通域）也生成 `未知异常` 框，大缺陷兜底不被 `max_unknown_boxes` 丢弃；
- 全套 **62 项单元测试通过**（含验证指标 / 未知缺陷框回归），compileall / ONNX checker / pico+N 导出通过。

---

## 10. 仍需注意

1. 伪目标来自织纹能量，对**极高对比的金属/经编背景**可能不稳定；若现场
   背景固定，建议补少量（50~200 张）布幅 mask 标注做微调，效果会明显提升。
2. `fold_peak_override=0.80` 是安全网硬保底：峰值超过它**一定报**，
   避免 attention 把大缺陷误判成折入。调低会更严格，调高会更依赖学习。
3. stage 0 蒸馏与 normality 记忆库不用前景头；只有 detector / 融合门用。
4. 注意力 token 数固定为 40×40+1（1280 输入）。若改 `img_size`，token 数
   会变但仍全静态；需重新导出 ONNX / TensorRT 引擎。
5. 旧 `edge_band_px` / `edge_attenuation` 配置键已从三个 yaml 移除；
   `DefectForegroundHead` 仍作为 `LocalGlobalForegroundAttention` 的兼容别名保留，
   `foreground_context_loss()` 是当前 `FabricContext.loss()` 实际调用的损失实现。
   `model.py` 不再调用旧的 `foreground_loss()` / `estimate_fabric_context()`。
