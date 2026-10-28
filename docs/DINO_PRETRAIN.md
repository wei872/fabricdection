# 用 DINO / DINOv2 做阶段 0 预训练（离线蒸馏）

> 配套：[DESIGN.md](DESIGN.md) §6 阶段 0、[TRAINING.md](TRAINING.md) §3。
> 实现：`jingwei/dino.py` + `tools/pretrain_dino.py` + 三个 `configs/jingwei_*.yaml` 的 `dino:` 段。

## 1. 为什么要做

**你的判断是对的：从零开始训练大概率效果差。** 原因不是网络结构不行，而是：

1. 19 类布匹缺陷是细粒度、低频、长尾任务；随机初始化的 CNN 在 1280² 上
   很容易先记住纹理/曝光，而不是缺陷语义；
2. 无监督正常性分支（PatchCore 记忆库 + Dinomaly 重建）需要先有一个
   **稳定的正常流形特征**；随机主干上的余弦距离没有意义，会造成误报或漏报；
3. `train/val` 的标注框可能不完整（大缺陷尤其容易被漏标），纯监督训练会把
   漏标当背景学进去。

DINOv2 是**自监督**模型，在 1.42 亿张无标注图片上训练，天然输出密集的
patch 特征，正好适合做「免费」的特征先验
（[DINOv2 仓库](https://github.com/facebookresearch/dinov2)、[论文](https://arxiv.org/abs/2304.07193)）。
Dinomaly 原版也正是用冻结 DINOv2 特征做异常检测。

## 2. 不能直接把 DINOv2 当推理主干

| 方案 | 精度 | 1280² 实时性 | 结论 |
|---|---|---|---|
| 推理时直接跑 DINOv2 ViT-B/14 | 高 | ❌ >100 G MACs，1660S 上不可能实时 | 放弃 |
| 用 DINOv2 权重初始化我们 CNN | 架构不匹配，ViT→CNN 无法直接加载 | ✅ | 不可行 |
| **离线 teacher + 在线轻量 student 蒸馏** | 接近前者的特征质量 | ✅ 推理图不变 | **采用** |

关键：**teacher 只在训练机上跑一次/每个 epoch 跑，导出 ONNX/TRT 时完全不存在**。
加入局部+全局注意力前景上下文后 `jingwei_p` 为 2.93 M / 8.09 G MACs、`jingwei_n` 为 6.36 M / 16.62 G MACs，
ONNX 输出张量数量、形状全部不变。

```
normaltrain 图片 (无标签)
      │
      ├── 弱增广 ──> [冻结 DINOv2 ViT] ──> patch tokens T (B,C,Ht,Wt)
      │                                         │
      └── 强增广 ──> [JingWei CNN student] ─────┼─ cosine 对齐
                         │  n3 / n5 / target_proj / q
                         └─ 1×1 辅助投影到 teacher 维度
      │
      ▼
stage 0 结束后：teacher 丢弃；target_proj 重新冻结；
                 student 权重作为阶段 A 的 --resume 初始化
```

## 3. 损失函数

teacher 的 patch token 图记为 \(T\in\mathbb{R}^{B\times C_t\times H_t\times W_t}\)，
student 的四个特征图记为 \(F_s\)（\(s\in\{n3,n5,rec,patch\}\)），
每个尺度配一个训练期专用的 \(1\times1\) 投影 \(P_s\)：

\[
\mathcal{L}_{0}=\sum_s w_s\,
\frac{1}{H_sW_s}\sum_{h,w}
\Big[1-\cos\big(P_s(F_s)_{h,w},\ \uparrow_s(T)_{h,w}\big)\Big]
\]

- \(\uparrow_s\)：把 teacher 图双线性插值到 student 特征图大小；
- 默认权重 `n3:1.0, n5:1.0, rec:1.0, patch:0.5`；
- `rec` 对齐的是 `normality.recon.target_proj`，也就是 Dinomaly 的「冻结基础
  特征头」。蒸馏完立刻 `requires_grad_(False)`，避免阶段 B 出现塌缩解；
- `patch` 对齐的是 `normality.patch`（PatchCore 记忆库的查询特征），让阶段 C
  的原型/中心点一开始就落在有意义的流形上。

训练时 teacher 看**弱增广**原图，student 看**同几何位置**的强增广
（曝光、低照度渐变、轻微模糊、随机擦除），因此 student 同时学到 DINOv2 的
语义和布匹光照/反光不变性。

## 4. 数据

- **只用 `normaltrain`**（可加 `normalval` 中未标定的正常图，但不能和阶段 D
  的阈值标定val泄漏；推荐只用 normaltrain）。
- **不要把已知缺陷图混进阶段 0**：阶段 0 会训练 `normality.patch` 和
  `target_proj`，混入异常会让「正常流形」把缺陷也学成正常。
- 图片会被随机裁成 `crop=448`（14 和 32 的公倍数，DINOv2 patch=14、
  CNN stride=32），所以原始 1280² 图不需要缩放标注。
- 建议 `normaltrain` 覆盖全部布种、光照、反光状态；每个布种 ≥200 张。

## 5. 快速开始

### 5.1 联网机器（最简单）

```bash
# 会由 torch.hub 自动拉取 facebookresearch/dinov2 的 backbone-only 权重
python tools/pretrain_dino.py \
    --cfg configs/jingwei_n.yaml \
    --data configs/fabric19.yaml \
    --teacher dinov2_vits14 \
    --iters 5000 --log-interval 100 --cache disk --out runs/dino

# 用蒸馏权重热启动阶段 A（注意是 --resume，不是 --ckpt）
python tools/train_detector.py \
    --cfg configs/jingwei_n.yaml \
    --data configs/fabric19.yaml \
    --resume runs/dino/dino_init.pt \
    --epochs 150 --log-interval 20 --out runs/detector
```

`jingwei_s` 默认 `dinov2_vitb14`（更强 teacher），`jingwei_p` 默认
`dinov2_vits14`（训练快、显存小）。三种配置都已在 `dino:` 段写好超参。

### 5.2 离线机器（工厂内网）

```bash
git clone https://github.com/facebookresearch/dinov2 /opt/dinov2
# 从官方 README 的 backbone-only 链接下载，例如 dinov2_vitb14_pretrain.pth
python tools/pretrain_dino.py \
    --cfg configs/jingwei_p.yaml \
    --data configs/fabric19.yaml \
    --teacher dinov2_vitb14 \
    --repo /opt/dinov2 \
    --weights /opt/weights/dinov2_vitb14_pretrain.pth \
    --iters 5000 --log-interval 100 --cache disk --out runs/dino
```

`--repo` 也接受 `facebookresearch/dinov2:main` 之类的 hub slug；`--weights`
接受本地 `.pth` 或 URL。代码会把 `weights` 直接传给 DINOv2 的 hub 函数。

### 5.3 无数据 / 无网络自检

```bash
python tools/pretrain_dino.py --smoke 4 --iters 2 --cfg configs/jingwei_p.yaml
# 用合成织纹 + synthetic teacher 跑通完整蒸馏循环，不下载任何权重
```

单元测试 `tests/test_model.py::test_20/21`（另见 `test_22/23/29/30/31/32` 动态局部+全局注意力前景/折入）覆盖同一路径：
`python -m unittest discover -s tests -v` 应输出 `OK (36 tests)`（含验证指标 / 图像缓存 / 未知缺陷框 / 标签校验）。
本地 teacher 权重与 checkpoint 统一通过 `jingwei.utils.torch_load` 读取，默认
`weights_only=True`，不再触发 PyTorch 2.6+ 的 `FutureWarning`。
阶段 0 启动即打印 teacher / 数据 / device / 参数量，并按
`--log-interval` 打印 `distill_cosine`、lr、it/s 与 ETA；默认 100 iters。

### 5.4 断点续训

阶段 0 每 `--save-interval` iters（默认 500）写 `runs/dino/last.pt` 与
`iter_XXXXXX.pt` 快照，最终仍写 `dino_init.pt`。中断后：

```bash
python tools/pretrain_dino.py --cfg configs/jingwei_p.yaml \
    --resume runs/dino/last.pt --log-interval 100 --save-interval 100
```

完整 checkpoint 会恢复 student、dino adaptor、optimizer、RNG、已完成的
iteration 以及 teacher / crop / batch / lr / loss weights 配置；不传 `--iters`
时沿用 checkpoint 里的总 iters，想延长训练则显式传更大的 `--iters`。
真实 DINOv2 teacher 不写入 checkpoint（从 hub / 本地权重重新加载），synthetic
teacher 会连同随机权重一起保存，保证 smoke 复现。详见
[DESIGN.md §12.8](DESIGN.md)。

## 6. teacher 选择

| teacher | 参数 | patch | 特征维度 | 建议 |
|---|---:|---:|---:|---|
| `dinov2_vits14` | 21 M | 14 | 384 | **默认**，训练快，显存小 |
| `dinov2_vitb14` | 86 M | 14 | 768 | 数据少、类别长尾时更稳 |
| `dinov2_vitl14` | 300 M | 14 | 1024 | 需要大显存训练机；推理图不变 |
| `dinov2_vits14_reg` / `_vitb14_reg` | 同上 | 14 | 同上 | 带 registers，特征更干净可选 |
| `dino_vitb16` / `dino_vits16` | 85/21 M | 16 | 768/384 | DINOv1 回退；Apache-2.0 |
| `synthetic` | 极小 | 14 | 可配 | CI / smoke，不能用于真实训练 |

`dino_resnet50` 没有密集 patch token，不支持像素对齐蒸馏，会直接报错。
`dinov3_*` 暂未接入：它使用自定义 DINOv3 License 和不同的 hub API，需先做
法务/工程评估；标准 DINOv2 主干权重已经是 **Apache-2.0**，可商用。

> ⚠️ **许可证**：facebookresearch/dinov2 仓库里同时存在多种权重。
> 标准 `dinov2_vits14/vitb14/vitl14/vitg14`（含 `_reg`）主干权重是
> Apache-2.0；但 **XRay-DINO、Cell-DINO、Channel-Adaptive DINO** 的权重/代码
> 仍在 FAIR Noncommercial / CC-BY-NC 下，产线商用请勿使用。
> 参考 [Meta 的 Apache-2.0 重授权公告](https://ai.meta.com/blog/dinov2-facet-computer-vision-fairness-evaluation/)
> 与 [仓库 LICENSE 说明](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md)。
> DINOv3 是另一份 [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)。

## 7. 预期收益与边界

**预期：**

- 小样本下 19 类分类 mAP 更高、收敛更快；
- 正常性分支的 `1 - max cos` 和重建残差不再基于随机特征，误报率更低；
- 大缺陷召回更稳：teacher 的全局语义能帮助区分「布种纹理变化」和「异常区域」；
- 对亮度/反光变化的鲁棒性更强。

**不能替代的东西：**

- 阶段 0 **不产生检测框**，也不训练检测头/ROI/记忆库阈值；
  `defect_fg` 前景头在阶段 0 保持冻结，由阶段 A 的 YOLO 框弱监督训练；
- 阶段 A–E 仍然必须跑；`dino_init.pt` 里的 head / bank / rec decoder 仍是随机的；
- 没有真实数据前，本仓库无法给出「mAP 提升 X%」的实测数字。请在你的 `val`
  上做 A/B：同一配置分别用随机初始化和 `dino_init.pt` 跑阶段 A，比较
  `Recall@large`、`mAP`、`FPR@normal`。

## 8. 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `torch.hub.load` 失败 / 超时 | 内网无外网 | 用 `--repo` 指向本地 clone，`--weights` 指向本地 `.pth` |
| OOM | teacher 太大 / crop 太大 / batch 太大 | 换 `dinov2_vits14`；`--batch 2`；`--crop 224` |
| loss 不下降 | 学习率太小 / iters 太少 / 数据太单一 | 先跑 5000 iters；`--lr 0.0005`；确认 normaltrain 覆盖多布种 |
| `no valid crop` | crop 不是 patch 和 32 的公倍数 | 用 224/448/672/896/1120；脚本会自动回退到 448 |
| `AttributeError: 'dict' object has no attribute 'tolist'`（第 500 iter 保存 `last.pt` 时） | `BitGenerator.state['state']` 在 Windows / numpy 2.x 下是 dict，不是 ndarray | 已修复为递归纯 Python 序列化；拉最新代码后重跑。崩溃发生在写文件之前，所以该次没有产生可用 `last.pt`；建议加 `--save-interval 100`，见 [DESIGN.md §12.8](DESIGN.md) |
| `UserWarning: xFormers is not available (SwiGLU/Attention/Block)` | DINOv2 hub 未检测到 xFormers，回退到标准 attention | 仅警告，不影响蒸馏；可忽略 |
| 想用自己无标签图做 DINO 自监督预训练 | 可行但昂贵 | 需要自己跑 DINOv2/DINO 训练；本工具的 teacher 权重是官方预训练好的，成本更低 |
| 能不能直接把 DINOv2 当 backbone | 不建议 | 1280² 下 ViT-B/14 >100 G MACs，1660S 无法实时；蒸馏后推理图不变 |

---

**一句话结论**：可以用、而且建议用 DINOv2；但要用「离线 teacher 蒸馏」而不是
「在线 DINOv2 推理」。阶段 0 的产物只是一个更好的 CNN 初始化，实时性和
大缺陷安全网机制都不变。
