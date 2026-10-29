# 训练与数据准备手册

> 配套设计说明见 [DESIGN.md](DESIGN.md)。所有命令在仓库根目录执行。

## 1. 环境

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# 可选
pip install onnx tensorrt ultralytics
# 阶段 0 用 DINOv2 teacher：需要 torchvision + 可访问 torch.hub；
# 内网机器请 clone https://github.com/facebookresearch/dinov2 并下载 backbone-only 权重。
```

最低验证环境（本仓库实测）：Python 3.11 + torch 2.2.2 (CPU 即可) +
numpy 1.26 + opencv-python-headless + PyYAML + scipy + tqdm。
跑 `python -m unittest discover -s tests -v` 应输出 `OK`（32 项）。
阶段 0 的 `--smoke` 模式用 synthetic teacher，不需要下载 DINOv2 权重。

## 2. 数据准备

### 2.1 目录约定

你的 `dataset/dataset.yaml` 里是 Windows 绝对路径：

```yaml
train: D:/tyhn/yolo/dataset/source5_augmented/images/train/
val:   D:/tyhn/yolo/dataset/source5_augmented/images/val/
normaltrain: D:/tyhn/yolo/dataset/source5_augmented/normal/images/train/
normalval:   D:/tyhn/yolo/dataset/source5_augmented/normal/images/val/
```

推荐整理成（相对路径，跨平台）：

```
dataset/
  images/train/   images/val/              # 缺陷图
  labels/train/   labels/val/              # YOLO .txt，每行: cls cx cy w h (归一化 0~1)
  normal/images/train/  normal/images/val/ # 纯正常图，无标签
```

然后修改 `configs/fabric19.yaml` 的四个路径。类别名已与 `dataset/dataset.yaml` 完全一致（19 类）。
**原始 `dataset/dataset.yaml` 不要改**，它作为类别真源保留。

### 2.2 上线前的数据检查

```bash
# 1) 图片与标签一一对应；2) 每类样本数；3) bbox 面积分布（决定"大缺陷"阈值）
python - <<'PY'
from pathlib import Path
from collections import Counter
import numpy as np
root = Path('dataset')
for split in ('train','val'):
    imgs = list((root/'images'/split).glob('*.jpg'))
    labs = list((root/'labels'/split).glob('*.txt'))
    print(split, 'images', len(imgs), 'labels', len(labs),
          'missing', len({p.stem for p in imgs} - {p.stem for p in labs}))
    c, areas = Counter(), []
    for f in labs:
        for line in f.read_text().splitlines():
            p = line.split()
            if len(p) >= 5:
                c[int(float(p[0]))] += 1
                areas.append(float(p[3]) * float(p[4]))     # 归一化面积
    a = np.array(areas)
    print('  per-class:', dict(sorted(c.items())))
    print('  area ratio quantiles:', np.round(np.quantile(a, [.5,.9,.95,.99]), 4))
PY
```

若 `area ratio` 的中位数远小于 `0.001`，说明大缺陷很少，
需要在 `configs/fabric19.yaml` 里调 `large_area_ratio`（同步改两个
`jingwei_*.yaml` 的 `fusion.large_area_ratio`）。

图片、标签、目录、视频文件名可以包含中文 / 空格 / Unicode，扩展名大小写
（`.JPG/.PNG/.WEBP`）也能识别；`jingwei/data.py::list_images` 与
`jingwei/image_io.py` 已统一处理，Windows 下**不需要重命名数据**，也不要把
工具里的 `imread_unicode` / `imwrite_unicode` 改回 `cv2.imread` / `cv2.imwrite`。
图像也不需要预先裁成正方形：`1040×2448` 这类长条布幅帧在
`fabric_photometric` 反光增广下已修复广播问题（见 [DESIGN.md §12.6](DESIGN.md)）。
训练命令启动即打印配置/数据/设备，并按 `--log-interval` 打印 batch 级进度和 ETA；
详见 [DESIGN.md §12.7](DESIGN.md)。

### 2.3 normaltrain 的关键要求

- **必须只含真正无缺陷的图**，包含所有布种、所有光照/反光状态；
- **必须包含布幅变化和边缘正常折入样本**（真实图或 `data.py` 合成增广），
  否则前景/折入上下文学不到正常流形，产线上容易把折入误判成疵点；
- 建议每个布种 ≥ 200 张；新布种上线只需追加图片并重跑阶段 C；
- `normalval` 与 `normaltrain` 不重叠，用于阈值标定（目标 FPR = 0.5%，
  折入子集单独看 `FPR@fold`）。

## 3. 阶段 0（推荐）+ 五阶段训练

**从零训练效果通常较差**，建议先跑阶段 0：用冻结的 DINOv2 当离线 teacher，
把它的密集 patch 特征蒸馏进我们的轻量 CNN。teacher 只在训练时存在，
推理图 / MACs / 1660S 实时性完全不变。完整说明见
[DINO_PRETRAIN.md](DINO_PRETRAIN.md)。

```bash
# 阶段 0：DINOv2 离线蒸馏（只用 normaltrain，无标签）
python tools/pretrain_dino.py \
    --cfg configs/jingwei_n.yaml --data configs/fabric19.yaml \
    --teacher dinov2_vits14 --iters 5000 --log-interval 100 \
    --num-workers 4 --out runs/dino
# 无数据 / 无网络自检：python tools/pretrain_dino.py --smoke 4 --iters 2

# 目标卡若是 GTX 1660 SUPER / 1650 / 无 Tensor Core 老卡，
# 下面所有 --cfg 都改为 configs/jingwei_p.yaml（训练和预测保持一致）。

# 阶段 A：有监督检测（train + labels），从阶段 0 权重热启动
# --num-workers 4 多线程读图；--amp bf16 混合精度（CUDA 默认）；--val-interval 1 每 epoch 输出精确率/召回率/mAP
python tools/train_detector.py \
    --cfg configs/jingwei_n.yaml --data configs/fabric19.yaml \
    --resume runs/dino/dino_init.pt --num-workers 4 --amp bf16 \
    --epochs 150 --batch 8 --log-interval 20 --val-interval 1 \
    --val-max 300 --normal-max 300 --out runs/detector

# 阶段 A 同时训练局部+全局注意力的 defect_fg / fabric_mask / fold_band
# （YOLO 框级弱监督 + 逐帧动态布幅伪目标）；data.py 也会以 25% 概率合成
# 布幅边缘正常折入增广，让 normaltrain / detector 都把它见成背景。
# 详见 docs/FOREGROUND.md 与 docs/CACHE_AND_VALIDATION.md。
# 阶段 B：无监督正常性（normaltrain，主干冻结）
python tools/train_normality.py \
    --cfg configs/jingwei_n.yaml --data configs/fabric19.yaml \
    --ckpt runs/detector/best.pt --iters 10000 --log-interval 100 \
    --num-workers 4 --out runs/normality

# 阶段 C：构建布种条件记忆库 + 阈值标定（normaltrain + normalval）
python tools/build_memory_bank.py \
    --cfg configs/jingwei_n.yaml --data configs/fabric19.yaml \
    --ckpt runs/normality/normality.pt --out runs/bank

# 阶段 D：融合 / 安全网阈值网格搜索（val + normalval）
python tools/calibrate_fusion.py \
    --cfg configs/jingwei_n.yaml --data configs/fabric19.yaml \
    --ckpt runs/bank/memory_bank.pt --max-val 500 \
    --out configs/fusion_calibrated.yaml

# 阶段 E：评估 + 导出
python tools/evaluate.py  --cfg configs/jingwei_n.yaml --ckpt runs/bank/memory_bank.pt
python tools/export.py    --cfg configs/jingwei_n.yaml --ckpt runs/bank/memory_bank.pt \
    --proto 0 --check --out runs/export/jingwei_n.onnx
```

把 `configs/fusion_calibrated.yaml` 里的 `fusion.anomaly_thresh` /
`fusion.final_score_thresh` 回填到 `configs/jingwei_n.yaml` 后再导出。

### 3.1 阶段 A 细节

- `model.set_stage("detector")` 跳过正常性分支，训练更快；
- 损失 = o2m 密集 + o2o 匈牙利 + 前景对齐（YOLO27 训练辅助，见 DESIGN.md §7）
  + 局部+全局注意力前景上下文损失（见 FOREGROUND.md）；
- 优化器 AdamW + cosine，梯度裁剪 5.0；
- 若某类 AP 特别低，在 `train.class_weights` 里填 19 个权重（如按 `1/sqrt(样本数)` 归一化）；
- 出框门槛（v0.2.2 起只有 **一个** 绝对阈值）：部署分数
  `sigmoid(obj_o2o)·sigmoid(cls)` 越过 `fusion.final_score_thresh`（0.15）才出框，
  `model.score_thresh` 必须与它相等（`test_61` 会检查）。所以这几项不要随手改：

  | 配置 | 默认 | 作用 |
  |---|---|---|
  | `model.obj_head_from` | `reg` | **objectness 读哪层特征**。`reg` = 回归 stem + 自带 3×3→5×5 DWConv（9 cell ≈72 px 感受野）；`neck` = 旧的裸 1×1 卷积——实测用 `neck` 时 `score` 300 步只有 0.033，门**永远打不开**（见 CODE_REVIEW §0.8）。除非做 A/B，不要改成 `neck` |
  | `model.obj_bias_init` / `cls_bias_init` | -4.0 / -2.0（σ≈0.018 / 0.119） | 初始分数 ≈0.002，前几十步不出框是正常的（实测第 7 步越过 0.15）。调高到 -2.0 **不会**更快学会，只会让第 0 步就吐出 99~150 个框、背景电平停在 0.5（见 CODE_REVIEW §0.8） |
  | `train.obj_quality_floor` | 0.0 | 质量目标下限。目标是 `w·(qual/qmax)+(1-w)`、`w=qmax`：回归没学时自动恒为 1.0，不需要再垫底线；设成 0.5 会把"定位很差的匹配"教成 objectness 0.5 |
  | `train.cls_quality_target` | false | true 时分类目标是 `one-hot × 质量`，冷启动会把缺陷的分类学成 ~0.02 |
  | `train.obj_o2o_mode / gamma / alpha` | focal / 2.0 / 0.5 | 正样本只占 0.1%–1%，alpha=0.25 是在压前景 |
  | `train.obj_o2o_neg_weight` | 2.5 | 背景项权重。平衡归一化给正负各 50% 份额，等于"输出 0.5 不亏"；>1 才有人为压低背景的梯度。**这是背景电平（`gate: p99`）的直接旋钮** |
  | `train.obj_o2m_neg_weight` / `cls_o2m_neg_weight` | 2.0 / 2.0 | 同上，作用在 o2m 分支与分类上；`score = obj·cls` 是乘积，两个因子都要低 |
  | `loss_weights.obj_o2o` | 6.0 | 门控分支的份额。低于 1.0 时它在总 loss 里的占比会掉到 1% 以下 |

- 每个 epoch 的 val 块多两行：

  ```
        gate: max_score=0.9312 p99=0.0431 (conf=0.15)  obj_max=0.98 obj_mean=0.061  ...
        normal: boxes/img=0.4 max_score=0.079
  ```

  * `max_score < conf` ⇒ **一个框都没出**（训练信号问题，不是阈值问题）；
  * `p99` 是**背景电平**（全网 99 分位）：它不降，precision 不会涨。健康值 < 0.05；
  * `normal: boxes/img` 是"正常布上出几个框"，比 `normal_fpr` 这个 0/1 指标敏感得多；
  * `-> P=… R=…` 是 F1 最优点上的 precision/recall——下发阈值处偏低但这里已经很高，
    说明该重标定阈值（`tools/calibrate_fusion.py`），而不是训练出了问题。
- **建议**：先用 `normaltrain` 里的图做"纯负样本"加入训练（无框），可显著降低 FPR。

### 3.2 阶段 B 细节（最容易出错）

- 主干 / neck / 头 / ROI **全部冻结**，BN 置 eval；
- 只训练 `PatchFeatureHead`、`NoisyBottleneck`、4 个 `VitBlock`、输出投影、融合门；
- 损失 `HardMiningCosineLoss` 的 `p` 在前 1000 iter 从 0 升到 0.9；
- 如果 `rec_loss` 不下降或降到 ~0：检查是否有参数漏冻结（塌缩解）；
- 可选合成异常增广：在正常图上随机 CutPaste / 画细线（模拟断纬）/ 局部抹平（模拟稀缝），
  但**不要用真实缺陷图**训练 B 分支，否则会失去"发现新异常"的能力。

### 3.3 阶段 C 输出

```
runs/bank/
  fabric_prototypes.npz     # K 个布种原型中心 + 描述符归一化统计量
  memory_bank.pt            # centroids / mu / sigma / tau / 重建残差标定 + 完整权重
```

日志示例（实测 smoke）：

```
prototype  1:    6400 patch features -> 256 centroids
prototype  1: mu=0.18120 sigma=0.06895 tau=2.755
prototype  0: mu=0.18852 sigma=0.21816 tau=3.703 (conservative fallback)
reconstruction: tau=1.03675 slope=40.59
```

`(conservative fallback)` 表示 `normalval` 里没有该布种，用最保守阈值兜底，属正常现象。

### 3.4 阶段 D 细节

网络对 val / normalval **只前向一次**并缓存特征，之后 15 组阈值只在缓存上跑
decode + 安全网，秒级完成。目标函数：

```
maximise   Recall@large - 2·max(0, FPR - target_fpr)
subject to FPR ≤ max(3·target_fpr, 0.02)
```

若没有一组满足约束，说明 `target_fpr` 设得太严，或 `normaltrain` 污染 /
布种覆盖不足（回到 §2.3）。

### 3.5 断点续训

阶段 0 / A / B 都支持完整 checkpoint 恢复。训练过程中会写：

- `runs/dino/last.pt`、`runs/detector/last.pt`、`runs/normality/last.pt`；
- 可选编号快照：`--save-interval N` 生成 `iter_XXXXXX.pt` / `epoch_XXXX.pt`；
- 阶段 0 最终仍写 `dino_init.pt`，阶段 B 最终仍写 `normality.pt`，供下一阶段加载。

完整 checkpoint 包含模型、优化器、scheduler、torch/CUDA/Python/NumPy RNG、
当前 epoch/iter 和训练配置。中断后用同一条命令把 `--resume` 指向 `last.pt`：

```bash
# 阶段 0：不传 --iters 时沿用 checkpoint 里的总 iters
python tools/pretrain_dino.py --cfg configs/jingwei_n.yaml \
    --resume runs/dino/last.pt --log-interval 100

# 阶段 A：从上次完成的 epoch 继续；可调大 --epochs 延长训练
python tools/train_detector.py --cfg configs/jingwei_n.yaml \
    --data configs/fabric19.yaml --resume runs/detector/last.pt \
    --epochs 200 --batch 8 --log-interval 20

# 阶段 B：从上次完成的 iteration 继续；可调大 --iters
python tools/train_normality.py --cfg configs/jingwei_n.yaml \
    --data configs/fabric19.yaml --resume runs/normality/last.pt \
    --iters 12000 --log-interval 100
```

`--resume` 会自动区分两种文件：

- 完整 checkpoint → 日志显示 `mode=full`，恢复全部状态并继续；
- 仅权重的 checkpoint（如 `dino_init.pt`）→ 日志显示 `mode=weights-only`，
  只热启动模型，从第 1 个 epoch/iter 开始。

长任务建议加 `--save-interval 100`（阶段 0 / B）或 `--save-interval 5`（阶段 A），
避免默认 500 iters / 每 epoch 才写一次导致中断损失过多。保存发生在
`save_checkpoint` 原子写之前若崩溃，不会产生新的 `last.pt`，此时重新运行即可。

恢复阶段 0 时，teacher / crop / batch / lr / loss weights 会优先使用 checkpoint
里的配置；若 CLI 传入冲突参数会给出警告。阶段 A/B 若 cfg、data、seed、batch、lr
与 checkpoint 不同也会警告。详见 [DESIGN.md §12.8](DESIGN.md)。

## 4. 快速自检（无真实数据）

```bash
python -m unittest discover -s tests -v                 # 62 项，约 130 s（含验证指标 / 未知缺陷框 / 冷启动门控）
python tools/benchmark.py --all                         # p/n/s 参数量 / MACs / 延迟估算
python tools/bench_device.py --cfg configs/jingwei_p.yaml --precision both  # 在你的 GPU 上实测
python tools/train_detector.py   --smoke 2 --epochs 1 --batch 1 --device cpu --log-interval 1
python tools/train_normality.py  --smoke 4 --iters 2 --batch 2 --device cpu --log-interval 1
python tools/build_memory_bank.py --smoke 4 --device cpu
python tools/calibrate_fusion.py --smoke --device cpu
python tools/evaluate.py --smoke --device cpu
python tools/export.py --check --out runs/export/jingwei_n.onnx
```

`--smoke` 用随机权重 + 合成织纹图，只验证"代码能跑通"，
其 mAP/recall 为 0 是预期的。

## 5. 常见故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| val 一直 `precision=0 recall=0 boxes/img=0.0`，且 `gate: max_score < conf` | 部署门没开：objectness 分支没有可用的特征（`obj_head_from: neck`）或质量目标塌陷 | v0.2.1/0.2.2 已修（[CODE_REVIEW.md §0.7](CODE_REVIEW.md) / §0.8）；确认 `model.obj_head_from: reg` 后重训。v0.2.2 之前保存的 checkpoint 与新 objectness 头**形状不同**：weight-only warm start（`dino_init.pt`）会自动跳过那 4 个张量并打印报告，完整 checkpoint 则降级为 weight-only（optimizer/LR schedule 重置） |
| `boxes/img` 几十上百、`normal_fpr=1.0000`、`gate: max_score≈1` 但 `p99≈0.5` | 背景电平太高：`score = obj·cls` 在每个 cell 都过阈值 | 看 `gate: p99`；调 `train.obj_o2o_neg_weight`（2.5→4）、`obj_o2m_neg_weight` / `cls_o2m_neg_weight`；**不要**调高 `model.obj_bias_init`（那正是病因） |
| `precision`（下发阈值处）很低，但 `-> P=… R=…` 很高 | 阈值没标定，不是模型差 | 跑 `tools/calibrate_fusion.py`，把 `fusion.final_score_thresh` 回填（decode 用的是同一个值） |
| val 一直 `precision=0` 但 `gate: max_score ≫ conf` | 框出来了但都是错框 | 跑 `tools/diagnose_precision.py` 看 FP 分解（duplicate / wrong_class / localization / background），再按 [CODE_REVIEW.md §0.5/§0.6](CODE_REVIEW.md) 的表格处理 |
| `loss share` 里 `box_o2m` 长期 >60% | v0.2.1 之前像素 L1 没有归一化，量级随输入分辨率线性变化 | 已改成按图像边长归一化；若仍 >60%，说明 GT 框质量差或 `box_o2m` 权重过高 |
| `no images under ...` | 路径没改 / 图片扩展名不在列表 | 检查 `configs/fabric19.yaml`；支持 jpg/jpeg/png/bmp/tif/tiff/webp（含大写） |
| 中文图片名报错 / 读图返回 `None` / 无法保存中文结果 | Windows 下 `cv2.imread` / `cv2.imwrite` / `cv2.VideoCapture` 直接吃非 ASCII 路径失败 | 已统一改用 `jingwei/image_io.py` 的 `imread_unicode` / `imwrite_unicode` / `VideoCaptureUnicode`；升级代码后若仍报错，贴完整 traceback，不要回退到直接 `cv2.imread`；详见 [DESIGN.md §12.5](DESIGN.md) |
| `ValueError: operands could not be broadcast together with shapes (H,W,3) (H,W)` | 旧版 `fabric_photometric` 反光场经 OpenCV 后变成 2D，未补通道维 | 已修复：`field[:, :, None]` 后再与 RGB 相乘；`test_25` 覆盖 1040×2448 类长条图，见 [DESIGN.md §12.6](DESIGN.md) |
| `FutureWarning: You are using torch.load with weights_only=False` | PyTorch 2.6+ 提示未来默认值改为 `True` | 已统一走 `jingwei.utils.torch_load`（默认 `weights_only=True`）；含自定义对象的第三方权重会自动回退并给出 `RuntimeWarning`，仅加载可信文件 |
| 训练时长时间没有任何日志 | 旧版只在 epoch 结束打印；Windows 控制台 / 重定向 stdout 还会缓冲 | 已加入启动日志 + `--log-interval` batch 日志 + `flush=True`；用 `--log-interval 20`（默认）观察进度，`--log-interval 0` 关闭 iter 日志；仍不显示时用 `python -u tools/train_detector.py ...` 或设置 `PYTHONUNBUFFERED=1`，见 [DESIGN.md §12.7](DESIGN.md) |
| 日志只有 loss，没有验证集精确率 / 召回率 | 旧版 `train_detector.py` 不带 val；或运行的是旧提交 | 拉取最新代码后默认 `--val-interval 1` 每 epoch 打印 `mAP50 / precision / recall / f1 / large_recall / normal_fpr` 并保存 `best.pt`；`--val-interval 0` 可关闭，`--val-max` 可抽样加速 |
| iter 日志 `loss=0.0000` 但 epoch 汇总非零 | `loss=` 是短日志区间平均，末尾若连续几张无标注图就会是 0 | 看同一行新增的 `epoch_avg=`（整轮平均）与 epoch 汇总；ETA 也改为长平均+短区间的混合速度，避免单个慢 batch 让 ETA 跳到几百小时 |
| 图片读取太慢 / GPU 利用率低 | 单进程读图 + 旧版 `fabric_photometric` 全分辨率大 sigma 高斯模糊；`--num-workers 0` | 用 `--num-workers 4`（Windows 下每个 worker 由 `make_worker_init_fn` 独立播种），CUDA 上加 `--amp bf16 --batch 8`；缓存功能已移除（实测更慢，详见 [CACHE_AND_VALIDATION.md](CACHE_AND_VALIDATION.md)） |
| 未知疵点类型没有预测框 | 旧安全网只兜底 `large_area_ratio` 以上的大异常 | 配置 `fusion.emit_unknown_boxes: true`（默认已开）、按需调小 `unknown_min_area_ratio`；`predict.py --no-unknown` 可临时关闭 |
| 1660 SUPER 上预测不够实时 | 用了 `n/s` 配置或假设 FP16 一定更快 | 训练和预测都用 `configs/jingwei_p.yaml`，预测加 `--realtime-1660`；用 `bench_device.py --precision both` 实测，详见 [CACHE_AND_VALIDATION.md](CACHE_AND_VALIDATION.md) |
| 训练中断后要从头开始 | 旧版 `last.pt` 只有模型权重 | 现在 `last.pt` 含优化器 / scheduler / RNG / epoch/iter；用 `--resume .../last.pt` 继续，日志出现 `mode=full` 才表示完整恢复；见 [DESIGN.md §12.8](DESIGN.md) |
| `--resume dino_init.pt` 显示 `weights-only` | `dino_init.pt` 是阶段 0 交给阶段 A 的权重文件，不含训练状态 | 这是正常的 weight-only 热启动；若要继续阶段 0 训练，应 `--resume runs/dino/last.pt` |
| `AttributeError: 'dict' object has no attribute 'tolist'`（在 `_generator_state` / `collect_rng_state` 保存 checkpoint 时） | `np.random.Generator.bit_generator.state` 的 `state` 字段在不同 numpy / BitGenerator 下可能是 dict 或 ndarray；旧代码假设 ndarray 并调用 `.tolist()` | 已修复为递归 `_to_plain` / `_from_plain` 转换，纯 Python 容器保存且 `weights_only=True` 友好；`test_28` 覆盖 PCG64/PCG64DXSM/Philox/SFC64/MT19937 与真实 `NormalImageDataset`。若崩溃发生在写 `last.pt` 之前，则没有生成新 checkpoint，拉最新代码后重新运行即可；建议加 `--save-interval 100` 缩短中断损失。见 [DESIGN.md §12.8](DESIGN.md) |
| xFormers is not available (SwiGLU / Attention / Block) UserWarning | DINOv2 hub 代码检测到未安装 xFormers，自动回退到标准 PyTorch attention | 仅警告，不影响 teacher 加载或蒸馏正确性；可忽略，或安装与 torch/CUDA 匹配的 xFormers |
| 标签全空 | `labels` 目录推断失败 | 显式传 `--labels`（工具会自动把 `/images/` 换成 `/labels/`）；中文标签文件名同样支持 |
| `rec_loss` → 0 | 主干没冻结，出现塌缩解 | 检查阶段 B 只把 `normality_parameters()` 加入优化器 |
| 全图误报 | 记忆库没标定 / 用了未训练权重 | 必须先跑阶段 C；`tau` 由 `normalval` 标定 |
| 大缺陷仍漏 | `anomaly_thresh` 太高 / `large_area_ratio` 太大 | 跑阶段 D；降低 `large_area_ratio` |
| OOM | 1280² 训练 batch 太大 | 减小 `detector_batch`；开 AMP（`--amp`，可自行加） |
| ONNX 导出失败 | 缺 `onnx` | `pip install onnx`；注意保持 `numpy<2`（torch 2.2 要求） |
| `AttributeError: config has no key ...` | yaml 缺键 | 对照 `configs/jingwei_n.yaml` 补齐 |
| 阶段 0 `torch.hub.load` 失败 | 内网无法访问 GitHub / 权重 URL | clone DINOv2 到本地，用 `--repo /path/to/dinov2 --weights /path/to/xxx.pth`；见 [DINO_PRETRAIN.md](DINO_PRETRAIN.md) |
| 阶段 0 OOM | teacher 太大 / crop / batch 太大 | 换 `dinov2_vits14`、`--batch 2`、`--crop 224`；推理图不受影响 |
| 阶段 0 loss 不降 | iters 太少 / normaltrain 布种单一 | 先跑 5000 iters；确认 normaltrain 覆盖全部布种与光照 |
| 布边正常折入仍误报 | 动态 `fold_band` 没学到 / `defect_fg` 把折入当前景 | 提高 `fold_loss_weight` / `pos_weight`；补充 normaltrain 折入图；见 [FOREGROUND.md](FOREGROUND.md) |
| 真实边缘缺陷被压掉 | 抑制过强 | 降低 `fold_attenuation`、提高 `fold_peak_override`，或增加带框边缘缺陷样本训练 `defect_fg` |
| 1660 SUPER 上 FP16 比 FP32 还慢 | TU116 能用 TensorRT，但没有 Tensor Core；TRT FP16 可能落到慢 kernel | 用 `--precision both` 实测，取快的；先构建 FP32 engine 再 A/B FP16，见 [DEPLOYMENT.md §4.1](DEPLOYMENT.md) |
| 1660 SUPER 上不到 40 FPS | n/s 变体算力过大 | 换 `configs/jingwei_p.yaml`（8.09 G MACs） |
| 后处理占了 3–4 ms | decode + 连通域在 CPU 上 | 用 uint8 输入、`top_k` 降到 100、`out_grid` 降到 96 |
