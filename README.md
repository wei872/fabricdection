# fabricdection · FADNet-RT

实时布匹缺陷检测系统设计项目。

- **总体设计文档**：[`docs/FADNet-RT_design.md`](docs/FADNet-RT_design.md)（必读）
- 目标三要素：① 大缺陷零漏检　② 已知缺陷分类精准　③ 1280×1280 实时（≥30 FPS）
- 架构：预处理层 + 监督检测分支（YOLO27s 风格）+ 无监督异常分支（Dinomaly 重构 × PatchCore 记忆库）+ **免登记布种自适应**（逐 tile 自归一化/特征中心化，全局单库单阈值）+ 融合裁决 F-Head（**逐类置信度标定，置信度↔严重度**）
- 强制参考：[Dinomaly](https://github.com/guojiajeremy/Dinomaly)（CVPR 2025）、[PatchCore](https://github.com/amazon-science/patchcore-inspection)、[Ultralytics YOLO27](https://docs.ultralytics.com/models/yolo27/)

## 目录

```
configs/                 # 总体配置、检测器配置、数据/类别配置
src/fadnet/
  pipeline/              # 平场校正（实测暗角2.8×）· ROI · 1280²重叠切块
  btype/                 # 布种条件化 FC（FFT谱 → 嵌入/类型ID · FiLM）
  ustream/               # U分支：Dinomaly重构分支 · PatchCore记忆库 · 热力图融合
  fusion/                # F-Head 融合裁决（"大缺陷不漏"规则引擎）
  augment/               # 布料感知合成缺陷引擎（频域锁定 CutPaste/DRAEM 变体）
data/                    # 原始样本（BMP 条带帧）
```

## 快速开始（骨架冒烟）

```bash
pip install pillow numpy torch
PYTHONPATH=src python -c "
import numpy as np
from PIL import Image
from fadnet.pipeline.tiling import slice_tiles, detect_fabric_roi
from fadnet.augment.synthdefects import inject_defect
im = np.array(Image.open('data/KBA26050009_20261014_101648_109095.bmp').convert('L'))
tile = slice_tiles(im)[0]
print(inject_defect(tile.image, 'broken_warp').boxes)
"
```

## 脚本

```bash
python scripts/prepare_dataset.py --src data --dst dataset_fabric --inject 1 --flatfield  # 原始条带→规范数据集
python scripts/train_ustream.py  --root dataset_fabric --device cuda      # U分支：重构训练+全局记忆库+全局阈值标定
python scripts/train_sstream.py  --data dataset_fabric/data_ultralytics.yaml --device 0   # S分支：YOLO27s风格(未发布自动回退yolo26)
python scripts/test_e2e.py       --src data --det runs/.../best.pt --device cuda           # 端到端测试：verdicts/overlay/时延
python scripts/eval_metrics.py   --mode fhead --preds runs/test/<ts>/verdicts.jsonl        # 大缺陷召回/分类/兜底/误报
python scripts/benchmark_latency.py --device cuda                                          # 实时性实测
```

数据集规范见 [`docs/DATASET_SPEC.md`](docs/DATASET_SPEC.md)（**免布种登记**：全局单记忆库 + 全局阈值）。

> 状态：v0.2 设计评审稿。训练侧需按 `docs/FADNet-RT_design.md` §8 澄清 Q1–Q8
>（标注笔迹确认、类别终版、目标硬件等）后进入 M1 里程碑。
