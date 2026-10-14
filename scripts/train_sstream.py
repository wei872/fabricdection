#!/usr/bin/env python3
"""S 分支训练：YOLO27s 风格监督检测器（design §3.3 / §5.1）

YOLO27 官方权重未发布（2026-10 核查），脚本按优先级自动选择：
  yolo27s.yaml（发布后）→ yolo26s.yaml（现网可用，接口一致）→ 自定义 dual-scale yaml。
数据集：docs/DATASET_SPEC.md（prepare_dataset.py 生成 data_ultralytics.yaml）。

用法：
  python scripts/train_sstream.py --data dataset_fabric/data_ultralytics.yaml \
      --weights yolo26s --epochs 300 --imgsz 1280 --device 0
  # 冒烟：
  python scripts/train_sstream.py --data dataset_fabric/data_ultralytics.yaml \
      --weights yolo26n --epochs 1 --imgsz 320 --device cpu --workers 0 --smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def pick_weights(requested: str) -> tuple[str, str]:
    """返回 (用于 YOLO() 的名称, 说明)。yolo27 未发布时自动回退 yolo26。"""
    from ultralytics import ASSETS
    import ultralytics.cfg as cfg
    root = Path(cfg.__file__).resolve().parent / "models"
    if "yolo27" in requested and not (root / "27" / f"{requested}.yaml").exists():
        note = (f"{requested} 尚未发布，自动回退 "
                f"{requested.replace('27', '26')}（YOLO27 发布后无需改动，weights 传回即可）")
        return requested.replace("27", "26"), note
    return requested, f"using {requested}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_fabric/data_ultralytics.yaml")
    ap.add_argument("--weights", default="yolo27s", help="yolo27s|yolo26s|...|*.pt|--cfg *.yaml")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=-1)      # -1 = ultralytics auto
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--project", default="runs/sstream")
    ap.add_argument("--name", default="det")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="透传 ultralytics 超参，如 cls=1.2 box=8.0")
    args = ap.parse_args()

    from ultralytics import YOLO

    w, note = pick_weights(args.weights)
    print(f"[init] {note}")
    model = YOLO(f"{w}.yaml")                              # 从结构新建；预训练权重用 .pt 时直接传路径
    kwargs = dict(data=args.data, epochs=1 if args.smoke else args.epochs,
                  imgsz=args.imgsz, batch=args.batch, device=args.device,
                  workers=args.workers, project=args.project, name=args.name,
                  optimizer="auto", patience=50, cos_lr=True,
                  close_mosaic=15, exist_ok=True)
    for kv in args.extra:                                  # 透传 k=v
        k, v = kv.split("="); kwargs[k] = float(v) if "." in v else int(v)
    results = model.train(**kwargs)

    # 训练后验证（含小目标/分桶报表见 scripts/eval_metrics.py）
    metrics = model.val(data=args.data, imgsz=args.imgsz, device=args.device,
                        project=args.project, name=f"{args.name}_val")
    print(f"[done] best={results}\n[val] {metrics}")


if __name__ == "__main__":
    main()
