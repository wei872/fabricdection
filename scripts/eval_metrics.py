#!/usr/bin/env python3
"""指标评估（design §7 验收）：

mode fhead   —— F-Head 结果 vs GT 框（dataset_fabric/abnormal/*/labels）
  · 大缺陷召回 @IoU0.3 / IoU0.5（验收红线：=100%）
  · 分类 P/R/F1（已知类 @IoU0.5）
  · unknown_anomaly 兜底召回（GT 有、S 分支无框但被 U 分支报出）
  · 正常 tile 误报率（FP/strip）
mode ustream —— U 分支单独 AUROC（tile 级 peak score vs GT 有无缺陷），
                并扫描 tau_u 给出"召回-误报"工作点建议。

用法：
  python scripts/eval_metrics.py --mode fhead --preds runs/test/<ts>/verdicts.jsonl \
      --root dataset_fabric
  python scripts/eval_metrics.py --mode ustream --preds runs/test/<ts>/verdicts.jsonl \
      --root dataset_fabric
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if inter <= 0:
        return 0.0
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def load_gt(root: Path) -> dict[str, np.ndarray]:
    """strip stem → [K,5] cls,x1,y1,x2,y2（全局坐标：tile 偏移已含在文件名 _t{x}_{y}）。"""
    gt: dict[str, list] = defaultdict(list)
    for lab in sorted((root / "abnormal").rglob("labels/*.txt")) if (root / "abnormal").exists() else []:
        stem = lab.stem
        # 文件名 <strip>_t<xx>_<x0>_<y0>；GT 框为 tile 内归一化 → 还原全局
        parts = stem.split("_")
        try:
            x0, y0 = float(parts[-2]), float(parts[-1])
        except ValueError:
            x0 = y0 = 0.0
        strip = "_".join(parts[:-4]) if len(parts) > 4 else stem
        for ln in lab.read_text().strip().splitlines():
            c, xc, yc, w, h = map(float, ln.split())
            gx, gy = (xc - w / 2) * 1280 + x0, (yc - h / 2) * 1280 + y0
            gt[strip].append([c, gx, gy, gx + w * 1280, gy + h * 1280])
    return {k: np.array(v, np.float32) for k, v in gt.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fhead", "ustream"], default="fhead")
    ap.add_argument("--preds", required=True)
    ap.add_argument("--root", default="dataset_fabric")
    ap.add_argument("--large-area", type=float, default=2500.0)
    ap.add_argument("--iou-large", type=float, default=0.3)
    ap.add_argument("--iou-cls", type=float, default=0.5)
    args = ap.parse_args()

    preds: list[dict] = [json.loads(l) for l in Path(args.preds).read_text().splitlines() if l]
    gt = load_gt(Path(args.root))
    if not gt:
        print("[warn] 无 GT 标注（dataset_fabric/abnormal 为空）：仅输出预测统计。"
              "真实异常样本入库并标注后重跑本脚本。")
    strips = {p["strip"].rsplit(".", 1)[0] for p in preds}

    if args.mode == "ustream":
        pos, neg = [], []
        for p in preds:
            key = p["strip"].rsplit(".", 1)[0]
            (pos if len(gt.get(key, [])) else neg).append(p["peak_score"])
        if pos and neg:
            pos_a, neg_a = np.array(pos), np.array(neg)
            thr = np.linspace(0, 1, 201)
            tpr = [(pos_a > t).mean() for t in thr]
            fpr = [(neg_a > t).mean() for t in thr]
            order = np.argsort(fpr)
            auc = float(np.trapezoid(np.array(tpr)[order], np.array(fpr)[order]))
            t99 = thr[np.searchsorted(-np.array(tpr), -0.99)]
            print(f"[ustream] tile-AUROC={auc:.4f} | 召回99%处 tau_u≈{t99:.3f} "
                  f"(该点 FPR={(neg_a > t99).mean():.3f})")
        else:
            print(f"[ustream] preds={len(preds)} peak 分布: "
                  f"min={min(p['peak_score'] for p in preds):.3f} "
                  f"max={max(p['peak_score'] for p in preds):.3f}")
        return

    # ---- fhead：大缺陷召回 / 分类 / unknown 兜底 / 误报 ----
    matched_large = tot_large = 0
    cls_tp = defaultdict(int); cls_fp = defaultdict(int); cls_fn = defaultdict(int)
    unknown_catch = 0
    n_fp_normal = n_normal = 0
    for p in preds:
        key = p["strip"].rsplit(".", 1)[0]
        g = gt.get(key, np.zeros((0, 5), np.float32))
        vs = [(v["box_xywh"], v["cls"], v["grade"]) for v in p["verdicts"]]
        boxes = [(b[0] + p["tile_xy"][0], b[1] + p["tile_xy"][1],
                  b[0] + b[2] + p["tile_xy"][0], b[1] + b[3] + p["tile_xy"][1])
                 for b, _, _ in vs]
        used = [False] * len(boxes)
        for row in g:
            c, gx0, gy0, gx1, gy1 = row
            area = (gx1 - gx0) * (gy1 - gy0)
            best, bi = 0.0, -1
            for i, bb in enumerate(boxes):
                v = iou((gx0, gy0, gx1, gy1), bb)
                if v > best:
                    best, bi = v, i
            large = area >= args.large_area
            if large:
                tot_large += 1
                if bi >= 0 and best >= args.iou_large and vs[bi][2] in ("critical", "major"):
                    matched_large += 1
            if bi >= 0 and best >= args.iou_cls:
                used[bi] = True
                if vs[bi][1] == CLS_NAMES.get(int(c), "?"):
                    cls_tp[vs[bi][1]] += 1
                else:
                    cls_fp[vs[bi][1]] += 1
                    cls_fn[CLS_NAMES.get(int(c), "?")] += 1
            elif bi < 0:
                cls_fn[CLS_NAMES.get(int(c), "?")] += 1
            if bi >= 0 and vs[bi][1] == "unknown_anomaly":
                unknown_catch += 1
        if len(g) == 0:
            n_normal += 1
            n_fp_normal += sum(1 for i, u in enumerate(used)
                               if u is False and vs[i][2] in ("critical", "major"))
    print(f"[fhead] 大缺陷召回 = {matched_large}/{tot_large}"
          + (f" = {matched_large/tot_large:.1%}" if tot_large else "（无大缺陷 GT）")
          + f" @IoU{args.iou_large}")
    print(f"[fhead] unknown 兜底命中 = {unknown_catch}")
    for c in sorted(set(cls_tp) | set(cls_fp) | set(cls_fn)):
        tp, fp, fn = cls_tp[c], cls_fp[c], cls_fn[c]
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        print(f"[fhead] {c:16s} P={prec:.2f} R={rec:.2f} F1={2*prec*rec/max(prec+rec,1e-9):.2f}"
              f" (tp={tp} fp={fp} fn={fn})")
    print(f"[fhead] 正常 tile 误报 = {n_fp_normal}/{max(n_normal,1)} strips")


CLS_NAMES = {0: "broken_warp", 1: "broken_weft", 2: "thin_weft", 3: "thick_weft",
             4: "foreign_fiber", 5: "stain_oil", 6: "hole", 7: "knot",
             8: "missing_weft_pick", 9: "slack_selvage", 10: "reed_mark",
             11: "oil_stain_spread", 12: "color_deviation", 13: "unknown_anomaly"}

if __name__ == "__main__":
    main()
