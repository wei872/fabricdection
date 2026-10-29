"""Pinpoint WHERE the false positives come from — one command, one report.

    python tools/diagnose_precision.py --cfg configs/jingwei_p.yaml \
        --ckpt runs/detector/best.pt --max-val 300
    python tools/diagnose_precision.py --cfg configs/jingwei_n.yaml --smoke

Why this exists
---------------
``tools/evaluate.py`` tells you precision is low; it does not tell you *why*.
Tuning blind (raising the threshold, adding class weights, retraining) is how
a week disappears.  This tool answers, for a given checkpoint, the five
questions that actually decide what to change next:

1. at which score threshold is F1 maximal (and what precision/recall comes
   with it)              -> is it a *threshold* problem?
2. how many boxes per image and of which kind are FP
   (duplicate / wrong_class / localization / background)
                          -> is it *decoding*, *classification* or *detection*?
3. are the FPs dominated by one scale (P3 vs P5)
                          -> is one branch flooding the frame?
4. do FPs have high objectness but low class score (or vice versa)
                          -> is the *objectness* or the *classifier* wrong?
5. which (GT class -> predicted class) pairs dominate
                          -> which classes need weighting / more data?

Every section ends with the concrete next action it implies.
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from _common import (ROOT, SyntheticNormalDataset, data_paths, load_model,
                     make_dataset, resolve, subset_indices)
from jingwei.data import collate, make_worker_init_fn
from jingwei.image_io import ascii_safe, console_safe
from jingwei.utils import box_iou

CONF_GRID = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60)


def _match(boxes, labels, scores, gt_boxes, gt_labels, nc, iou_thr):
    """Greedy class-aware matching, highest score first.

    Returns list of (thr_reason) tuples: one per prediction, in score order:
    ``"tp"``, ``"duplicate"``, ``"wrong_class"``, ``"localization"``,
    ``"background"`` (same definition as ``tools/evaluate.py``).
    """
    out = []
    if boxes.numel() == 0:
        return out
    used = torch.zeros(len(gt_boxes), dtype=torch.bool, device=boxes.device)
    order = torch.argsort(scores, descending=True)
    for i in order.tolist():
        c = int(labels[i])
        if c >= nc:                      # 未知异常: not a class, not scored
            out.append(("unknown", i))
            continue
        if len(gt_boxes):
            iou = box_iou(boxes[i:i + 1], gt_boxes)[0]
            masked = iou.clone()
            masked[used] = -1
            j = int(masked.argmax())
            if bool(masked[j] >= iou_thr) and int(gt_labels[j]) == c:
                used[j] = True
                out.append(("tp", i))
                continue
            best = float(iou.max())
            same = (gt_labels == c) & (iou >= iou_thr)
            if bool((same & used).any()):
                out.append(("duplicate", i))
            elif best >= iou_thr:
                out.append(("wrong_class", i))
            elif best >= 0.1:
                out.append(("localization", i))
            else:
                out.append(("background", i))
        else:
            out.append(("background", i))
    return out


@torch.no_grad()
def collect(model, loader, device, nc, iou_thr=0.5):
    """Run the val loader once and keep everything needed for the report."""
    frames = []
    for batch in loader:
        dets = model(batch["image"].to(device))
        for b, d in enumerate(dets):
            frames.append(dict(
                boxes=d["boxes"].detach().cpu(),
                scores=d["scores"].detach().cpu(),
                labels=d["labels"].detach().cpu(),
                scale=d.get("scale_idx", torch.full((len(d["labels"]),), -1,
                                                    dtype=torch.long)).detach().cpu(),
                gt_boxes=batch["boxes"][b].cpu(),
                gt_labels=batch["labels"][b].cpu(),
            ))
    return frames


def report_at(frames, conf, nc, iou_thr):
    tp = fp = fn = n_det = 0
    cats = Counter()
    per_scale = Counter()          # scale -> [tp, fp]
    per_class_gt = Counter()
    per_class_tp = Counter()
    per_class_fp = Counter()
    confusions = Counter()
    fp_scores = []                 # (obj, cls) proxies: fused score only
    tp_scores = []
    for f in frames:
        keep = f["scores"] >= conf
        boxes, labels, scores = f["boxes"][keep], f["labels"][keep], f["scores"][keep]
        scale = f["scale"][keep]
        marks = _match(boxes, labels, scores, f["gt_boxes"], f["gt_labels"],
                       nc, iou_thr)
        for l in f["gt_labels"].tolist():
            per_class_gt[int(l)] += 1
        n_det += int(keep.sum())
        hit = set()
        for kind, i in marks:
            if kind == "unknown":
                continue
            s = int(scale[i]) if i < len(scale) else -1
            if kind == "tp":
                tp += 1
                per_scale[(s, "tp")] += 1
                per_class_tp[int(labels[i])] += 1
                tp_scores.append(float(scores[i]))
                hit.add(i)
            else:
                fp += 1
                cats[kind] += 1
                per_scale[(s, "fp")] += 1
                per_class_fp[int(labels[i])] += 1
                fp_scores.append(float(scores[i]))
                if kind in ("wrong_class", "duplicate"):
                    # nearest GT class, for the confusion table
                    if len(f["gt_boxes"]):
                        iou = box_iou(boxes[i:i + 1], f["gt_boxes"])[0]
                        j = int(iou.argmax())
                        if float(iou[j]) >= 0.1:
                            confusions[(int(f["gt_labels"][j]), int(labels[i]))] += 1
        fn += max(0, len(f["gt_boxes"]) - len(hit))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return dict(tp=tp, fp=fp, fn=fn, n_det=n_det, precision=prec, recall=rec,
                f1=f1, cats=cats, per_scale=per_scale,
                per_class_gt=per_class_gt, per_class_tp=per_class_tp,
                per_class_fp=per_class_fp, confusions=confusions,
                fp_score=float(np.mean(fp_scores)) if fp_scores else 0.0,
                tp_score=float(np.mean(tp_scores)) if tp_scores else 0.0)


def normal_fpr_at(norm_frames, conf):
    """Image-level FPR from already-collected normal frames (no extra forward)."""
    n = len(norm_frames)
    fp = sum(int((f["scores"] >= conf).sum() > 0) for f in norm_frames)
    return fp / max(1, n), fp, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/jingwei_n.yaml")
    ap.add_argument("--data", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-val", type=int, default=300)
    ap.add_argument("--normal-max", type=int, default=100)
    ap.add_argument("--subset-seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--conf", type=float, default=None,
                    help="report the detailed sections at this threshold "
                         "(default: the F1-optimal one)")
    ap.add_argument("--top-confusions", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    model, cfg = load_model(args.cfg, args.ckpt, args.device)
    model.eval()
    nc = int(cfg.data.nc)
    names = list(cfg.data.names) + ["未知异常"]
    size = int(cfg.input_size[0])
    paths = data_paths(cfg, args.data)

    def short(i):
        return ascii_safe(names[i], f"cls{i}") if 0 <= i < len(names) else str(i)

    if args.smoke:
        base = SyntheticNormalDataset(n=8, size=size, seed=3)

        class Wrap(torch.utils.data.Dataset):
            def __len__(self):
                return len(base)

            def __getitem__(self, i):
                d = base[i]
                d["boxes"] = torch.tensor([[100., 100., 400., 300.]])
                d["labels"] = torch.tensor([3])
                return d
        val_loader = DataLoader(Wrap(), batch_size=1, collate_fn=collate)
        norm_loader = DataLoader(
            SyntheticNormalDataset(n=8, size=size, seed=11), batch_size=1,
            collate_fn=collate)
    else:
        vds = make_dataset("val", resolve(paths["val"]), None, size, args,
                           has_labels=True, augment=False, seed=0)
        n_key = "normalval" if "normalval" in paths else "normaltrain"
        nds = make_dataset(n_key, resolve(paths[n_key]), None, size, args,
                           has_labels=False, augment=False, seed=0)
        vidx = subset_indices(len(vds), args.max_val, args.subset_seed)
        if vidx is not None:
            vds = Subset(vds, vidx)
        nidx = subset_indices(len(nds), args.normal_max, args.subset_seed + 1)
        if nidx is not None:
            nds = Subset(nds, nidx)
        kw = dict(num_workers=max(0, args.workers), collate_fn=collate,
                  pin_memory=True)
        if args.workers > 0:
            kw["worker_init_fn"] = make_worker_init_fn(args.subset_seed)
            kw["prefetch_factor"] = 2
        val_loader = DataLoader(vds, batch_size=max(1, args.batch),
                                shuffle=False, **kw)
        norm_loader = DataLoader(nds, batch_size=max(1, args.batch),
                                 shuffle=False, **kw)

    print(console_safe(
        f"\n=== precision diagnosis: {cfg.name}  ckpt={args.ckpt or 'none'}  "
        f"device={args.device} ==="))
    print(f"peak_filter={bool(cfg.model.get('peak_filter', True))}  "
          f"dedup={bool(cfg.model.get('dedup', True))}  "
          f"top_k={int(cfg.model.top_k)}  "
          f"score_thresh={float(cfg.model.score_thresh):.2f}  "
          f"final_score_thresh={float(cfg.fusion.final_score_thresh):.2f}  "
          f"obj_o2o_mode={cfg.train.get('obj_o2o_mode', 'focal')}")
    print(f"val images={len(val_loader.dataset)}  normal images={len(norm_loader.dataset)}")
    print(f"candidates collected with final_score_thresh=0 (shipped value is "
          f"{shipped_thresh:.2f}); decode's own model.score_thresh still applies")

    # ``JingWei.eval`` already drops boxes below fusion.final_score_thresh, which
    # would make any sweep below it meaningless.  Collect UNFILTERED candidates
    # (still >= model.score_thresh inside decode) and threshold offline.
    shipped_thresh = float(cfg.fusion.final_score_thresh)
    cfg.fusion.final_score_thresh = 0.0
    frames = collect(model, val_loader, args.device, nc, args.iou)
    norm_frames = collect(model, norm_loader, args.device, nc, args.iou)
    cfg.fusion.final_score_thresh = shipped_thresh
    n_img = max(1, len(frames))

    # ------------------------------------------------ 1. threshold sweep -----
    print("\n[1] score-threshold sweep  (class-aware TP/FP, IoU "
          f"{args.iou:.2f})")
    print("    conf   boxes/img  precision  recall     F1     normal_FPR")
    rows = []
    for conf in CONF_GRID:
        r = report_at(frames, conf, nc, args.iou)
        fpr, _, _ = normal_fpr_at(norm_frames, conf)
        rows.append((conf, r, fpr))
        print(f"    {conf:4.2f}   {r['n_det'] / n_img:8.2f}  "
              f"{r['precision']:9.4f}  {r['recall']:7.4f}  {r['f1']:6.4f}  "
              f"{fpr:10.4f}")
    best_conf, best, best_fpr = max(rows, key=lambda t: t[1]["f1"])
    chosen = args.conf if args.conf is not None else best_conf
    det = report_at(frames, chosen, nc, args.iou) if args.conf is not None else best
    print(f"    -> F1-optimal threshold = {best_conf:.2f} "
          f"(P={best['precision']:.4f} R={best['recall']:.4f} "
          f"F1={best['f1']:.4f}); detailed sections below use conf={chosen:.2f}")

    # ------------------------------------------------ 2. FP anatomy ----------
    cats = det["cats"]
    total_fp = max(1, det["fp"])
    print(f"\n[2] where the {det['fp']} false positives come from "
          f"(conf>={chosen:.2f})")
    for k in ("duplicate", "wrong_class", "localization", "background"):
        print(f"    {k:<14s} {cats.get(k, 0):6d}  {cats.get(k, 0) / total_fp:6.1%}")
    print(f"    TP={det['tp']}  FN={det['fn']}  "
          f"boxes/img={det['n_det'] / n_img:.2f}  "
          f"mean score TP={det['tp_score']:.3f} vs FP={det['fp_score']:.3f}")

    # ------------------------------------------------ 3. per scale -----------
    print("\n[3] per detection scale (0 = P3 stride 8, 1 = P5 stride 32, "
          "-1 = safety net)")
    for s in (-1, 0, 1):
        t = det["per_scale"].get((s, "tp"), 0)
        f_ = det["per_scale"].get((s, "fp"), 0)
        if t + f_ == 0:
            continue
        print(f"    scale {s:+d}: boxes={t + f_:5d}  TP={t:5d}  FP={f_:5d}  "
              f"precision={t / max(1, t + f_):.3f}")

    # ------------------------------------------------ 4. per class -----------
    print("\n[4] per class at conf>="f"{chosen:.2f}  (only classes with GT or "
          "predictions)")
    print("    id  name        GT   det  TP   FP   precision  recall")
    ids = sorted(set(det["per_class_gt"]) | set(det["per_class_fp"]))
    worst = []
    for i in ids:
        gt = det["per_class_gt"].get(i, 0)
        tp = det["per_class_tp"].get(i, 0)
        fp = det["per_class_fp"].get(i, 0)
        p = tp / max(1, tp + fp)
        r = tp / max(1, gt)
        print(console_safe(
            f"    {i:2d}  {short(i):<10.10s}  {gt:4d}  {tp + fp:4d}  {tp:4d}  "
            f"{fp:4d}   {p:8.3f}  {r:6.3f}"))
        if fp >= 5:
            worst.append((fp, i, p))
    worst.sort(reverse=True)

    # ------------------------------------------------ 5. confusions ----------
    if det["confusions"]:
        print(f"\n[5] top {args.top_confusions} class confusions "
              "(GT class -> predicted class)")
        for (g, p), c in det["confusions"].most_common(args.top_confusions):
            print(console_safe(f"    {short(g):<10.10s} -> {short(p):<10.10s} {c:5d}"))

    # ------------------------------------------------ verdict -----------------
    print("\n[verdict] what to change next")
    advice = []
    if cats.get("duplicate", 0) / total_fp > 0.3:
        advice.append("* duplicates dominate: keep model.peak_filter/dedup on, and "
                      "raise fusion.final_score_thresh to the F1-optimal value above.")
    if cats.get("wrong_class", 0) / total_fp > 0.3:
        advice.append("* wrong_class dominates: the box is right, the label is not. "
                      "Set train.class_weights (inverse frequency of the train split) "
                      "or enable fusion.roi_head for a second-stage classifier.")
    if cats.get("localization", 0) / total_fp > 0.3:
        advice.append("* localization dominates: raise train.loss_weights.box_o2o / "
                      "box_o2m, check model.reg_scale, and check GT box quality.")
    if cats.get("background", 0) / total_fp > 0.3:
        advice.append("* background dominates: the deployed objectness is firing on "
                      "normal weave. Add normal images to stage A (hard negatives) and "
                      "keep train.obj_o2o_mode=focal so every cell is supervised.")
    if det["n_det"] / n_img > 6.0:
        advice.append("* boxes/img is high (>6): decoding is still producing far more "
                      "boxes than there are defects in a frame.")
    if best_conf > float(cfg.fusion.final_score_thresh) * 1.5:
        advice.append(f"* the shipped threshold ({cfg.fusion.final_score_thresh:.2f}) "
                      f"is much lower than the F1 optimum ({best_conf:.2f}); "
                      "re-run tools/calibrate_fusion.py or lower it deliberately.")
    if not advice:
        advice.append("* nothing dominant: precision is limited by the model itself — "
                      "more data / longer training / a stronger variant (jingwei_s).")
    for a in advice:
        print(console_safe("  " + a))
    if worst:
        print(console_safe("  worst classes by FP count: " +
                           ", ".join(f"{short(i)} ({fp})" for fp, i, _ in worst[:5])))
    print()


if __name__ == "__main__":
    main()
