"""Evaluate DINOv2 defect detection: mAP50, precision/recall and optional normal FPR."""
from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from _common import (configure_inference, data_paths, load_model,
                     make_dataset, resolve, subset_indices)
from jingwei.data import collate, make_worker_init_fn
from jingwei.image_io import ascii_safe, console_safe
from jingwei.utils import box_iou, count_parameters

PR_THRESHOLDS = (0.05, 0.10, 0.15, 0.25, 0.40)


def average_precision(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    x = np.linspace(0, 1, 101)
    return float(np.trapz(np.interp(x, mrec, mpre), x))


def _flatten_records(records):
    flat = [(float(s), bool(t)) for rec in records for s, t in rec]
    flat.sort(key=lambda x: -x[0])
    return flat


def _best_f1(records, n_gt):
    """Best global F1 over the detector's own PR curve."""
    flat = _flatten_records(records)
    if not flat or n_gt <= 0:
        return 0.0, 0.0
    cum_tp = np.cumsum([1.0 if t else 0.0 for _, t in flat])
    cum_fp = np.cumsum([0.0 if t else 1.0 for _, t in flat])
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1.0)
    recall = cum_tp / float(n_gt)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    i = int(np.argmax(f1))
    return float(f1[i]), float(flat[i][0])


def _pr_at_thresholds(records, n_gt, thresholds):
    """Global precision/recall at fixed score thresholds."""
    flat = _flatten_records(records)
    out = {}
    if not flat:
        for t in thresholds:
            out[float(t)] = {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                             "tp": 0, "fp": 0, "ndet": 0}
        return out
    scores = np.asarray([s for s, _ in flat], dtype=np.float64)
    hit = np.asarray([1.0 if t else 0.0 for _, t in flat], dtype=np.float64)
    for t in thresholds:
        sel = scores >= float(t)
        tp = float(hit[sel].sum())
        fp = float(sel.sum() - tp)
        p = tp / max(tp + fp, 1.0)
        r = tp / max(float(n_gt), 1.0)
        out[float(t)] = {
            "precision": p,
            "recall": r,
            "f1": 2.0 * p * r / max(p + r, 1e-12),
            "tp": int(tp),
            "fp": int(fp),
            "ndet": int(sel.sum()),
        }
    return out


@torch.no_grad()
def evaluate_map(model, loader, device, nc, iou_thr=0.5, conf_thresh=0.0,
                 large_min_area=None, pr_thresholds=PR_THRESHOLDS):
    """mAP50 + global precision/recall/F1 at ``conf_thresh``.

    When ``large_min_area`` is given, large-defect recall (a GT box counts as
    hit when *any* prediction has IoU > 0.5) is computed in the SAME pass, so
    validation no longer decodes every image twice.
    """
    records = [[] for _ in range(nc)]
    n_gt = np.zeros(nc, np.int64)
    tp = fp = 0
    n_det = 0
    large_hit = large_total = 0
    fp_breakdown = {"duplicate_same_gt": 0, "wrong_class": 0,
                    "localization": 0, "background": 0}
    for batch in loader:
        dets = model(batch["image"].to(device=device, dtype=next(model.parameters(), batch["image"]).dtype))
        for b, d in enumerate(dets):
            gt_boxes = batch["boxes"][b].to(device)
            gt_labels = batch["labels"][b].to(device)
            for c in range(nc):
                n_gt[c] += int((gt_labels == c).sum())
            used = torch.zeros(len(gt_boxes), dtype=torch.bool, device=device)
            order = torch.argsort(d["scores"], descending=True)
            for i in order.tolist():
                score = float(d["scores"][i])
                c = int(d["labels"][i])
                if not 0 <= c < nc:
                    raise ValueError(f"prediction label {c} outside configured classes")
                box = d["boxes"][i:i + 1]
                if len(gt_boxes):
                    iou = box_iou(box, gt_boxes)[0]
                    masked = iou.clone()
                    masked[used | (gt_labels != c)] = -1
                    j = int(masked.argmax())
                    is_tp = bool(masked[j] >= iou_thr and gt_labels[j] == c and
                                 not used[j])
                    if is_tp:
                        used[j] = True
                    if score >= conf_thresh and not is_tp:
                        best_iou = float(iou.max().item()) if iou.numel() else 0.0
                        same = (gt_labels == c) & (iou >= iou_thr)
                        if bool((same & used).any()):
                            fp_breakdown["duplicate_same_gt"] += 1
                        elif best_iou >= iou_thr:
                            fp_breakdown["wrong_class"] += 1
                        elif best_iou >= 0.1:
                            fp_breakdown["localization"] += 1
                        else:
                            fp_breakdown["background"] += 1
                else:
                    is_tp = False
                    if score >= conf_thresh:
                        fp_breakdown["background"] += 1
                records[c].append((score, is_tp))
                n_det += 1
                if score >= conf_thresh:
                    tp += int(is_tp)
                    fp += int(not is_tp)

            if large_min_area is not None and len(gt_boxes):
                area = ((gt_boxes[:, 2] - gt_boxes[:, 0]) *
                        (gt_boxes[:, 3] - gt_boxes[:, 1]))
                sel = area >= float(large_min_area)
                if bool(sel.any()):
                    gb = gt_boxes[sel]
                    pb = d["boxes"]
                    iou = box_iou(pb, gb) if pb.numel() else \
                        torch.zeros(0, len(gb), device=device)
                    large_hit += int((iou.max(0).values > 0.5).sum()) if iou.numel() else 0
                    large_total += len(gb)

    aps = []
    for c in range(nc):
        if n_gt[c] == 0:
            aps.append(float("nan"))
            continue
        rec = sorted(records[c], key=lambda x: -x[0])
        cum_tp = np.cumsum([r[1] for r in rec])
        cum_fp = np.cumsum([not r[1] for r in rec])
        aps.append(average_precision(cum_tp / n_gt[c],
                                     cum_tp / np.maximum(cum_tp + cum_fp, 1)))
    total_gt = int(n_gt.sum())
    fn = max(0, total_gt - tp)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    best_f1, best_f1_thr = _best_f1(records, total_gt)
    pr_at = _pr_at_thresholds(records, total_gt, pr_thresholds or ())
    pr = {"precision": float(precision), "recall": float(recall),
          "f1": float(f1), "tp": int(tp), "fp": int(fp), "fn": int(fn),
          "n_gt": total_gt, "n_det": int(n_det),
          "best_f1": best_f1, "best_f1_thr": best_f1_thr,
          "fp_breakdown": fp_breakdown, "pr_at": pr_at}
    if large_min_area is not None:
        pr["large_recall"] = large_hit / max(1, large_total)
    valid_aps = [a for a in aps if np.isfinite(a)]
    map50 = float(np.mean(valid_aps)) if valid_aps else 0.0
    return map50, aps, pr


@torch.no_grad()
def normal_fpr(model, loader, device):
    n = fp = known_fp = unknown_fp = 0
    for batch in loader:
        dets = model(batch["image"].to(device=device, dtype=next(model.parameters(), batch["image"]).dtype))
        n += len(dets)
        known_fp += sum(len(d["boxes"]) > 0 for d in dets)
        unknown_fp += sum(len(d.get("unknown_boxes", [])) > 0 for d in dets)
        fp += sum(len(d["boxes"]) > 0 or len(d.get("unknown_boxes", [])) > 0 for d in dets)
    return fp / max(1, n), fp, n, known_fp, unknown_fp


@torch.no_grad()
def validate_detector(model, val_loader, norm_loader, device, cfg,
                      conf_thresh=None, iou_thr=0.5,
                      pr_thresholds=PR_THRESHOLDS):
    """One validation pass returning all acceptance metrics as a plain dict."""
    conf = float(conf_thresh if conf_thresh is not None
                 else cfg.inference.final_score_thresh)
    h, w = (int(v) for v in cfg.input_size[:2])
    large_area = float(cfg.inference.large_area_ratio) * h * w
    map50, aps, pr = evaluate_map(model, val_loader, device, int(cfg.data.nc),
                                  iou_thr=iou_thr, conf_thresh=conf,
                                  large_min_area=large_area,
                                  pr_thresholds=pr_thresholds)
    rec = pr.pop("large_recall", 0.0)
    if norm_loader is not None:
        fpr, nfp, nn, known_fp, unknown_fp = normal_fpr(model, norm_loader, device)
    else:
        fpr, nfp, nn, known_fp, unknown_fp = float("nan"), 0, 0, 0, 0
    novelty = getattr(model, "novelty", None)
    unknown_active = novelty is not None and bool(novelty.calibrated) and not getattr(model, "known_only", False)
    out = {
        "unknown_active": unknown_active,
        "normal_known_fpr": known_fp / nn if nn else float("nan"),
        "normal_unknown_fpr": unknown_fp / nn if nn and unknown_active else float("nan"),
        "mAP50": float(map50),
        "AP50": [float(a) for a in aps],
        "large_recall": float(rec),
        "normal_fpr": float(fpr),
        "normal_fp": int(nfp),
        "normal_n": int(nn),
        "conf_thresh": conf,
        "iou_thr": float(iou_thr),
    }
    out.update(pr)
    return out


@torch.no_grad()
def evaluate_unknown(model, loader, device, iou_thr=.5):
    """Held-out novel categories only; labels ignored, one-to-one box matching."""
    tp = fp = gt_count = images = image_hits = 0
    for batch in loader:
        detections = model(batch["image"].to(device=device, dtype=next(model.parameters(), batch["image"]).dtype))
        for i, det in enumerate(detections):
            gt = batch["boxes"][i].to(device)
            boxes = det.get("unknown_boxes", gt.new_empty(0, 4))
            scores = det.get("unknown_scores", gt.new_empty(0))
            gt_count += len(gt)
            images += int(len(gt) > 0)
            used = torch.zeros(len(gt), device=device, dtype=torch.bool)
            hit = False
            for j in scores.argsort(descending=True).tolist():
                if len(gt):
                    overlap = box_iou(boxes[j:j + 1], gt)[0]
                    overlap[used] = -1
                    match = int(overlap.argmax())
                    if overlap[match] >= iou_thr:
                        used[match] = True
                        tp += 1
                        hit = True
                        continue
                fp += 1
            image_hits += int(hit)
    return dict(tp=tp, fp=fp, gt=gt_count,
                box_precision=tp / max(tp + fp, 1),
                box_recall=tp / gt_count if gt_count else float("nan"),
                positive_images=images,
                image_localization_recall=image_hits / images if images else float("nan"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cfg", default="configs/jingwei_dinov2.yaml")
    ap.add_argument("--data", default=None)
    ap.add_argument("--ckpt", required=True, help="trained DINOv2 detector checkpoint")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-val", type=int, default=0)
    ap.add_argument("--normal-max", type=int, default=0)
    ap.add_argument("--subset-seed", type=int, default=0)
    ap.add_argument("--known-only", action="store_true")
    ap.add_argument("--half", action="store_true", help="CUDA FP16, must match calibration")
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--val-conf", type=float, default=None)
    ap.add_argument("--val-iou", type=float, default=0.5)
    args = ap.parse_args()
    model, cfg = load_model(args.cfg, args.ckpt, args.device)
    configure_inference(model, args.known_only, args.half)
    paths = data_paths(cfg, args.data)
    size = int(cfg.input_size[0])
    kw = dict(num_workers=max(0, args.workers), collate_fn=collate, pin_memory=True)
    if args.workers > 0:
        kw.update(worker_init_fn=make_worker_init_fn(args.subset_seed), prefetch_factor=2)
    def loader(split, labeled, limit, seed):
        ds = make_dataset(resolve(paths[split]), size,
                          has_labels=labeled, augment=False, seed=seed)
        idx = subset_indices(len(ds), limit, seed)
        return DataLoader(Subset(ds, idx) if idx is not None else ds,
                          batch_size=max(1, args.batch), shuffle=False, **kw)
    vl = loader("val", True, args.max_val, args.subset_seed)
    nl = (loader("normalval", False, args.normal_max, args.subset_seed + 1)
          if paths.get("normalval") else None)
    m = validate_detector(model, vl, nl, args.device, cfg,
                          conf_thresh=args.val_conf, iou_thr=args.val_iou)
    print("\n================ DINOv2 detector evaluation ================")
    print(f"params    : {count_parameters(model) / 1e6:.2f} M")
    print(f"conf / IoU: {m['conf_thresh']:.3f} / {m['iou_thr']:.2f}")
    print(f"mAP50     : {m['mAP50']:.4f}")
    print(f"precision : {m['precision']:.4f} (TP={m['tp']} FP={m['fp']})")
    print(f"recall    : {m['recall']:.4f} (FN={m['fn']} GT={m['n_gt']})")
    print(f"F1        : {m['f1']:.4f}")
    print(f"best F1   : {m['best_f1']:.4f} @ conf={m['best_f1_thr']:.2f}")
    print(f"large localization recall: {m['large_recall']:.4f}")
    if nl is None:
        print("normal FPR: N/A (normalval not configured)")
    else:
        print(f"normal FPR: {m['normal_fpr']:.4f} ({m['normal_fp']}/{m['normal_n']})")
    print(f"normal known/unknown FPR: {m['normal_known_fpr']:.4f} / {m['normal_unknown_fpr']:.4f}")
    print(f"unknown active: {m['unknown_active']}")
    if paths.get("unknownval") and m["unknown_active"]:
        uv = loader("unknownval", True, args.max_val, args.subset_seed)
        print("held-out unknown localization:", evaluate_unknown(model, uv, args.device, args.val_iou))
    else:
        print("held-out unknown localization: N/A (requires calibrated branch and unknownval YOLO boxes)")
    if args.diag:
        for t, v in sorted(m['pr_at'].items()):
            print(f"conf {t:.2f}: P={v['precision']:.4f} R={v['recall']:.4f} "
                  f"F1={v['f1']:.4f} TP={v['tp']} FP={v['fp']}")
        print("FP breakdown:", m['fp_breakdown'])
    for i, a in enumerate(m['AP50']):
        print(console_safe(f"AP50[{i}] {cfg.data.names[i]}: {a:.4f}"))


if __name__ == "__main__":
    main()
