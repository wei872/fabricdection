"""Evaluation: precision / recall / F1 / mAP50 / large recall / FPR / FPS.

    python tools/evaluate.py --cfg configs/jingwei_n.yaml --ckpt runs/bank/memory_bank.pt
    python tools/evaluate.py --cfg configs/jingwei_n.yaml --smoke
    python tools/evaluate.py --cfg configs/jingwei_p.yaml --max-val 200

``validate_detector()`` is also called from ``tools/train_detector.py`` so every
training epoch prints val metrics instead of only loss.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from _common import (ROOT, SyntheticNormalDataset, data_paths, load_model,
                     make_dataset, resolve, subset_indices)
from jingwei.data import collate, make_worker_init_fn
from jingwei.image_io import ascii_safe, console_safe
from jingwei.utils import box_iou, count_parameters, profile_macs

PR_THRESHOLDS = (0.05, 0.10, 0.15, 0.25, 0.40)


# ``np.trapz`` was renamed to ``np.trapezoid`` in NumPy 2.0 and removed from the
# namespace; the old call raised ``AttributeError: module 'numpy' has no
# attribute 'trapz'`` and killed every validation pass on a NumPy 2 install.
_TRAPZ = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def average_precision(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    x = np.linspace(0, 1, 101)
    return float(_TRAPZ(np.interp(x, mrec, mpre), x))


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
    if not flat or n_gt <= 0:
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

    Unknown-class boxes (label >= nc) are counted separately: they are the
    deterministic safety net / normality fallback and must NOT be scored as a
    normal class, but they still count for large-defect recall.

    When ``large_min_area`` is given, large-defect recall (a GT box counts as
    hit when *any* prediction has IoU > 0.5) is computed in the SAME pass, so
    validation no longer decodes every image twice.
    """
    records = [[] for _ in range(nc)]
    n_gt = np.zeros(nc, np.int64)
    tp = fp = 0
    unknown_boxes = 0
    n_det = 0
    large_hit = large_total = 0
    n_images = 0
    fp_breakdown = {"duplicate_same_gt": 0, "wrong_class": 0,
                    "localization": 0, "background": 0}
    # Liveness of the deployed score gate (pre-threshold, straight from
    # head.decode): ``gate_max`` below the confidence threshold means the model
    # is not producing detections at all, which is a *different* failure from
    # "detections are wrong" and must not be debugged as a precision problem.
    liveness = {"gate_max": 0.0, "obj_max": 0.0, "cls_max": 0.0,
                "n_above": 0, "n_out": 0}
    for batch in loader:
        dets = model(batch["image"].to(device))
        st = getattr(getattr(model, "head", None), "last_stats", None) or {}
        if st:
            liveness["gate_max"] = max(liveness["gate_max"], float(st.get("score_max", 0.0)))
            liveness["obj_max"] = max(liveness["obj_max"], float(st.get("obj_max", 0.0)))
            liveness["cls_max"] = max(liveness["cls_max"], float(st.get("cls_max", 0.0)))
            liveness["n_above"] += int(st.get("n_above", 0))
            liveness["n_out"] += int(st.get("n_out", 0))
        n_images += len(dets)
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
                if c >= nc:            # 未知异常 is not one of the 19 classes
                    if score >= conf_thresh:
                        unknown_boxes += 1
                        n_det += 1
                    continue
                box = d["boxes"][i:i + 1]
                if len(gt_boxes):
                    iou = box_iou(box, gt_boxes)[0]
                    masked = iou.clone()
                    masked[used] = -1
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
          "unknown_boxes": int(unknown_boxes),
          "dets_per_image": float(n_det) / max(1.0, float(n_images)),
          "best_f1": best_f1, "best_f1_thr": best_f1_thr,
          "fp_breakdown": fp_breakdown, "pr_at": pr_at}
    pr.update(liveness)
    if large_min_area is not None:
        pr["large_recall"] = large_hit / max(1, large_total)
    valid_aps = [a for a in aps if np.isfinite(a)]
    map50 = float(np.mean(valid_aps)) if valid_aps else 0.0
    return map50, aps, pr


@torch.no_grad()
def normal_fpr(model, loader, device):
    n = fp = 0
    for batch in loader:
        dets = model(batch["image"].to(device))
        n += len(dets)
        fp += sum(len(d["boxes"]) > 0 for d in dets)
    return fp / max(1, n), fp, n


@torch.no_grad()
def validate_detector(model, val_loader, norm_loader, device, cfg,
                      conf_thresh=None, iou_thr=0.5,
                      pr_thresholds=PR_THRESHOLDS):
    """One validation pass returning all acceptance metrics as a plain dict."""
    conf = float(conf_thresh if conf_thresh is not None
                 else cfg.fusion.final_score_thresh)
    h, w = (int(v) for v in cfg.input_size[:2])
    large_area = float(cfg.fusion.large_area_ratio) * h * w
    map50, aps, pr = evaluate_map(model, val_loader, device, int(cfg.data.nc),
                                  iou_thr=iou_thr, conf_thresh=conf,
                                  large_min_area=large_area,
                                  pr_thresholds=pr_thresholds)
    rec = pr.pop("large_recall", 0.0)
    if norm_loader is not None:
        fpr, nfp, nn = normal_fpr(model, norm_loader, device)
    else:
        fpr, nfp, nn = 0.0, 0, 0
    out = {
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
def benchmark(model, device, size, iters=5, warmup=2):
    # ``--device cuda:0`` used to skip every synchronize() (the check compared
    # the string to plain "cuda"), so the reported latency was the time to
    # *enqueue* kernels, not to run them.
    on_cuda = str(device).startswith("cuda")
    x = torch.randn(1, 3, size, size, device=device)
    if on_cuda:
        torch.cuda.synchronize()
    for _ in range(warmup):
        model(x)
    if on_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        model(x)
    if on_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/jingwei_n.yaml")
    ap.add_argument("--data", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-val", type=int, default=0)
    ap.add_argument("--normal-max", type=int, default=0)
    ap.add_argument("--subset-seed", type=int, default=0,
                    help="fixed seed for random --max-val / --normal-max subsets")
    ap.add_argument("--detector-only", action="store_true",
                    help="skip normality/fusion for stage-A detector metrics")
    ap.add_argument("--diag", action="store_true",
                    help="print multi-threshold PR and FP breakdown")
    ap.add_argument("--batch", type=int, default=8,
                    help="validation batch size")
    ap.add_argument("--workers", type=int, default=2,
                    help="DataLoader workers for image decoding")
    ap.add_argument("--val-conf", type=float, default=None,
                    help="score threshold for precision/recall "
                         "(default: fusion.final_score_thresh)")
    ap.add_argument("--val-iou", type=float, default=0.5)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--names", default=None)
    ap.add_argument("--emit-unknown", action="store_true",
                    help="force unknown-defect boxes on")
    ap.add_argument("--no-unknown", action="store_true",
                    help="force unknown-defect boxes off")
    args = ap.parse_args()

    model, cfg = load_model(args.cfg, args.ckpt, args.device)
    if args.emit_unknown:
        cfg.fusion.emit_unknown_boxes = True
    if args.no_unknown:
        cfg.fusion.emit_unknown_boxes = False
    if args.detector_only:
        model.set_stage("detector")
    paths = data_paths(cfg, args.data)
    model.eval()
    size = int(cfg.input_size[0])
    names = list(cfg.data.names) if not args.names else args.names.split(",")

    if args.smoke:
        ds = SyntheticNormalDataset(n=8, size=size)
        class Wrap(torch.utils.data.Dataset):
            def __len__(self): return len(ds)
            def __getitem__(self, i):
                d = ds[i]
                d["boxes"] = torch.tensor([[100., 100., 400., 300.]])
                d["labels"] = torch.tensor([3])
                return d
        val_loader = DataLoader(Wrap(), batch_size=1, collate_fn=collate)
        norm_loader = DataLoader(SyntheticNormalDataset(n=8, size=size, seed=42),
                                 batch_size=1, collate_fn=collate)
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

    m = validate_detector(model, val_loader, norm_loader, args.device, cfg,
                          conf_thresh=args.val_conf, iou_thr=args.val_iou)
    dev = next(model.parameters()).device
    macs = profile_macs(model, [torch.zeros(1, 3, size, size, device=dev)])

    print("\n================ JingWei evaluation ================")
    print(f"params            : {count_parameters(model) / 1e6:.2f} M")
    print(f"MACs @ {size}     : {macs['total'] / 1e9:.2f} G "
          f"(conv {macs.get('conv', 0)/1e9:.2f} / linear {macs.get('linear', 0)/1e9:.2f} "
          f"/ attention {macs.get('attention', 0)/1e9:.2f})")
    print(f"conf / IoU        : {m['conf_thresh']:.3f} / {m['iou_thr']:.2f}")
    print(f"stage             : {'detector-only' if args.detector_only else 'full pipeline'}")
    print(f"mAP50             : {m['mAP50']:.4f}")
    print(f"precision         : {m['precision']:.4f}  (TP={m['tp']} FP={m['fp']})")
    print(f"recall            : {m['recall']:.4f}  (FN={m['fn']} GT={m['n_gt']})")
    print(f"F1                : {m['f1']:.4f}")
    print(f"best F1           : {m['best_f1']:.4f} @ conf={m['best_f1_thr']:.2f}")
    print(f"large-defect recall: {m['large_recall']:.4f}   <-- primary acceptance metric")
    print(f"normal image FPR  : {m['normal_fpr']:.4f}  "
          f"({m['normal_fp']}/{m['normal_n']})")
    print(f"unknown boxes     : {m['unknown_boxes']}  "
          f"(label={int(cfg.data.nc)} / 未知异常)")
    if args.diag:
        print("P/R at fixed conf :")
        for t, v in sorted(m.get("pr_at", {}).items(), key=lambda kv: float(kv[0])):
            print(f"  conf {float(t):.2f}     : "
                  f"P={v['precision']:.4f} R={v['recall']:.4f} "
                  f"F1={v['f1']:.4f} (TP={v['tp']} FP={v['fp']})")
        fb = m.get("fp_breakdown", {})
        print("FP breakdown      : " +
              " ".join(f"{k}={v}" for k, v in fb.items()))
    for i, a in enumerate(m["AP50"]):
        label = ascii_safe(names[i], f"cls{i}") if i < len(names) else str(i)
        print(console_safe(f"  AP50[{i:2d}] {label:<8.8s}: {a:.4f}"))
    if not args.smoke:
        ms = benchmark(model, args.device, size)
        print(f"latency (full pipeline, {args.device}): {ms:.2f} ms -> {1000 / ms:.1f} FPS")
    print("====================================================\n")


if __name__ == "__main__":
    main()
