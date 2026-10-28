"""Evaluation: precision / recall / F1 / mAP50 / large recall / FPR / FPS.

    python tools/evaluate.py --cfg configs/jingwei_n.yaml --ckpt runs/bank/memory_bank.pt
    python tools/evaluate.py --cfg configs/jingwei_n.yaml --smoke
    python tools/evaluate.py --cfg configs/jingwei_p.yaml --cache disk --max-val 200

``validate_detector()`` is also called from ``tools/train_detector.py`` so every
training epoch prints val metrics instead of only loss.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from _common import (ROOT, SyntheticNormalDataset, add_cache_args, data_paths,
                     load_model, make_dataset, resolve, resolve_label_offset)
from jingwei.data import collate
from jingwei.image_io import ascii_safe, console_safe
from jingwei.losses import sanitize_detection_targets
from jingwei.utils import box_iou, count_parameters, profile_macs


def average_precision(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    x = np.linspace(0, 1, 101)
    return float(np.trapz(np.interp(x, mrec, mpre), x))


@torch.no_grad()
def evaluate_map(model, loader, device, nc, iou_thr=0.5, conf_thresh=0.0,
                 label_offset=0):
    """mAP50 + global precision/recall/F1 at ``conf_thresh``.

    Unknown-class boxes (label >= nc) are counted separately: they are the
    deterministic safety net / normality fallback and must NOT be scored as a
    normal class, but they still count for large-defect recall.
    """
    records = [[] for _ in range(nc)]
    n_gt = np.zeros(nc, np.int64)
    tp = fp = 0
    unknown_boxes = 0
    n_det = 0
    for batch in loader:
        images = batch["image"]
        bsz = images.size(0)
        raw_boxes = batch.get("boxes", [torch.zeros((0, 4))] * bsz)
        raw_labels = batch.get("labels", [torch.zeros((0,), dtype=torch.long)] * bsz)
        targets = sanitize_detection_targets(
            [{"boxes": raw_boxes[b], "labels": raw_labels[b]}
             for b in range(bsz)],
            nc, (images.shape[-2], images.shape[-1]), device, label_offset)
        dets = model(images.to(device))
        for b, d in enumerate(dets):
            gt_boxes = targets[b]["boxes"]
            gt_labels = targets[b]["labels"]
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
                    iou[used] = -1
                    j = int(iou.argmax())
                    is_tp = bool(iou[j] >= iou_thr and gt_labels[j] == c and not used[j])
                    if is_tp:
                        used[j] = True
                else:
                    is_tp = False
                records[c].append((score, is_tp))
                n_det += 1
                if score >= conf_thresh:
                    tp += int(is_tp)
                    fp += int(not is_tp)

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
    pr = {"precision": float(precision), "recall": float(recall),
          "f1": float(f1), "tp": int(tp), "fp": int(fp), "fn": int(fn),
          "n_gt": total_gt, "n_det": int(n_det),
          "unknown_boxes": int(unknown_boxes)}
    valid_aps = [a for a in aps if np.isfinite(a)]
    map50 = float(np.mean(valid_aps)) if valid_aps else 0.0
    return map50, aps, pr


@torch.no_grad()
def large_defect_recall(model, loader, device, ratio, nc=19, label_offset=0):
    hit, total = 0, 0
    for batch in loader:
        images = batch["image"]
        bsz = images.size(0)
        raw_boxes = batch.get("boxes", [torch.zeros((0, 4))] * bsz)
        raw_labels = batch.get("labels", [torch.zeros((0,), dtype=torch.long)] * bsz)
        targets = sanitize_detection_targets(
            [{"boxes": raw_boxes[b], "labels": raw_labels[b]}
             for b in range(bsz)],
            nc, (images.shape[-2], images.shape[-1]), device, label_offset)
        dets = model(images.to(device))
        for b, d in enumerate(dets):
            g = targets[b]["boxes"]
            if g.numel() == 0:
                continue
            area = ((g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])) / (1280 * 1280)
            sel = area >= ratio
            if not sel.any():
                continue
            gb = g[sel]
            pb = d["boxes"]
            iou = box_iou(pb, gb) if pb.numel() else torch.zeros(0, len(gb))
            hit += int((iou.max(0).values > 0.5).sum()) if iou.numel() else 0
            total += len(gb)
    return hit / max(1, total)


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
                      conf_thresh=None, iou_thr=0.5, label_offset=0):
    """One validation pass returning all acceptance metrics as a plain dict."""
    conf = float(conf_thresh if conf_thresh is not None
                 else cfg.fusion.final_score_thresh)
    map50, aps, pr = evaluate_map(model, val_loader, device,
                                  int(cfg.data.nc), iou_thr=iou_thr,
                                  conf_thresh=conf, label_offset=label_offset)
    rec = large_defect_recall(model, val_loader, device,
                              float(cfg.fusion.large_area_ratio),
                              nc=int(cfg.data.nc), label_offset=label_offset)
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
    x = torch.randn(1, 3, size, size, device=device)
    if device == "cuda":
        torch.cuda.synchronize()
    for _ in range(warmup):
        model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        model(x)
    if device == "cuda":
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
    ap.add_argument("--val-conf", type=float, default=None,
                    help="score threshold for precision/recall "
                         "(default: fusion.final_score_thresh)")
    ap.add_argument("--val-iou", type=float, default=0.5)
    ap.add_argument("--label-offset", choices=["0", "1", "auto"], default=None,
                    help="YOLO class-id offset: 0=0..18, 1=1..19, auto=infer")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--names", default=None)
    ap.add_argument("--emit-unknown", action="store_true",
                    help="force unknown-defect boxes on")
    ap.add_argument("--no-unknown", action="store_true",
                    help="force unknown-defect boxes off")
    add_cache_args(ap)
    args = ap.parse_args()

    model, cfg = load_model(args.cfg, args.ckpt, args.device)
    label_offset = (args.label_offset if args.label_offset is not None
                    else cfg.data.get("label_offset", 0))
    # make_dataset() normalises cached labels with the same offset.
    args.nc = int(cfg.data.nc)
    if args.emit_unknown:
        cfg.fusion.emit_unknown_boxes = True
    if args.no_unknown:
        cfg.fusion.emit_unknown_boxes = False
    paths = data_paths(cfg, args.data)
    # Resolve 'auto' once for the whole split (never per batch).
    if not args.smoke:
        label_offset = resolve_label_offset(label_offset, cfg.data.nc,
                                            [resolve(paths["val"])])
    args.label_offset = label_offset
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
        if args.max_val:
            vds = Subset(vds, range(min(args.max_val, len(vds))))
        if args.normal_max:
            nds = Subset(nds, range(min(args.normal_max, len(nds))))
        val_loader = DataLoader(vds, batch_size=1, shuffle=False,
                                collate_fn=collate)
        norm_loader = DataLoader(nds, batch_size=1, shuffle=False,
                                 collate_fn=collate)

    m = validate_detector(model, val_loader, norm_loader, args.device, cfg,
                          conf_thresh=args.val_conf, iou_thr=args.val_iou,
                          label_offset=label_offset)
    macs = profile_macs(model, [torch.zeros(1, 3, size, size)])

    print("\n================ JingWei evaluation ================")
    print(f"params            : {count_parameters(model) / 1e6:.2f} M")
    print(f"MACs @ {size}     : {macs['total'] / 1e9:.2f} G "
          f"(conv {macs.get('conv', 0)/1e9:.2f} / linear {macs.get('linear', 0)/1e9:.2f} "
          f"/ attention {macs.get('attention', 0)/1e9:.2f})")
    print(f"conf / IoU        : {m['conf_thresh']:.3f} / {m['iou_thr']:.2f}")
    print(f"mAP50             : {m['mAP50']:.4f}")
    print(f"precision         : {m['precision']:.4f}  (TP={m['tp']} FP={m['fp']})")
    print(f"recall            : {m['recall']:.4f}  (FN={m['fn']} GT={m['n_gt']})")
    print(f"F1                : {m['f1']:.4f}")
    print(f"large-defect recall: {m['large_recall']:.4f}   <-- primary acceptance metric")
    print(f"normal image FPR  : {m['normal_fpr']:.4f}  "
          f"({m['normal_fp']}/{m['normal_n']})")
    print(f"unknown boxes     : {m['unknown_boxes']}  "
          f"(label={int(cfg.data.nc)} / 未知异常)")
    for i, a in enumerate(m["AP50"]):
        label = ascii_safe(names[i], f"cls{i}") if i < len(names) else str(i)
        print(console_safe(f"  AP50[{i:2d}] {label:<8.8s}: {a:.4f}"))
    if not args.smoke:
        ms = benchmark(model, args.device, size)
        print(f"latency (full pipeline, {args.device}): {ms:.2f} ms -> {1000 / ms:.1f} FPS")
    print("====================================================\n")


if __name__ == "__main__":
    main()
