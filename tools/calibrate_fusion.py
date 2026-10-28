"""Stage D — tune fusion / safety-net hyperparameters on val + normalval.

Objective (explicitly matches the user's priority order):

    maximise  recall(large defects)
    subject to  image-level FPR on normalval <= target_fpr

    python tools/calibrate_fusion.py --cfg configs/jingwei_n.yaml \
        --ckpt runs/bank/memory_bank.pt --out configs/fusion_calibrated.yaml

The network is run ONCE per image; only decode + safety net are repeated for
each candidate threshold, so the grid search costs milliseconds, not minutes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from _common import (ROOT, data_paths, load_model, resolve,
                     resolve_label_offset)
from jingwei.data import (NormalImageDataset, YOLODefectDataset, collate,
                          image_hw, to_model_input,
                          infer_labels_dir)
from jingwei.fusion import fuse_detections
from jingwei.losses import sanitize_detection_targets
from jingwei.utils import box_iou
from jingwei.image_io import console_safe


@torch.no_grad()
def precompute(model, loader, device, nc=19, label_offset=0):
    """Cache (feats, gated anomaly, context, preds, batch) per image."""
    cache = []
    for batch in loader:
        x = batch["image"]
        if "boxes" in batch:
            # Filter invalid class ids / boxes BEFORE any CUDA indexing.
            targets = sanitize_detection_targets(
                [{"boxes": batch["boxes"][b], "labels": batch["labels"][b]}
                 for b in range(x.size(0))],
                nc, (x.shape[-2], x.shape[-1]), device, label_offset)
            batch = dict(batch)
            batch["boxes"] = [t["boxes"] for t in targets]
            batch["labels"] = [t["labels"] for t in targets]
        x = to_model_input(x, device)
        feats = model.neck(model.backbone(x))
        raw = model.normality(feats, 0)["anomaly"] if model.normality is not None else None
        anomaly, context = raw, None
        if raw is not None and model.foreground is not None:
            anomaly, context, _ = model.foreground(x, feats, raw)
        preds = model.head(feats, training=False)
        cache.append((feats, anomaly, context, preds, batch))
    return cache


@torch.no_grad()
def apply_thresholds(model, cache, device, cfg):
    dets = []
    for feats, anomaly, context, preds, batch in cache:
        raw = model.head.decode(preds, model.img_size, anomaly_boost=anomaly)
        for b, d in enumerate(raw):
            if anomaly is not None:
                fb = {k: v[b:b + 1] for k, v in feats.items()}
                ctx = context[b:b + 1] if context is not None else None
                d = fuse_detections(d, anomaly[b:b + 1], model.img_size, cfg,
                                    model.unknown_label, feats=fb,
                                    roi_refiner=model.roi if cfg.fusion.roi_head else None,
                                    context=ctx)
            else:
                keep = d["scores"] >= float(cfg.fusion.final_score_thresh)
                d = {k: v[keep] for k, v in d.items()}
            dets.append(d)
    return dets


def large_recall_from_cache(dets, cache, device, ratio):
    hit = total = 0
    i = 0
    for _, _, _, _, batch in cache:
        for b in range(len(batch["image"])):
            g = batch["boxes"][b].to(device)
            d = dets[i]
            i += 1
            if g.numel() == 0:
                continue
            area = ((g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])) / (1280 * 1280)
            large = area >= ratio
            if not large.any():
                continue
            gb = g[large]
            pb = d["boxes"]
            iou = box_iou(pb, gb) if pb.numel() else torch.zeros(0, len(gb), device=device)
            hit += int((iou.max(0).values > 0.5).sum()) if iou.numel() else 0
            total += len(gb)
    return hit / max(1, total)


def normal_fpr_from_cache(dets):
    return sum(len(d["boxes"]) > 0 for d in dets) / max(1, len(dets))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/jingwei_n.yaml")
    ap.add_argument("--data", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="configs/fusion_calibrated.yaml")
    ap.add_argument("--max-val", type=int, default=200)
    ap.add_argument("--label-offset", choices=["0", "1", "auto"], default=None,
                    help="YOLO class-id offset: 0=0..18, 1=1..19, auto=infer")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    model, cfg = load_model(args.cfg, args.ckpt, args.device)
    label_offset = (args.label_offset if args.label_offset is not None
                    else cfg.data.get("label_offset", 0))
    args.nc = int(cfg.data.nc)
    paths = data_paths(cfg, args.data)
    # Resolve 'auto' once for the whole split (never per batch).
    if not args.smoke:
        label_offset = resolve_label_offset(label_offset, cfg.data.nc,
                                            [resolve(paths["val"])])
    args.label_offset = label_offset
    model.eval()
    size = int(cfg.input_size[0])

    if args.smoke:
        from _common import SyntheticNormalDataset
        val_ds = SyntheticNormalDataset(n=4, size=size)

        class Wrap(torch.utils.data.Dataset):
            def __len__(self): return len(val_ds)
            def __getitem__(self, i):
                d = val_ds[i]
                d["boxes"] = torch.tensor([[200., 200., 700., 500.]])
                d["labels"] = torch.tensor([10])
                return d
        val_loader = DataLoader(Wrap(), batch_size=1, collate_fn=collate)
        norm_loader = DataLoader(SyntheticNormalDataset(n=4, size=size, seed=7),
                                 batch_size=1, collate_fn=collate)
    else:
        vdir = resolve(paths["val"])
        ldir = infer_labels_dir(vdir)
        val_ds = YOLODefectDataset(vdir, ldir, img_size=size, augment=False)
        idx = list(range(min(args.max_val, len(val_ds))))
        val_loader = DataLoader(Subset(val_ds, idx), batch_size=1,
                                shuffle=False, collate_fn=collate)
        ndir = resolve(paths["normalval"])
        norm_loader = DataLoader(NormalImageDataset(ndir, img_size=size, augment=False),
                                 batch_size=1, shuffle=False, collate_fn=collate)

    print("precomputing network outputs ...")
    val_cache = precompute(model, val_loader, args.device,
                           nc=int(cfg.data.nc), label_offset=label_offset)
    norm_cache = precompute(model, norm_loader, args.device,
                            nc=int(cfg.data.nc), label_offset=label_offset)

    target_fpr = float(cfg.normality.target_fpr)
    best = None
    for ath in (0.35, 0.45, 0.55, 0.65, 0.75):
        cfg.fusion.anomaly_thresh = ath
        for sth in (0.10, 0.15, 0.25):
            cfg.fusion.final_score_thresh = sth
            val_dets = apply_thresholds(model, val_cache, args.device, cfg)
            norm_dets = apply_thresholds(model, norm_cache, args.device, cfg)
            rec = large_recall_from_cache(val_dets, val_cache, args.device,
                                          float(cfg.fusion.large_area_ratio))
            fpr = normal_fpr_from_cache(norm_dets)
            ok = fpr <= max(target_fpr * 3, 0.02)
            score = rec - 2.0 * max(0.0, fpr - target_fpr)
            print(f"anomaly_thresh={ath:.2f} score_thresh={sth:.2f} "
                  f"large_recall={rec:.4f} normal_FPR={fpr:.4f} "
                  f"score={score:.4f} {'OK' if ok else 'REJECT'}")
            if ok and (best is None or score > best[0]):
                best = (score, ath, sth, rec, fpr)

    if best is None:
        print("no setting met the FPR constraint; relax target_fpr and retry")
        return
    _, ath, sth, rec, fpr = best
    payload = {"fusion": {"anomaly_thresh": ath, "final_score_thresh": sth},
               "measured": {"large_defect_recall": rec, "normal_image_fpr": fpr,
                            "note": "measured by tools/calibrate_fusion.py on your val set"}}
    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, allow_unicode=True)
    print(console_safe(f"best -> {out}: {payload}"))


if __name__ == "__main__":
    main()
