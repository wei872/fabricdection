"""Evaluation: mAP50, per-class AP, large-defect recall, normal-image FPR, FPS.

    python tools/evaluate.py --cfg configs/jingwei_n.yaml --ckpt runs/bank/memory_bank.pt
    python tools/evaluate.py --cfg configs/jingwei_n.yaml --smoke
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from _common import ROOT, SyntheticNormalDataset, data_paths, load_model, resolve
from jingwei.data import NormalImageDataset, YOLODefectDataset, collate
from jingwei.image_io import ascii_safe, console_safe
from jingwei.utils import box_iou, count_parameters, profile_macs


def average_precision(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    x = np.linspace(0, 1, 101)
    return float(np.trapz(np.interp(x, mrec, mpre), x))


@torch.no_grad()
def evaluate_map(model, loader, device, nc, iou_thr=0.5):
    records = [[] for _ in range(nc)]
    n_gt = np.zeros(nc, np.int64)
    for batch in loader:
        dets = model(batch["image"].to(device))
        for b, d in enumerate(dets):
            gt_boxes = batch["boxes"][b].to(device)
            gt_labels = batch["labels"][b].to(device)
            for c in range(nc):
                n_gt[c] += int((gt_labels == c).sum())
            used = torch.zeros(len(gt_boxes), dtype=torch.bool, device=device)
            order = torch.argsort(d["scores"], descending=True)
            for i in order.tolist():
                c = int(d["labels"][i])
                if c >= nc:            # 未知异常 is not one of the 19 classes
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
                records[c].append((float(d["scores"][i]), is_tp))

    aps = []
    for c in range(nc):
        if n_gt[c] == 0:
            aps.append(float("nan"))
            continue
        rec = sorted(records[c], key=lambda x: -x[0])
        tp = np.cumsum([r[1] for r in rec])
        fp = np.cumsum([not r[1] for r in rec])
        aps.append(average_precision(tp / n_gt[c], tp / np.maximum(tp + fp, 1)))
    return float(np.nanmean(aps)), aps


@torch.no_grad()
def large_defect_recall(model, loader, device, ratio):
    hit, total = 0, 0
    for batch in loader:
        dets = model(batch["image"].to(device))
        for b, d in enumerate(dets):
            g = batch["boxes"][b].to(device)
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
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--names", default=None)
    args = ap.parse_args()

    model, cfg = load_model(args.cfg, args.ckpt, args.device)
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
        vdir, ndir = resolve(paths["val"]), resolve(paths["normalval"])
        ldir = str(vdir).replace("/images/", "/labels/")
        vds = YOLODefectDataset(vdir, ldir, img_size=size, augment=False)
        idx = list(range(min(args.max_val, len(vds)))) if args.max_val else list(range(len(vds)))
        val_loader = DataLoader(vds, batch_size=1, shuffle=False, collate_fn=collate, sampler=idx)
        norm_loader = DataLoader(NormalImageDataset(ndir, img_size=size, augment=False),
                                 batch_size=1, shuffle=False, collate_fn=collate)

    map50, aps = evaluate_map(model, val_loader, args.device, int(cfg.data.nc))
    rec = large_defect_recall(model, val_loader, args.device, float(cfg.fusion.large_area_ratio))
    fpr, fp, n = normal_fpr(model, norm_loader, args.device)
    macs = profile_macs(model, [torch.zeros(1, 3, size, size)])

    print("\n================ JingWei evaluation ================")
    print(f"params            : {count_parameters(model) / 1e6:.2f} M")
    print(f"MACs @ {size}     : {macs['total'] / 1e9:.2f} G "
          f"(conv {macs.get('conv', 0)/1e9:.2f} / linear {macs.get('linear', 0)/1e9:.2f} "
          f"/ attention {macs.get('attention', 0)/1e9:.2f})")
    print(f"mAP50             : {map50:.4f}")
    print(f"large-defect recall: {rec:.4f}   <-- primary acceptance metric")
    print(f"normal image FPR  : {fpr:.4f}  ({fp}/{n})")
    for i, a in enumerate(aps):
        label = ascii_safe(names[i], f"cls{i}") if i < len(names) else str(i)
        print(console_safe(f"  AP50[{i:2d}] {label:<8.8s}: {a:.4f}"))
    if not args.smoke:
        ms = benchmark(model, args.device, size)
        print(f"latency (full pipeline, {args.device}): {ms:.2f} ms -> {1000 / ms:.1f} FPS")
    print("====================================================\n")


if __name__ == "__main__":
    main()
