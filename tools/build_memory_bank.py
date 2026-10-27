"""Stage C — build the fabric-conditioned PatchCore memory bank and calibrate it.

    python tools/build_memory_bank.py --cfg configs/jingwei_n.yaml \
        --ckpt runs/normality/normality.pt --out runs/bank
    python tools/build_memory_bank.py --cfg configs/jingwei_n.yaml --smoke 16

Offline, once. Steps:
  1. descriptor -> k-means fabric prototypes (布种原型) on ``normaltrain``
  2. patch features -> per-prototype spherical k-means / coreset compression
  3. ``normalval`` calibration -> per-prototype (mu, sigma, tau) at a target FPR
     and the reconstruction-residual threshold
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from _common import (ROOT, SyntheticNormalDataset, data_paths,
                     descriptors_for_folder, extract_patch_features,
                     load_model, resolve, save_checkpoint)
from jingwei.data import list_images
from jingwei.fabric import (fabric_descriptor, fit_fabric_prototypes,
                            greedy_coreset, spherical_kmeans)
from jingwei.image_io import console_safe, imread_unicode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/jingwei_n.yaml")
    ap.add_argument("--data", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default="runs/bank")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-images", type=int, default=0)
    args = ap.parse_args()

    model, cfg = load_model(args.cfg, args.ckpt, args.device)
    paths = data_paths(cfg, args.data)
    bank = model.normality.bank
    K = int(cfg.normality.n_prototypes)
    C = int(cfg.normality.n_centroids)
    g = int(cfg.normality.grid)
    size = int(cfg.input_size[0])
    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    # ---------------- 1 + 2 : prototypes and per-prototype patch features -----------
    if args.smoke:
        ds = SyntheticNormalDataset(n=args.smoke, size=size)
        descs = np.stack([ds[i]["descriptor"].numpy() for i in range(len(ds))])
        proto, labels = fit_fabric_prototypes(descs, k=min(K, len(ds)))
        feats_by_k = {}
        model.eval()
        with torch.no_grad():
            for i in range(len(ds)):
                x = ds[i]["image"][None].to(args.device)
                q = model.normality.patch(model.neck(model.backbone(x)))
                q = q.permute(0, 2, 3, 1).reshape(-1, q.shape[1]).cpu().numpy()
                feats_by_k.setdefault(int(labels[i]), []).append(q)
        feats_by_k = {k: np.concatenate(v, 0) for k, v in feats_by_k.items()}
        val_ds = SyntheticNormalDataset(n=max(4, args.smoke // 2), size=size, seed=999)
        val_labels = proto.assign_many(
            np.stack([val_ds[i]["descriptor"].numpy() for i in range(len(val_ds))]))
    else:
        limit = args.max_images or None
        files, descs, proto, labels = descriptors_for_folder(
            resolve(paths["normaltrain"]), K, limit)
        feats_by_k = extract_patch_features(model, files, labels, args.device, cfg,
                                            batch_size=args.batch_size, limit=limit)
        val_files = list_images(resolve(paths["normalval"]))
        val_ds = []
        for f in val_files:
            gray = imread_unicode(f, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            val_ds.append((f, proto.assign(fabric_descriptor(gray))))
        if not val_ds:
            raise FileNotFoundError(
                f"no readable normalval images in {console_safe(resolve(paths['normalval']))}")

    for k, x in feats_by_k.items():
        x = np.asarray(x, np.float32)
        x /= np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8
        if str(cfg.normality.bank_method) == "coreset":
            keep = greedy_coreset(x, min(C, len(x)), seed=int(cfg.normality.seed))
            centers = spherical_kmeans(keep, C, iters=8)
        else:
            centers = spherical_kmeans(x, C, iters=20, seed=int(cfg.normality.seed))
        bank.set_prototype(k, centers)
        print(f"prototype {k:2d}: {len(x):7d} patch features -> {centers.shape[0]} centroids")
    # fill empty prototypes with the mean of the fitted ones
    fitted = [k for k in range(K) if bool(bank.valid[k])]
    if fitted:
        mean_c = bank.centroids[fitted].mean(0)
        for k in range(K):
            if k not in fitted:
                bank.set_prototype(k, mean_c.numpy())

    # ---------------- 3 : calibration on normalval ----------------------------------
    dists = {k: [] for k in range(K)}
    residuals = []
    model.eval()

    def collect(x, k):
        feats = model.neck(model.backbone(x))
        q = model.normality.patch(feats)
        d = bank.raw_distance(q, k).detach().cpu().numpy().ravel()
        dists[k].append(d)
        _, res, _, _ = model.normality.recon(feats["n5"], return_features=True)
        residuals.append(res.detach().cpu().numpy().ravel())

    if args.smoke:
        for i in range(len(val_ds)):
            collect(val_ds[i]["image"][None].to(args.device), int(val_labels[i]))
    else:
        for f, k in val_ds:
            im = imread_unicode(f, cv2.IMREAD_COLOR)
            if im is None:
                continue
            im = cv2.cvtColor(cv2.resize(im, (size, size)), cv2.COLOR_BGR2RGB)
            x = torch.from_numpy(im.transpose(2, 0, 1)).float()[None] / 255.0
            collect(x.to(args.device), k)

    z_pool, stats = [], {}
    for k in range(K):
        if not dists[k]:
            continue
        d = np.concatenate(dists[k])
        mu, sigma = float(d.mean()), max(float(d.std()), 1e-6)
        z = (d - mu) / sigma
        z_pool.append(z)
        stats[k] = (mu, sigma)
    z_pool = np.concatenate(z_pool) if z_pool else np.zeros(1, np.float32)
    fpr = float(cfg.normality.target_fpr)
    for k in range(K):
        mu, sigma = stats.get(k, (float(np.median([s[0] for s in stats.values()])) if stats else 0.0,
                                  float(np.median([s[1] for s in stats.values()])) if stats else 1.0))
        if k in stats:
            z = (np.concatenate(dists[k]) - mu) / sigma
            tau = float(np.quantile(z, 1.0 - fpr))
        else:
            # conservative: strictest threshold seen + 1 sigma, so an unseen
            # fabric prototype errs on the side of NOT raising false alarms
            tau = float(np.quantile(z_pool, 1.0 - fpr)) + 1.0
        bank.set_calibration(k, mu, sigma, tau, slope=1.0)
        print(f"prototype {k:2d}: mu={mu:.5f} sigma={sigma:.5f} tau={tau:.3f}"
              f"{' (conservative fallback)' if k not in stats else ''}")

    res = np.concatenate(residuals) if residuals else np.array([0.0], np.float32)
    r_tau = float(np.quantile(res, 1.0 - float(cfg.normality.target_fpr)))
    spread = max(float(np.quantile(res, 0.999) - np.quantile(res, 0.5)), 1e-4)
    model.normality.recon.rec_tau.fill_(r_tau)
    model.normality.recon.rec_slope.fill_(8.0 / spread)
    print(f"reconstruction: tau={r_tau:.5f} slope={8.0 / spread:.2f}")

    np.savez(out / "fabric_prototypes.npz",
             centers=proto.centers, mean=proto.mean, std=proto.std,
             descriptors=descs)
    save_checkpoint({"bank": bank.state_dict(),
                     "recon_calib": {"tau": r_tau, "slope": 8.0 / spread},
                     "model": model.state_dict(),
                     "n_prototypes": K, "n_centroids": C},
                    out / "memory_bank.pt")
    print(console_safe(f"saved -> {out}"))


if __name__ == "__main__":
    main()
