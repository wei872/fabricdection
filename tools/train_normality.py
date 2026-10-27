"""Stage B — normality branch training on NORMAL images ONLY (no labels).

    python tools/train_normality.py --cfg configs/jingwei_n.yaml --ckpt runs/detector/last.pt
    python tools/train_normality.py --cfg configs/jingwei_n.yaml --smoke 32 --iters 20

The backbone / neck / detection head are FROZEN (Dinomaly's "frozen foundation
encoder"). Only the patch-feature projection, the noisy bottleneck, the linear
attention decoder and the fusion gate are updated, using the *loose*
hard-mining cosine reconstruction loss.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from _common import (ROOT, SyntheticNormalDataset, data_paths, format_eta,
                     load_model, log, resolve)
from jingwei.data import NormalImageDataset, collate
from jingwei.losses import HardMiningCosineLoss
from jingwei.utils import count_parameters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/jingwei_n.yaml")
    ap.add_argument("--data", default=None)
    ap.add_argument("--ckpt", default=None, help="detector checkpoint from stage A")
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="runs/normality")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--log-interval", type=int, default=200,
                    help="print a progress line every N iterations (0 disables iter logs)")
    args = ap.parse_args()

    model, cfg = load_model(args.cfg, args.ckpt, args.device)
    paths = data_paths(cfg, args.data)
    iters = args.iters or int(cfg.train.normality_iters)
    batch = args.batch or int(cfg.train.normality_batch)
    lr = args.lr or float(cfg.train.normality_lr)
    size = int(cfg.input_size[0])

    ds = (SyntheticNormalDataset(n=args.smoke, size=size) if args.smoke else
          NormalImageDataset(resolve(paths["normaltrain"]), img_size=size,
                             augment=True, with_descriptor=False))
    loader = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0,
                        collate_fn=collate, drop_last=True)

    # freeze everything except the normality branch
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.normality.normality_parameters():
        p.requires_grad_(True)
    model.set_stage("full")
    model.train()
    model.backbone.eval()      # BN statistics stay fixed -> stable target features
    model.neck.eval()

    params = [p for p in model.normality.normality_parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    criterion = HardMiningCosineLoss(cfg.train.rec_p_final, cfg.train.rec_warmup,
                                     cfg.train.rec_factor)
    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    trainable = sum(p.numel() for p in params) / 1e6
    log("=" * 72)
    log("JingWei stage B - normality / reconstruction training")
    log(f"cfg={args.cfg}  data={args.data or cfg.data.dataset_yaml}  device={args.device}")
    log(f"ckpt={args.ckpt or 'none'}  out={out}")
    if args.device.startswith("cuda") and torch.cuda.is_available():
        log(f"gpu={torch.cuda.get_device_name(0)}")
    log(f"model={count_parameters(model) / 1e6:.2f} M params  "
        f"trainable={trainable:.2f} M  img_size={size}")
    log(f"normal images={len(ds)}  batches/loader={len(loader)}  iters={iters}  "
        f"batch={batch}  lr={lr:g}")
    log(f"log_interval={args.log_interval}  "
        "(use --log-interval 0 to disable iter logs; python -u disables buffering)")
    log("=" * 72)

    it = 0
    global_t0 = time.time()
    interval_t0, interval_loss, interval_n = global_t0, 0.0, 0
    while it < iters:
        for data in loader:
            if it >= iters:
                break
            x = data["image"].to(args.device, non_blocking=True)
            feats = model.neck(model.backbone(x))
            nrm = model.normality(feats, 0, return_parts=True)
            loss = criterion(nrm["target"], nrm["reconstructed"], it)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=0.1)
            opt.step()
            loss_val = float(loss.detach())
            interval_loss += loss_val
            interval_n += 1
            it += 1

            if args.log_interval and (it == 1 or
                                      it % args.log_interval == 0 or
                                      it == iters):
                elapsed = max(time.time() - interval_t0, 1e-6)
                ips = interval_n / elapsed
                eta = (iters - it) / max(ips, 1e-6)
                avg = interval_loss / max(interval_n, 1)
                lr_now = float(opt.param_groups[0]["lr"])
                log(f"iter {it:6d}/{iters}  rec_loss={avg:.5f}  lr={lr_now:.2e}  "
                    f"{ips:.2f} it/s  elapsed={format_eta(time.time() - global_t0)}  "
                    f"eta={format_eta(eta)}")
                interval_t0, interval_loss, interval_n = time.time(), 0.0, 0
    torch.save({"model": model.state_dict(), "iters": it, "cfg": args.cfg},
               out / "normality.pt")
    log(f"saved -> {out / 'normality.pt'}")


if __name__ == "__main__":
    main()
