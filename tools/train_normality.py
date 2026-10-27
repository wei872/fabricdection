"""Stage B — normality branch training on NORMAL images ONLY (no labels).

    python tools/train_normality.py --cfg configs/jingwei_n.yaml --ckpt runs/detector/best.pt --cache disk
    # full resume after an interrupted run
    python tools/train_normality.py --cfg configs/jingwei_n.yaml --resume runs/normality/last.pt --cache disk
    python tools/train_normality.py --cfg configs/jingwei_n.yaml --smoke 32 --iters 20

The backbone / neck / detection head are FROZEN (Dinomaly's "frozen foundation
encoder"). Only the patch-feature projection, the noisy bottleneck, the linear
attention decoder and the fusion gate are updated, using the *loose*
hard-mining cosine reconstruction loss.

Full checkpoints go to ``last.pt`` (and numbered ``iter_XXXXXX.pt`` snapshots);
the final weight-only ``normality.pt`` remains for stage C compatibility.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from _common import (ROOT, SyntheticNormalDataset, add_cache_args,
                     collect_rng_state, data_paths, format_eta, iter_speed,
                     load_checkpoint, load_model, log, make_dataset, resolve,
                     resume_checkpoint, save_checkpoint, set_global_seed)
from jingwei.data import collate
from jingwei.losses import HardMiningCosineLoss
from jingwei.utils import count_parameters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/jingwei_n.yaml")
    ap.add_argument("--data", default=None)
    ap.add_argument("--ckpt", default=None, help="detector checkpoint from stage A")
    ap.add_argument("--resume", default=None,
                    help="full training checkpoint (resume) or weight-only warm start")
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="runs/normality")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-interval", type=int, default=500,
                    help="write last.pt / iter snapshots every N iterations (0 only final)")
    ap.add_argument("--log-interval", type=int, default=200,
                    help="print a progress line every N iterations (0 disables iter logs)")
    add_cache_args(ap)
    args = ap.parse_args()

    set_global_seed(args.seed)
    # --resume supersedes --ckpt; both weight-only and full checkpoints are
    # handled by resume_checkpoint() after the optimizer exists.
    init_ckpt = None if args.resume else args.ckpt
    model, cfg = load_model(args.cfg, init_ckpt, args.device)
    paths = data_paths(cfg, args.data)
    iters = args.iters or int(cfg.train.normality_iters)
    batch = args.batch or int(cfg.train.normality_batch)
    lr = args.lr or float(cfg.train.normality_lr)
    size = int(cfg.input_size[0])

    ds = (SyntheticNormalDataset(n=args.smoke, size=size) if args.smoke else
          make_dataset("normaltrain", resolve(paths["normaltrain"]), None,
                       size, args, has_labels=False, augment=True,
                       seed=args.seed, with_descriptor=False))
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

    resume_state = load_checkpoint(args.resume) if args.resume else None
    info = (resume_checkpoint(resume_state, "normality", model, opt, None,
                              ds, device=args.device)
            if resume_state is not None else None)
    start_it = min(info["iter"], iters) if info and info["full"] else 0
    it = start_it
    meta = info["meta"] if info else {}
    resume_mode = ("full" if info and info["full"] else
                   "weights-only" if info else "none")

    trainable = sum(p.numel() for p in params) / 1e6
    log("=" * 72)
    log("JingWei stage B - normality / reconstruction training")
    log(f"cfg={args.cfg}  data={args.data or cfg.data.dataset_yaml}  device={args.device}")
    log(f"ckpt={args.ckpt or 'none'}  resume={args.resume or 'none'}  "
        f"mode={resume_mode}  out={out}")
    if args.device.startswith("cuda") and torch.cuda.is_available():
        log(f"gpu={torch.cuda.get_device_name(0)}")
    log(f"model={count_parameters(model) / 1e6:.2f} M params  "
        f"trainable={trainable:.2f} M  img_size={size}")
    log(f"normal images={len(ds)}  batches/loader={len(loader)}  iters={iters}  "
        f"batch={batch}  lr={lr:g}  seed={args.seed}")
    log(f"start_iter={start_it}/{iters}  save_interval={args.save_interval}  "
        f"log_interval={args.log_interval}")
    log(f"cache={args.cache}  cache_dir={args.cache_dir}")
    log("(use --log-interval 0 to disable iter logs; python -u disables buffering)")
    log("=" * 72)

    if info and info["full"]:
        log(f"resumed full checkpoint: {args.resume}")
        for key, now in (("batch", batch), ("lr", lr), ("cfg", args.cfg),
                         ("data", args.data or cfg.data.dataset_yaml),
                         ("seed", args.seed)):
            old = meta.get(key)
            if old is not None and old != now:
                log(f"[!] resume mismatch: {key} checkpoint={old!r}, now={now!r}; "
                    "continuing with current CLI values")
        if meta.get("iters") and meta["iters"] != iters:
            log(f"extending iters: checkpoint={meta['iters']}, now={iters}")
    elif info:
        log(f"[!] {args.resume}: {info['reason']}; starting from iter 1")

    if start_it >= iters:
        log(f"checkpoint already reached iter {start_it}/{iters}; nothing to do")
        return

    def make_state(it_now: int) -> dict:
        return {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "rng": collect_rng_state(ds),
            "train": {
                "stage": "normality",
                "format_version": 2,
                "iter": it_now,
                "iters": iters,
                "batch": batch,
                "lr": lr,
                "batches_per_loader": len(loader),
                "cfg": args.cfg,
                "data": args.data or cfg.data.dataset_yaml,
                "seed": args.seed,
                "args": vars(args),
            },
        }

    global_t0 = time.time()
    global_it0 = start_it
    running_loss, running_n = 0.0, 0
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
            running_loss += loss_val
            running_n += 1
            it += 1

            if args.log_interval and (it == 1 or
                                      it % args.log_interval == 0 or
                                      it == iters):
                elapsed = max(time.time() - interval_t0, 1e-6)
                ips = iter_speed(it - global_it0, time.time() - global_t0,
                                 interval_n, elapsed)
                eta = (iters - it) / max(ips, 1e-6)
                avg = interval_loss / max(interval_n, 1)
                run_avg = running_loss / max(running_n, 1)
                lr_now = float(opt.param_groups[0]["lr"])
                log(f"iter {it:6d}/{iters}  rec_loss={avg:.5f}  "
                    f"run_avg={run_avg:.5f}  lr={lr_now:.2e}  "
                    f"{ips:.2f} it/s  elapsed={format_eta(time.time() - global_t0)}  "
                    f"eta={format_eta(eta)}")
                interval_t0, interval_loss, interval_n = time.time(), 0.0, 0

            if (args.save_interval and it % args.save_interval == 0 and
                    it < iters):
                state = make_state(it)
                save_checkpoint(state, out / "last.pt")
                save_checkpoint(state, out / f"iter_{it:06d}.pt")
                log(f"checkpoint -> {out / 'last.pt'}")

    state = make_state(it)
    save_checkpoint(state, out / "last.pt")
    save_checkpoint({"model": model.state_dict(), "iters": it, "cfg": args.cfg},
                    out / "normality.pt")
    log(f"saved -> {out / 'last.pt'}")
    log(f"weights -> {out / 'normality.pt'}")


if __name__ == "__main__":
    main()
