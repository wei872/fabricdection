"""Stage A — supervised detector training on the YOLO-format defect data.

    # recommended: warm-start from stage-0 DINOv2 distillation
    python tools/train_detector.py --cfg configs/jingwei_n.yaml \
        --pretrained runs/dino/dino_init.pt --epochs 150 --num-workers 4
    # full resume after an interrupted run (optimizer / scheduler / RNG restored)
    python tools/train_detector.py --cfg configs/jingwei_n.yaml \
        --resume runs/detector/last.pt --epochs 150
    # or from scratch / smoke
    python tools/train_detector.py --cfg configs/jingwei_n.yaml --epochs 150
    python tools/train_detector.py --cfg configs/jingwei_n.yaml --smoke 4 --epochs 1

The one-to-many head, the one-to-one (deployed) head and the foreground
alignment auxiliary branch are trained together. The normality branch is
skipped here (``set_stage('detector')``) and trained in stage B.

``--pretrained`` explicitly loads weights only and starts a fresh optimizer /
scheduler, even for a full checkpoint. Raw ViT weights must first be distilled
with tools/pretrain_dino.py; they are not compatible with WeaveCSP.
``--resume`` auto-detects a full training checkpoint (``last.pt``) versus a
weight-only warm start (``dino_init.pt``). Full checkpoints are written at
every epoch; ``--save-interval N`` additionally keeps numbered snapshots.
``best.pt`` is saved when validation mAP50 + large-defect recall improves.

Data loading runs in parallel workers (``--num-workers``); set it to 0 only
when debugging. On Ada/Ampere GPUs ``--amp bf16`` (the default on CUDA) cuts
step time noticeably, especially at batch 8. Every epoch still prints val
precision / recall / F1 / mAP50 / large recall. Use ``--val-interval 0`` to
disable validation and ``--val-max 200`` to validate on a subset for speed.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from _common import (ROOT, SyntheticNormalDataset, collect_rng_state, data_paths,
                     format_eta, iter_speed, load_checkpoint, load_model, log,
                     make_dataset, resolve, resume_checkpoint, save_checkpoint,
                     set_global_seed, subset_indices)
from evaluate import validate_detector
from jingwei.data import collate, make_worker_init_fn
from jingwei.utils import count_parameters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/jingwei_n.yaml")
    ap.add_argument("--data", default=None, help="override dataset yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="runs/detector")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--prefetch", type=int, default=2,
                    help="batches prefetched per worker (0 uses the torch default)")
    ap.add_argument("--amp", choices=["off", "bf16", "fp16"], default="bf16",
                    help="mixed precision: bf16 (default on CUDA, no scaler) / "
                         "off / fp16")
    ap.add_argument("--smoke", type=int, default=0, help="synthetic images for a dry run")
    ap.add_argument("--max-batches", type=int, default=0,
                    help="stop each epoch after N batches (debug / benchmark only)")
    init = ap.add_mutually_exclusive_group()
    init.add_argument("--resume", default=None,
                      help="resume full training state (legacy weight-only warm start also supported)")
    init.add_argument("--pretrained", default=None,
                      help="JingWei checkpoint for weight-only initialization; resets "
                           "optimizer/scheduler/epoch. For raw DINOv2 weights run stage 0 first.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-interval", type=int, default=0,
                    help="also keep epoch_XXXX.pt every N epochs (0 disables snapshots)")
    ap.add_argument("--log-interval", type=int, default=20,
                    help="print a progress line every N batches (0 disables iter logs)")
    ap.add_argument("--val-interval", type=int, default=1,
                    help="run val precision/recall/mAP every N epochs (0 disables)")
    ap.add_argument("--val-max", type=int, default=0,
                    help="limit val images for fast periodic validation (0 = all)")
    ap.add_argument("--normal-max", type=int, default=0,
                    help="limit normalval images for FPR (0 = all)")
    ap.add_argument("--val-seed", type=int, default=0,
                    help="fixed seed for the --val-max / --normal-max subsets")
    ap.add_argument("--val-batch", type=int, default=8,
                    help="validation batch size (0 = batch 1)")
    ap.add_argument("--val-workers", type=int, default=2,
                    help="DataLoader workers for validation")
    ap.add_argument("--val-conf", type=float, default=None,
                    help="score threshold for val precision/recall "
                         "(default: fusion.final_score_thresh)")
    ap.add_argument("--val-iou", type=float, default=0.5,
                    help="IoU threshold for val TP matching")
    args = ap.parse_args()

    set_global_seed(args.seed)
    if str(args.device).startswith("cuda") and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    # Weights are loaded by resume_checkpoint() below so a full checkpoint can
    # restore model + optimizer + scheduler + RNG in one place.
    model, cfg = load_model(args.cfg, None, args.device)
    paths = data_paths(cfg, args.data)
    epochs = args.epochs or int(cfg.train.detector_epochs)
    batch = args.batch or int(cfg.train.detector_batch)
    lr = args.lr or float(cfg.train.detector_lr)
    size = int(cfg.input_size[0])

    use_amp = args.amp != "off" and str(args.device).startswith("cuda")
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    scale_enabled = use_amp and args.amp == "fp16"
    # torch.amp.GradScaler was added after the supported torch 2.0/2.2 releases.
    scaler = (torch.amp.GradScaler("cuda", enabled=scale_enabled)
              if hasattr(torch.amp, "GradScaler") else
              torch.cuda.amp.GradScaler(enabled=scale_enabled))

    if args.smoke:
        ds = SyntheticNormalDataset(n=args.smoke, size=size)
        # fake YOLO targets so the whole loss graph is exercised
        class Wrapped(torch.utils.data.Dataset):
            def __len__(self): return len(ds)
            def __getitem__(self, i):
                d = ds[i]
                d["boxes"] = torch.tensor([[100., 100., 260., 180.],
                                           [600., 700., 980., 740.]])
                d["labels"] = torch.tensor([1, 17])
                return d
        train_ds = Wrapped()
    else:
        train_ds = make_dataset("train", resolve(paths["train"]), None,
                                size, args, has_labels=True, augment=True,
                                seed=args.seed)

    nw = 0 if args.smoke else max(0, args.num_workers)
    loader_kw = dict(num_workers=nw, collate_fn=collate, drop_last=False,
                     pin_memory=True, worker_init_fn=make_worker_init_fn(args.seed))
    if nw > 0:
        loader_kw["persistent_workers"] = True
        if args.prefetch:
            loader_kw["prefetch_factor"] = int(args.prefetch)
    loader = DataLoader(train_ds, batch_size=batch, shuffle=True, **loader_kw)
    n_batches = len(loader)
    if args.max_batches:
        n_batches = min(n_batches, args.max_batches)

    # ---- validation loaders (precision / recall / mAP / FPR) -----------------
    val_batch = max(1, args.val_batch)

    def build_val_loaders():
        if args.smoke:
            base = SyntheticNormalDataset(n=max(2, min(args.smoke, 4)), size=size,
                                          seed=args.seed + 1)

            class Wrap(torch.utils.data.Dataset):
                def __len__(self): return len(base)

                def __getitem__(self, i):
                    d = base[i]
                    d["boxes"] = torch.tensor([[100., 100., 400., 300.]])
                    d["labels"] = torch.tensor([3])
                    return d
            vl = DataLoader(Wrap(), batch_size=1, shuffle=False,
                            collate_fn=collate)
            nl = DataLoader(SyntheticNormalDataset(n=4, size=size,
                                                   seed=args.seed + 2),
                            batch_size=1, shuffle=False, collate_fn=collate)
            return vl, nl
        vds = make_dataset("val", resolve(paths["val"]), None, size, args,
                           has_labels=True, augment=False, seed=args.seed)
        n_key = "normalval" if "normalval" in paths else "normaltrain"
        nds = make_dataset(n_key, resolve(paths[n_key]), None, size, args,
                           has_labels=False, augment=False, seed=args.seed)
        vidx = subset_indices(len(vds), args.val_max, args.val_seed)
        if vidx is not None:
            vds = Subset(vds, vidx)
        nidx = subset_indices(len(nds), args.normal_max, args.val_seed + 1)
        if nidx is not None:
            nds = Subset(nds, nidx)
        vw = max(0, args.val_workers)
        kw = dict(num_workers=vw, collate_fn=collate, pin_memory=True)
        if vw > 0:
            kw["worker_init_fn"] = make_worker_init_fn(args.seed + 1)
            kw["prefetch_factor"] = 2
        vl = DataLoader(vds, batch_size=val_batch, shuffle=False, **kw)
        nl = DataLoader(nds, batch_size=val_batch, shuffle=False, **kw)
        return vl, nl

    val_loader = norm_loader = None
    if args.val_interval:
        val_loader, norm_loader = build_val_loaders()

    model.set_stage("detector")
    model.train()
    params = [p for p in model.parameter_groups("detector") if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=float(cfg.train.weight_decay))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, epochs * n_batches))

    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    init_path = args.resume or args.pretrained
    resume_state = load_checkpoint(init_path) if init_path else None
    if args.pretrained:
        # A raw ViT state_dict cannot initialize the CNN backbone: distil it
        # with pretrain_dino.py first. Do not silently accept unrelated weights.
        sd = resume_state.get("model", resume_state)
        model_sd = model.state_dict()
        if not any(k.startswith("backbone.") and k in model_sd
                   and torch.is_tensor(v) and v.shape == model_sd[k].shape
                   for k, v in sd.items()):
            raise ValueError("--pretrained requires compatible JingWei backbone weights. "
                             "Raw DINOv2/YOLO weights cannot be loaded directly; "
                             "run tools/pretrain_dino.py to produce dino_init.pt.")
        resume_state = dict(resume_state)
        # Even a full last.pt is a fresh run when --pretrained is explicit.
        for key in ("optimizer", "scheduler", "rng"):
            resume_state.pop(key, None)
    info = (resume_checkpoint(resume_state, "detector", model, opt, sched,
                              train_ds, device=args.device)
            if resume_state is not None else None)
    start_epoch = min(info["epoch"], epochs) if info and info["full"] else 0
    it = info["iter"] if info and info["full"] else 0
    meta = info["meta"] if info else {}
    best_score = float(meta.get("best_score", -1.0)) if info and info["full"] else -1.0
    resume_mode = ("full" if info and info["full"] else
                   "weights-only" if info else "none")
    teacher = ((resume_state.get("teacher") or meta.get("teacher"))
               if resume_state is not None else None)

    trainable = sum(p.numel() for p in params) / 1e6
    log("=" * 72)
    log("JingWei stage A - supervised detector training")
    log(f"cfg={args.cfg}  data={args.data or cfg.data.dataset_yaml}  device={args.device}")
    log(f"resume={args.resume or 'none'}  pretrained={args.pretrained or 'none'}  "
        f"mode={resume_mode}  out={out}")
    log("DINOv2 teacher: NOT loaded in stage A (offline distillation is stage 0 only)")
    if resume_state is None:
        log("[!] initialization=random: no checkpoint supplied; dino: config alone "
            "does not enable DINOv2 pretraining")
    else:
        if teacher:
            log(f"initialization checkpoint teacher={teacher} "
                "(synthetic means smoke-only, not pretrained DINOv2)")
        else:
            log("initialization=checkpoint; DINOv2 provenance not recorded in this file")
    if args.device.startswith("cuda") and torch.cuda.is_available():
        log(f"gpu={torch.cuda.get_device_name(0)}")
    log(f"model={count_parameters(model) / 1e6:.2f} M params  "
        f"trainable={trainable:.2f} M  img_size={size}  amp={args.amp}")
    log(f"images={len(train_ds)}  batches/epoch={n_batches}  epochs={epochs}  "
        f"batch={batch}  lr={lr:g}  workers={nw}  prefetch={args.prefetch}  "
        f"seed={args.seed}")
    log(f"start_epoch={start_epoch + 1}/{epochs}  global_iter={it}  "
        f"save_interval={args.save_interval}  log_interval={args.log_interval}")
    log(f"val_interval={args.val_interval}  val_batch={val_batch}  "
        f"val_workers={0 if args.smoke else args.val_workers}  "
        f"val_max={args.val_max}  normal_max={args.normal_max}  "
        f"val_seed={args.val_seed}")
    log("(use --log-interval 0 to disable iter logs; --val-interval 0 disables val; "
        "python -u disables buffering)")
    log("=" * 72)

    if info and info["full"]:
        log(f"resumed full checkpoint: {args.resume}")
        for key, now in (("epochs", epochs), ("batch", batch), ("lr", lr),
                         ("batches_per_epoch", n_batches), ("cfg", args.cfg),
                         ("data", args.data or cfg.data.dataset_yaml),
                         ("seed", args.seed)):
            old = meta.get(key)
            if old is not None and old != now:
                log(f"[!] resume mismatch: {key} checkpoint={old!r}, now={now!r}; "
                    "continuing with current CLI values")
    elif info:
        log(f"[!] {init_path}: {info['reason']}; starting from epoch 1")
        if args.pretrained:
            log("--pretrained: fresh optimizer / scheduler; using current --lr / --epochs")

    if start_epoch >= epochs:
        log(f"checkpoint already reached epoch {start_epoch}/{epochs}; nothing to do")
        return

    global_t0 = time.time()
    global_it0 = it
    check_every = max(1, args.log_interval or 100)
    for ep in range(start_epoch, epochs):
        t0 = time.time()
        interval_t0 = t0
        running = {}                     # loss name -> GPU scalar (no per-iter sync)
        interval_loss = None
        interval_n = 0
        for bi, batch_data in enumerate(loader):
            if bi >= n_batches:
                break
            images = batch_data["image"].to(args.device, non_blocking=True)
            targets = [dict(boxes=b.to(args.device), labels=l.to(args.device))
                       for b, l in zip(batch_data["boxes"], batch_data["labels"])]
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                result = model(images, targets=targets, it=it)
                loss = result["losses"]["total"]
            do_log = bool(args.log_interval and (it == 0 or
                                                 (bi + 1) % args.log_interval == 0 or
                                                 (bi + 1) == n_batches))
            if (bi == 0 or do_log or (bi + 1) % check_every == 0) and \
                    not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at iter {it}")
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            if scaler.is_enabled():
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            sched.step()

            detached = {k: v.detach() for k, v in result["losses"].items()}
            for k, v in detached.items():
                running[k] = running.get(k, v.new_zeros(())) + v
            total_loss = detached["total"]
            interval_loss = total_loss if interval_loss is None else interval_loss + total_loss
            interval_n += 1
            it += 1

            if do_log:
                elapsed = max(time.time() - interval_t0, 1e-6)
                ips = iter_speed(it - global_it0, time.time() - global_t0,
                                 interval_n, elapsed)
                remain = (n_batches - (bi + 1)) + (epochs - ep - 1) * n_batches
                eta = remain / max(ips, 1e-6)
                avg = float(interval_loss) / max(interval_n, 1)
                epoch_avg = float(running.get("total", 0.0)) / max(bi + 1, 1)
                lr_now = float(opt.param_groups[0]["lr"])
                log(f"epoch {ep + 1:3d}/{epochs}  iter {bi + 1:5d}/{n_batches}  "
                    f"loss={avg:.4f}  epoch_avg={epoch_avg:.4f}  "
                    f"lr={lr_now:.2e}  {ips:.2f} it/s  "
                    f"elapsed={format_eta(time.time() - t0)}  eta={format_eta(eta)}")
                interval_t0, interval_loss, interval_n = time.time(), None, 0

        n = max(1, n_batches)
        msg = " | ".join(f"{k}={float(v) / n:.3f}" for k, v in running.items()
                         if k != "total")
        log(f"epoch {ep + 1:3d}/{epochs}  {msg}  total={float(running.get('total', 0.0)) / n:.3f} "
            f"({time.time() - t0:.1f}s)  elapsed_total={format_eta(time.time() - global_t0)}")

        # ---- validation: precision / recall / F1 / mAP50 / large recall -----
        val_metrics = None
        if (args.val_interval and val_loader is not None and
                ((ep + 1) % args.val_interval == 0 or ep + 1 == epochs)):
            tv = time.time()
            model.eval()
            model.set_stage("detector")
            try:
                val_metrics = validate_detector(
                    model, val_loader, norm_loader, args.device, cfg,
                    conf_thresh=args.val_conf, iou_thr=args.val_iou)
            finally:
                model.train()
                model.set_stage("detector")
            val_metrics["seconds"] = round(time.time() - tv, 2)
            log(f"val   epoch {ep + 1:3d}/{epochs}  "
                f"mAP50={val_metrics['mAP50']:.4f}  "
                f"precision={val_metrics['precision']:.4f}  "
                f"recall={val_metrics['recall']:.4f}  "
                f"f1={val_metrics['f1']:.4f}  "
                f"best_f1={val_metrics['best_f1']:.4f}@{val_metrics['best_f1_thr']:.2f}  "
                f"large_recall={val_metrics['large_recall']:.4f}  "
                f"normal_fpr={val_metrics['normal_fpr']:.4f}  "
                f"unknown_boxes={val_metrics['unknown_boxes']}  "
                f"({val_metrics['seconds']:.1f}s)")
            score = 0.5 * val_metrics["mAP50"] + 0.5 * val_metrics["large_recall"]
            if score > best_score:
                best_score = score
                save_checkpoint({"model": model.state_dict(),
                                 "metrics": val_metrics,
                                 "train": {"stage": "detector",
                                           "teacher": teacher,
                                           "epoch": ep + 1,
                                           "best_score": best_score}},
                                out / "best.pt")
                log(f"best   -> {out / 'best.pt'}  (score={score:.4f})")

        state = {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "rng": collect_rng_state(train_ds),
            "metrics": val_metrics,
            "train": {
                "stage": "detector",
                "teacher": teacher,
                "initialization_checkpoint": str(init_path) if init_path else None,
                "format_version": 3,
                "epoch": ep + 1,          # next epoch to run on resume
                "iter": it,
                "epochs": epochs,
                "batch": batch,
                "lr": lr,
                "batches_per_epoch": n_batches,
                "cfg": args.cfg,
                "data": args.data or cfg.data.dataset_yaml,
                "seed": args.seed,
                "best_score": best_score,
                "args": vars(args),
            },
        }
        save_checkpoint(state, out / "last.pt")
        if args.save_interval and (ep + 1) % args.save_interval == 0:
            save_checkpoint(state, out / f"epoch_{ep + 1:04d}.pt")
        log(f"checkpoint -> {out / 'last.pt'}")
    log(f"saved -> {out / 'last.pt'}")


if __name__ == "__main__":
    main()
