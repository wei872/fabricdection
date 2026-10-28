"""Stage A — supervised detector training on the YOLO-format defect data.

    # recommended: warm-start from stage-0 DINOv2 distillation
    python tools/train_detector.py --cfg configs/jingwei_n.yaml \
        --resume runs/dino/dino_init.pt --epochs 150
    # full resume after an interrupted run (optimizer / scheduler / RNG restored)
    python tools/train_detector.py --cfg configs/jingwei_n.yaml \
        --resume runs/detector/last.pt --epochs 150
    # or from scratch / smoke
    python tools/train_detector.py --cfg configs/jingwei_n.yaml --epochs 150
    python tools/train_detector.py --cfg configs/jingwei_n.yaml --smoke 4 --epochs 1

The one-to-many head, the one-to-one (deployed) head and the foreground
alignment auxiliary branch are trained together. The normality branch is
skipped here (``set_stage('detector')``) and trained in stage B.

``--resume`` auto-detects a full training checkpoint (``last.pt``) versus a
weight-only warm start (``dino_init.pt``). Full checkpoints are written at
every epoch; ``--save-interval N`` additionally keeps numbered snapshots.
``best.pt`` is saved when validation mAP50 + large-defect recall improves.

Add ``--cache disk`` (or ``ram`` / ``bytes``) to use a YOLO-style image cache;
every epoch still prints val precision / recall / F1 / mAP50 / large recall.
Use ``--val-interval 0`` to disable validation and ``--val-max 200`` to
validate on a subset for speed.

Speed: the augmentation pipeline is now a uint8 LUT / ``cv2.multiply`` path
(~4.5 ms per 1280x1280 sample instead of ~420 ms), the uint8->float CHW cast
happens once per batch on the GPU, and cached datasets no longer pickle their
memmap into DataLoader workers, so ``--num-workers 4``/8 works on Windows too.
``--bench-data 20`` measures the loader before training starts and every
progress line shows ``data=<ms> step=<ms>`` so you can see which side is the
bottleneck. ``--amp auto`` enables bf16 (Ampere+) or fp16 + GradScaler.
See docs/SPEED.md.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader, Subset

from _common import (ROOT, SyntheticNormalDataset, add_cache_args,
                     collect_rng_state, data_paths, format_eta, iter_speed,
                     load_checkpoint, load_model, log, make_dataset, resolve,
                     resolve_label_offset, resume_checkpoint,
                     save_checkpoint, set_global_seed)
from evaluate import validate_detector
from jingwei.data import (collate, image_hw, to_model_input, worker_init)
from jingwei.utils import count_parameters


def pick_amp_dtype(mode, device):
    """Resolve --amp into a torch dtype (None = full fp32)."""
    mode = str(mode).strip().lower()
    if mode in ("off", "none", "false", "0") or not str(device).startswith("cuda"):
        return None
    if mode == "bf16":
        return torch.bfloat16
    if mode == "fp16":
        return torch.float16
    # auto: bf16 needs no loss scaling, so prefer it on Ampere and newer
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def bench_dataloader(loader, n_batches, batch, log_fn=log):
    """Measure raw loader throughput (augmentation + cache read + collate)."""
    log(f"bench-data: timing {n_batches} batches of {batch} ...")
    state = {"it": iter(loader)}

    def take():
        """next() that restarts the epoch, so tiny debug splits still measure."""
        try:
            return next(state["it"])
        except StopIteration:
            state["it"] = iter(loader)
            return next(state["it"])

    for _ in range(2):                       # warm up workers / page cache
        take()
    t0 = time.time()
    seen = 0
    for i in range(n_batches):
        b = take()
        seen += int(b["image"].shape[0])
    dt = max(time.time() - t0, 1e-6)
    log(f"bench-data: {seen} images in {dt:.2f}s -> {seen / dt:.1f} img/s  "
        f"({1000 * dt / max(1, i + 1):.1f} ms/batch)")
    log("bench-data: if this is slower than the GPU step, raise --num-workers "
        "or use --cache disk; if it is much faster, the GPU is the bottleneck")
    return seen / dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/jingwei_n.yaml")
    ap.add_argument("--data", default=None, help="override dataset yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="runs/detector")
    ap.add_argument("--num-workers", type=int, default=None,
                    help="DataLoader worker processes (default: 4 on Linux, 2 on "
                         "Windows; cached datasets are memmapped per worker)")
    ap.add_argument("--prefetch-factor", type=int, default=2,
                    help="batches prefetched per worker (default 2; every "
                         "prefetched 1280x1280 batch of 8 is ~39 MB)")
    ap.add_argument("--amp", default="auto", choices=["auto", "off", "bf16", "fp16"],
                    help="mixed precision on CUDA: auto = bf16 when supported "
                         "(Ampere+), else fp16 with loss scaling")
    ap.add_argument("--cv-threads", type=int, default=0,
                    help="OpenCV threads in the main process (0 = OpenCV default)")
    ap.add_argument("--mp-start", default="spawn",
                    choices=["spawn", "fork", "forkserver", "default"],
                    help="DataLoader worker start method. spawn (default) is "
                         "what Windows uses anyway and avoids the fork+OpenMP "
                         "deadlock that can freeze the first batch")
    ap.add_argument("--bench-data", type=int, default=0, metavar="N",
                    help="measure dataloader throughput on N batches before "
                         "training and exit (0 = off)")
    ap.add_argument("--epoch-iters", type=int, default=0,
                    help="limit iterations per epoch (0 = full epoch); useful to "
                         "see validation metrics faster on a 29k-image dataset")
    ap.add_argument("--smoke", type=int, default=0, help="synthetic images for a dry run")
    ap.add_argument("--resume", default=None,
                    help="full training checkpoint (resume) or weight-only warm start")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-interval", type=int, default=0,
                    help="also keep epoch_XXXX.pt every N epochs (0 disables snapshots)")
    ap.add_argument("--log-interval", type=int, default=20,
                    help="print a progress line every N batches (0 disables iter logs)")
    add_cache_args(ap)
    ap.add_argument("--val-interval", type=int, default=1,
                    help="run val precision/recall/mAP every N epochs (0 disables)")
    ap.add_argument("--val-max", type=int, default=0,
                    help="limit val images for fast periodic validation (0 = all)")
    ap.add_argument("--normal-max", type=int, default=0,
                    help="limit normalval images for FPR (0 = all)")
    ap.add_argument("--val-conf", type=float, default=None,
                    help="score threshold for val precision/recall "
                         "(default: fusion.final_score_thresh)")
    ap.add_argument("--val-iou", type=float, default=0.5,
                    help="IoU threshold for val TP matching")
    ap.add_argument("--label-offset", choices=["0", "1", "auto"], default=None,
                    help="YOLO class-id offset: 0=0..18 (standard), "
                         "1=1..19 labels, auto=infer 1-based when cls>=nc")
    args = ap.parse_args()

    set_global_seed(args.seed)
    # Weights are loaded by resume_checkpoint() below so a full checkpoint can
    # restore model + optimizer + scheduler + RNG in one place.
    model, cfg = load_model(args.cfg, None, args.device)
    if args.label_offset is not None:
        cfg.data.label_offset = args.label_offset
    paths = data_paths(cfg, args.data)
    # Resolve 'auto' ONCE for the whole run (see tools/_common.resolve_label_offset);
    # a per-batch decision would mix 0-based and 1-based ids in one model.
    if not args.smoke:
        cfg.data.label_offset = resolve_label_offset(
            cfg.data.label_offset, cfg.data.nc, [resolve(paths["train"])])
    model.det_loss.label_offset = cfg.data.label_offset
    # make_dataset() reads these to normalise labels while building a cache.
    args.label_offset = cfg.data.label_offset
    args.nc = int(cfg.data.nc)
    epochs = args.epochs or int(cfg.train.detector_epochs)
    batch = args.batch or int(cfg.train.detector_batch)
    lr = args.lr or float(cfg.train.detector_lr)
    size = int(cfg.input_size[0])
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    amp_dtype = pick_amp_dtype(args.amp, args.device)

    if args.num_workers is None:
        args.num_workers = 2 if os.name == "nt" else 4
    if args.smoke:
        args.num_workers = 0
    args.num_workers = max(0, int(args.num_workers))

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

    if args.cv_threads:
        cv2.setNumThreads(int(args.cv_threads))

    def make_loader(ds, bs, shuffle, persistent=True):
        kw = {}
        if args.num_workers > 0:
            kw.update(prefetch_factor=max(1, int(args.prefetch_factor)),
                      worker_init_fn=worker_init,
                      persistent_workers=persistent)
            if args.mp_start != "default":
                kw.update(multiprocessing_context=args.mp_start)
        return DataLoader(ds, batch_size=bs, shuffle=shuffle,
                          num_workers=args.num_workers, collate_fn=collate,
                          drop_last=False, pin_memory=torch.cuda.is_available(),
                          **kw)

    loader = make_loader(train_ds, batch, True)
    if args.bench_data:
        bench_dataloader(loader, int(args.bench_data), batch)
        return

    # ---- validation loaders (precision / recall / mAP / FPR) -----------------
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
        if args.val_max:
            vds = Subset(vds, range(min(args.val_max, len(vds))))
        if args.normal_max:
            nds = Subset(nds, range(min(args.normal_max, len(nds))))
        # Validation runs in the main process (batch 1): creating a SECOND
        # worker pool while the persistent train workers are alive deadlocks on
        # some platforms, and per-image cost here is a cached memmap read.
        vl = DataLoader(vds, batch_size=1, shuffle=False, collate_fn=collate)
        nl = DataLoader(nds, batch_size=1, shuffle=False, collate_fn=collate)
        return vl, nl

    val_loader = norm_loader = None
    if args.val_interval:
        val_loader, norm_loader = build_val_loaders()

    model.set_stage("detector")
    model.train()
    params = [p for p in model.parameter_groups("detector") if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=float(cfg.train.weight_decay))
    steps_per_epoch = min(int(args.epoch_iters) or len(loader), len(loader))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, epochs * steps_per_epoch))

    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    n_batches = len(loader)
    resume_state = load_checkpoint(args.resume) if args.resume else None
    info = (resume_checkpoint(resume_state, "detector", model, opt, sched,
                              train_ds, device=args.device)
            if resume_state is not None else None)
    start_epoch = min(info["epoch"], epochs) if info and info["full"] else 0
    it = info["iter"] if info and info["full"] else 0
    meta = info["meta"] if info else {}
    ckpt_offset = meta.get("label_offset") if info and info["full"] else None
    if ckpt_offset is not None and str(ckpt_offset) != str(cfg.data.label_offset):
        log(f"WARN: checkpoint label_offset={ckpt_offset}, current="
            f"{cfg.data.label_offset}; pass --label-offset {ckpt_offset} "
            f"to keep class ids identical to the resumed run")
    best_score = float(meta.get("best_score", -1.0)) if info and info["full"] else -1.0
    resume_mode = ("full" if info and info["full"] else
                   "weights-only" if info else "none")

    trainable = sum(p.numel() for p in params) / 1e6
    log("=" * 72)
    log("JingWei stage A - supervised detector training")
    log(f"cfg={args.cfg}  data={args.data or cfg.data.dataset_yaml}  device={args.device}")
    log(f"resume={args.resume or 'none'}  mode={resume_mode}  out={out}")
    if args.device.startswith("cuda") and torch.cuda.is_available():
        log(f"gpu={torch.cuda.get_device_name(0)}")
    log(f"model={count_parameters(model) / 1e6:.2f} M params  "
        f"trainable={trainable:.2f} M  img_size={size}")
    log(f"images={len(train_ds)}  batches/epoch={n_batches}  epochs={epochs}  "
        f"batch={batch}  lr={lr:g}  workers={0 if args.smoke else args.num_workers}  "
        f"seed={args.seed}")
    log(f"start_epoch={start_epoch + 1}/{epochs}  global_iter={it}  "
        f"save_interval={args.save_interval}  log_interval={args.log_interval}")
    log(f"cache={args.cache}  cache_dir={args.cache_dir}  "
        f"val_interval={args.val_interval}  val_max={args.val_max}  "
        f"normal_max={args.normal_max}  label_offset={cfg.data.label_offset}")
    log(f"data: workers={args.num_workers}  prefetch={args.prefetch_factor}  "
        f"amp={'off' if amp_dtype is None else str(amp_dtype).split('.')[-1]}  "
        f"epoch_iters={steps_per_epoch}  "
        f"augmentation=fast-uint8  (--bench-data 20 measures loader throughput)")
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
        log(f"[!] {args.resume}: {info['reason']}; starting from epoch 1")

    if start_epoch >= epochs:
        log(f"checkpoint already reached epoch {start_epoch}/{epochs}; nothing to do")
        return

    if args.num_workers > 0:
        t_warm = time.time()
        next(iter(loader))          # start worker processes before the clock runs
        log(f"workers ready in {time.time() - t_warm:.1f}s "
            f"({args.num_workers} processes, start={args.mp_start})")

    use_scaler = amp_dtype == torch.float16
    try:                                   # torch >= 2.4
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    def step_model(batch_data, it):
        images = to_model_input(batch_data["image"], args.device)
        targets = [dict(boxes=b.to(args.device), labels=l.to(args.device))
                   for b, l in zip(batch_data["boxes"], batch_data["labels"])]
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype,
                            enabled=amp_dtype is not None):
            result = model(images, targets=targets, it=it)
            loss = result["losses"]["total"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at iter {it}")
        if use_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            opt.step()
        sched.step()
        return result

    batch_left = time.perf_counter()
    global_t0 = time.time()
    global_it0 = it
    for ep in range(start_epoch, epochs):
        t0 = time.time()
        interval_t0 = t0
        running = {}
        interval_loss, interval_n = 0.0, 0
        data_s, step_s = 0.0, 0.0
        iters_this_epoch = steps_per_epoch
        for bi, batch_data in enumerate(loader):
            t_got = time.perf_counter()
            data_s += t_got - batch_left
            result = step_model(batch_data, it)
            batch_left = time.perf_counter()
            step_s += batch_left - t_got
            for k, v in result["losses"].items():
                running[k] = running.get(k, 0.0) + float(v.detach())
            total_loss = float(result["losses"]["total"].detach())
            interval_loss += total_loss
            interval_n += 1
            it += 1

            if args.log_interval and (it == 1 or
                                      (bi + 1) % args.log_interval == 0 or
                                      (bi + 1) == n_batches):
                elapsed = max(time.time() - interval_t0, 1e-6)
                ips = iter_speed(it - global_it0, time.time() - global_t0,
                                 interval_n, elapsed)
                remain = ((iters_this_epoch - (bi + 1)) +
                         (epochs - ep - 1) * iters_this_epoch)
                eta = remain / max(ips, 1e-6)
                avg = interval_loss / max(interval_n, 1)
                epoch_avg = running.get("total", 0.0) / max(bi + 1, 1)
                lr_now = float(opt.param_groups[0]["lr"])
                n_done = max(1, interval_n)
                log(f"epoch {ep + 1:3d}/{epochs}  iter {bi + 1:5d}/"
                    f"{iters_this_epoch}  "
                    f"loss={avg:.4f}  epoch_avg={epoch_avg:.4f}  "
                    f"lr={lr_now:.2e}  {ips:.2f} it/s  "
                    f"data={1000 * data_s / n_done:.0f}ms "
                    f"step={1000 * step_s / n_done:.0f}ms  "
                    f"elapsed={format_eta(time.time() - t0)}  eta={format_eta(eta)}")
                interval_t0, interval_loss, interval_n = time.time(), 0.0, 0
                data_s, step_s = 0.0, 0.0

            if bi + 1 >= iters_this_epoch:
                break

        n = max(1, len(running))
        msg = " | ".join(f"{k}={v / n:.3f}" for k, v in running.items() if k != "total")
        log(f"epoch {ep + 1:3d}/{epochs}  {msg}  total={running.get('total', 0) / n:.3f} "
            f"({time.time() - t0:.1f}s)  elapsed_total={format_eta(time.time() - global_t0)}")

        # ---- validation: precision / recall / F1 / mAP50 / large recall -----
        val_metrics = None
        if (args.val_interval and val_loader is not None and
                ((ep + 1) % args.val_interval == 0 or ep + 1 == epochs)):
            tv = time.time()
            model.eval()
            try:
                val_metrics = validate_detector(
                    model, val_loader, norm_loader, args.device, cfg,
                    conf_thresh=args.val_conf, iou_thr=args.val_iou,
                    label_offset=cfg.data.label_offset)
            finally:
                model.train()
                model.set_stage("detector")
            val_metrics["seconds"] = round(time.time() - tv, 2)
            log(f"val   epoch {ep + 1:3d}/{epochs}  "
                f"mAP50={val_metrics['mAP50']:.4f}  "
                f"precision={val_metrics['precision']:.4f}  "
                f"recall={val_metrics['recall']:.4f}  "
                f"f1={val_metrics['f1']:.4f}  "
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
                "label_offset": str(cfg.data.label_offset),
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
