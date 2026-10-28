"""Stage 0 — OPTIONAL but strongly recommended DINOv2 offline distillation.

    # real teacher (needs torch.hub / local DINOv2 clone + weights)
    python tools/pretrain_dino.py --cfg configs/jingwei_n.yaml \
        --teacher dinov2_vits14 --iters 5000 --num-workers 4 --out runs/dino

    # then warm-start the supervised detector from the distilled student
    python tools/train_detector.py --cfg configs/jingwei_n.yaml \
        --resume runs/dino/dino_init.pt

    # full resume after an interrupted distillation run
    python tools/pretrain_dino.py --cfg configs/jingwei_n.yaml \
        --resume runs/dino/last.pt

    # no internet / CI: synthetic teacher exercises the whole loop
    python tools/pretrain_dino.py --smoke 4 --iters 2

The DINOv2 ViT is a FROZEN TEACHER at training time only.  The exported engine
still contains only the lightweight CNN student (jingwei_p = 8.09 G MACs).
After distillation ``normality.recon.target_proj`` is re-frozen, exactly as
Dinomaly's frozen foundation encoder requires (docs/DESIGN.md §6 stage 0).

Full checkpoints go to ``last.pt`` (and numbered ``iter_XXXXXX.pt`` snapshots);
the final weight-only ``dino_init.pt`` remains the hand-off to stage A.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from _common import (ROOT, SyntheticNormalDataset,
                     checkpoint_meta, collect_rng_state, data_paths,
                     format_eta, is_resumable, iter_speed, load_checkpoint,
                     load_model, log, make_dataset, resolve, resume_checkpoint,
                     save_checkpoint, set_global_seed)
from jingwei.data import collate, make_worker_init_fn
from jingwei.dino import (DINODistillLoss, DINOStudentAdaptor, load_teacher,
                          make_views)
from jingwei.utils import count_parameters


def valid_crop(crop: int, patch: int, size: int) -> bool:
    return (crop <= size and crop > 0 and crop % patch == 0 and crop % 32 == 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/jingwei_n.yaml")
    ap.add_argument("--data", default=None, help="override dataset yaml")
    ap.add_argument("--ckpt", default=None, help="optional warm start checkpoint")
    ap.add_argument("--resume", default=None,
                    help="full training checkpoint (resume) or weight-only warm start")
    ap.add_argument("--teacher", default=None,
                    help="dinov2_vits14 | dinov2_vitb14 | dino_vitb16 | synthetic ...")
    ap.add_argument("--repo", default=None, help="local DINOv2/DINO clone or hub slug")
    ap.add_argument("--weights", default=None, help="local backbone-only .pth (offline)")
    ap.add_argument("--crop", type=int, default=None,
                    help="square crop; must be a multiple of patch_size and 32")
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="runs/dino")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--smoke", type=int, default=0,
                    help="synthetic images + synthetic teacher; no data / no download")
    ap.add_argument("--save-interval", type=int, default=500,
                    help="write last.pt / iter snapshots every N iterations (0 only final)")
    ap.add_argument("--log-interval", type=int, default=100,
                    help="print a progress line every N iterations (0 disables iter logs)")
    args = ap.parse_args()

    set_global_seed(args.seed)
    # Load once: full checkpoints must restore teacher/crop/lr BEFORE the
    # teacher is constructed; the same state is reused by resume_checkpoint().
    resume_state = load_checkpoint(args.resume) if args.resume else None
    resume_meta = checkpoint_meta(resume_state) if resume_state else {}
    resume_full = bool(resume_state and is_resumable(resume_state, "dino"))
    init_ckpt = None if args.resume else args.ckpt
    model, cfg = load_model(args.cfg, init_ckpt, args.device)
    size = int(cfg.input_size[0])
    dino_cfg = cfg.get("dino", {}) or {}

    teacher_name = "synthetic" if args.smoke else (args.teacher or dino_cfg.get("teacher", "dinov2_vits14"))
    repo = args.repo or dino_cfg.get("repo")
    weights = args.weights or dino_cfg.get("weights")
    crop = args.crop or int(dino_cfg.get("crop", 448))
    iters = args.iters or int(dino_cfg.get("iters", 5000))
    batch = args.batch or int(dino_cfg.get("batch", 4))
    lr = args.lr or float(dino_cfg.get("lr", 5e-4))
    loss_weights = dict(dino_cfg.get("loss_weights") or {})

    if resume_full:
        def _conflict(flag, value, stored, label):
            if value is not None and value != stored:
                log(f"[!] resume: ignoring --{flag}={value!r}; "
                    f"using checkpoint {label}={stored!r}")

        stored_teacher = resume_meta.get("teacher", teacher_name)
        stored_repo = resume_meta.get("repo", repo)
        stored_weights = resume_meta.get("weights", weights)
        stored_crop = int(resume_meta.get("crop", crop))
        stored_batch = int(resume_meta.get("batch", batch))
        stored_lr = float(resume_meta.get("lr", lr))
        stored_loss_weights = dict(resume_meta.get("loss_weights") or loss_weights)
        _conflict("teacher", args.teacher, stored_teacher, "teacher")
        _conflict("repo", args.repo, stored_repo, "repo")
        _conflict("weights", args.weights, stored_weights, "weights")
        _conflict("crop", args.crop, stored_crop, "crop")
        _conflict("batch", args.batch, stored_batch, "batch")
        _conflict("lr", args.lr, stored_lr, "lr")
        teacher_name, repo, weights = stored_teacher, stored_repo, stored_weights
        crop, batch, lr = stored_crop, stored_batch, stored_lr
        loss_weights = stored_loss_weights
        if args.iters is None:
            iters = int(resume_meta.get("iters", iters))
        log(f"resume config: teacher={teacher_name}  crop={crop}  "
            f"batch={batch}  lr={lr:g}  iters={iters}")

    if args.smoke:
        teacher_name = "synthetic"
        repo, weights = None, None
        batch = min(batch, max(1, args.smoke))
        if args.iters is None and not resume_full:
            iters = min(iters, 2)

    teacher = load_teacher(teacher_name, repo=repo, weights=weights,
                           device=args.device)
    patch = teacher.spec.patch_size
    # Real DINOv2/ViT teachers are frozen and reloaded from hub/weights; only
    # the tiny synthetic smoke teacher needs its random weights persisted.
    teacher_state = (teacher.state_dict()
                     if teacher.spec.family == "synthetic" else None)
    if not valid_crop(crop, patch, size):
        # 448 is the smallest practical size that is a multiple of 14 and 32
        candidates = [c for c in (224, 448, 672, 896, 1120, 1344)
                      if valid_crop(c, patch, size)]
        if not candidates:
            raise SystemExit(f"no valid crop for patch={patch}, size={size}")
        crop = candidates[min(1, len(candidates) - 1)]
        log(f"[!] adjusted crop -> {crop} (multiple of {patch} and 32)")

    log(f"stage 0: teacher={teacher.spec.name} ({teacher.spec.family}, "
        f"patch={patch}, dim={teacher.spec.embed_dim})  crop={crop}  "
        f"iters={iters}  batch={batch}  lr={lr}")

    paths = data_paths(cfg, args.data)
    if args.smoke:
        ds = SyntheticNormalDataset(n=args.smoke, size=size)
    else:
        ds = make_dataset("normaltrain", resolve(paths["normaltrain"]), None,
                          size, args, has_labels=False, augment=False,
                          seed=args.seed, with_descriptor=False)
    nw = 0 if args.smoke else max(0, args.num_workers)
    loader = DataLoader(ds, batch_size=batch, shuffle=True,
                        num_workers=nw,
                        collate_fn=collate, drop_last=not args.smoke,
                        pin_memory=args.device.startswith("cuda"),
                        worker_init_fn=make_worker_init_fn(args.seed),
                        persistent_workers=bool(nw))

    # ---- freeze everything, then unfreeze ONLY the student feature pathway ----
    for p in model.parameters():
        p.requires_grad_(False)
    trainable_modules = (model.backbone, model.neck,
                         model.normality.patch,
                         model.normality.recon.target_proj)
    for mod in trainable_modules:
        for p in mod.parameters():
            p.requires_grad_(True)

    adaptor = DINOStudentAdaptor(model, teacher.spec.embed_dim).to(args.device)
    criterion = DINODistillLoss(loss_weights)
    params = [p for p in model.parameters() if p.requires_grad] + list(adaptor.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    model.train()

    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    info = (resume_checkpoint(resume_state, "dino", model, opt, None, ds,
                              extra_modules={"dino_adaptor": adaptor,
                                             "teacher_model": teacher},
                              device=args.device)
            if resume_state is not None else None)
    start_it = min(info["iter"], iters) if info and info["full"] else 0
    it = start_it
    meta = info["meta"] if info else {}
    resume_mode = ("full" if info and info["full"] else
                   "weights-only" if info else "none")

    trainable = sum(p.numel() for p in params) / 1e6
    log("=" * 72)
    log("JingWei stage 0 - offline DINO/DINOv2 teacher distillation")
    log(f"cfg={args.cfg}  data={args.data or cfg.data.dataset_yaml}  device={args.device}")
    log(f"ckpt={args.ckpt or 'none'}  resume={args.resume or 'none'}  "
        f"mode={resume_mode}  out={out}")
    log(f"teacher={teacher.spec.name} ({teacher.spec.family})  "
        f"repo={repo or 'torch.hub'}  weights={weights or 'default/pretrained'}")
    if args.device.startswith("cuda") and torch.cuda.is_available():
        log(f"gpu={torch.cuda.get_device_name(0)}")
    log(f"model={count_parameters(model) / 1e6:.2f} M params  "
        f"trainable={trainable:.2f} M  img_size={size}  crop={crop}")
    log(f"normal images={len(ds)}  batches/loader={len(loader)}  iters={iters}  "
        f"batch={batch}  lr={lr:g}  workers={0 if args.smoke else args.num_workers}  "
        f"seed={args.seed}")
    log(f"start_iter={start_it}/{iters}  save_interval={args.save_interval}  "
        f"log_interval={args.log_interval}")
    log("(use --log-interval 0 to disable iter logs; python -u disables buffering)")
    log("=" * 72)

    if info and info["full"]:
        log(f"resumed full checkpoint: {args.resume}")
        for key, now in (("cfg", args.cfg),
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

    def make_state(it_now: int) -> dict:
        state = {
            "model": model.state_dict(),
            "dino_adaptor": adaptor.state_dict(),
            "optimizer": opt.state_dict(),
            "rng": collect_rng_state(ds),
            "train": {
                "stage": "dino",
                "format_version": 2,
                "iter": it_now,
                "iters": iters,
                "teacher": teacher.spec.name,
                "teacher_family": teacher.spec.family,
                "patch": patch,
                "embed_dim": teacher.spec.embed_dim,
                "repo": repo,
                "weights": weights,
                "crop": crop,
                "batch": batch,
                "lr": lr,
                "loss_weights": dict(loss_weights),
                "batches_per_loader": len(loader),
                "cfg": args.cfg,
                "data": args.data or cfg.data.dataset_yaml,
                "seed": args.seed,
                "args": vars(args),
            },
        }
        if teacher_state is not None:
            state["teacher_model"] = teacher_state
        return state

    def finalize(it_now: int) -> None:
        # Dinomaly stage B expects a FROZEN foundation projection.
        model.normality.recon.target_proj.requires_grad_(False)
        model.eval()
        init_state = {
            "model": model.state_dict(),
            "dino_adaptor": adaptor.state_dict(),
            "teacher": teacher.spec.name,
            "teacher_family": teacher.spec.family,
            "crop": crop,
            "iters": it_now,
            "cfg": args.cfg,
        }
        if teacher_state is not None:
            init_state["teacher_model"] = teacher_state
        save_checkpoint(init_state, out / "dino_init.pt")
        log(f"saved -> {out / 'dino_init.pt'}")
        log("next: python tools/train_detector.py --cfg "
            f"{args.cfg} --resume {out / 'dino_init.pt'}")

    if start_it >= iters:
        log(f"checkpoint already reached iter {start_it}/{iters}; nothing to do")
        finalize(start_it)
        return

    global_t0 = time.time()
    global_it0 = start_it
    running_loss, running_n = 0.0, 0
    interval_t0, interval_loss, interval_n = global_t0, 0.0, 0
    while it < iters:
        for data in loader:
            if it >= iters:
                break
            x = data["image"].to(args.device, non_blocking=True)
            tv, sv = make_views(x, crop, strong=True)
            with torch.no_grad():
                tmap = teacher.patch_map(tv)

            bb = model.backbone(sv)
            feats = model.neck(bb)
            q = model.normality.patch(feats)
            rec_target = model.normality.recon.target_proj(feats["n5"])
            smaps = adaptor(feats, q, rec_target)
            loss = criterion(smaps, tmap)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
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
                log(f"iter {it:6d}/{iters}  distill_cosine={avg:.5f}  "
                    f"run_avg={run_avg:.5f}  lr={lr_now:.2e}  {ips:.2f} it/s  "
                    f"elapsed={format_eta(time.time() - global_t0)}  "
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
    log(f"checkpoint -> {out / 'last.pt'}")
    finalize(it)


if __name__ == "__main__":
    main()
