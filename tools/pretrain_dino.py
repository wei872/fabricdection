"""Stage 0 — OPTIONAL but strongly recommended DINOv2 offline distillation.

    # real teacher (needs torch.hub / local DINOv2 clone + weights)
    python tools/pretrain_dino.py --cfg configs/jingwei_n.yaml \
        --teacher dinov2_vits14 --iters 5000 --out runs/dino

    # then warm-start the supervised detector from the distilled student
    python tools/train_detector.py --cfg configs/jingwei_n.yaml \
        --resume runs/dino/dino_init.pt

    # no internet / CI: synthetic teacher exercises the whole loop
    python tools/pretrain_dino.py --smoke 4 --iters 2

The DINOv2 ViT is a FROZEN TEACHER at training time only.  The exported engine
still contains only the lightweight CNN student (jingwei_p = 7.59 G MACs).
After distillation ``normality.recon.target_proj`` is re-frozen, exactly as
Dinomaly's frozen foundation encoder requires (docs/DESIGN.md §6 stage 0).
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from _common import (ROOT, SyntheticNormalDataset, data_paths, format_eta,
                     load_model, log, resolve)
from jingwei.data import NormalImageDataset, collate
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
    ap.add_argument("--log-interval", type=int, default=100,
                    help="print a progress line every N iterations (0 disables iter logs)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    model, cfg = load_model(args.cfg, args.ckpt, args.device)
    size = int(cfg.input_size[0])
    dino_cfg = cfg.get("dino", {}) or {}

    teacher_name = "synthetic" if args.smoke else (args.teacher or dino_cfg.get("teacher", "dinov2_vits14"))
    repo = args.repo or dino_cfg.get("repo")
    weights = args.weights or dino_cfg.get("weights")
    crop = args.crop or int(dino_cfg.get("crop", 448))
    iters = args.iters or int(dino_cfg.get("iters", 5000))
    batch = args.batch or int(dino_cfg.get("batch", 4))
    lr = args.lr or float(dino_cfg.get("lr", 5e-4))
    loss_weights = dino_cfg.get("loss_weights")

    if args.smoke:
        batch = min(batch, max(1, args.smoke))
        iters = min(iters, 2) if not args.iters else args.iters

    teacher = load_teacher(teacher_name, repo=repo, weights=weights,
                           device=args.device)
    patch = teacher.spec.patch_size
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
        ds = NormalImageDataset(resolve(paths["normaltrain"]), img_size=size,
                                augment=False, with_descriptor=False)
    loader = DataLoader(ds, batch_size=batch, shuffle=True,
                        num_workers=0 if args.smoke else args.num_workers,
                        collate_fn=collate, drop_last=not args.smoke,
                        pin_memory=args.device.startswith("cuda"))

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

    trainable = sum(p.numel() for p in params) / 1e6
    log("=" * 72)
    log("JingWei stage 0 - offline DINO/DINOv2 teacher distillation")
    log(f"cfg={args.cfg}  data={args.data or cfg.data.dataset_yaml}  device={args.device}")
    log(f"ckpt={args.ckpt or 'none'}  repo={repo or 'torch.hub'}  "
        f"weights={weights or 'default/pretrained'}  out={out}")
    if args.device.startswith("cuda") and torch.cuda.is_available():
        log(f"gpu={torch.cuda.get_device_name(0)}")
    log(f"model={count_parameters(model) / 1e6:.2f} M params  "
        f"trainable={trainable:.2f} M  img_size={size}  crop={crop}")
    log(f"normal images={len(ds)}  batches/loader={len(loader)}  iters={iters}  "
        f"batch={batch}  lr={lr:g}  workers={0 if args.smoke else args.num_workers}")
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
            it += 1

            if args.log_interval and (it == 1 or
                                      it % args.log_interval == 0 or
                                      it == iters):
                elapsed = max(time.time() - interval_t0, 1e-6)
                ips = interval_n / elapsed
                eta = (iters - it) / max(ips, 1e-6)
                avg = interval_loss / max(interval_n, 1)
                lr_now = float(opt.param_groups[0]["lr"])
                log(f"iter {it:6d}/{iters}  distill_cosine={avg:.5f}  "
                    f"lr={lr_now:.2e}  {ips:.2f} it/s  "
                    f"elapsed={format_eta(time.time() - global_t0)}  "
                    f"eta={format_eta(eta)}")
                interval_t0, interval_loss, interval_n = time.time(), 0.0, 0

    # Dinomaly stage B expects a FROZEN foundation projection — re-freeze it.
    model.normality.recon.target_proj.requires_grad_(False)
    model.eval()
    torch.save({
        "model": model.state_dict(),
        "dino_adaptor": adaptor.state_dict(),
        "teacher": teacher.spec.name,
        "teacher_family": teacher.spec.family,
        "crop": crop,
        "iters": it,
        "cfg": args.cfg,
    }, out / "dino_init.pt")
    log(f"saved -> {out / 'dino_init.pt'}")
    log("next: python tools/train_detector.py --cfg "
        f"{args.cfg} --resume {out / 'dino_init.pt'}")


if __name__ == "__main__":
    main()
