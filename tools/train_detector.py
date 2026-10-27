"""Stage A — supervised detector training on the YOLO-format defect data.

    # recommended: warm-start from stage-0 DINOv2 distillation
    python tools/train_detector.py --cfg configs/jingwei_n.yaml \
        --resume runs/dino/dino_init.pt --epochs 150
    # or from scratch / smoke
    python tools/train_detector.py --cfg configs/jingwei_n.yaml --epochs 150
    python tools/train_detector.py --cfg configs/jingwei_n.yaml --smoke 4 --epochs 1

The one-to-many head, the one-to-one (deployed) head and the foreground
alignment auxiliary branch are trained together. The normality branch is
skipped here (``set_stage('detector')``) and trained in stage B.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from _common import ROOT, data_paths, format_eta, load_model, log, resolve
from jingwei.data import YOLODefectDataset, collate
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
    ap.add_argument("--smoke", type=int, default=0, help="synthetic images for a dry run")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--log-interval", type=int, default=20,
                    help="print a progress line every N batches (0 disables iter logs)")
    args = ap.parse_args()

    model, cfg = load_model(args.cfg, args.resume, args.device)
    paths = data_paths(cfg, args.data)
    epochs = args.epochs or int(cfg.train.detector_epochs)
    batch = args.batch or int(cfg.train.detector_batch)
    lr = args.lr or float(cfg.train.detector_lr)
    size = int(cfg.input_size[0])

    if args.smoke:
        from _common import SyntheticNormalDataset
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
        train_dir = resolve(paths["train"])
        lab = str(train_dir).replace("/images/", "/labels/")
        train_ds = YOLODefectDataset(train_dir, lab, img_size=size, augment=True)

    loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                        num_workers=0 if args.smoke else args.num_workers,
                        collate_fn=collate, drop_last=False, pin_memory=True)

    model.set_stage("detector")
    model.train()
    params = [p for p in model.parameter_groups("detector") if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=float(cfg.train.weight_decay))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs * len(loader)))

    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    n_batches = len(loader)
    trainable = sum(p.numel() for p in params) / 1e6
    log("=" * 72)
    log("JingWei stage A - supervised detector training")
    log(f"cfg={args.cfg}  data={args.data or cfg.data.dataset_yaml}  device={args.device}")
    log(f"resume={args.resume or 'none'}  out={out}")
    if args.device.startswith("cuda") and torch.cuda.is_available():
        log(f"gpu={torch.cuda.get_device_name(0)}")
    log(f"model={count_parameters(model) / 1e6:.2f} M params  "
        f"trainable={trainable:.2f} M  img_size={size}")
    log(f"images={len(train_ds)}  batches/epoch={n_batches}  epochs={epochs}  "
        f"batch={batch}  lr={lr:g}  workers={0 if args.smoke else args.num_workers}")
    log(f"log_interval={args.log_interval}  "
        "(use --log-interval 0 to disable iter logs; python -u disables buffering)")
    log("=" * 72)

    it = 0
    global_t0 = time.time()
    for ep in range(epochs):
        t0 = time.time()
        interval_t0 = t0
        running = {}
        interval_loss, interval_n = 0.0, 0
        for bi, batch_data in enumerate(loader):
            images = batch_data["image"].to(args.device, non_blocking=True)
            targets = [dict(boxes=b.to(args.device), labels=l.to(args.device))
                       for b, l in zip(batch_data["boxes"], batch_data["labels"])]
            opt.zero_grad(set_to_none=True)
            result = model(images, targets=targets, it=it)
            loss = result["losses"]["total"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at iter {it}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            opt.step()
            sched.step()
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
                ips = interval_n / elapsed
                remain = (n_batches - (bi + 1)) + (epochs - ep - 1) * n_batches
                eta = remain / max(ips, 1e-6)
                avg = interval_loss / max(interval_n, 1)
                lr_now = float(opt.param_groups[0]["lr"])
                log(f"epoch {ep + 1:3d}/{epochs}  iter {bi + 1:5d}/{n_batches}  "
                    f"loss={avg:.4f}  lr={lr_now:.2e}  {ips:.2f} it/s  "
                    f"elapsed={format_eta(time.time() - t0)}  eta={format_eta(eta)}")
                interval_t0, interval_loss, interval_n = time.time(), 0.0, 0

        n = max(1, len(loader))
        msg = " | ".join(f"{k}={v / n:.3f}" for k, v in running.items() if k != "total")
        log(f"epoch {ep + 1:3d}/{epochs}  {msg}  total={running.get('total', 0) / n:.3f} "
            f"({time.time() - t0:.1f}s)  elapsed_total={format_eta(time.time() - global_t0)}")
        torch.save({"model": model.state_dict(), "epoch": ep, "cfg": args.cfg},
                   out / "last.pt")
        log(f"checkpoint -> {out / 'last.pt'}")
    log(f"saved -> {out / 'last.pt'}")


if __name__ == "__main__":
    main()
