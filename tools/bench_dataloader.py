"""Measure raw dataloader throughput (cache + augmentation + collate).

Training speed on a 29k-image fabric dataset is limited by whichever side is
slower: the GPU step, or feeding it.  This tool isolates the feeding side so you
can pick ``--cache`` / ``--num-workers`` / batch size with numbers instead of
guesses.

    python tools/bench_dataloader.py --cfg configs/jingwei_p.yaml \
        --data configs/fabric19.yaml --cache disk \
        --cache-dir D:/tyhn/yolo/dataset/runs/cache \
        --batch 8 --num-workers 4 --batches 20
    python tools/bench_dataloader.py --cfg configs/jingwei_p.yaml --cache none \
        --batches 10 --num-workers 0          # original-image baseline

Prints img/s, ms/batch and the implied epoch time for the whole train split.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader

from _common import (ROOT, add_cache_args, data_paths, format_eta, log,
                     make_dataset, resolve, resolve_label_offset)
from jingwei import load_config
from jingwei.data import collate, to_model_input, worker_init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/jingwei_p.yaml")
    ap.add_argument("--data", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--prefetch-factor", type=int, default=2)
    ap.add_argument("--mp-start", default="spawn",
                    choices=["spawn", "fork", "forkserver", "default"])
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--no-aug", action="store_true", help="disable augmentation")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--to-device", action="store_true",
                    help="also measure the uint8->float CHW cast + H2D copy")
    ap.add_argument("--label-offset", choices=["0", "1", "auto"], default=None)
    add_cache_args(ap)
    args = ap.parse_args()

    if args.num_workers is None:
        args.num_workers = 2 if os.name == "nt" else 4
    if args.num_workers <= 0:
        # OpenCV inside the main process may then use all cores
        cv2.setNumThreads(0)
    else:
        cv2.setNumThreads(1)

    cfg = load_config(args.cfg)
    paths = data_paths(cfg, args.data)
    args.nc = int(paths.get("nc", cfg.data.nc))
    images_dir = resolve(paths[args.split])
    if args.label_offset:
        args.label_offset = resolve_label_offset(args.label_offset, args.nc,
                                                 [images_dir])
    size = int(cfg.input_size[0])
    ds = make_dataset(args.split, images_dir, None, size, args,
                      has_labels=True, augment=not args.no_aug, seed=0)
    kwargs = {}
    if args.num_workers > 0:
        kwargs.update(prefetch_factor=max(1, args.prefetch_factor),
                      worker_init_fn=worker_init, persistent_workers=True)
        if args.mp_start != "default":
            kwargs.update(multiprocessing_context=args.mp_start)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.num_workers, collate_fn=collate,
                        pin_memory=args.device.startswith("cuda"), **kwargs)

    log("=" * 72)
    log(f"bench dataloader: split={args.split}  images={len(ds)}  "
        f"batch={args.batch}  workers={args.num_workers}  "
        f"cache={args.cache}  aug={not args.no_aug}")
    log("=" * 72)

    state = {"it": iter(loader)}

    def take():
        """next() that restarts the epoch, so tiny debug splits still measure."""
        try:
            return next(state["it"])
        except StopIteration:
            state["it"] = iter(loader)
            return next(state["it"])

    for _ in range(2):                      # warm-up: workers + page cache
        take()
    seen = 0
    t0 = time.perf_counter()
    for i in range(args.batches):
        batch = take()
        if args.to_device:
            to_model_input(batch["image"], args.device)
        seen += int(batch["image"].shape[0])
    dt = max(time.perf_counter() - t0, 1e-6)
    nb = max(1, i + 1)
    ips = seen / dt
    per_epoch = len(ds) / max(ips, 1e-6)
    log(f"result: {seen} images in {dt:.2f}s -> {ips:.1f} img/s  "
        f"{1000 * dt / nb:.1f} ms/batch of {args.batch}")
    log(f"implied: one epoch over {len(ds)} images = {format_eta(per_epoch)}  "
        f"(loader only, GPU step excluded)")
    log("tips: --cache disk removes JPEG decode; raise --num-workers until img/s "
        "stops improving; if img/s >> GPU it/s the GPU is the bottleneck")


if __name__ == "__main__":
    main()
