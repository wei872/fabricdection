"""Pre-build YOLO-style train/val image caches before training.

Examples (Windows paths with Chinese characters are supported)::

    python tools/cache_dataset.py --cfg configs/jingwei_p.yaml \
        --splits train,val,normaltrain,normalval --mode disk
    python tools/cache_dataset.py --cfg configs/jingwei_p.yaml \
        --splits train --mode ram --rebuild

Then train with the SAME mode::

    python tools/train_detector.py --cfg configs/jingwei_p.yaml \
        --cache disk --cache-dir runs/cache

The cache is a ``*.json`` metadata file plus one ``*.npy`` (disk/ram) or
``*.bin`` (bytes) data file.  It is keyed by image path/size/mtime and label
content, so adding or relabelling images automatically triggers a rebuild.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from _common import ROOT, add_cache_args, data_paths, load_config, log, resolve
from jingwei.cache import build_dataset_cache, cache_prefix
from jingwei.data import infer_labels_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/jingwei_n.yaml")
    ap.add_argument("--data", default=None, help="override dataset yaml")
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--splits", default="train,val,normaltrain,normalval")
    add_cache_args(ap)
    ap.add_argument("--mode", choices=["ram", "disk", "bytes"], default=None,
                    help="alias for --cache")
    ap.add_argument("--label-offset", choices=["0", "1", "auto"], default=None,
                    help="YOLO class-id offset stored in cache: 0=0..18, "
                         "1=1..19 labels, auto=infer; cached labels are always 0-based")
    ap.add_argument("--nc", type=int, default=None,
                    help="number of classes (default: cfg.data.nc)")
    ap.add_argument("--rebuild", action="store_true",
                    help="alias for --rebuild-cache")
    args = ap.parse_args()
    if args.mode:
        args.cache = args.mode
    if args.cache == "none":
        ap.error("choose --cache/--mode ram / disk / bytes (not none) when building")
    if args.rebuild:
        args.rebuild_cache = True

    cfg = load_config(args.cfg)
    label_offset = (args.label_offset if args.label_offset is not None
                    else cfg.data.get("label_offset", 0))
    nc = int(args.nc if args.nc is not None else cfg.data.nc)
    args.label_offset, args.nc = label_offset, nc
    paths = data_paths(cfg, args.data)
    size = int(args.img_size or cfg.input_size[0])
    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 72)
    log("JingWei dataset cache builder")
    log(f"cfg={args.cfg}  data={args.data or cfg.data.dataset_yaml}  "
        f"img_size={size}  mode={args.cache}  dir={cache_dir}  "
        f"nc={nc}  label_offset={label_offset}")
    log("=" * 72)

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        if split not in paths:
            log(f"[skip] {split}: not in dataset yaml")
            continue
        img_dir = resolve(paths[split])
        has_labels = split in ("train", "val")
        labels_dir = infer_labels_dir(img_dir) if has_labels else None
        tag = hashlib.sha1(str(img_dir).encode("utf-8", "replace")).hexdigest()[:8]
        prefix = cache_prefix(cache_dir, tag, split, size, args.cache)
        log(f"[{split}] images={img_dir}")
        try:
            meta = build_dataset_cache(
                img_dir, labels_dir, size, prefix, args.cache, has_labels,
                rebuild=bool(args.rebuild_cache), log_fn=log,
                label_offset=label_offset, nc=nc)
            log(f"[{split}] done: {meta['n']} images, "
                f"labels={'yes' if has_labels else 'no'}")
        except FileNotFoundError as e:
            log(f"[{split}] FAILED: {e}")


if __name__ == "__main__":
    main()
