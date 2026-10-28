"""YOLO-style dataset cache for fast, Unicode-safe training.

Ultralytics YOLO can pre-generate a ``train.cache`` so each epoch does not
re-open / re-decode thousands of images and re-parse labels.  This module does
the same for JingWei, with three modes:

* ``disk``  - decoded, resized RGB uint8 images in one memory-mapped ``.npy``;
             no per-image filesystem access after the cache is built.
* ``ram``   - same file, loaded into RAM once (fastest, uses ~N*H*W*3 bytes).
* ``bytes`` - original encoded image bytes concatenated in one ``.bin`` file;
             avoids disk seeks but still decodes JPEG/PNG in memory.  Much
             smaller on disk than ``disk``/``ram``.

All paths are stored as Unicode JSON (``ensure_ascii=False``), so Chinese image
and folder names still work on Windows.  The cache is keyed by file path, size,
mtime and label-file content hash; if any of those change the cache is rebuilt.
The deployed model / ONNX graph is unaffected.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .data import (fabric_photometric, geometric_augment, infer_labels_dir,
                   list_images, read_image, to_chw_float)
from .fabric import fabric_descriptor
from .losses import resolve_label_offset

CACHE_VERSION = 3
_CACHE_WARNED = False


# --------------------------------------------------------------------------------------
# Paths / metadata
# --------------------------------------------------------------------------------------
def label_path_for(img_path, labels_dir) -> str:
    return os.path.join(str(labels_dir), Path(img_path).stem + ".txt")


def cache_prefix(cache_dir, tag, split, img_size, mode) -> Path:
    """Unique prefix for this split / size / mode / source directory."""
    key = f"{tag}|{split}|{int(img_size)}|{mode}"
    h = hashlib.sha1(key.encode("utf-8", "replace")).hexdigest()[:10]
    safe_split = str(split).replace("/", "_").replace("\\", "_")
    return Path(cache_dir) / f"{safe_split}_{int(img_size)}_{mode}_{h}"


def cache_paths(prefix: Path, mode: str):
    prefix = Path(prefix)
    ext = ".npy" if mode in ("disk", "ram") else ".bin"
    return Path(str(prefix) + ".json"), Path(str(prefix) + ext)


def _file_stat(p) -> str:
    try:
        st = os.stat(str(p))
        return f"{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        return "missing"


def _label_hash(path) -> str:
    try:
        return hashlib.sha1(Path(path).read_bytes()).hexdigest()
    except OSError:
        return "missing"


def dataset_signature(files, label_files=None) -> str:
    h = hashlib.sha1()
    for f in files:
        h.update(str(f).encode("utf-8", "replace"))
        h.update(b"|")
        h.update(_file_stat(f).encode("ascii"))
        h.update(b"\n")
    for lf in (label_files or []):
        h.update(str(lf).encode("utf-8", "replace"))
        h.update(b"|")
        h.update(_label_hash(lf).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def _read_yolo_label_rows(path):
    """Parse a .txt into raw rows (no filtering / warning / offset)."""
    out, n_bad = [], 0
    p = Path(path)
    if not p.is_file():
        return out, n_bad
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 5:
                n_bad += 1
                continue
            try:
                c, cx, cy, bw, bh = map(float, parts[:5])
            except ValueError:
                n_bad += 1
                continue
            if not all(math.isfinite(v) for v in (c, cx, cy, bw, bh)) or not float(c).is_integer():
                n_bad += 1
                continue
            ci = int(c)
            x0, y0 = cx - bw / 2, cy - bh / 2
            x1, y1 = cx + bw / 2, cy + bh / 2
            if bw <= 0 or bh <= 0 or x0 < -1e-6 or y0 < -1e-6 or x1 > 1 + 1e-6 or y1 > 1 + 1e-6:
                n_bad += 1
                continue
            out.append([ci, float(cx), float(cy), float(bw), float(bh)])
    return out, n_bad


def infer_label_offset_from_label_files(label_files, nc) -> int:
    """Resolve 'auto' ONCE for the whole dataset, not per image / per batch.

    Per-file auto detection would shift only the files that happen to contain
    class id == nc, leaving the rest of a 1-based dataset unshifted.
    """
    ids = []
    for lf in label_files:
        rows, _ = _read_yolo_label_rows(lf)
        ids.extend(int(r[0]) for r in rows if 0 <= int(r[0]) <= int(nc))
    return resolve_label_offset(torch.tensor(ids, dtype=torch.long), "auto", int(nc))


def _load_yolo_labels(path, label_offset=0, nc=None):
    """Return (rows, n_bad) with normalised YOLO coords and 0-based class ids.

    A ``*.cache`` must never contain a class id >= nc: on CUDA that becomes an
    opaque device-side assert in the detection loss.  Invalid rows are dropped
    here (with ONE warning) so the training loop sees clean labels even if the
    user has not run ``tools/validate_labels.py`` yet.
    """
    global _CACHE_WARNED
    p = Path(path)
    out, n_bad = _read_yolo_label_rows(p)

    if nc:
        # Drop absurd ids (e.g. a stray 99) BEFORE inferring a 1-based offset;
        # otherwise they make ``auto`` think the whole file is not 1-based.
        kept = [r for r in out if r[0] <= int(nc)]
        n_bad += len(out) - len(kept)
        out = kept
    if out and nc and str(label_offset) not in ("0", "none", "false", ""):
        ids = torch.tensor([r[0] for r in out], dtype=torch.long)
        off = resolve_label_offset(ids, label_offset, int(nc))
        if off:
            for r in out:
                r[0] -= off
    if nc:
        good = [r for r in out if 0 <= r[0] < int(nc)]
        n_bad += len(out) - len(good)
        out = good
    if n_bad and not _CACHE_WARNED:
        _CACHE_WARNED = True
        warnings.warn(
            f"YOLO cache is dropping malformed/out-of-range label rows "
            f"(e.g. {n_bad} in {p}). Run tools/validate_labels.py for a full "
            f"report; if labels are 1..{nc}, use --label-offset 1 or "
            f"--fix-one-based.", RuntimeWarning, stacklevel=2)
    return out, n_bad


# --------------------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------------------
def build_dataset_cache(images_dir, labels_dir, img_size, prefix, mode,
                        has_labels, rebuild=False, log_fn=print,
                        label_offset=0, nc=None):
    """Build (or validate) a cache.  Returns the metadata dict.

    ``label_offset`` / ``nc`` normalise 1-based or malformed YOLO labels once,
    at cache-build time, so the cached ``*.json`` never feeds ``cls >= nc`` to
    a CUDA indexing kernel.
    """
    if mode not in ("disk", "ram", "bytes"):
        raise ValueError(f"unknown cache mode: {mode!r}")
    img_size = int(img_size)
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    meta_path, data_path = cache_paths(prefix, mode)

    files = list_images(images_dir)
    if not files:
        raise FileNotFoundError(f"no images under {images_dir}")
    label_files = ([label_path_for(f, labels_dir) for f in files]
                   if has_labels else [])
    # 'auto' must be resolved once for the WHOLE split.  Otherwise a 1-based
    # dataset whose individual .txt files lack class id == nc would be shifted
    # only partially, which is worse than dropping those rows.
    effective_offset = label_offset
    if has_labels and nc and str(label_offset).strip().lower() == "auto":
        effective_offset = infer_label_offset_from_label_files(label_files, nc)
    sig = dataset_signature(files, label_files)

    if not rebuild and meta_path.is_file() and data_path.is_file():
        try:
            with meta_path.open("r", encoding="utf-8") as fh:
                meta = json.load(fh)
            if (meta.get("version") == CACHE_VERSION and
                    meta.get("mode") == mode and
                    meta.get("signature") == sig and
                    meta.get("img_size") == img_size and
                    meta.get("label_offset", "0") == str(effective_offset) and
                    meta.get("nc", nc) == nc and
                    int(meta.get("n", -1)) == len(files)):
                log_fn(f"cache hit  -> {meta_path.name}  "
                       f"({len(files)} images, mode={mode})")
                return meta
            log_fn(f"cache stale -> rebuilding {meta_path.name} "
                   f"(dataset changed or version mismatch)")
        except (OSError, ValueError, json.JSONDecodeError):
            log_fn(f"cache unreadable -> rebuilding {meta_path.name}")

    log_fn(f"building cache -> {data_path.name}  "
           f"({len(files)} images, {img_size}x{img_size}, mode={mode})")
    labels = []
    orig_shapes = []
    n_bad_labels = 0
    paths = [str(f) for f in files]

    if mode in ("disk", "ram"):
        arr = np.lib.format.open_memmap(
            str(data_path), mode="w+", dtype=np.uint8,
            shape=(len(files), img_size, img_size, 3))
        for i, f in enumerate(files):
            img = read_image(f)
            orig_shapes.append([int(img.shape[0]), int(img.shape[1])])
            img = cv2.resize(img, (img_size, img_size),
                             interpolation=cv2.INTER_LINEAR)
            arr[i] = img
            if has_labels:
                rows, bad = _load_yolo_labels(label_files[i], effective_offset, nc)
                labels.append(rows)
                n_bad_labels += bad
            if (i + 1) % 200 == 0 or i + 1 == len(files):
                log_fn(f"  cached {i + 1}/{len(files)}")
        arr.flush()
        del arr
        offsets = None
    else:  # bytes
        blobs = []
        for i, f in enumerate(files):
            b = Path(f).read_bytes()
            blobs.append(b)
            # decode once here only to record the original shape
            tmp = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
            orig_shapes.append([int(tmp.shape[0]), int(tmp.shape[1])]
                               if tmp is not None else [0, 0])
            if has_labels:
                rows, bad = _load_yolo_labels(label_files[i], effective_offset, nc)
                labels.append(rows)
                n_bad_labels += bad
        total = sum(len(b) for b in blobs)
        arr = np.memmap(str(data_path), dtype=np.uint8, mode="w+",
                        shape=(max(1, total),))
        offsets = [0]
        pos = 0
        for b in blobs:
            n = len(b)
            arr[pos:pos + n] = np.frombuffer(b, dtype=np.uint8)
            pos += n
            offsets.append(pos)
        arr.flush()
        del arr
        if (len(files)) % 200 == 0 or len(files) < 200:
            log_fn(f"  cached {len(files)}/{len(files)}")

    meta = {
        "version": CACHE_VERSION,
        "mode": mode,
        "has_labels": bool(has_labels),
        "img_size": img_size,
        "n": len(files),
        "paths": paths,
        "orig_shapes": orig_shapes,
        "labels": labels,
        "offsets": offsets,
        "signature": sig,
        "label_offset": str(effective_offset),
        "label_offset_requested": str(label_offset),
        "nc": int(nc) if nc is not None else None,
        "n_bad_labels": int(n_bad_labels),
        "labels_dir": str(labels_dir) if labels_dir is not None else None,
        "images_dir": str(images_dir),
    }
    tmp_meta = meta_path.with_name(meta_path.name + ".tmp")
    with tmp_meta.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp_meta, meta_path)
    size_mb = data_path.stat().st_size / 1e6
    log_fn(f"cache ready -> {data_path.name}  {size_mb:.1f} MB  "
           f"(reuse with --cache {mode})")
    if n_bad_labels:
        log_fn(f"cache warning -> dropped {n_bad_labels} invalid YOLO rows; "
               f"run tools/validate_labels.py for details")
    return meta


# --------------------------------------------------------------------------------------
# Load / datasets
# --------------------------------------------------------------------------------------
def load_cached_dataset(prefix, mode, augment, seed, with_descriptor=False):
    prefix = Path(prefix)
    meta_path, data_path = cache_paths(prefix, mode)
    with meta_path.open("r", encoding="utf-8") as fh:
        meta = json.load(fh)
    if mode in ("disk", "ram"):
        images = np.load(str(data_path),
                         mmap_mode="r" if mode == "disk" else None)
    else:
        images = np.memmap(str(data_path), dtype=np.uint8, mode="r")
    if meta.get("has_labels"):
        return CachedYOLODefectDataset(meta, images, mode, augment, seed)
    return CachedNormalImageDataset(meta, images, mode, augment, seed,
                                    with_descriptor)


class _CachedBase(Dataset):
    def __init__(self, meta, images, mode, augment, seed):
        self.meta = meta
        self.images = images
        self.mode = mode
        self.img_size = int(meta["img_size"])
        self.augment = bool(augment)
        self.paths = list(meta["paths"])
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return int(self.meta["n"])

    def _get_image(self, idx):
        if self.mode == "bytes":
            off = self.meta["offsets"]
            s, e = int(off[idx]), int(off[idx + 1])
            buf = np.asarray(self.images[s:e]).tobytes()
            bgr = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(f"cannot decode cached image: "
                                        f"{self.paths[idx]}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return cv2.resize(rgb, (self.img_size, self.img_size),
                              interpolation=cv2.INTER_LINEAR)
        # writable copy: augmentation uses in-place flips / negative strides
        return np.array(self.images[idx], copy=True)


class CachedYOLODefectDataset(_CachedBase):
    """Cached YOLO labels/images; same output dict as ``YOLODefectDataset``."""

    def __init__(self, meta, images, mode, augment, seed):
        super().__init__(meta, images, mode, augment, seed)
        self.labels = list(meta.get("labels", []))

    def __getitem__(self, idx):
        img = self._get_image(idx)
        h, w = img.shape[:2]
        boxes, labels = [], []
        for c, cx, cy, bw, bh in self.labels[idx]:
            boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                          (cx + bw / 2) * w, (cy + bh / 2) * h])
            labels.append(int(c))
        boxes = np.array(boxes, np.float32).reshape(-1, 4)
        labels = np.array(labels, np.int64)

        if self.augment:
            img = fabric_photometric(img, self.rng)
            img, boxes = geometric_augment(img, boxes, self.rng, self.img_size)
        else:
            if len(boxes):
                boxes[:, [0, 2]] *= self.img_size / w
                boxes[:, [1, 3]] *= self.img_size / h
            img = cv2.resize(img, (self.img_size, self.img_size),
                             interpolation=cv2.INTER_LINEAR)
        return dict(image=to_chw_float(img),
                    boxes=torch.from_numpy(boxes).float(),
                    labels=torch.from_numpy(labels).long(),
                    path=self.paths[idx])


class CachedNormalImageDataset(_CachedBase):
    """Cached normal-only images; same output dict as ``NormalImageDataset``."""

    def __init__(self, meta, images, mode, augment, seed,
                 with_descriptor=False, resize_for_descriptor=512):
        super().__init__(meta, images, mode, augment, seed)
        self.with_descriptor = bool(with_descriptor)
        self.resize_for_descriptor = int(resize_for_descriptor)

    def __getitem__(self, idx):
        img = self._get_image(idx)
        out = {"path": self.paths[idx]}
        if self.with_descriptor:
            gray = cv2.cvtColor(
                cv2.resize(img, (self.resize_for_descriptor,
                                 self.resize_for_descriptor)),
                cv2.COLOR_RGB2GRAY)
            out["descriptor"] = torch.from_numpy(fabric_descriptor(gray))
        if self.augment:
            img = fabric_photometric(img, self.rng)
            empty = np.zeros((0, 4), np.float32)
            img, _ = geometric_augment(img, empty, self.rng, self.img_size)
        else:
            img = cv2.resize(img, (self.img_size, self.img_size),
                             interpolation=cv2.INTER_LINEAR)
        out["image"] = to_chw_float(img)
        return out
