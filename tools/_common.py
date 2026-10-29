"""Shared helpers for the CLI tools.

Image / video reads go through :mod:`jingwei.image_io`, so Chinese / Unicode
dataset paths work on Windows as well as Linux.  Training scripts use
:func:`log` so progress appears immediately even when stdout is buffered,
:func:`iter_speed` for stable ETA, :func:`make_dataset` for the YOLO readers,
and :func:`save_checkpoint` / :func:`resume_checkpoint` for full-state resume.
"""
from __future__ import annotations

import math
import os
import random
import sys
from pathlib import Path

# allow `python tools/xxx.py` from the repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from jingwei import build_model, load_config
from jingwei.data import (NormalImageDataset, YOLODefectDataset, collate,
                          infer_labels_dir, list_images, make_worker_init_fn)
from jingwei.fabric import fabric_descriptor, fit_fabric_prototypes
from jingwei.image_io import console_safe, imread_unicode
from jingwei.utils import torch_load


def log(msg: str = "", *, flush: bool = True) -> None:
    """Print an ASCII-safe training line and flush immediately.

    Windows consoles and redirected logs often buffer ``print`` output, which
    makes a long epoch look like a hang.  ``flush=True`` makes every line show
    up as soon as it is produced.
    """
    print(console_safe(msg), flush=flush)


def format_eta(seconds: float) -> str:
    """Human-readable duration for progress / ETA logs."""
    if not math.isfinite(seconds) or seconds < 0:
        return "n/a"
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def iter_speed(done: int, elapsed: float, interval_n: int,
               interval_elapsed: float) -> float:
    """Stable it/s for ETA: mostly epoch-average, slightly instantaneous.

    A single slow batch (checkpoint, GC, OS page fault) used to make the ETA
    jump from 1h to 700h.  Blending the running average with the short interval
    keeps the estimate useful without hiding real slowdowns.
    """
    avg = float(done) / max(float(elapsed), 1e-6)
    inst = float(interval_n) / max(float(interval_elapsed), 1e-6)
    return 0.85 * avg + 0.15 * inst


def make_dataset(split, images_dir, labels_dir, img_size, args, *,
                 has_labels: bool, augment: bool, seed: int,
                 with_descriptor: bool = False, log_fn=log):
    """Return the YOLO / normal-image dataset for a split."""
    images_dir = Path(images_dir)
    if has_labels:
        labels_dir = str(labels_dir) if labels_dir else infer_labels_dir(images_dir)
        return YOLODefectDataset(images_dir, labels_dir, img_size=img_size,
                                 augment=augment, seed=seed)
    return NormalImageDataset(images_dir, img_size=img_size,
                              augment=augment, seed=seed,
                              with_descriptor=with_descriptor)


class NegativeMixDataset(Dataset):
    """Yield defect-free frames as *pure negatives* for a fraction of samples.

    Stage A used to see defect images only, so every frame it ever saw
    contained at least one defect and every "background" cell it learned from
    was fabric that happened to sit inside a defective frame.  The model
    therefore learned "something is always here", which is exactly the
    ``boxes/img = 55`` / ``normal_fpr = 1.0`` failure of the v0.2.1 run.

    Mixing in ``normaltrain`` frames with **empty** targets fixes the prior
    directly: a whole frame of normal texture now has to be scored as
    background.  ``--normal-frac 0.25`` is the usual setting (one frame in
    four); the length stays that of the positive split, so an epoch still
    covers every defect image, just with fewer repeats.
    """

    def __init__(self, pos_ds, neg_ds, frac: float = 0.25, seed: int = 0):
        self.pos_ds = pos_ds
        self.neg_ds = neg_ds
        self.frac = min(1.0, max(0.0, float(frac)))
        self._rng = np.random.default_rng(int(seed))

    def __len__(self) -> int:
        return len(self.pos_ds)

    def __getitem__(self, i):
        if (self.neg_ds is not None and len(self.neg_ds) and
                float(self._rng.random()) < self.frac):
            d = dict(self.neg_ds[i % len(self.neg_ds)])
            # every sample needs both keys: ``jingwei.data.collate`` only looks
            # at ``batch[0]`` to decide, so a key missing from ONE sample means
            # a KeyError from the others.
            d["boxes"] = torch.zeros(0, 4)
            d["labels"] = torch.zeros(0, dtype=torch.long)
            return d
        return self.pos_ds[i]


def subset_indices(n: int, limit: int | None, seed: int = 0) -> list[int] | None:
    """Return a deterministic random subset of ``range(n)``.

    ``--val-max`` used to take the first N images of a sorted split, which on
    an augmented dataset selected mostly near-duplicate source images.  A
    fixed-seed sample is still comparable across epochs but covers the split
    uniformly.  ``None`` means "use the full dataset".
    """
    if not limit or int(limit) >= int(n):
        return None
    rng = np.random.default_rng(int(seed))
    return rng.choice(int(n), size=int(limit), replace=False).tolist()


def data_paths(cfg, override: str | None = None) -> dict:
    p = Path(override or cfg.data.dataset_yaml)
    if not p.is_absolute():
        p = ROOT / p
    with p.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def set_global_seed(seed: int) -> None:
    """Seed Python / NumPy / CPU+CUDA torch RNGs for reproducible resumes."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def save_checkpoint(state: dict, path) -> Path:
    """Atomically write a checkpoint (safe if the process is killed mid-save)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        torch.save(state, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return path


def load_checkpoint(path) -> dict:
    """Load a checkpoint relative to the repo root when given a relative path."""
    return torch_load(resolve(path), map_location="cpu")


def checkpoint_meta(state: dict) -> dict:
    return dict(state.get("train", {}) or {})


def is_resumable(state: dict, stage: str) -> bool:
    meta = checkpoint_meta(state)
    return "optimizer" in state and meta.get("stage") == stage


# --------------------------------------------------------------------------------------
# RNG state (kept weights_only=True friendly: no raw numpy arrays / custom objects)
#
# numpy's bit_generator.state is version / BitGenerator dependent.  PCG64 / PCG64DXSM
# store ``state`` as a *dict* of Python ints, Philox / SFC64 store ndarray fields,
# MT19937 stores a big uint32 ``key`` ndarray, while older numpy builds may expose a
# plain uint32 ndarray directly.  The old code assumed ndarray.tolist() and crashed on
# Windows / numpy 2.x with:
#   AttributeError: 'dict' object has no attribute 'tolist'
# We now recursively convert every ndarray / numpy scalar to pure Python containers.
# --------------------------------------------------------------------------------------
_RNG_FORMAT_KEY = "__jingwei_rng_format__"
_NP_ARRAY_KEY = "__jingwei_np_array__"


def _to_plain(obj):
    """Recursively turn numpy arrays / scalars into weights_only-safe Python data."""
    if isinstance(obj, np.ndarray):
        return {
            _NP_ARRAY_KEY: True,
            "dtype": str(obj.dtype),
            "data": obj.tolist(),
        }
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, (int, float, str, bytes, bool)) or obj is None:
        return obj
    # Unknown objects must not be pickled into a weights_only checkpoint.
    return str(obj)


def _from_plain(obj):
    """Inverse of :func:`_to_plain` for numpy BitGenerator state dicts."""
    if isinstance(obj, dict):
        if obj.get(_NP_ARRAY_KEY) is True and "dtype" in obj and "data" in obj:
            return np.asarray(obj["data"], dtype=np.dtype(obj["dtype"]))
        return {k: _from_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_plain(v) for v in obj]
    return obj


def _generator_state(rng) -> dict:
    st = _to_plain(rng.bit_generator.state)
    st[_RNG_FORMAT_KEY] = 2
    return st


def _restore_generator_state(rng, st: dict) -> None:
    if st.get(_RNG_FORMAT_KEY) == 2:
        state = _from_plain({k: v for k, v in st.items()
                             if k != _RNG_FORMAT_KEY})
    else:
        # Backward compatibility with v1 checkpoints written before the fix.
        state = dict(st)
        state["state"] = np.asarray(state["state"], dtype=np.uint32)
        state["has_uint32"] = int(state["has_uint32"])
        state["uinteger"] = int(state["uinteger"])
    rng.bit_generator.state = state


def collect_rng_state(dataset=None) -> dict:
    out = {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "cuda": (torch.cuda.get_rng_state_all()
                 if torch.cuda.is_available() else []),
    }
    rng = getattr(dataset, "rng", None)
    if rng is not None and hasattr(rng, "bit_generator"):
        out["numpy"] = _generator_state(rng)
    return out


def restore_rng_state(state: dict, dataset=None) -> None:
    if not state:
        return
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "python" in state:
        try:
            random.setstate(tuple(state["python"]))
        except (TypeError, ValueError):
            pass
    if state.get("cuda") and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state["cuda"])
        except (RuntimeError, TypeError):
            pass
    if "numpy" in state:
        rng = getattr(dataset, "rng", None)
        if rng is not None and hasattr(rng, "bit_generator"):
            try:
                _restore_generator_state(rng, state["numpy"])
            except Exception as exc:  # pragma: no cover - corrupt / foreign ckpt
                log(f"[!] could not restore numpy RNG state ({exc}); "
                    "training continues with the seeded generator")


def resume_checkpoint(state_or_path, stage: str, model, optimizer=None,
                      scheduler=None, dataset=None, extra_modules=None,
                      device="cpu") -> dict:
    """Load weights; additionally restore full training state when available.

    ``state_or_path`` may be a path or an already-loaded state dict.  A full
    checkpoint contains ``optimizer`` and ``train.stage == stage``; otherwise
    the file is treated as a weight-only warm start (e.g. ``dino_init.pt``).
    """
    state = (state_or_path if isinstance(state_or_path, dict)
             else load_checkpoint(state_or_path))
    meta = checkpoint_meta(state)
    full = bool(optimizer is not None and "optimizer" in state and
                meta.get("stage") == stage)

    sd = state.get("model", state)
    _, reshaped = split_shape_mismatch(model, sd)
    load_state_dict_report(model, sd, tag=str(stage))
    if extra_modules:
        for key, module in extra_modules.items():
            if key in state:
                load_state_dict_report(module, state[key], tag=f"{stage}/{key}")

    # A checkpoint whose parameter *shapes* changed cannot carry its optimizer
    # state either: ``AdamW.load_state_dict`` is indexed by parameter position,
    # so an added module (v0.2.2's ``obj_stem``) makes it raise
    # "loaded state dict contains a parameter group that doesn't match the size
    # of the optimizer's group" -- or, worse, silently mis-assign momentum.
    # Downgrade such a checkpoint to a weight-only warm start instead of dying
    # 4 seconds into a 150-epoch run.
    if full and (reshaped or not optimizer_state_compatible(optimizer,
                                                            state["optimizer"])):
        full = False
        if reshaped:
            log(f"[!] {len(reshaped)} tensor(s) changed shape vs. this model, so "
                "the optimizer / scheduler state is NOT restored "
                "(weight-only warm start; the epoch counter and LR schedule "
                "restart at 0).")
        else:
            log("[!] the checkpoint's optimizer state does not match this "
                "model's parameter groups -> weight-only warm start.")
    elif full:
        optimizer.load_state_dict(state["optimizer"])
        for st in optimizer.state.values():
            if isinstance(st, dict):
                for k, v in st.items():
                    if torch.is_tensor(v):
                        st[k] = v.to(device)
        if scheduler is not None and "scheduler" in state:
            scheduler.load_state_dict(state["scheduler"])
        restore_rng_state(state.get("rng", {}), dataset)

    if full:
        reason = "ok"
    elif meta.get("stage") and meta.get("stage") != stage:
        reason = f"stage mismatch: checkpoint is {meta.get('stage')!r}, expected {stage!r}"
    elif "optimizer" not in state:
        reason = "weight-only checkpoint (no optimizer / scheduler state)"
    else:
        reason = "missing optimizer"
    return {
        "full": full,
        "reason": reason,
        "meta": meta,
        "epoch": int(meta.get("epoch", 0)) if full else 0,
        "iter": int(meta.get("iter", 0)) if full else 0,
        "state": state,
    }


def split_shape_mismatch(model, sd) -> tuple[dict, list]:
    """Split ``sd`` into (loadable tensors, tensors whose *shape* changed).

    ``load_state_dict(strict=False)`` still raises ``RuntimeError`` on a shape
    mismatch -- it only tolerates missing / unexpected *keys*.  That made
    ``--resume runs/dino/dino_init.pt`` crash outright in v0.2.2, where the
    objectness heads moved from the neck (``c`` channels) to the regression
    stem (``h = c*head_width``): the backbone/nest weights are still perfectly
    reusable, only 4 tensors are not.
    """
    own = model.state_dict()
    keep, reshaped = {}, []
    for key, value in sd.items():
        target = own.get(key)
        if (target is not None and torch.is_tensor(value)
                and tuple(value.shape) != tuple(target.shape)):
            reshaped.append((key, tuple(value.shape), tuple(target.shape)))
        else:
            keep[key] = value
    return keep, reshaped


def optimizer_state_compatible(optimizer, opt_state) -> bool:
    """Do the checkpoint's param groups have the same layout as ``optimizer``?"""
    if optimizer is None or not isinstance(opt_state, dict):
        return False
    groups = opt_state.get("param_groups") or []
    if len(groups) != len(optimizer.param_groups):
        return False
    return all(len(g.get("params", ())) == len(og["params"])
               for g, og in zip(groups, optimizer.param_groups))


def load_state_dict_report(model, sd, tag: str = "checkpoint") -> list[str]:
    """``load_state_dict(strict=False)`` + a loud report of what did not match.

    Silently ignoring missing / unexpected keys is how a run that "loaded fine"
    ends up with randomly initialised layers (e.g. ``--ckpt`` from a different
    width, or a detector checkpoint used for the pico config).  Shape-mismatched
    tensors are *skipped* (they keep the model's own init) rather than raising,
    because partial warm starts are the normal case across architecture
    changes.  Returns the collected warning lines so callers can also surface
    them in a JSON report.
    """
    sd, reshaped = split_shape_mismatch(model, sd)
    result = model.load_state_dict(sd, strict=False)
    missing = list(getattr(result, "missing_keys", []) or [])
    unexpected = list(getattr(result, "unexpected_keys", []) or [])
    msgs = []
    if missing:
        msgs.append(f"[!] {tag}: {len(missing)} missing tensor(s) "
                    f"(kept at their current init), e.g. {missing[:3]}")
    if unexpected:
        msgs.append(f"[!] {tag}: {len(unexpected)} unexpected tensor(s) "
                    f"(ignored), e.g. {unexpected[:3]}")
    if reshaped:
        detail = ", ".join(f"{k} {list(a)}->{list(b)}" for k, a, b in reshaped[:3])
        msgs.append(
            f"[!] {tag}: {len(reshaped)} tensor(s) skipped because their shape "
            f"changed in this model version (they keep their own init): {detail}"
            + ("" if len(reshaped) <= 3 else f" ... (+{len(reshaped) - 3} more)"))
    for m in msgs:
        log(m)
    return msgs


def load_model(cfg_path: str, ckpt: str | None = None, device="cpu"):
    cfg = load_config(cfg_path)
    model = build_model(cfg)
    if ckpt:
        state = load_checkpoint(ckpt)
        sd = state.get("model", state)
        load_state_dict_report(model, sd, tag=str(ckpt))
    return model.to(device), cfg


class SyntheticNormalDataset(Dataset):
    """On-the-fly woven-texture images for smoke tests (no files needed)."""

    def __init__(self, n=32, size=1280, seed=0):
        self.n, self.size, self.seed = n, size, seed

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.seed + idx)
        s = 320                       # generate at low res then upscale -> fast
        yy, xx = np.mgrid[0:s, 0:s]
        px, py = rng.uniform(4, 12), rng.uniform(5, 15)
        weave = (np.sin(2 * np.pi * xx / px) * 0.5 + np.sin(2 * np.pi * yy / py) * 0.5)
        g = np.clip(128 + rng.uniform(20, 55) * weave, 0, 255)
        field = rng.uniform(0, 40) * np.sin(2 * np.pi * xx / s + rng.uniform(0, 6))
        img = np.clip(np.stack([g + field] * 3, -1), 0, 255).astype(np.uint8)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        return dict(image=torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0,
                    descriptor=torch.from_numpy(fabric_descriptor(img[:, :, 0])),
                    path=f"synthetic_{idx:05d}")


@torch.no_grad()
def descriptors_for_folder(folder, k_prototypes, limit=None):
    files = list_images(folder)
    if limit:
        files = files[:limit]
    readable, descs = [], []
    for f in files:
        img = imread_unicode(f, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        readable.append(f)
        descs.append(fabric_descriptor(img))
    if not descs:
        raise FileNotFoundError(f"no readable images in {console_safe(folder)}")
    d = np.stack(descs)
    proto, labels = fit_fabric_prototypes(d, k=k_prototypes)
    return readable, d, proto, labels


@torch.no_grad()
def extract_patch_features(model, files, proto_labels, device, cfg, batch_size=1, limit=None):
    """Return {prototype_id: (N, D) L2-normalised patch features}."""
    model.eval()
    g = int(cfg.normality.grid)
    size = int(cfg.input_size[0])
    out = {}
    n = min(len(files), limit) if limit else len(files)
    for i in range(0, n, batch_size):
        chunk = files[i:i + batch_size]
        imgs = []
        for f in chunk:
            im = imread_unicode(f, cv2.IMREAD_COLOR)
            if im is None:
                raise FileNotFoundError(f"cannot read image: {console_safe(f)}")
            im = cv2.cvtColor(cv2.resize(im, (size, size)), cv2.COLOR_BGR2RGB)
            imgs.append(torch.from_numpy(im.transpose(2, 0, 1)).float() / 255.0)
        x = torch.stack(imgs).to(device)
        feats = model.neck(model.backbone(x))
        q = model.normality.patch(feats)                  # (B, D, g, g)
        q = q.permute(0, 2, 3, 1).reshape(-1, q.shape[1]).cpu().numpy()
        for j in range(len(chunk)):
            k = int(proto_labels[i + j])
            out.setdefault(k, []).append(q[j * g * g:(j + 1) * g * g])
    return {k: np.concatenate(v, 0) for k, v in out.items()}
