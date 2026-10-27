"""Shared helpers for the CLI tools.

Image / video reads go through :mod:`jingwei.image_io`, so Chinese / Unicode
dataset paths work on Windows as well as Linux.  Training scripts use
:func:`log` so progress appears immediately even when stdout is buffered, and
:func:`save_checkpoint` / :func:`resume_checkpoint` for full-state resume.
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
from jingwei.data import NormalImageDataset, YOLODefectDataset, collate, list_images
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
# --------------------------------------------------------------------------------------
def _generator_state(rng) -> dict:
    st = rng.bit_generator.state
    return {
        "bit_generator": str(st["bit_generator"]),
        "state": st["state"].tolist(),
        "has_uint32": int(st["has_uint32"]),
        "uinteger": int(st["uinteger"]),
    }


def _restore_generator_state(rng, st: dict) -> None:
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
            except (KeyError, TypeError, ValueError):
                pass


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
    model.load_state_dict(sd, strict=False)
    if extra_modules:
        for key, module in extra_modules.items():
            if key in state:
                module.load_state_dict(state[key], strict=False)

    if full:
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


def load_model(cfg_path: str, ckpt: str | None = None, device="cpu"):
    cfg = load_config(cfg_path)
    model = build_model(cfg)
    if ckpt:
        state = load_checkpoint(ckpt)
        sd = state.get("model", state)
        model.load_state_dict(sd, strict=False)
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
