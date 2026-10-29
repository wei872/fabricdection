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
from torch.utils.data import Dataset

from jingwei import build_model, load_config
from jingwei.data import NormalImageDataset, YOLODefectDataset, infer_labels_dir
from jingwei.image_io import console_safe
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


def make_dataset(images_dir, img_size, *, has_labels: bool, augment: bool,
                 seed: int, labels_dir=None):
    """Return the YOLO / normal-image dataset for a split."""
    images_dir = Path(images_dir)
    if has_labels:
        labels_dir = str(labels_dir) if labels_dir else infer_labels_dir(images_dir)
        return YOLODefectDataset(images_dir, labels_dir, img_size=img_size,
                                 augment=augment, seed=seed)
    return NormalImageDataset(images_dir, img_size=img_size,
                              augment=augment, seed=seed)


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
        "numpy_global": _to_plain(np.random.get_state()),
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
    if "numpy_global" in state:
        np.random.set_state(tuple(_from_plain(state["numpy_global"])))
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


def restore_detector_weights(model, weights, allow_detail_upgrade=False):
    """Strict restore except for explicit, complete new-detail additions.

    Never permit a partly missing branch or missing old encoder/head tensors.
    Evaluation and full resume remain strict: random detail weights must not
    silently enter an inference artifact or reuse an incompatible optimizer.
    """
    current = model.state_dict()
    missing = set(current) - set(weights)
    unexpected = set(weights) - set(current)
    allowed = set()
    upgraded = []
    for prefix in ("backbone.local_detail.", "novelty.texture."):
        branch = {k for k in current if k.startswith(prefix)}
        if branch and branch <= missing:
            allowed.update(branch)
            upgraded.append(prefix)
    if allow_detail_upgrade and missing and missing == allowed and not unexpected:
        # Check shapes before loading anything, including old tensors.
        if any(current[k].shape != weights[k].shape for k in weights):
            raise RuntimeError("Checkpoint tensor shapes differ; detail-only migration refused")
        model.load_state_dict(weights, strict=False)
        if getattr(model, "novelty", None) is not None:
            model.novelty.calibrated.fill_(False)
        log("--pretrained: initialized NEW detail branches: " + ", ".join(upgraded) +
            "; old encoder/adapters/head restored; unknown must be recalibrated")
    else:
        try:
            model.load_state_dict(weights, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(str(exc) + "\nUse --pretrained with a NEW --out for detail-branch migration. "
                               "For legacy inference use a config with model.local_detail=false and "
                               "unknown.texture_detail=false; never silently initialize inference weights.") from exc


def resume_checkpoint(state_or_path, model, optimizer=None, scheduler=None,
                      dataset=None, device="cpu", scaler=None) -> dict:
    """Strict detector restore. Passing an optimizer requires full train state.

    Weight-only initialization is explicit: omit the optimizer. Never silently
    convert an incomplete --resume into a new training schedule.
    """
    state = (state_or_path if isinstance(state_or_path, dict)
             else load_checkpoint(state_or_path))
    meta = checkpoint_meta(state)
    full = optimizer is not None
    if full:
        required = ["model", "optimizer", "rng", "train"]
        if scheduler is not None:
            required.append("scheduler")
        missing = [key for key in required if key not in state]
        if missing or not all(key in meta for key in ("epoch", "iter")):
            raise ValueError("Incomplete training checkpoint for --resume; "
                             "use --pretrained for weight-only initialization. "
                             f"Missing state: {missing or ['epoch/iter']}")
        if meta.get("backbone") != "dinov2":
            raise ValueError("--resume requires a direct DINOv2 detector checkpoint")
        if hasattr(model, "architecture_compatible") and not model.architecture_compatible(meta.get("architecture")):
            raise ValueError("--resume geometry differs from checkpoint; use --pretrained with a new --out")

    restore_detector_weights(model, state.get("model", state), allow_detail_upgrade=not full)
    if not full and getattr(model, "novelty", None) is not None:
        saved_arch = state.get("architecture") or meta.get("architecture")
        if saved_arch != model.architecture_signature:
            model.novelty.reset()
            log("--pretrained: geometry/architecture changed or was unrecorded; reset normal prototypes "
                "for normaltrain and independent recalibration")
    if full:
        optimizer.load_state_dict(state["optimizer"])
        for st in optimizer.state.values():
            if isinstance(st, dict):
                for k, v in st.items():
                    if torch.is_tensor(v):
                        st[k] = v.to(device)
        if scheduler is not None:
            scheduler.load_state_dict(state["scheduler"])
        if scaler is not None and scaler.is_enabled():
            if state.get("scaler"):
                scaler.load_state_dict(state["scaler"])
            else:
                log("[!] checkpoint has no FP16 scaler state; using a fresh loss scale")
        restore_rng_state(state["rng"], dataset)

    return {
        "full": full,
        "reason": "ok" if full else "explicit weight-only initialization",
        "meta": meta,
        "epoch": int(meta["epoch"]) if full else 0,
        "iter": int(meta["iter"]) if full else 0,
        "state": state,
    }


def load_model(cfg_path: str, ckpt: str | None = None, device="cpu",
               initialize_pretrained: bool = True):
    cfg = load_config(cfg_path)
    # Complete detector checkpoints include the encoder: construct architecture
    # without downloading foundation weights, then restore all tensors strictly.
    original = cfg.model.get("dino_pretrained", True)
    if ckpt or not initialize_pretrained:
        cfg.model.dino_pretrained = False
    try:
        model = build_model(cfg)
    finally:
        cfg.model.dino_pretrained = original
    if ckpt:
        state = load_checkpoint(ckpt)
        saved_arch = state.get("architecture") or (state.get("train") or {}).get("architecture")
        if model.backbone.pyramid_ratio > 1 and (not saved_arch or "detector_input" not in saved_arch):
            raise ValueError("Capped high-resolution inference requires recorded checkpoint geometry. "
                             "First train this profile with --pretrained and a new --out; "
                             "do not silently evaluate legacy weights with different decoder strides.")
        if saved_arch and "detector_input" in saved_arch and saved_arch != model.architecture_signature:
            raise ValueError("Checkpoint geometry does not match --cfg (input/pyramid/strides/detail). "
                             "Use the config that trained this checkpoint; for a new geometry use "
                             "train_detector.py --pretrained with a new --out.")
        restore_detector_weights(model, state.get("model", state))
        model.loaded_checkpoint_epoch = ((state.get("train") or {}).get("epoch") or
                                         (state.get("calibration") or {}).get("source_epoch"))
    return model.to(device), cfg


class SyntheticFabricDataset(Dataset):
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
                    path=f"synthetic_{idx:05d}")


def configure_inference(model, known_only=False, half=False, fuse=True):
    """Fail loudly rather than benchmark/predict an uncalibrated unknown path."""
    model.eval()
    model.known_only = known_only
    novelty = getattr(model, "novelty", None)
    if novelty is not None and not known_only:
        if not bool(novelty.calibrated) or int(novelty.updates) == 0:
            raise ValueError("Unknown detection is not calibrated. Run tools/calibrate_anomaly.py "
                             "or explicitly use --known-only (no unknown detection).")
        if novelty.texture is not None and int(novelty.texture.updates) == 0:
            raise ValueError("Native texture prototypes are untrained; warm-start training on normaltrain "
                             "and recalibrate before enabling unknown detection")
        if novelty.calibration_shape.tolist() != [model.backbone.encoder_size, int(model.img_size[0])]:
            raise ValueError("Input sizes differ from novelty calibration; recalibrate for this configuration")
        if int(novelty.precision_bits) != (16 if half else 32):
            raise ValueError("Inference precision differs from novelty calibration; recalibrate with matching --half")
    if half and next(model.parameters()).device.type != "cuda":
        raise ValueError("--half requires CUDA")
    if fuse and hasattr(model, "fuse_for_inference"):
        model.fuse_for_inference()
    if half:
        model.half()
    return model
