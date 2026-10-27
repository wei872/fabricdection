"""Shared helpers for the CLI tools.

Image / video reads go through :mod:`jingwei.image_io`, so Chinese / Unicode
dataset paths work on Windows as well as Linux.
"""
from __future__ import annotations

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


def data_paths(cfg, override: str | None = None) -> dict:
    p = Path(override or cfg.data.dataset_yaml)
    if not p.is_absolute():
        p = ROOT / p
    with p.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def load_model(cfg_path: str, ckpt: str | None = None, device="cpu"):
    cfg = load_config(cfg_path)
    model = build_model(cfg)
    if ckpt:
        state = torch_load(ckpt, map_location="cpu")
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
