"""Dataset readers for the two data sources in ``dataset/dataset.yaml``.

* ``train`` / ``val``      — YOLO-format defect data (images + ``.txt`` labels)
* ``normaltrain`` / ``normalval`` — normal images ONLY (no labels), used by
  the unsupervised normality branch and for false-positive calibration.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .fabric import fabric_descriptor

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")


def list_images(root: str | Path):
    root = Path(root)
    files = []
    for ext in IMG_EXTS:
        files.extend(root.rglob(ext))
    return sorted(str(f) for f in files)


def read_image(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_chw_float(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float() / 255.0


# --------------------------------------------------------------------------------------
# Fabric-specific appearance augmentation
# --------------------------------------------------------------------------------------
def fabric_photometric(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simulate the three fabric-type nuisance factors WITHOUT moving boxes.

    反光 (reflectance): smooth low-frequency illumination field + specular patch
    经纬粗细 (yarn width): slight blur / unsharp masking
    经纬密 (density): handled by anisotropic resize in the geometric aug
    """
    x = img.astype(np.float32) / 255.0
    h, w = x.shape[:2]

    # global exposure / contrast
    x = np.clip((x - 0.5) * rng.uniform(0.75, 1.35) + 0.5 * rng.uniform(0.85, 1.15), 0, 1)

    # smooth illumination field (反光)
    if rng.random() < 0.5:
        gh, gw = max(2, h // 16), max(2, w // 16)
        field = rng.normal(0, 1, (gh, gw, 1)).astype(np.float32)
        field = cv2.resize(field, (w, h), interpolation=cv2.INTER_CUBIC)
        field = cv2.GaussianBlur(field, (0, 0), sigmaX=max(4, w / 12))
        x = np.clip(x * (1.0 + 0.18 * field), 0, 1)

    # yarn width: mild blur or sharpen
    if rng.random() < 0.25:
        x = cv2.GaussianBlur(x, (0, 0), sigmaX=rng.uniform(0.4, 1.1))
    elif rng.random() < 0.25:
        x = np.clip(x + 0.6 * (x - cv2.GaussianBlur(x, (0, 0), 1.2)), 0, 1)

    # 布幅变化导致的边缘正常折入 / 卷边：只改亮度，不移动任何标注框。
    # 让 normaltrain 与 detector 的 hard-negative 都见到这种“像缺陷但不是缺陷”的模式。
    if rng.random() < 0.25:
        side = int(rng.integers(0, 2))
        band = int(rng.integers(max(4, w // 50), max(8, w // 18)))
        ramp = np.linspace(0.30, 1.0, band, dtype=np.float32)[None, :, None]
        if side == 0:
            x[:, :band, :] *= ramp
            x[:, max(0, band - 2):band, :] *= 0.65          # fold crease
        else:
            x[:, -band:, :] *= ramp[:, ::-1, :]
            x[:, -band:min(w, -band + 2), :] *= 0.65
    x = np.clip(x, 0, 1)

    return (x * 255.0).astype(np.uint8)


def geometric_augment(img: np.ndarray, boxes: np.ndarray, rng,
                      img_size: int, anisotropic: bool = True):
    """Flips, rot90 and anisotropic scaling with consistent box transforms."""
    h, w = img.shape[:2]
    if rng.random() < 0.5:
        img = img[:, ::-1]
        if len(boxes):
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
    if rng.random() < 0.5:
        img = img[::-1, :]
        if len(boxes):
            boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
    if rng.random() < 0.2:
        k = int(rng.integers(1, 4))
        img = np.rot90(img, k)
        for _ in range(k):
            if len(boxes):
                old = boxes.copy()
                nb = np.empty_like(old)
                nb[:, 0] = old[:, 1]
                nb[:, 2] = old[:, 3]
                nb[:, 1] = w - old[:, 2]
                nb[:, 3] = w - old[:, 0]
                boxes = nb
            h, w = w, h                     # rot90 swaps width/height
        h, w = img.shape[:2]

    # aniso-resize changes thread density along warp/weft, boxes scale with it
    if anisotropic and rng.random() < 0.3:
        sx, sy = rng.uniform(0.85, 1.18), rng.uniform(0.85, 1.18)
        nw, nh = int(w * sx), int(h * sy)
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        if len(boxes):
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy
        h, w = img.shape[:2]

    # final resize to the fixed 1280x1280 network input
    fx, fy = img_size / w, img_size / h
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    if len(boxes):
        boxes[:, [0, 2]] *= fx
        boxes[:, [1, 3]] *= fy
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, img_size)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, img_size)
    return img, boxes


# --------------------------------------------------------------------------------------
# YOLO-format defect dataset
# --------------------------------------------------------------------------------------
class YOLODefectDataset(Dataset):
    def __init__(self, images_dir, labels_dir=None, img_size=1280, augment=True, seed=0):
        self.img_size, self.augment = img_size, augment
        self.files = list_images(images_dir)
        if labels_dir is None:
            p = str(images_dir).replace("\\", "/")
            labels_dir = p.replace("/images/", "/labels/")
        self.labels_dir = str(labels_dir)
        self.rng = np.random.default_rng(seed)
        if not self.files:
            raise FileNotFoundError(f"no images under {images_dir}")

    def __len__(self):
        return len(self.files)

    def _label_path(self, img_path):
        stem = Path(img_path).stem
        return os.path.join(self.labels_dir, stem + ".txt")

    def _load_label(self, path, w, h):
        boxes, labels = [], []
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    c, cx, cy, bw, bh = map(float, parts[:5])
                    x0, y0 = (cx - bw / 2) * w, (cy - bh / 2) * h
                    x1, y1 = (cx + bw / 2) * w, (cy + bh / 2) * h
                    if x1 > x0 and y1 > y0:
                        boxes.append([x0, y0, x1, y1])
                        labels.append(int(c))
        return (np.array(boxes, np.float32).reshape(-1, 4),
                np.array(labels, np.int64))

    def __getitem__(self, idx):
        path = self.files[idx]
        img = read_image(path)
        h, w = img.shape[:2]
        boxes, labels = self._load_label(self._label_path(path), w, h)

        if self.augment:
            img = fabric_photometric(img, self.rng)
            img, boxes = geometric_augment(img, boxes, self.rng, self.img_size)
        else:
            if len(boxes):
                boxes[:, [0, 2]] *= self.img_size / w
                boxes[:, [1, 3]] *= self.img_size / h
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        return dict(image=to_chw_float(img),
                    boxes=torch.from_numpy(boxes).float(),
                    labels=torch.from_numpy(labels).long(),
                    path=path)


# --------------------------------------------------------------------------------------
# Normal-only image dataset
# --------------------------------------------------------------------------------------
class NormalImageDataset(Dataset):
    def __init__(self, root, img_size=1280, augment=True, seed=0,
                 with_descriptor=False, resize_for_descriptor=512):
        self.img_size, self.augment = img_size, augment
        self.with_descriptor = with_descriptor
        self.resize_for_descriptor = resize_for_descriptor
        self.files = list_images(root)
        self.rng = np.random.default_rng(seed)
        if not self.files:
            raise FileNotFoundError(f"no normal images under {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = read_image(path)
        out = {"path": path}
        if self.with_descriptor:
            gray = cv2.cvtColor(cv2.resize(img, (self.resize_for_descriptor,
                                                 self.resize_for_descriptor)),
                                cv2.COLOR_RGB2GRAY)
            out["descriptor"] = torch.from_numpy(fabric_descriptor(gray))

        if self.augment:
            img = fabric_photometric(img, self.rng)
            empty = np.zeros((0, 4), np.float32)
            img, _ = geometric_augment(img, empty, self.rng, self.img_size)
        else:
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        out["image"] = to_chw_float(img)
        return out


def collate(batch):
    out = {"image": torch.stack([b["image"] for b in batch], 0)}
    for key in ("boxes", "labels", "path", "descriptor"):
        if key in batch[0]:
            out[key] = ([b[key] for b in batch] if key in ("boxes", "labels", "path")
                        else torch.stack([b[key] for b in batch], 0))
    return out
