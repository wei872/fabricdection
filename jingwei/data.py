"""Dataset readers for the two data sources in ``dataset/dataset.yaml``.

* ``train`` / ``val``      — YOLO-format defect data (images + ``.txt`` labels)
* ``normaltrain`` / ``normalval`` — normal images ONLY (no labels), used by
  the unsupervised normality branch and for false-positive calibration.

All file reads go through :mod:`jingwei.image_io`, so Chinese / Unicode image
and folder names work on Windows as well as Linux.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .fabric import fabric_descriptor
from .image_io import console_safe, imread_unicode

IMG_EXTS = (
    "*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp",
    "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.TIF", "*.TIFF", "*.WEBP",
)


def list_images(root: str | Path):
    root = Path(root)
    files = set()
    for ext in IMG_EXTS:
        files.update(root.rglob(ext))
    return sorted(str(f) for f in files)


def infer_labels_dir(images_dir) -> str:
    """YOLO convention: .../images/... -> .../labels/... (Windows-safe).

    Also handles paths that END in ``/images`` with no trailing slash, which is
    what ``dataset.yaml`` usually contains.
    """
    p = str(images_dir).replace("\\", "/")
    if "/images/" in p:
        return p.replace("/images/", "/labels/")
    if p.endswith("/images"):
        return p[:-len("images")] + "labels"
    return p


def read_image(path: str | Path) -> np.ndarray:
    img = imread_unicode(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {console_safe(path)}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_chw_float(img: np.ndarray) -> torch.Tensor:
    """Kept for compatibility: uint8 HWC -> float32 CHW in [0, 1]."""
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float() / 255.0


def to_image_tensor(img: np.ndarray) -> torch.Tensor:
    """uint8 HWC -> uint8 HWC tensor (zero-copy when the array is contiguous).

    Datasets hand uint8 tensors to the DataLoader instead of float CHW ones:
    a 1280x1280 sample is 4.7 MB as uint8 but 18.9 MB as float32, and the
    ``/255`` normalisation is one extra pass over 4.7 M pixels *per sample*.
    The cast now happens once per batch on the GPU (see ``to_model_input``),
    which is ~6x cheaper and moves 3x less data over PCIe.
    """
    if not img.flags["C_CONTIGUOUS"] or not img.flags["WRITEABLE"]:
        # torch refuses to wrap read-only arrays (e.g. a disk-cache memmap cell)
        img = np.array(img, copy=True, order="C")
    return torch.from_numpy(img)


def to_model_input(batch, device=None, dtype=torch.float32) -> torch.Tensor:
    """Turn a dataloader item / batch into the float NCHW tensor the model wants.

    Accepts uint8 HWC/HWC-batched tensors (the fast path), numpy arrays, or
    already-float CHW tensors, so every tool can share one code path.
    """
    x = batch["image"] if isinstance(batch, dict) else batch
    if isinstance(x, np.ndarray):
        x = to_image_tensor(x)
    if x.dtype == torch.uint8:
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.ndim != 4:
            raise ValueError(f"expected HWC/HWCN image tensor, got {tuple(x.shape)}")
        if device is not None:
            # 3x less PCIe traffic than shipping float32
            x = x.to(device, non_blocking=True)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.to(dtype).div_(255.0)
        return x
    x = x.to(device=device, dtype=dtype, non_blocking=True)
    if x.ndim == 3:
        x = x.unsqueeze(0)
    return x


def image_hw(x: torch.Tensor) -> tuple[int, int]:
    """(H, W) of a batch, whether it is uint8 HWC or float CHW."""
    if x.dtype == torch.uint8 and x.ndim == 4:
        return int(x.shape[1]), int(x.shape[2])
    return int(x.shape[-2]), int(x.shape[-1])


def worker_init(worker_id: int) -> None:
    """Make DataLoader workers predictable and stop OpenCV thread oversubscription.

    Without this, every worker spawns one thread per core inside ``resize`` /
    ``GaussianBlur`` and the workers fight each other (and the GPU feed loop).
    """
    cv2.setNumThreads(1)
    np.random.seed((worker_id + 1) * 100003 % (2 ** 32))


# --------------------------------------------------------------------------------------
# Fabric-specific appearance augmentation  (fast uint8 path)
# --------------------------------------------------------------------------------------
_LUT_RANGE = np.arange(256, dtype=np.float32)


def photometric_lut(contrast: float, exposure: float) -> np.ndarray:
    """8-bit LUT for ``(x - 0.5) * contrast + 0.5 * exposure`` (old float math)."""
    table = np.clip((_LUT_RANGE - 127.5) * float(contrast) + 127.5 * float(exposure),
                    0.0, 255.0)
    return table.astype(np.uint8)


def apply_illumination(img: np.ndarray, rng: np.random.Generator,
                       amplitude: float = 0.18) -> np.ndarray:
    """Smooth 反光 field as a fixed-point uint8 multiply.

    The previous implementation built a 1280x1280 *float32* field and blurred it
    with ``sigmaX = max(4, width / 12)`` (~106 px).  On OpenCV that is a
    ~0.5 s kernel per image -- with ``num_workers=0`` it alone capped training at
    "<1 it/s".  A smooth field with that sigma is indistinguishable from a very
    low-resolution field upsampled with cubic interpolation, which costs <1 ms.
    """
    h, w = img.shape[:2]
    # One control point per ~110 px reproduces the old Gaussian sigma = w / 12
    # while keeping the float math on a ~12x12 grid instead of 1280x1280.
    gh = max(2, int(round(h / 110.0)) + int(rng.integers(-1, 2)))
    gw = max(2, int(round(w / 110.0)) + int(rng.integers(-1, 2)))
    field = rng.normal(0.0, 1.0, (gh, gw)).astype(np.float32)
    field -= float(field.mean())
    std = float(field.std())
    if std > 1e-6:
        field /= std
    gain8 = np.clip((1.0 + amplitude * field) * 64.0 + 0.5, 1.0, 255.0).astype(np.uint8)
    gain8 = cv2.resize(gain8, (w, h), interpolation=cv2.INTER_LINEAR)   # uint8, ~0.5 ms
    gain3 = cv2.cvtColor(gain8, cv2.COLOR_GRAY2BGR)
    return cv2.multiply(img, gain3, scale=1.0 / 64.0)


def _fold_band(img: np.ndarray, rng: np.random.Generator, min_w: int = 4,
               max_w: int = 8) -> np.ndarray:
    """布幅/卷边: brighten-toward-edge band with a dark crease, on slices only."""
    h, w = img.shape[:2]
    lo = min(max(min_w, w // 50), max(1, w // 2))
    hi = min(max(max_w, w // 18), max(lo + 1, w // 2))
    band = int(rng.integers(lo, max(lo + 1, hi)))
    ramp = np.linspace(0.30, 1.0, band, dtype=np.float32)
    if rng.integers(0, 2) == 0:
        seg = img[:, :band]
        img[:, :band] = (seg.astype(np.float32) * ramp[None, :, None]).astype(np.uint8)
        c0 = max(0, band - 3)
        img[:, c0:c0 + 2] = (img[:, c0:c0 + 2].astype(np.float32) * 0.65).astype(np.uint8)
    else:
        seg = img[:, w - band:]
        img[:, w - band:] = (seg.astype(np.float32) *
                             ramp[::-1][None, :, None]).astype(np.uint8)
        c1 = min(w, w - band + 3)
        img[:, max(0, c1 - 2):c1] = (img[:, max(0, c1 - 2):c1].astype(np.float32)
                                     * 0.65).astype(np.uint8)
    return img


def fabric_photometric(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simulate the three fabric-type nuisance factors WITHOUT moving boxes.

    反光 (reflectance): smooth low-frequency illumination field + specular patch
    经纬粗细 (yarn width): slight blur / unsharp masking
    经纬密 (density): handled by anisotropic resize in the geometric aug

    Everything runs on uint8 with LUT / ``cv2.multiply`` instead of float32
    arrays: ~5 ms per 1280x1280 image instead of ~250-600 ms, with the same
    augmentation distribution.
    """
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    elif img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    if not img.flags["C_CONTIGUOUS"]:
        img = np.ascontiguousarray(img)

    # global exposure / contrast in ONE 8-bit lookup table
    out = cv2.LUT(img, photometric_lut(rng.uniform(0.75, 1.35),
                                       rng.uniform(0.85, 1.15)))

    # smooth illumination field (反光)
    if rng.random() < 0.5:
        out = apply_illumination(out, rng)

    # yarn width: mild blur or sharpen (3x3 kernels: a 3x3 box filter is
    # visually the same as a Gaussian with sigma 0.4-1.1 and ~2x cheaper)
    if rng.random() < 0.25:
        out = cv2.blur(out, (3, 3))
    elif rng.random() < 0.25:
        out = cv2.addWeighted(out, 1.6, cv2.blur(out, (3, 3)), -0.6, 0.0)

    # 布幅变化导致的边缘正常折入 / 卷边：只改亮度，不移动任何标注框。
    # 让 normaltrain 与 detector 的 hard-negative 都见到这种“像缺陷但不是缺陷”的模式。
    if rng.random() < 0.25:
        _fold_band(out, rng)             # in place, on a fresh array
    return out


def geometric_augment(img: np.ndarray, boxes: np.ndarray, rng, img_size: int,
                      anisotropic: bool = True):
    """Flips, rot90 and anisotropic scaling with consistent box transforms.

    All spatial jitter happens BEFORE the single resize to ``img_size`` and the
    flips/rotations happen AFTER it, on the (small, fixed size) output: the old
    order copied / rotated the full-resolution fabric frame several times.
    """
    h, w = img.shape[:2]
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)

    # aniso-resize changes thread density along warp/weft, boxes scale with it
    if anisotropic and rng.random() < 0.3:
        sx, sy = float(rng.uniform(0.85, 1.18)), float(rng.uniform(0.85, 1.18))
        nw, nh = max(8, int(round(w * sx))), max(8, int(round(h * sy)))
        shrink = nw * nh < w * h
        img = cv2.resize(img, (nw, nh),
                         interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)
        if len(boxes):
            boxes[:, [0, 2]] *= nw / float(w)
            boxes[:, [1, 3]] *= nh / float(h)
        h, w = nh, nw

    # final resize to the fixed 1280x1280 network input
    if (h, w) != (img_size, img_size):
        img = cv2.resize(img, (img_size, img_size),
                         interpolation=cv2.INTER_AREA if img_size < w else cv2.INTER_LINEAR)
        fx, fy = img_size / float(w), img_size / float(h)
        if len(boxes):
            boxes[:, [0, 2]] *= fx
            boxes[:, [1, 3]] *= fy

    # spatial flips / rotations on the square network input
    if rng.random() < 0.5:
        img = cv2.flip(img, 1)
        if len(boxes):
            boxes[:, [0, 2]] = img_size - boxes[:, [2, 0]]
    if rng.random() < 0.5:
        img = cv2.flip(img, 0)
        if len(boxes):
            boxes[:, [1, 3]] = img_size - boxes[:, [3, 1]]
    if rng.random() < 0.2:
        k = int(rng.integers(1, 4))
        for _ in range(k):
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            if len(boxes):
                old = boxes.copy()
                boxes[:, 0] = old[:, 1]                  # x0' = y0
                boxes[:, 2] = old[:, 3]                  # x1' = y1
                boxes[:, 1] = img_size - old[:, 2]       # y0' = S - x1
                boxes[:, 3] = img_size - old[:, 0]       # y1' = S - x0

    if len(boxes):
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
            labels_dir = infer_labels_dir(images_dir)
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

        return dict(image=to_image_tensor(img),
                    boxes=torch.from_numpy(np.ascontiguousarray(boxes)).float(),
                    labels=torch.from_numpy(np.ascontiguousarray(labels)).long(),
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
        out["image"] = to_image_tensor(img)
        return out


def collate(batch):
    """Stack uint8 HWC samples into one ``(N, H, W, 3)`` uint8 tensor.

    The uint8 -> float CHW cast happens later on the GPU (``to_model_input``),
    which is ~6x cheaper than doing it per sample on the CPU.
    """
    out = {"image": torch.stack([b["image"] for b in batch], 0)}
    for key in ("boxes", "labels", "path", "descriptor"):
        if key in batch[0]:
            out[key] = ([b[key] for b in batch] if key in ("boxes", "labels", "path")
                        else torch.stack([b[key] for b in batch], 0))
    return out
