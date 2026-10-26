"""Fabric-type (布种) modelling.

A fabric type changes three things: yarn thickness (经纬线粗细), thread density
(经纬密) and specular reflectance (反光). Those are exactly the nuisances that
break a *single* normality model. We therefore

1. describe every image by a small, photometrically-normalised descriptor
   (2-D FFT period along warp & weft + texture statistics),
2. cluster the normal training images into K 布种原型 (fabric prototypes),
3. build **one compressed PatchCore memory bank per prototype** (see
   ``normality.py``), and
4. select the prototype at inference in O(1) from the same descriptor.

All clustering / subsampling runs offline, on CPU, once.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------------------
# 1. Per-image descriptor
# --------------------------------------------------------------------------------------
def _spectrum_periods(hp: np.ndarray, min_lag: int = 3, max_lag: int = 96):
    """Dominant warp/weft periods from the 2-D power spectrum.

    ``hp`` is the high-pass (illumination-free) image. Summing |FFT(hp)|^2 over
    the orthogonal frequency axis yields a 1-D energy profile per direction;
    the strongest non-DC peak gives the thread period.
    """
    h, w = hp.shape
    spec = np.abs(np.fft.fft2(hp)) ** 2
    spec = np.fft.fftshift(spec)
    px = spec.sum(axis=0)          # (w,) energy per horizontal frequency
    py = spec.sum(axis=1)          # (h,) energy per vertical frequency

    def pick(profile, n):
        f = np.abs(np.fft.fftshift(np.fft.fftfreq(n)))
        valid = (f > 1.0 / max_lag) & (f < 1.0 / min_lag) & (np.arange(n) != n // 2)
        if not valid.any():
            return float(max_lag), 0.0
        idx = np.argmax(np.where(valid, profile, -1.0))
        period = 1.0 / (f[idx] + 1e-9)
        peak = float(profile[idx] / (profile.sum() + 1e-9))
        return float(period), peak

    return (*pick(px, w), *pick(py, h))   # (period_x, peak_x, period_y, peak_y)


def cv2_box_blur_2d(x: np.ndarray, k: int) -> np.ndarray:
    """Integral-image box filter, numpy only (no cv2 dependency here)."""
    p = np.pad(x, ((k // 2, k // 2), (k // 2, k // 2)), mode="edge")
    ii = np.pad(p.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    h, w = x.shape
    return (ii[k:k + h, k:k + w] - ii[:h, k:k + w] -
            ii[k:k + h, :w] + ii[:h, :w]) / float(k * k)


def fabric_descriptor(gray: np.ndarray, resize: int = 512) -> np.ndarray:
    """Compute an 8-D fabric descriptor from a uint8/float grayscale image.

    dims: [period_warp, period_weft, peak_warp, peak_weft, ratio, grad_mean,
           lap_var, local_contrast]
    """
    import cv2

    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    h, w = gray.shape[:2]
    scale = resize / max(h, w)
    if scale < 1.0:
        gray = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))),
                          interpolation=cv2.INTER_AREA)
    # High-pass FIRST, then z-score on the high-pass signal. Doing the z-score
    # on the raw image would divide the weave by the (large) reflectance
    # variance and make the descriptor depend on 反光 amplitude.
    g = gray.astype(np.float32) / 255.0
    hp = g - cv2_box_blur_2d(g, 25)
    hp = (hp - hp.mean()) / (hp.std() + 1e-6)
    # ALL descriptor dims are computed on this illumination-free, normalised
    # weave structure.
    period_x, peak_x, period_y, peak_y = _spectrum_periods(hp)  # 经密 / 纬密

    gx = np.abs(np.diff(hp, axis=1)).mean()
    gy = np.abs(np.diff(hp, axis=0)).mean()
    lap = float((np.diff(hp, n=2, axis=0) ** 2).mean() + (np.diff(hp, n=2, axis=1) ** 2).mean())
    local = float(np.abs(hp - _box_blur(hp, 15)).mean())

    desc = np.array([period_x, period_y, peak_x, peak_y,
                     period_x / (period_y + 1e-6), (gx + gy) / 2.0,
                     lap, local], dtype=np.float32)
    desc[:2] /= scale          # express periods in original-image pixels
    return np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0)


def _box_blur(x: np.ndarray, k: int) -> np.ndarray:
    import cv2
    return cv2.blur(x, (k, k))


# --------------------------------------------------------------------------------------
# 2. Prototype clustering
# --------------------------------------------------------------------------------------
@dataclass
class FabricPrototypes:
    centers: np.ndarray         # (K, D) in normalised descriptor space
    mean: np.ndarray            # (D,)
    std: np.ndarray             # (D,)

    def __post_init__(self):
        self.centers = np.asarray(self.centers, np.float32)
        self.mean = np.asarray(self.mean, np.float32)
        self.std = np.asarray(self.std, np.float32)

    def normalise(self, desc: np.ndarray) -> np.ndarray:
        return (np.asarray(desc, np.float32) - self.mean) / self.std

    def assign(self, desc: np.ndarray) -> int:
        z = self.normalise(desc)
        return int(np.argmin(((self.centers - z) ** 2).sum(-1)))

    def assign_many(self, descs: np.ndarray) -> np.ndarray:
        z = (np.asarray(descs, np.float32) - self.mean) / self.std
        d = ((z[:, None, :] - self.centers[None, :, :]) ** 2).sum(-1)
        return d.argmin(-1).astype(np.int64)


def kmeans_np(x: np.ndarray, k: int, iters: int = 50, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic k-means (numpy only). Returns (centers, labels)."""
    x = np.asarray(x, np.float32)
    n = len(x)
    k = min(k, n)
    rng = np.random.default_rng(seed)
    # k-means++ style seeding
    centers = [x[rng.integers(n)]]
    d2 = ((x - centers[0]) ** 2).sum(-1)
    for _ in range(1, k):
        probs = d2 / (d2.sum() + 1e-12)
        idx = int(rng.choice(n, p=probs))
        centers.append(x[idx])
        d2 = np.minimum(d2, ((x - centers[-1]) ** 2).sum(-1))
    centers = np.stack(centers)

    labels = np.zeros(n, np.int64)
    for _ in range(iters):
        dist = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        new_labels = dist.argmin(-1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if mask.any():
                centers[j] = x[mask].mean(0)
            else:  # re-seed the empty cluster at the worst-covered sample
                centers[j] = x[dist.min(-1).argmax()]
    return centers, labels


def fit_fabric_prototypes(descs: np.ndarray, k: int = 8, seed: int = 0) -> tuple[FabricPrototypes, np.ndarray]:
    x = np.asarray(descs, np.float32)
    mean = x.mean(0)
    std = x.std(0) + 1e-6
    z = (x - mean) / std
    centers, labels = kmeans_np(z, k, seed=seed)
    return FabricPrototypes(centers=centers, mean=mean, std=std), labels


# --------------------------------------------------------------------------------------
# 3. Memory-bank compression
# --------------------------------------------------------------------------------------
def spherical_kmeans(x: np.ndarray, k: int, iters: int = 20, seed: int = 0) -> np.ndarray:
    """Compress L2-normalised patch features into k unit-norm centroids.

    The centroids are the deployable form of PatchCore's memory bank: a 1x1
    convolution followed by a channel-wise max reproduces the nearest-neighbour
    cosine score exactly, with no FAISS / k-NN search at inference.
    """
    x = np.asarray(x, np.float32)
    n = len(x)
    k = min(k, n)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=k, replace=False)
    c = x[idx].copy()
    c /= np.linalg.norm(c, axis=-1, keepdims=True) + 1e-8
    for _ in range(iters):
        sim = x @ c.T                          # (n, k)
        lab = sim.argmax(1)
        new = c.copy()
        for j in range(k):
            mask = lab == j
            if mask.any():
                new[j] = x[mask].mean(0)
            else:
                new[j] = x[sim.max(1).argmin()]  # worst-covered sample
        new /= np.linalg.norm(new, axis=-1, keepdims=True) + 1e-8
        if np.max(np.abs(new - c)) < 1e-5:
            c = new
            break
        c = new
    return c


def greedy_coreset(x: np.ndarray, k: int, seed: int = 0, chunk: int = 8192) -> np.ndarray:
    """Approximate greedy farthest-point coreset (PatchCore's subsampler).

    Keeps the exact *representative* patch features rather than centroids. Used
    when ``normality.bank_method == 'coreset'``. The result is then converted
    to centroids for deployment, or used directly by an exact k-NN scorer.
    """
    x = np.asarray(x, np.float32)
    n = len(x)
    k = min(k, n)
    rng = np.random.default_rng(seed)
    chosen = np.empty(k, np.int64)
    chosen[0] = rng.integers(n)
    dist = np.full(n, np.inf, np.float32)
    for i in range(1, k):
        last = x[chosen[i - 1]]
        for s in range(0, n, chunk):
            d = ((x[s:s + chunk] - last) ** 2).sum(-1)
            dist[s:s + chunk] = np.minimum(dist[s:s + chunk], d)
        dist[chosen[i - 1]] = -1
        chosen[i] = int(dist.argmax())
    return x[chosen[:k]]
