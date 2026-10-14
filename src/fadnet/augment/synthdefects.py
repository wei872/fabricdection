"""合成缺陷引擎（布料感知版 CutPaste/DRAEM，design §4.2）

频域锁定：FFT 估计主频（纱距/线宽），沿经纬轴生成 1–3 纱宽的断线/条带；
形态库：断经/断纬、稀密档、异纤细线（贝塞尔）、油污团块（低频 Perlin）、
破洞、结头。同时输出 YOLO 格式框标签，供 S 分支冷启动（合成:真实 ≈ 2:1）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CLASSES = ["broken_warp", "broken_weft", "thin_thick_weft", "double_pick", "slub",
           "skip_draw", "hole", "knot", "stain_oil", "foreign_fiber", "streak_bar",
           "edge_defect", "other_known", "unknown_anomaly"]

# 合成引擎已实现的形态（prepare_dataset 注入池仅从此采样；其余类待 M2 扩充）
IMPLEMENTED_KINDS = ["broken_warp", "broken_weft", "thin_thick_weft",
                     "foreign_fiber", "stain_oil", "hole", "knot"]


def _autocorr_pitch(profile: np.ndarray, min_lag: int = 1, max_lag: int = 64,
                    thresh: float = 0.15, fallback: float = 3.0) -> float:
    """轮廓去均值 → 归一化自相关 → 第一个显著峰对应的 lag 即纱距（px）。

    对布种变化（细 1.5px ~ 粗几十 px）比 FFT 主频更稳：
    FFT 低频能量常被格子/光照场主导，会淹没仅 2px 周期的细纹理峰。
    """
    p = profile.astype(np.float64)
    p = p - p.mean()
    if p.std() < 1e-6:
        return fallback
    ac = np.correlate(p, p, mode="full")[len(p) - 1:]
    ac /= ac[0] + 1e-12
    max_lag = min(max_lag, len(ac) - 1)
    for lag in range(min_lag, max_lag):
        if ac[lag] >= thresh and ac[lag] >= ac[lag - 1] and ac[lag] >= ac[lag + 1]:
            return float(lag)
    return fallback


def estimate_pitch(gray: np.ndarray) -> tuple[float, float]:
    """自相关法估计 (warp_pitch 经距, weft_pitch 纬距)（px）。

    样本实测细纹理 ≈1.6–2.8 px（逼近奈奎斯特极限，见 design Q4）。
    对 1/4 下采样轮廓先找粗尺度的格子周期，再在 [1,64]px 内找细纹理纱距，
    取细尺度为主；粗织物自动退化到真实纱距。
    """
    g = gray.astype(np.float32)
    k = 4  # 下采样提速
    h, w = (g.shape[0] // k) * k, (g.shape[1] // k) * k
    small = g[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3))
    # 行向轮廓（跨列均值）→ 经向纱距（相邻经纱行间距）；列向轮廓 → 纬距
    row_profile = small.mean(axis=1)
    col_profile = small.mean(axis=0)
    pw = _autocorr_pitch(row_profile, max_lag=16) * k
    ph = _autocorr_pitch(col_profile, max_lag=16) * k
    return pw, ph


@dataclass
class SynResult:
    image: np.ndarray            # float32
    boxes: list[tuple[int, float, float, float, float]]  # (cls_id, xc, yc, w, h) 归一化


def _yolo_box(mask: np.ndarray, cls_id: int) -> tuple[int, float, float, float, float] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) < 2:
        return None
    h, w = mask.shape
    x0, x1, y0, y1 = xs.min() / w, xs.max() / w, ys.min() / h, ys.max() / h
    return (cls_id, (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, 1e-4), max(y1 - y0, 1e-4))


def _bezier_fiber(shape: tuple[int, int], rng: np.random.Generator,
                  length: int, thickness: int) -> np.ndarray:
    """贝塞尔细线（异纤/毛发形态，≈样本中深色弧线）。"""
    h, w = shape
    p0 = rng.uniform(0.1, 0.9, 2) * [w, h]
    p1 = p0 + rng.normal(0, 0.25, 2) * [w, h]
    p2 = p0 + rng.uniform(-1, 1, 2) * [w, h] * rng.uniform(0.3, 1.0)
    t = np.linspace(0, 1, length)[:, None]
    pts = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2
    mask = np.zeros((h, w), np.uint8)
    for x, y in pts.astype(int):
        if 0 <= x < w and 0 <= y < h:
            y0, y1 = max(0, y - thickness), min(h, y + thickness + 1)
            x0, x1 = max(0, x - thickness), min(w, x + thickness + 1)
            mask[y0:y1, x0:x1] = 1
    return mask


def _perlin_blob(shape: tuple[int, int], rng: np.random.Generator, scale: int = 6) -> np.ndarray:
    """低频随机团块（油污）；scale 越小越低频。"""
    h, w = shape
    small = rng.random((max(2, h // scale), max(2, w // scale)))
    from PIL import Image
    blob = np.asarray(Image.fromarray((small * 255).astype(np.uint8))
                      .resize((w, h), Image.BILINEAR), np.float32) / 255
    return (blob - blob.min()) / (np.ptp(blob) + 1e-6)


def inject_defect(
    tile: np.ndarray,
    kind: str,
    rng: np.random.Generator | None = None,
) -> SynResult:
    """在 1280² 正常块上注入一种缺陷，返回图像与 YOLO 标签。

    tile: [H,W] 灰度或 [H,W,3]；kind ∈ {broken_warp, broken_weft, foreign_fiber,
    stain_oil, hole, knot, thin_thick_weft}
    """
    rng = rng or np.random.default_rng()
    gray = tile[..., 0] if tile.ndim == 3 else tile
    h, w = gray.shape
    img = tile.astype(np.float32).copy()
    g_target = img[..., 0] if img.ndim == 3 else img
    pw, ph = estimate_pitch(gray)                      # 纬距(px/列向), 经距(px/行向)
    cls_id = CLASSES.index(kind)

    if kind in ("broken_warp", "broken_weft", "thin_thick_weft"):
        # 断经：竖直暗线（沿经向=纵向减少纱）；断纬：水平暗线；稀密档：横/竖调亮带
        n_yarn = rng.integers(1, 4)
        t = int(max(1, round(n_yarn * (pw if kind == "broken_warp" else ph))))
        length = int(rng.integers(60, h))              # 断纱长度
        if kind == "broken_warp":
            x = rng.integers(w // 8, 7 * w // 8); y = rng.integers(0, h - length)
            mask = np.zeros((h, w), np.uint8)
            mask[y:y + length, max(0, x - t // 2):x + t // 2 + 1] = 1
        elif kind == "broken_weft":
            y = rng.integers(h // 8, 7 * h // 8); x = rng.integers(0, w - length)
            mask = np.zeros((h, w), np.uint8)
            mask[max(0, y - t // 2):y + t // 2 + 1, x:x + length] = 1
        else:
            band = int(rng.integers(8, 30)); y = rng.integers(0, h - band)
            mask = np.zeros((h, w), np.uint8); mask[y:y + band, :] = 1
        delta = rng.uniform(0.25, 0.55) * g_target.mean()
        sign = -1.0 if kind != "thin_thick_weft" else (-1.0 if rng.random() < 0.7 else 0.5)
        alpha = mask.astype(np.float32) * rng.uniform(0.6, 1.0)
        g_target[...] = np.clip(g_target + sign * delta * alpha, 0, 255)
    elif kind == "foreign_fiber":
        thickness = int(max(1, round(rng.uniform(0.5, 1.5) * pw)))
        mask = _bezier_fiber((h, w), rng, int(rng.integers(150, 700)), thickness)
        dark = rng.uniform(30, 70)                     # ≈样本实测对比度 56 灰阶
        g_target[...] = np.clip(g_target - dark * mask, 0, 255)
    elif kind == "stain_oil":
        blob = _perlin_blob((h, w), rng, scale=int(rng.integers(4, 10)))
        blob = (blob > rng.uniform(0.45, 0.6)).astype(np.float32)
        blob *= _perlin_blob((h, w), rng, scale=8)
        # 随机椭圆窗约束空间范围：污渍为局部团块，不应覆盖整块 tile
        cx, cy = rng.integers(w // 6, 5 * w // 6), rng.integers(h // 6, 5 * h // 6)
        rw, rh = rng.integers(30, 220), rng.integers(30, 160)
        yy, xx = np.ogrid[:h, :w]
        win = (((xx - cx) / rw) ** 2 + ((yy - cy) / rh) ** 2 < 1).astype(np.float32)
        blob *= win
        mask = (blob > 0.15).astype(np.uint8)
        if mask.sum() < 16:                       # 偶发空窗：退化为窗口本身
            blob, mask = win, win.astype(np.uint8)
        g_target[...] = np.clip(g_target - rng.uniform(15, 45) * blob, 0, 255)
    elif kind == "hole":
        rw, rh = rng.integers(6, 40), rng.integers(6, 40)
        cx, cy = rng.integers(w // 8, 7 * w // 8), rng.integers(h // 8, 7 * h // 8)
        yy, xx = np.ogrid[:h, :w]
        mask = (((xx - cx) / rw) ** 2 + ((yy - cy) / rh) ** 2 < 1).astype(np.uint8)
        rim = (((xx - cx) / (rw + 2)) ** 2 + ((yy - cy) / (rh + 2)) ** 2 < 1) - mask
        g_target[...] = np.clip(g_target * (1 - 0.85 * mask) + 40 * rim, 0, 255)
    elif kind == "knot":
        r = int(rng.integers(2, 6))
        cx, cy = rng.integers(w // 8, 7 * w // 8), rng.integers(h // 8, 7 * h // 8)
        mask = np.zeros((h, w), np.uint8)
        mask[max(0, cy - r):cy + r, max(0, cx - r):cx + r] = 1
        g_target[...] = np.clip(g_target + rng.uniform(-40, 50) * mask, 0, 255)
    else:
        raise ValueError(f"未实现的缺陷形态: {kind}")

    box = _yolo_box(mask, cls_id)
    return SynResult(image=img, boxes=[box] if box else [])
