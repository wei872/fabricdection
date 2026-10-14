"""ROI 检测与 Tiling（design §3.2-P1/P3）

条带（≤2440×~850）→ 有效布面 ROI → 1280×1280 15% 重叠切块，
块内记录全局坐标，供 F-Head 跨块归并与米级聚合。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Tile:
    image: np.ndarray          # [1280,1280,(C)] float32
    x0: int                    # 在条带全局坐标系中的左上角
    y0: int
    fabric_id: str = ""


def detect_fabric_roi(gray: np.ndarray, min_ratio: float = 0.05) -> tuple[int, int, int, int]:
    """用纹理能量找有效布面区域，剔除样本中可见的大片空白边。

    返回 (x0, y0, x1, y1)。简易实现：列×行方差能量阈值；
    生产版建议换轻量分割或用相机 FOV 标定直接给定。
    """
    g = gray.astype(np.float32)
    col_energy = g.std(axis=0)
    row_energy = g.std(axis=1)
    cx = np.where(col_energy > col_energy.max() * min_ratio)[0]
    ry = np.where(row_energy > row_energy.max() * min_ratio)[0]
    if len(cx) == 0 or len(ry) == 0:
        return 0, 0, gray.shape[1], gray.shape[0]
    return int(cx[0]), int(ry[0]), int(cx[-1]) + 1, int(ry[-1]) + 1


def slice_tiles(
    strip: np.ndarray,
    tile_size: int = 1280,
    overlap: float = 0.15,
    roi: tuple[int, int, int, int] | None = None,
    fabric_id: str = "",
) -> list[Tile]:
    """按重叠率切块；右/下边缘不足处镜像补齐。"""
    h, w = strip.shape[:2]
    x0, y0, x1, y1 = roi if roi is not None else (0, 0, w, h)
    stride = int(tile_size * (1 - overlap))
    tiles: list[Tile] = []
    ys = np.unique(np.clip(np.arange(y0, max(y1 - tile_size, y0) + 1, stride), y0, y1 - tile_size)) \
        if y1 - y0 > tile_size else np.array([y0])
    xs = np.unique(np.clip(np.arange(x0, max(x1 - tile_size, x0) + 1, stride), x0, x1 - tile_size)) \
        if x1 - x0 > tile_size else np.array([x0])
    for ty in ys:
        for tx in xs:
            patch = strip[ty:ty + tile_size, tx:tx + tile_size]
            if patch.shape[0] < tile_size or patch.shape[1] < tile_size:
                pad_h, pad_w = tile_size - patch.shape[0], tile_size - patch.shape[1]
                patch = np.pad(patch, ((0, pad_h), (0, pad_w)) + ((0, 0),) * (patch.ndim - 2),
                               mode="reflect")
            tiles.append(Tile(image=patch.astype(np.float32), x0=int(tx), y0=int(ty),
                              fabric_id=fabric_id))
    return tiles
