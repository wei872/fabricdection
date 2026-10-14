"""U 分支·双信号融合与异常区提取（design §3.4.3 / F-Head 规则 1）

A = max( z_rec(A_rec), z_knn(A_knn) )
z_*：逐布种稳健标准化（正常训练集 1%/99% 分位 → [0,1] 截断）。
阈值 τ_u 标定见 design §7.2（以"大缺陷最小分 − 5% 裕量"为 τ_large 并保证 τ_u ≤ τ_large）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


class RobustNorm:
    """逐布种稳健标准化表（参数存 configs/fabric_thresholds.yaml）。"""

    def __init__(self, table: dict[str, dict[str, tuple[float, float]]]):
        self.table = table  # {fabric_id: {"rec": (p1, p99), "knn": (p1, p99)}}

    def z(self, x: torch.Tensor, fabric_id: str, channel: str) -> torch.Tensor:
        p1, p99 = self.table.get(fabric_id, self.table["*"])[channel]
        return ((x - p1) / max(p99 - p1, 1e-6)).clamp(0, 1)

    @staticmethod
    def fit(samples: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> tuple[float, float]:
        return float(np.percentile(samples, lo)), float(np.percentile(samples, hi))


def fuse_heatmaps(a_rec: torch.Tensor, a_knn: torch.Tensor, norm: RobustNorm,
                  fabric_id: str) -> torch.Tensor:
    """双信号 max 级联（召回优先，design §3.4.3）。"""
    return torch.maximum(norm.z(a_rec, fabric_id, "rec"), norm.z(a_knn, fabric_id, "knn"))


@dataclass
class AnomRegion:
    x0: int
    y0: int
    x1: int
    y1: int
    score: float      # 区内最高归一化异常分
    area: int         # px


def extract_regions(a: torch.Tensor, tau_u: float, min_area: int = 9) -> list[AnomRegion]:
    """二值化 → 连通域 → 候选异常区（F-Head 规则 1）。

    实现：最大池化提速后在 1/4 分辨率上找峰值簇，再回投原分辨率开窗细化；
    生产版可换 GPU connected-components（kornia/cucim）。这里为无依赖简易版。
    """
    m = (a[0, 0] > tau_u).cpu().numpy().astype(np.uint8)
    h, w = m.shape
    regions: list[AnomRegion] = []
    visited = np.zeros_like(m)
    for yy, xx in zip(*np.where(m & ~visited)):
        if visited[yy, xx]:
            continue
        stack, xs, ys = [(yy, xx)], [], []
        while stack:                                   # BFS 连通域
            cy, cx = stack.pop()
            if cy < 0 or cx < 0 or cy >= h or cx >= w or visited[cy, cx] or not m[cy, cx]:
                continue
            visited[cy, cx] = 1
            xs.append(cx); ys.append(cy)
            stack += [(cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)]
        if len(xs) >= min_area:
            regions.append(AnomRegion(min(xs), min(ys), max(xs) + 1, max(ys) + 1,
                                      float(a[0, 0, min(ys):max(ys) + 1,
                                               min(xs):max(xs) + 1].max()), len(xs)))
    return regions
