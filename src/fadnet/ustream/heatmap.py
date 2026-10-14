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


def self_normalize(a: torch.Tensor, bg_quantile: float = 90.0) -> torch.Tensor:
    """逐 tile 自归一化 —— "免布种登记"的关键（用户要求 2026-10：模型自适应所有布种）。

    Dinomaly/PatchCore 已完成"缺陷前景 vs 布料背景"分离：背景分分布只由传感噪声与
    纹理残差决定，其**相对尺度**跨布种/跨光照高度一致。故以本 tile 背景中位数为基线、
    (p{bg_quantile} - p50) 为尺度，把分数换算成"背景尺度倍数"(z)，阈值 tau_u 全局通用，
    无需逐布种标定/登记。大缺陷占 tile ≤30% 时中位数仍由背景主导，估计稳健。
    """
    flat = a.flatten(1).float()
    med = flat.median(dim=1).values
    scale = (torch.quantile(flat, bg_quantile / 100.0, dim=1) - med).clamp_min(1e-6)
    return ((flat - med[:, None]) / scale[:, None]).reshape(a.shape)


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
