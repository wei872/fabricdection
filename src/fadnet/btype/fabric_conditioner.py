"""布种条件化 FC（design §3.5）

布种变化三要素 = 经纬线粗细 / 经纬密 / 反光 → 皆可参数化：
  * FFT 对数谱主频间距与线宽比 ≈ 粗细、经纬密（样本实测 main pitch ≈1.6–2.8 px）
  * 平场前亮度/高光统计 ≈ 反光
  * 可选 MES 布种代码（Q5）直接给定布种 ID
输出：布种嵌入（FiLM 用，32d）+ 布种 ID（阈值表/记忆库分桶查表用）。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def fft_profile(gray: np.ndarray, bins: int = 64) -> np.ndarray:
    """FFT 对数谱 → 径向 bins 特征 + 主轴主频位置（经纬密/粗细的先验特征）。"""
    g = gray.astype(np.float32)
    g = g - g.mean()
    P = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    P = np.log1p(P)
    h, w = P.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.int32)
    r = np.clip(r, 0, bins - 1)
    radial = np.zeros(bins, dtype=np.float32)
    np.add.at(radial, r, P)
    cnt = np.bincount(r.ravel(), minlength=bins).astype(np.float32) + 1e-6
    return radial / cnt


class FabricConditioner(nn.Module):
    """轻量布种估计器：FFT 谱特征 + 亮度统计 → 嵌入(32d) + 布种 ID logits。"""

    def __init__(self, fft_bins: int = 64, embed_dim: int = 32, n_fabrics: int = 8):
        super().__init__()
        in_dim = fft_bins + 6  # 谱特征 + [mean,std,p95,p99,高光比例,暗角比]
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU(),
        )
        self.embed_head = nn.Linear(128, embed_dim)
        self.type_head = nn.Linear(128, n_fabrics)

    @staticmethod
    def brightness_stats(gray: np.ndarray) -> np.ndarray:
        g = gray.astype(np.float32)
        c = 100
        corners = np.concatenate([g[:c, :c].ravel(), g[:c, -c:].ravel(),
                                  g[-c:, :c].ravel(), g[-c:, -c:].ravel()])
        specular = float((g > g.mean() + 3 * g.std()).mean())
        return np.array([g.mean(), g.std(), np.percentile(g, 95), np.percentile(g, 99),
                         specular, g.mean() / (corners.mean() + 1e-6)], dtype=np.float32)

    def forward(self, feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """feats: [B, fft_bins+6] → (embed [B,32], logits [B,n_fabrics])"""
        h = self.net(feats)
        return self.embed_head(h), self.type_head(h)

    @torch.no_grad()
    def infer(self, gray: np.ndarray, fft_bins: int = 64) -> tuple[np.ndarray, int]:
        x = np.concatenate([fft_profile(gray, fft_bins), self.brightness_stats(gray)])
        embed, logits = self(torch.from_numpy(x)[None])
        return embed[0].numpy(), int(logits.argmax(-1))


class FiLM(nn.Module):
    """Feature-wise Linear Modulation：布种嵌入调制特征（<0.1M 参数，design §3.5）。"""

    def __init__(self, embed_dim: int, feat_dim: int):
        super().__init__()
        self.to_gamma_beta = nn.Linear(embed_dim, 2 * feat_dim)

    def forward(self, feat: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.to_gamma_beta(embed).chunk(2, dim=-1)  # [B,D] each
        return feat * (1 + gamma[:, None, :]) + beta[:, None, :]
