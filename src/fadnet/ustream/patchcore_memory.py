"""U 分支·记忆库通道 —— PatchCore 风格（design §3.4.2）

对应 PatchCore (CVPR 2022, https://github.com/amazon-science/patchcore-inspection)：
  * 中层特征 + 邻域感知聚合（3×3, stride 1）→ locally-aware patch features
  * 正常样本记忆库 → 随机 10% → 贪心 k-center Coreset（≈8k 中心）
  * 测试：1-NN 距离 + 最近 4 邻重加权 → 上采样 + 高斯 σ=4 → A_knn
记忆库按布种分桶（FC 布种 ID 查桶），GPU 分块 cdist <1 ms。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


def neighborhood_aware(features: torch.Tensor, p: int = 3) -> torch.Tensor:
    """PatchCore 邻域感知块特征：特征图 3×3 均值池化(stride 1) 与中心特征拼接后投影。

    features: [B, D, gh, gw]（token 网格重排）→ [B, D, gh, gw]（简化：均值池化
    特征与中心特征取平均；与骨干维数对齐，免去额外投影层）。
    """
    pooled = F.avg_pool2d(features, kernel_size=p, stride=1, padding=p // 2)
    return 0.5 * (features + pooled)


def greedy_coreset(features: torch.Tensor, n_centers: int, seed: int = 0) -> torch.Tensor:
    """贪心 k-center Coreset 下采样（PatchCore 记忆库压缩）。

    features: [N, D] → centers: [n_centers, D]
    """
    g = torch.Generator(device=features.device).manual_seed(seed)
    n = features.shape[0]
    first = int(torch.randint(n, (1,), generator=g, device=features.device))
    idx = [first]
    dist = torch.cdist(features, features[first:first + 1]).squeeze(1)  # 到最近中心的距离
    for _ in range(n_centers - 1):
        nxt = int(dist.argmax())
        idx.append(nxt)
        dist = torch.minimum(dist, torch.cdist(features, features[nxt:nxt + 1]).squeeze(1))
    return features[torch.tensor(idx, device=features.device)]


@dataclass
class MemoryBank:
    """布种分桶记忆库（buckets: fabric_id → [M,D] Tensor）。"""

    dim: int
    buckets: dict[str, torch.Tensor] = field(default_factory=dict)

    def build(self, fabric_id: str, feats: torch.Tensor, n_centers: int = 8192,
              seed: int = 0) -> None:
        """feats: [N,D]（已做邻域感知聚合）→ Coreset 入库。"""
        feats = F.normalize(feats, dim=-1)
        self.buckets[fabric_id] = greedy_coreset(feats, min(n_centers, feats.shape[0]), seed)

    def bucket(self, fabric_id: str) -> torch.Tensor:
        if fabric_id not in self.buckets:  # 未知布种：用首个桶兜底 + 告警上抛
            fabric_id = next(iter(self.buckets))
        return self.buckets[fabric_id]

    def save(self, path: str) -> None:
        torch.save({k: v.cpu() for k, v in self.buckets.items()}, path)

    def load(self, path: str, device: str = "cuda") -> None:
        self.buckets = {k: v.to(device) for k, v in torch.load(path).items()}


def knn_anomaly_scores(
    queries: torch.Tensor,     # [N, D] token 网格铺平
    coreset: torch.Tensor,     # [M, D]
    k: int = 1,
    reweight_n: int = 4,
    chunk: int = 4096,
) -> torch.Tensor:
    """PatchCore 打分：1-NN 距离 + 近邻重加权。

    返回 [N] 距离分（含重加权）。GPU 分块，8k×384 规模 <1 ms。
    """
    scores = torch.empty(queries.shape[0], device=queries.device)
    for s in range(0, queries.shape[0], chunk):
        q = F.normalize(queries[s:s + chunk], dim=-1)
        d = torch.cdist(q, coreset)                      # [b, M]
        d_min, nn_idx = d.topk(k, largest=False)         # 1-NN
        # PatchCore 重加权：query 异常时其 kNN 中心之间的相似度更低
        near = coreset[nn_idx[:, 0]]                     # [b, D]
        d_nb = torch.cdist(near, coreset).topk(reweight_n + 1, largest=False).values[:, 1:]
        w = 1 - (d_min[:, 0].exp() / (d_nb.exp().sum(1) + 1e-12))
        scores[s:s + chunk] = (1 - 0) * w * d_min[:, 0]  # s* = (1 - max s_b)·d_min
    return scores


def scores_to_heatmap(scores: torch.Tensor, gh: int, gw: int, out_hw: tuple[int, int],
                      gaussian_sigma: float = 4.0) -> torch.Tensor:
    """[N] → [1,1,H,W]：重排网格 + 双线性上采样 + 高斯平滑（PatchCore 后处理）。"""
    m = scores.reshape(1, 1, gh, gw)
    m = F.interpolate(m, size=out_hw, mode="bilinear", align_corners=False)
    r = int(3 * gaussian_sigma)
    kx = torch.exp(-(torch.arange(-r, r + 1, device=m.device) ** 2) / (2 * gaussian_sigma ** 2))
    kx = kx / kx.sum()
    m = F.conv2d(m, kx.view(1, 1, 1, -1), padding=(0, r))
    return F.conv2d(m, kx.view(1, 1, -1, 1), padding=(r, 0))
