"""U 分支·记忆库通道 —— PatchCore 风格（design §3.4.2）

对应 PatchCore (CVPR 2022, https://github.com/amazon-science/patchcore-inspection)：
  * 中层特征 + 邻域感知聚合（3×3, stride 1）→ locally-aware patch features
  * 正常样本记忆库 → 贪心 k-center Coreset（≈8k 中心）
  * 测试：1-NN 距离 + 最近 4 邻重加权 → 上采样 + 高斯 σ=4 → A_knn

免登记布种自适应（用户要求 2026-10）：入库/查询前对**每个 tile 做特征中心化**
（style_center，减去该 tile 的均值特征方向），把"布种风格"（经纬密/粗细/反光造成的
整体特征偏移）从特征中剥离，只留缺陷前景-背景相对结构 → **单一全局记忆库**即可
覆盖所有布种，无需布种 ID / 分桶 / 登记。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def neighborhood_aware(features: torch.Tensor, p: int = 3) -> torch.Tensor:
    """PatchCore 邻域感知块特征：特征图 3×3 均值池化(stride 1) 与中心特征取平均。

    features: [B, D, gh, gw]（token 网格重排）→ 同形状（与骨干维数对齐，免投影层）。
    """
    pooled = F.avg_pool2d(features, kernel_size=p, stride=1, padding=p // 2)
    return 0.5 * (features + pooled)


def style_center(features: torch.Tensor) -> torch.Tensor:
    """逐 tile 特征中心化：剥离布种风格偏移（免登记泛化的特征侧手段）。

    支持 [B, D, gh, gw] 与 [N, D]（后者视为单 tile）。
    """
    if features.dim() == 4:
        return features - features.mean(dim=(2, 3), keepdim=True)
    return features - features.mean(dim=0, keepdim=True)


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
    """全局单一记忆库（centers: [M, D]；style_center 后布种无关）。"""

    dim: int
    centers: torch.Tensor | None = None

    def build(self, feats: torch.Tensor, n_centers: int = 8192, seed: int = 0) -> None:
        """feats: [N, D]（已 style_center + 邻域感知）→ Coreset 入库。"""
        feats = F.normalize(feats, dim=-1)
        self.centers = greedy_coreset(feats, min(n_centers, feats.shape[0]), seed)

    def bucket(self, fabric_id: str = "*") -> torch.Tensor:
        """兼容旧调用名；现全局单库，忽略 fabric_id。"""
        assert self.centers is not None, "MemoryBank 未构建/加载"
        return self.centers

    def save(self, path: str) -> None:
        torch.save({"dim": self.dim, "centers": self.centers.cpu()}, path)

    def load(self, path: str, device: str = "cuda") -> None:
        d = torch.load(path, map_location=device)
        self.dim, self.centers = d["dim"], d["centers"].to(device)


def knn_anomaly_scores(
    queries: torch.Tensor,     # [N, D] token 网格铺平（已 style_center）
    coreset: torch.Tensor,     # [M, D]
    k: int = 1,
    reweight_n: int = 4,
    chunk: int = 4096,
) -> torch.Tensor:
    """PatchCore 打分：1-NN 距离 + 近邻重加权。返回 [N]（含重加权）。"""
    scores = torch.empty(queries.shape[0], device=queries.device)
    for s in range(0, queries.shape[0], chunk):
        q = F.normalize(queries[s:s + chunk], dim=-1)
        d = torch.cdist(q, coreset)                      # [b, M]
        d_min, nn_idx = d.topk(k, largest=False)         # 1-NN
        # PatchCore 重加权：query 异常时其 kNN 中心之间的相似度更低
        near = coreset[nn_idx[:, 0]]                     # [b, D]
        d_nb = torch.cdist(near, coreset).topk(reweight_n + 1, largest=False).values[:, 1:]
        w = 1 - (d_min[:, 0].exp() / (d_nb.exp().sum(1) + 1e-12))
        scores[s:s + chunk] = w * d_min[:, 0]            # s* = (1 - max s_b)·d_min
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
