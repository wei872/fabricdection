"""Normality branch: PatchCore memory bank + Dinomaly reconstruction residual.

Both signals are computed from the SAME shared backbone features, so the branch
adds only a few GFLOPs to the detector. Nothing here runs a second heavy
encoder at inference.

B1 — Compressed PatchCore memory bank
    Locally aggregated (3x3) patch features from n3/n5 are L2-normalised and
    projected to D dims on an 80x80 grid. For each fabric prototype we store K
    unit-norm centroids. The nearest-neighbour cosine score is exactly

        d(q) = 1 - max_k <q, c_k>  =  1 - max(conv1x1(q, C))

    so the whole memory bank is ONE 1x1 convolution + a channel max — fully
    TensorRT-exportable, deterministic, O(D*K*80^2).

B2 — Dinomaly reconstruction residual
    A frozen "foundation" projection of the stride-32 map provides the target.
    A noisy bottleneck (dropout) + 4 linear-attention (no softmax) blocks try
    to reconstruct it from a jittered input. Trained on normal images only with
    a *loose* hard-mining cosine loss. Linear attention "cannot focus", so an
    anomalous pattern is pulled back to the normal manifold -> large residual.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import GaussianSmooth


# ======================================================================================
# B1 — patch features + compressed memory bank
# ======================================================================================
class PatchFeatureHead(nn.Module):
    """PatchCore-style locally-aware patch features on a fixed grid."""

    def __init__(self, cfg, channels):
        super().__init__()
        n = cfg.normality
        self.grid = int(n.grid)
        self.dim = int(n.patch_dim)
        # Fixed 1280² deployment: n3 is stride-8. Decide the ONNX-safe pooling
        # path from config, not from a traced tensor shape (no TracerWarning).
        self.n3_size = int(cfg.input_size[0]) // 8
        half = self.dim // 2
        self.local = nn.AvgPool2d(int(n.local_pool), 1, int(n.local_pool) // 2)
        self.proj3 = nn.Conv2d(channels[1], half, 1, bias=False)
        self.proj5 = nn.Conv2d(channels[3], self.dim - half, 1, bias=False)
        self.fuse = nn.Conv2d(self.dim, self.dim, 1, bias=False)

    def forward(self, feats):
        g = self.grid
        f3 = F.normalize(self.proj3(self.local(feats["n3"])), dim=1)
        f5 = F.normalize(self.proj5(self.local(feats["n5"])), dim=1)
        # ONNX only supports adaptive_avg_pool2d when the output size divides
        # the input size. jingwei_p uses grid=64 on a 160x160 n3 map (2.5x),
        # which would otherwise fail ONNX/TensorRT export. Use bilinear resize
        # in that case; both paths keep a fully static output shape.
        if self.n3_size % g == 0:
            f3 = F.adaptive_avg_pool2d(f3, (g, g))
        else:
            f3 = F.interpolate(f3, size=(g, g), mode="bilinear", align_corners=False)
        f5 = F.interpolate(f5, size=(g, g), mode="bilinear", align_corners=False)
        q = self.fuse(torch.cat([f3, f5], 1))
        return F.normalize(q, dim=1)                  # (B, D, g, g)


class FabricPrototypeBank(nn.Module):
    """K unit-norm centroids per fabric prototype + per-prototype calibration.

    Buffers (not parameters) so they can be written by ``tools/build_memory_bank.py``
    and travel with the checkpoint / exported graph.
    """

    def __init__(self, dim: int, n_prototypes: int, n_centroids: int, seed: int = 0):
        super().__init__()
        self.dim, self.n_proto, self.n_centroids = dim, n_prototypes, n_centroids
        g = torch.Generator().manual_seed(seed)
        c = F.normalize(torch.randn(n_prototypes, n_centroids, dim, generator=g), dim=-1)
        self.register_buffer("centroids", c)
        self.register_buffer("mu", torch.zeros(n_prototypes))
        self.register_buffer("sigma", torch.ones(n_prototypes))
        self.register_buffer("tau", torch.full((n_prototypes,), 3.0))
        self.register_buffer("valid", torch.zeros(n_prototypes, dtype=torch.bool))
        self.slope = 1.0
        # A REAL frozen 1x1 conv realises the whole memory bank. Its weights are
        # swapped in from ``centroids[k]`` when the fabric prototype changes.
        # For deployment, freeze one prototype (``set_active``) and the bank is
        # just an ordinary Conv2d that TensorRT can fuse.
        self.scorer = nn.Conv2d(dim, n_centroids, 1, bias=False)
        self.scorer.weight.requires_grad_(False)
        self.scorer.weight.data.copy_(c[0].unsqueeze(-1).unsqueeze(-1))
        self._active = 0

    @torch.no_grad()
    def set_prototype(self, k: int, centroids: np.ndarray):
        c = torch.as_tensor(centroids, dtype=torch.float32)
        c = F.normalize(c, dim=-1)
        # A rare fabric prototype may yield fewer patches than n_centroids.
        # Pad by near-duplicates (tiny jitter) so the Conv2d shape stays fixed.
        if c.shape[0] < self.n_centroids:
            reps = int(np.ceil(self.n_centroids / c.shape[0]))
            c = c.repeat(reps, 1)[:self.n_centroids]
            c = F.normalize(c + 1e-3 * torch.randn_like(c), dim=-1)
        else:
            c = c[:self.n_centroids]
        self.centroids[k] = c
        self.valid[k] = True
        if k == self._active:
            self.scorer.weight.data.copy_(c.unsqueeze(-1).unsqueeze(-1))

    @torch.no_grad()
    def set_calibration(self, k: int, mu: float, sigma: float, tau: float, slope: float = 1.0):
        self.mu[k] = float(mu)
        self.sigma[k] = max(float(sigma), 1e-6)
        self.tau[k] = float(tau)
        self.slope = float(slope)

    @torch.no_grad()
    def set_active(self, k: int):
        """Bake prototype ``k`` into the scorer (use once before export)."""
        k = max(0, min(self.n_proto - 1, int(k)))
        self.scorer.weight.data.copy_(self.centroids[k].unsqueeze(-1).unsqueeze(-1))
        self._active = k
        return self

    def raw_distance(self, q: torch.Tensor, k: int) -> torch.Tensor:
        """d = 1 - max_k cos(q, c_k), computed as a 1x1 conv + channel max."""
        if k != self._active:
            self.set_active(k)
        cos = self.scorer(q)                                       # (B, K, g, g)
        return 1.0 - cos.max(dim=1, keepdim=True).values          # (B, 1, g, g)

    def score(self, q: torch.Tensor, proto_id=0) -> torch.Tensor:
        """Calibrated anomaly probability in [0,1]."""
        B = q.shape[0]
        if torch.is_tensor(proto_id):
            ids = [int(v) for v in proto_id.detach().cpu().flatten().tolist()][:B]
        else:
            ids = [int(proto_id)] * B
        outs = []
        for b, k in enumerate(ids):
            k = max(0, min(self.n_proto - 1, k))
            d = self.raw_distance(q[b:b + 1], k)
            z = (d - self.mu[k]) / self.sigma[k]
            outs.append(torch.sigmoid((z - self.tau[k]) * self.slope))
        return torch.cat(outs, 0)


# ======================================================================================
# B2 — Dinomaly-style lightweight reconstruction
# ======================================================================================
class NoisyBottleneck(nn.Module):
    """Dinomaly's bMlp: pre-existing Dropouts do all the noise injection."""

    def __init__(self, dim: int, ratio: float = 4.0, drop: float = 0.2):
        super().__init__()
        h = int(dim * ratio)
        self.fc1 = nn.Linear(dim, h)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(h, dim)
        self.drop = nn.Dropout(drop)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        y = self.drop(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.drop(y)
        y = self.fc2(y)
        y = self.drop(y)
        return self.norm(y)


class LinearAttention2(nn.Module):
    """Linear attention WITHOUT softmax (ELU+1 feature map).

    Dinomaly's key observation: attention that "naturally cannot focus" cannot
    copy an anomalous input, which is exactly what a reconstruction-based
    anomaly detector needs. Complexity is O(N d^2) instead of O(N^2 d).
    """

    def __init__(self, dim: int, heads: int = 8):
        super().__init__()
        assert dim % heads == 0
        self.dim, self.heads, self.d = dim, heads, dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        h, d = self.heads, self.d
        qkv = self.qkv(x).reshape(B, N, 3, h, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        kv = torch.einsum("bhsd,bhse->bhde", k, v)
        z = 1.0 / (torch.einsum("bhsd,bhd->bhs", q, k.sum(dim=-2)) + 1e-6)
        out = torch.einsum("bhde,bhsd,bhs->bhse", kv, q, z)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)

    def macs(self, x, out) -> int:
        B, N, C = x.shape
        return 2 * B * self.heads * N * self.d * self.d + B * self.heads * N * self.d


class VitBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 2.0, drop: float = 0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = LinearAttention2(dim, heads)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
            nn.Dropout(drop), nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


class FeatureJitter(nn.Module):
    """Dinomaly's feature-space noise (train only)."""

    def __init__(self, scale: float = 0.3):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        if self.training and self.scale > 0:
            norm = x.norm(dim=-1, keepdim=True) / x.shape[-1]
            x = x + torch.randn_like(x) * norm * self.scale
        return x


class ReconstructionResidual(nn.Module):
    """Frozen target projection + noisy bottleneck + linear-attention decoder."""

    def __init__(self, cfg, c5: int):
        super().__init__()
        n = cfg.normality
        d = int(n.recon_dim)
        # The "foundation feature" projector is FROZEN (Dinomaly's frozen
        # encoder). Stage 0 (tools/pretrain_dino.py, docs/DINO_PRETRAIN.md)
        # can first align it to a frozen DINOv2 teacher, then re-freeze it.
        self.target_proj = nn.Conv2d(c5, d, 1, bias=False)
        nn.init.orthogonal_(self.target_proj.weight.view(d, c5))
        self.target_proj.requires_grad_(False)

        self.jitter = FeatureJitter(n.jitter)
        self.bottleneck = NoisyBottleneck(d, n.recon_ratio, n.dropout)
        self.blocks = nn.ModuleList([
            VitBlock(d, int(n.recon_heads), n.recon_mlp_ratio, n.dropout * 0.5)
            for _ in range(int(n.recon_blocks))])
        self.out_norm = nn.LayerNorm(d)
        self.out_proj = nn.Linear(d, d)

        self.register_buffer("rec_tau", torch.tensor(float(n.rec_tau)))
        self.register_buffer("rec_slope", torch.tensor(float(n.rec_slope)))

    def forward(self, p5: torch.Tensor, return_features: bool = False):
        f = F.normalize(self.target_proj(p5).detach(), dim=1)   # frozen, clean target
        B, D, H, W = f.shape
        tgt = f.flatten(2).transpose(1, 2)                       # (B, N, D)

        x = self.jitter(tgt)
        x = self.bottleneck(x)
        for blk in self.blocks:
            x = blk(x)
        rec = F.normalize(self.out_proj(self.out_norm(x)), dim=-1)

        cos = (tgt * rec).sum(-1)                                # (B, N)
        res = (1.0 - cos).reshape(B, 1, H, W)
        a = torch.sigmoid((res - self.rec_tau) * self.rec_slope)
        if return_features:
            return a, res, tgt, rec
        return a

    def trainable_parameters(self):
        for name, p in self.named_parameters():
            if not name.startswith("target_proj"):
                yield p


# ======================================================================================
# Fusion of the two normality signals
# ======================================================================================
class AnomalyFusion(nn.Module):
    """Combine the two calibrated anomaly probabilities.

    ``fixed``  — ``alpha * a_nn + beta * a_rec`` (linear, calibration preserving).
    ``gate``   — a learnable 1x1 conv, but applied in **logit space**:
                 ``sigmoid(w1 * logit(a_nn) + w2 * logit(a_rec) + b)``.

    Why logit space matters
    -----------------------
    ``a_nn`` and ``a_rec`` are already probabilities produced by *calibrated*
    sigmoids (per-prototype ``mu/sigma/tau`` and ``rec_tau``), and the
    deployments compare the fused map against ``fusion.anomaly_thresh=0.5``.
    The old ``gate`` fed the probabilities into another conv + sigmoid, i.e. a
    second squashing.  Because ``sigmoid(0) == 0.5``, *every* non-zero input
    mapped above the 0.5 threshold (0.01 -> 0.5025, 0.10 -> 0.525, the
    calibrated 0.5 -> 0.6225), so the threshold stopped meaning "beyond the
    calibrated tau" and the connected-component safety net saw an almost
    fully-anomalous frame.

    Blending the logits instead keeps the calibration point: with
    ``w1 + w2 == 1`` and ``b == 0`` (the initialisation) the mapping is the
    identity for ``a_nn == a_rec``, so ``a == tau`` still fuses to exactly
    ``tau`` and the whole downstream threshold chain stays valid.
    """

    EPS = 1e-4

    def __init__(self, mode: str = "gate", alpha: float = 0.6, beta: float = 0.4):
        super().__init__()
        self.mode = mode
        if mode == "gate":
            self.gate = nn.Conv2d(2, 1, 1)
            with torch.no_grad():
                self.gate.weight[0, 0, 0, 0] = alpha
                self.gate.weight[0, 1, 0, 0] = beta
                self.gate.bias.fill_(0.0)
        else:
            self.register_buffer("w", torch.tensor([alpha, beta]).reshape(1, 2, 1, 1))

    @classmethod
    def _logit(cls, p: torch.Tensor) -> torch.Tensor:
        p = p.clamp(cls.EPS, 1.0 - cls.EPS)
        return torch.log(p) - torch.log1p(-p)

    def forward(self, a_nn: torch.Tensor, a_rec: torch.Tensor) -> torch.Tensor:
        if self.mode == "gate":
            z = torch.cat([self._logit(a_nn), self._logit(a_rec)], 1)
            return torch.sigmoid(self.gate(z))
        x = torch.cat([a_nn, a_rec], 1)
        return (x * self.w).sum(1, keepdim=True)


class NormalityBranch(nn.Module):
    def __init__(self, cfg, channels):
        super().__init__()
        n = cfg.normality
        self.grid = int(n.grid)
        self.out_grid = int(n.out_grid)
        assert self.out_grid % self.grid == 0, "out_grid must be an integer multiple of grid"
        self.up_factor = self.out_grid // self.grid
        self.patch = PatchFeatureHead(cfg, channels)
        self.bank = FabricPrototypeBank(int(n.patch_dim), int(n.n_prototypes),
                                        int(n.n_centroids), int(n.seed))
        self.recon = ReconstructionResidual(cfg, channels[3])
        self.fusion = AnomalyFusion(n.fusion, float(n.alpha), float(n.beta))
        self.smooth = GaussianSmooth(k=7, sigma=2.0)

    def forward(self, feats, proto_id=0, return_parts: bool = False):
        q = self.patch(feats)                                    # (B, D, g, g)
        a_nn = self.bank.score(q, proto_id)                      # (B, 1, g, g)
        a_rec, res, tgt, rec = self.recon(feats["n5"], return_features=True)
        a_rec = F.interpolate(a_rec, size=(self.grid, self.grid),
                              mode="bilinear", align_corners=False)
        a = self.fusion(a_nn, a_rec)
        a = self.smooth(a)
        # integer scale_factor -> ONNX Resize with CONSTANT scales, so the
        # exported anomaly tensor has a fully static shape for TensorRT.
        a = F.interpolate(a, scale_factor=self.up_factor,
                          mode="bilinear", align_corners=False)
        out = {"anomaly": a, "q": q, "a_nn": a_nn, "a_rec": a_rec}
        if return_parts:
            out.update({"residual": res, "target": tgt, "reconstructed": rec})
        return out

    def normality_parameters(self):
        """Parameters trained in stage B (normal images only)."""
        yield from self.patch.parameters()
        yield from self.recon.trainable_parameters()
        if self.fusion.mode == "gate":
            yield from self.fusion.parameters()
