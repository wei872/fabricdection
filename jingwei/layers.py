"""Reusable convolutional blocks for the shared backbone / neck.

Design notes
------------
* All stride-4 work uses **depthwise separable** convolutions. At 1280x1280 a
  dense 3x3 conv on a 320x320 map is expensive, so we keep the widened early
  stage (YOLO27's "stronger small-object detection") but make it cheap.
* ``AxialAttention`` is the only attention in the detector. It runs on the 40x40
  stride-32 map and factorises full attention into row + column attention,
  O(HW(H+W)C) instead of O((HW)^2 C). Row attention is exactly what long weft
  defects (断纬 / 纬斜) need.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2 if p is None else p, groups=g, bias=bias)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(nn.Module):
    """3x3 depthwise + 1x1 pointwise, BN + SiLU."""

    def __init__(self, c1, c2, k=3, s=1, act=True):
        super().__init__()
        self.dw = nn.Conv2d(c1, c1, k, s, k // 2, groups=c1, bias=False)
        self.pw = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


def make_downsample(c1, c2, k=3, s=1, act=True, dw=True):
    """Switchable depthwise-separable / dense conv.

    On GPUs **without Tensor Cores** (e.g. GTX 1660 SUPER / TU116) depthwise
    convolutions can be memory-bound and poorly served by cuDNN's Winograd
    path. Setting ``model.dwconv: false`` in the config swaps them for plain
    dense convs for an easy A/B measurement on the target device.
    """
    return DWConv(c1, c2, k, s, act=act) if dw else ConvBNAct(c1, c2, k, s, act=act)


class Bottleneck(nn.Module):
    def __init__(self, c, shortcut=True, e=1.0, dw=True):
        super().__init__()
        h = max(8, int(c * e))
        self.cv1 = ConvBNAct(c, h, 1)
        self.cv2 = DWConv(h, c, 3) if dw else ConvBNAct(h, c, 3)
        self.add = shortcut

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class CSPBlock(nn.Module):
    """C2f / C3k2 style cross-stage-partial block."""

    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5, dw=True):
        super().__init__()
        self.c = max(8, int(c2 * e))
        self.cv1 = ConvBNAct(c1, 2 * self.c, 1)
        self.cv2 = ConvBNAct((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, shortcut, e=1.0, dw=dw) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """Fast spatial pyramid pooling (YOLO lineage)."""

    def __init__(self, c1, c2, k=5):
        super().__init__()
        h = c1 // 2
        self.cv1 = ConvBNAct(c1, h, 1)
        self.cv2 = ConvBNAct(h * 4, c2, 1)
        self.pool = nn.MaxPool2d(k, 1, k // 2)

    def forward(self, x):
        y = self.cv1(x)
        y1 = self.pool(y)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([y, y1, y2, y3], 1))


class LocalContrastNorm(nn.Module):
    """Reflectance / illumination normalisation (fabric-type robustness, layer 1).

    Removes slow photometric variation (反光, 光源不均) while keeping the
    high-frequency weave structure::

        mu    = G(x)                      (two stacked box filters ~ Gaussian)
        sigma = sqrt(G((x - mu)^2))
        y     = (x - mu) / (sigma + eps)

    Implemented with ``avg_pool2d`` only, so it folds cleanly into TensorRT and
    costs ~2 memory-bound passes instead of any learned parameters.
    """

    def __init__(self, k: int = 15, eps: float = 1e-3, learnable: bool = True, gain: float = 0.25):
        super().__init__()
        self.k, self.eps, self.gain = k, eps, gain
        if learnable:
            self.gamma = nn.Parameter(torch.ones(1, 3, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, 3, 1, 1))
        else:
            self.register_buffer("gamma", torch.ones(1, 3, 1, 1))
            self.register_buffer("beta", torch.zeros(1, 3, 1, 1))

    def _box(self, x):
        """Separable box filter: two 1-D passes instead of one k x k window.

        A box kernel is exactly separable, and with ``count_include_pad=False``
        the 2-D result divides by the number of *valid* cells (n_rows*n_cols),
        which is precisely what the two 1-D passes produce -- so this is
        numerically identical (max |diff| ~ 2e-7 in float32) while costing
        ``2k`` instead of ``k*k`` multiply-adds per pixel.

        At 1280x1280 the old k=15 window did 4 x 225 = 900 ops/pixel/channel
        (~4.4 GMAC for the whole layer, ~26% of the network's 16.6 GMACs) --
        the profiler does not count pooling, so it was invisible in the MACs
        table. The separable form needs ~120 ops/pixel/channel.
        """
        k = self.k
        y = F.avg_pool2d(x, (1, k), 1, (0, k // 2), count_include_pad=False)
        return F.avg_pool2d(y, (k, 1), 1, (k // 2, 0), count_include_pad=False)

    def forward(self, x):
        mu = self._box(self._box(x))
        xc = x - mu
        sigma = self._box(self._box(xc * xc)).clamp_min(0).sqrt() + self.eps
        y = xc / sigma
        return (y * self.gamma + self.beta) * self.gain


class AxialAttention(nn.Module):
    """Row + column self-attention on a low-resolution feature map."""

    def __init__(self, dim: int, heads: int = 8):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.dim, self.heads = dim, heads
        self.d = dim // heads
        self.scale = self.d ** -0.5
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.bn = nn.BatchNorm2d(dim)

    @staticmethod
    def _attend(q, k, v, scale):
        # q/k/v: (batch, L, d) -> (batch, L, d)
        a = torch.bmm(q, k.transpose(1, 2)) * scale
        a = a.softmax(dim=-1)
        return torch.bmm(a, v)

    def forward(self, x):
        B, C, H, W = x.shape
        h, d = self.heads, self.d
        qkv = self.qkv(x).reshape(B, 3, h, d, H, W).permute(1, 0, 2, 4, 5, 3)
        q, k, v = (t.contiguous() for t in qkv.unbind(0))  # (B, h, H, W, d)

        # --- row attention: sequence = W ---
        q_r = q.reshape(B * h * H, W, d)
        k_r = k.reshape(B * h * H, W, d)
        v_r = v.reshape(B * h * H, W, d)
        o_r = self._attend(q_r, k_r, v_r, self.scale).reshape(B, h, H, W, d)

        # --- column attention: sequence = H ---
        q_c = q.permute(0, 1, 3, 2, 4).contiguous().reshape(B * h * W, H, d)
        k_c = k.permute(0, 1, 3, 2, 4).contiguous().reshape(B * h * W, H, d)
        v_c = v.permute(0, 1, 3, 2, 4).contiguous().reshape(B * h * W, H, d)
        o_c = self._attend(q_c, k_c, v_c, self.scale)
        o_c = o_c.reshape(B, h, W, H, d).permute(0, 1, 3, 2, 4)  # (B,h,H,W,d)

        y = (o_r + o_c).reshape(B, h, H, W, d).permute(0, 1, 4, 2, 3).reshape(B, C, H, W)
        return x + self.bn(self.proj(y.contiguous()))

    def macs(self, x, out) -> int:
        """Multiply-accumulates of the two attention matmuls (qk + a@v)."""
        B, _, H, W = x.shape
        return 2 * B * self.heads * self.d * H * W * (H + W)
