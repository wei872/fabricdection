"""Convolutional adapters and PAN/head building blocks."""
from __future__ import annotations

import torch
import torch.nn as nn


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
    """Depthwise-separable or dense downsampling used by the PAN neck."""
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
