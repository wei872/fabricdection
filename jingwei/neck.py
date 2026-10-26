"""Dual-scale PAN neck.

Following YOLO27 n/s, we *fuse* on three scales (P3/P4/P5) but only *predict*
on two (P3 for tiny defects, P5 for large ones). P4 exists to carry information
between them and is never attached to a detection head; a fixed scalar keeps the
two surviving scales balanced in the head.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .layers import CSPBlock, make_downsample


class DualScalePAN(nn.Module):
    def __init__(self, cfg, channels):
        super().__init__()
        c3, c4, c5 = channels[1], channels[2], channels[3]
        m = cfg.model
        n = max(1, round(m.neck_blocks * m.depth))
        dw = bool(m.get("dwconv", True))

        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        self.reduce_p5 = nn.Conv2d(c5, c4, 1, bias=False)
        self.top4 = CSPBlock(c4 + c4, c4, n, dw=dw)
        self.reduce_p4 = nn.Conv2d(c4, c3, 1, bias=False)
        self.top3 = CSPBlock(c3 + c3, c3, n, dw=dw)

        self.down3 = make_downsample(c3, c4, 3, 2, dw=dw)
        self.bot4 = CSPBlock(c4 + c4, c4, n, dw=dw)
        self.down4 = make_downsample(c4, c5, 3, 2, dw=dw)
        self.bot5 = CSPBlock(c5 + c5, c5, n, dw=dw)

        # "fixed scaling on the fused features keeps the two scales balanced"
        self.bal3 = nn.Parameter(torch.tensor(float(m.balance_p3)))
        self.bal5 = nn.Parameter(torch.tensor(float(m.balance_p5)))

    def forward(self, feats):
        p3, p4, p5 = feats["p3"], feats["p4"], feats["p5"]

        f4 = self.top4(torch.cat([self.up(self.reduce_p5(p5)), p4], 1))
        n3 = self.top3(torch.cat([self.up(self.reduce_p4(f4)), p3], 1))

        n4 = self.bot4(torch.cat([self.down3(n3), f4], 1))
        n5 = self.bot5(torch.cat([self.down4(n4), p5], 1))

        return {"n3": self.bal3 * n3, "n4": n4, "n5": self.bal5 * n5}
