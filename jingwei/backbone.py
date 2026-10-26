"""Shared convolutional backbone: WeaveCSP.

Input 1280x1280 -> D(stride4, 320), P3(stride8, 160), P4(stride16, 80),
P5(stride32, 40). Only ONE backbone is executed at inference; both the
supervised head and the normality branch reuse its features. This is the single
most important real-time decision in the design.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .layers import (AxialAttention, CSPBlock, ConvBNAct,
                     LocalContrastNorm, SPPF, make_downsample)


class WeaveCSP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg.model
        c0 = int(m.stem_ch * m.width)          # stride-2 channels
        cd = int(m.detail_ch * m.width)        # stride-4 detail map
        c3 = int(m.p3_ch * m.width)
        c4 = int(m.p4_ch * m.width)
        c5 = int(m.p5_ch * m.width)
        n3, n4, n5 = (max(1, round(n * m.depth)) for n in m.num_blocks)
        # model.dwconv is optional so existing n/s configs keep behaving as before
        dw = bool(m.get("dwconv", True)) if hasattr(m, "get") else True
        Conv = lambda c1, c2, k, s, act=True: make_downsample(c1, c2, k, s, act, dw)

        self.norm = LocalContrastNorm(k=m.lcn_k) if m.reflection_norm else nn.Identity()

        # --- stem: dense conv only at 640 (cheap), cheap conv at 320 (widened) ---
        self.stem0 = ConvBNAct(3, c0, 3, 2)
        self.stem1 = Conv(c0, cd, 3, 2)

        # --- stage 1 (stride 8) + detail injection from the stride-4 map ---
        self.stage1_down = Conv(cd, c3, 3, 2)
        self.detail_down = Conv(cd, cd, 3, 2, act=False)   # 320 -> 160
        self.detail_fuse = ConvBNAct(c3 + cd, c3, 1)
        self.stage1 = CSPBlock(c3, c3, n3, dw=dw)

        # --- stage 2 (stride 16) ---
        self.stage2_down = Conv(c3, c4, 3, 2)
        self.stage2 = CSPBlock(c4, c4, n4, dw=dw)

        # --- stage 3 (stride 32) with global context ---
        self.stage3_down = Conv(c4, c5, 3, 2)
        self.stage3 = CSPBlock(c5, c5, n5, dw=dw)
        self.sppf = SPPF(c5, c5)
        self.attn = AxialAttention(c5, heads=m.attn_heads) if m.attn_p5 else nn.Identity()

        self.out_channels = (cd, c3, c4, c5)
        self.strides = (4, 8, 16, 32)

    def forward(self, x):
        x = self.norm(x)

        d = self.stem1(self.stem0(x))            # (B, cd, 320, 320)

        p3 = self.stage1_down(d)                 # (B, c3, 160, 160)
        det = self.detail_down(d)                # (B, cd, 160, 160)
        p3 = self.stage1(self.detail_fuse(torch.cat([p3, det], 1)))

        p4 = self.stage2(self.stage2_down(p3))   # (B, c4, 80, 80)
        p5 = self.stage3(self.stage3_down(p4))   # (B, c5, 40, 40)
        p5 = self.attn(self.sppf(p5))

        return {"d": d, "p3": p3, "p4": p4, "p5": p5}
