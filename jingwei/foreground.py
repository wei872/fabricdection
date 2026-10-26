"""Fabric foreground / background separation + edge fold-in suppression.

Problem this module solves
--------------------------
On a loom the fabric web (布幅) shifts and changes width.  A normal fold-in /
turned edge (折入) at the left/right border creates a long, dark, high-contrast
strip.  A purely anomaly-driven detector sees that strip as a large defect and
the deterministic safety-net emits a false ``未知异常`` box.

We therefore give the model an explicit **foreground context**:

* ``fabric_mask``  — soft, per-frame estimate of the valid cloth region
  (background / non-fabric = 0).  Estimated from weave texture energy, so it
  follows 布幅变化 without any fixed ROI.
* ``edge_band``    — soft band just *inside* the cloth boundary, where normal
  fold-in / curling / shadow lives.
* ``defect_fg``    — a small learned head supervised by the YOLO defect boxes.
  It learns "defect foreground"; normal fold-in has no boxes, so it stays low.

Effective anomaly used by decode + safety net:

    A_eff = A * fabric_mask * (1 - edge_attenuation * edge_band * (1 - defect_fg))

So a normal fold-in (low ``defect_fg``, inside ``edge_band``) is attenuated,
while a real edge defect (high ``defect_fg`` or very high anomaly peak) survives.
The safety net additionally skips background components and fold-like
edge components unless the learned foreground or the anomaly peak overrides it.

All ops are static-shape / ONNX-friendly (avg_pool, cumsum, sigmoid, interpolate).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# Deterministic, label-free cloth / boundary estimation
# --------------------------------------------------------------------------------------
def estimate_fabric_context(x: torch.Tensor, out_grid: int, band_cells: float,
                            texture_thresh: float = 0.2, slope: float = 30.0,
                            eps: float = 1e-6):
    """Return (fabric_mask, edge_band), both (B,1,G,G) in [0,1].

    ``x`` is RGB float in [0,1], shape (B,3,H,W).  No parameters, no labels:
    woven cloth has high local high-frequency energy, background / machine is
    comparatively flat.  The boundary is estimated per frame, so 布幅变化 is
    handled automatically.  ``band_cells`` is the fold-in band width expressed
    in anomaly-grid cells (precomputed from the fixed deployment input size).
    """
    g = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
    # local high-pass energy (same spirit as LocalContrastNorm)
    low = F.avg_pool2d(g, kernel_size=15, stride=1, padding=7,
                       count_include_pad=False)
    energy = (g - low) * (g - low)
    # static-shape downsample to the anomaly grid (1280/160=8, 1280/128=10)
    e = F.adaptive_avg_pool2d(energy, (out_grid, out_grid))

    col = e.mean(dim=2, keepdim=True)                 # (B,1,1,G)
    row = e.mean(dim=3, keepdim=True)                 # (B,1,G,1)
    # Smooth the 1-D profiles first: the sharp cloth/background boundary and a
    # fold-in shadow create isolated energy spikes that would otherwise drown
    # the normal weave when computing a robust scale.
    col_s = F.avg_pool2d(col, kernel_size=(1, 15), stride=1,
                         padding=(0, 7), count_include_pad=False)
    row_s = F.avg_pool2d(row, kernel_size=(15, 1), stride=1,
                         padding=(7, 0), count_include_pad=False)
    # remove the flat-background floor before normalising
    col_s = (col_s - col_s.amin(dim=-1, keepdim=True)).clamp(min=0.0)
    row_s = (row_s - row_s.amin(dim=-2, keepdim=True)).clamp(min=0.0)
    cscale = col_s.mean(dim=-1, keepdim=True) + 2.0 * col_s.std(
        dim=-1, keepdim=True, unbiased=False) + eps
    rscale = row_s.mean(dim=-2, keepdim=True) + 2.0 * row_s.std(
        dim=-2, keepdim=True, unbiased=False) + eps
    col_norm = (col_s / cscale).clamp(0.0, 1.5)
    row_norm = (row_s / rscale).clamp(0.0, 1.5)
    fabric_col = torch.sigmoid((col_norm - texture_thresh) * slope)
    fabric_row = torch.sigmoid((row_norm - texture_thresh) * slope)
    fabric_mask = fabric_col * fabric_row             # (B,1,G,G) via broadcast

    # distance (in grid cells) from the nearest boundary, *inside* the cloth.
    d_left = torch.cumsum(fabric_col, dim=-1)
    total_col = d_left[..., -1:]
    d_right = (total_col - d_left).clamp(min=0.0)
    d_col = torch.minimum(d_left, d_right)            # (B,1,1,G)
    edge_col = torch.exp(-d_col / band_cells) * fabric_col

    # 布幅变化 mainly moves the left/right selvedge; top/bottom are usually
    # continuous cloth in a web inspection camera.  Only the column boundary
    # is treated as the fold-in band, otherwise every frame's top/bottom rows
    # would be falsely marked as edge.
    e_col = edge_col.expand(-1, -1, out_grid, -1)
    edge_band = e_col * fabric_mask
    return fabric_mask.contiguous(), edge_band.contiguous()


# --------------------------------------------------------------------------------------
# Learned defect-foreground head (weakly supervised by YOLO boxes)
# --------------------------------------------------------------------------------------
class DefectForegroundHead(nn.Module):
    """Tiny conv head: n3 + n5 -> 1-channel defect-foreground probability.

    Trained with box-filled targets from ``train`` (defect images).  Normal
    images / normal fold-in have no boxes, so they are explicit negatives.
    """

    def __init__(self, c3: int, c5: int, hidden: int, out_grid: int):
        super().__init__()
        self.reduce5 = nn.Conv2d(c5, c3, 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(c3 * 2, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1))
        # start as "background everywhere"; normal fold-in must NOT look like
        # defect foreground before the head has learned from boxes.
        nn.init.constant_(self.fuse[-1].bias, -3.0)
        self.out_grid = int(out_grid)

    def forward(self, feats: dict) -> torch.Tensor:
        n3 = feats["n3"]
        n5 = F.interpolate(self.reduce5(feats["n5"]), size=n3.shape[-2:],
                           mode="nearest")
        y = self.fuse(torch.cat([n3, n5], dim=1))
        y = F.interpolate(y, size=(self.out_grid, self.out_grid),
                          mode="bilinear", align_corners=False)
        return torch.sigmoid(y)


class FabricContext(nn.Module):
    """Combine deterministic cloth/edge masks with the learned defect fg."""

    def __init__(self, cfg, channels):
        super().__init__()
        fg = cfg.foreground
        self.out_grid = int(cfg.normality.out_grid)
        self.edge_attenuation = float(fg.edge_attenuation)
        self.band_cells = max(0.5, float(fg.edge_band_px) * self.out_grid /
                              float(cfg.input_size[0]))
        self.texture_thresh = float(fg.get("texture_thresh", 0.2))
        self.slope = float(fg.get("slope", 30.0))
        self.defect_head = DefectForegroundHead(
            channels[1], channels[3], int(fg.hidden), self.out_grid)

    def context_only(self, x, feats):
        fabric_mask, edge_band = estimate_fabric_context(
            x, self.out_grid, self.band_cells,
            texture_thresh=self.texture_thresh, slope=self.slope)
        defect_fg = self.defect_head(feats)
        context = torch.cat([defect_fg, fabric_mask, edge_band], dim=1)
        return context, defect_fg

    def forward(self, x, feats, anomaly):
        context, defect_fg = self.context_only(x, feats)
        fabric_mask, edge_band = context[:, 1:2], context[:, 2:3]
        eff = anomaly * fabric_mask * (
            1.0 - self.edge_attenuation * edge_band * (1.0 - defect_fg))
        return eff.clamp(0.0, 1.0), context, defect_fg


# --------------------------------------------------------------------------------------
# Weak box supervision for the learned foreground head (training only)
# --------------------------------------------------------------------------------------
def build_foreground_target(targets, img_size, grid, device, dtype,
                            shrink: float = 0.08):
    """(B,1,G,G) soft target: 1 inside (slightly eroded) GT defect boxes."""
    b = len(targets)
    tgt = torch.zeros(b, 1, grid, grid, device=device, dtype=dtype)
    H, W = img_size
    for i, t in enumerate(targets):
        boxes = t["boxes"]
        if boxes is None or boxes.numel() == 0:
            continue
        boxes = boxes.to(device=device, dtype=torch.float32)
        for box in boxes:
            x0, y0, x1, y1 = [float(v) for v in box.tolist()]
            dx, dy = (x1 - x0) * shrink, (y1 - y0) * shrink
            x0, x1, y0, y1 = x0 + dx, x1 - dx, y0 + dy, y1 - dy
            gx0 = max(0, min(grid - 1, int(math.floor(x0 / W * grid))))
            gy0 = max(0, min(grid - 1, int(math.floor(y0 / H * grid))))
            gx1 = max(gx0 + 1, min(grid, int(math.ceil(x1 / W * grid))))
            gy1 = max(gy0 + 1, min(grid, int(math.ceil(y1 / H * grid))))
            tgt[i, 0, gy0:gy1, gx0:gx1] = 1.0
    return tgt


def foreground_loss(pred: torch.Tensor, target: torch.Tensor,
                    pos_weight: float = 5.0) -> torch.Tensor:
    """BCE (positive-weighted, boxes are sparse) + soft Dice.

    ``pred`` is already a sigmoid probability, so use functional BCE with an
    explicit per-pixel weight instead of ``binary_cross_entropy_with_logits``.
    """
    w = 1.0 + (float(pos_weight) - 1.0) * target
    bce = F.binary_cross_entropy(pred, target, weight=w)
    inter = (pred * target).sum()
    dice = 1.0 - (2.0 * inter + 1.0) / (pred.sum() + target.sum() + 1.0)
    return bce + 0.5 * dice
