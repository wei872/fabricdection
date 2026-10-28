"""Local + global attention foreground / background context.

Problem this module solves
--------------------------
On a loom the fabric web (布幅) shifts and changes width.  A normal fold-in /
turned edge (折入) can appear at *any* image column, not at a fixed pixel band.
A purely anomaly-driven detector sees that long dark strip as a large defect and
the deterministic safety-net emits a false ``未知异常`` box.

The OLD implementation used a fixed ``edge_band_px`` strip measured from the
image border.  That is explicitly wrong for moving 布幅.  This module replaces
it with a **learned, transformer-like local + global attention context**:

* A ``[CLS]`` token + linear-attention ``VitBlock`` stack over stride-32 tokens
  gives every cell a **global view** of the whole frame: where the cloth is,
  where the background is, and how wide the web is in this particular frame.
* A depthwise-conv local branch over stride-8 features gives **local texture /
  boundary awareness** (weave, fold shadow, defect edges).
* The global token modulates the fused feature map (FiLM-style), and three 1x1
  heads output dynamic per-frame maps at the anomaly grid:

  ``defect_fg``   - learned defect foreground (weakly supervised by YOLO boxes)
  ``fabric_mask`` - soft cloth / non-cloth (background) mask
  ``fold_band``   - dynamic band where a NORMAL fold-in is plausible; it follows
                    the predicted cloth boundary, NOT fixed image-border pixels.

Effective anomaly used by decode + safety net:

    A_eff = A * fabric_mask * (1 - fold_attenuation * fold_band * (1 - defect_fg))

So a normal fold-in (low ``defect_fg``, high ``fold_band``) is attenuated,
while a real edge defect (high ``defect_fg`` or very high anomaly peak) survives.
The safety net additionally rejects background components using the dynamic
``fabric_mask``.

Training is weakly supervised, so no pixel-level cloth masks are required:
``build_fabric_context_targets`` derives a per-frame pseudo cloth mask from
brightness-normalised weave energy and a dynamic morphological boundary.  Those
targets are used ONLY in the loss; the deployed graph is all learned convs +
linear attention with static shapes (ONNX / TensorRT friendly).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .normality import LinearAttention2, VitBlock


# --------------------------------------------------------------------------------------
# Learned local + global attention context (deployed)
# --------------------------------------------------------------------------------------
class LocalGlobalForegroundAttention(nn.Module):
    """Transformer-like context head with local conv + global linear attention.

    Input dict uses ``n3`` (stride 8) and ``n5`` (stride 32) from
    :class:`jingwei.neck.DualScalePAN`.  All grid sizes are precomputed from
    the fixed deployment config, so there are no tensor-dependent control
    flows or dynamic ``Resize(size=...)`` nodes in the exported graph.
    """

    def __init__(self, cfg, channels):
        super().__init__()
        fg = cfg.foreground
        self.out_grid = int(cfg.normality.out_grid)
        self.global_grid = int(cfg.input_size[0]) // 32     # 40 @1280
        self.hidden = int(fg.get("hidden", 32))
        self.attn_dim = int(fg.get("attention_dim", 64))
        heads = int(fg.get("attention_heads", 4))
        n_blocks = max(1, int(fg.get("attention_blocks", 2)))
        if self.attn_dim % heads != 0:
            raise ValueError("foreground.attention_dim must be divisible by attention_heads")
        c3, c5 = int(channels[1]), int(channels[3])

        self.local = nn.Sequential(
            nn.Conv2d(c3, c3, 3, padding=1, groups=c3, bias=False),
            nn.GroupNorm(max(1, min(8, c3)), c3),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, self.hidden, 1),
            nn.ReLU(inplace=True),
        )
        self.token_proj = nn.Conv2d(c3 + c5, self.attn_dim, 1)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.attn_dim) * 0.02)
        self.blocks = nn.ModuleList([
            VitBlock(self.attn_dim, heads, mlp_ratio=2.0, drop=0.0)
            for _ in range(n_blocks)
        ])
        self.global_proj = nn.Linear(self.attn_dim, self.hidden)
        self.global_head = nn.Linear(self.attn_dim, 1)       # cloth / background prior

        self.fuse = nn.Sequential(
            nn.Conv2d(self.hidden + self.attn_dim, self.hidden, 3, padding=1),
            nn.GroupNorm(max(1, min(8, self.hidden)), self.hidden),
            nn.ReLU(inplace=True),
        )
        self.fg_head = nn.Conv2d(self.hidden, 1, 1)
        self.mask_head = nn.Conv2d(self.hidden, 1, 1)
        self.fold_head = nn.Conv2d(self.hidden, 1, 1)
        # Safe initial state: no defect foreground, cloth mostly present,
        # fold band sparse.  Training sharpens all three from data.
        nn.init.constant_(self.fg_head.bias, -3.0)
        nn.init.constant_(self.mask_head.bias, 2.0)
        nn.init.constant_(self.fold_head.bias, -2.0)

    def forward(self, feats: dict) -> dict:
        n3, n5 = feats["n3"], feats["n5"]
        B = n3.shape[0]
        G, g = self.out_grid, self.global_grid

        # ---- local branch: high-res texture / boundary detail -----------------
        local = F.interpolate(n3, size=(G, G), mode="bilinear", align_corners=False)
        local = self.local(local)                              # (B, hidden, G, G)

        # ---- global branch: tokens over the WHOLE frame ----------------------
        n3g = F.interpolate(n3, size=(g, g), mode="bilinear", align_corners=False)
        n5g = F.interpolate(n5, size=(g, g), mode="bilinear", align_corners=False)
        tok = self.token_proj(torch.cat([n3g, n5g], dim=1))  # (B, D, g, g)
        tok = tok.flatten(2).transpose(1, 2)                  # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)
        tok = torch.cat([cls, tok], dim=1)                    # (B, N+1, D)
        for blk in self.blocks:
            tok = blk(tok)                                    # LinearAttention2
        cls_out = tok[:, 0]                                   # (B, D)
        spatial = tok[:, 1:].transpose(1, 2).reshape(B, self.attn_dim, g, g)

        global_up = F.interpolate(spatial, size=(G, G), mode="bilinear",
                                  align_corners=False)
        global_ctx = torch.tanh(self.global_proj(cls_out)).view(B, self.hidden, 1, 1)
        fused = self.fuse(torch.cat([local, global_up], dim=1))
        fused = fused * (1.0 + 0.5 * global_ctx)              # FiLM-style global prior

        defect_fg = torch.sigmoid(self.fg_head(fused))
        fabric_mask = torch.sigmoid(self.mask_head(fused))
        fold_band = torch.sigmoid(self.fold_head(fused))
        context = torch.cat([defect_fg, fabric_mask, fold_band], dim=1)
        global_logit = self.global_head(cls_out)              # (B, 1)
        return {
            "context": context,
            "defect_fg": defect_fg,
            "fabric_mask": fabric_mask,
            "fold_band": fold_band,
            "global_logit": global_logit,
        }


class FabricContext(nn.Module):
    """Deployed foreground context + anomaly gating.

    ``context_only`` / ``forward`` keep the old call signatures so
    :class:`jingwei.model.JingWei`, ``tools/calibrate_fusion.py`` and ONNX
    export do not change.  Internally everything now comes from
    :class:`LocalGlobalForegroundAttention`; the RGB image ``x`` is ignored at
    inference (it is only used to build weak training targets).
    """

    def __init__(self, cfg, channels):
        super().__init__()
        fg = cfg.foreground
        self.out_grid = int(cfg.normality.out_grid)
        self.fold_attenuation = float(
            fg.get("fold_attenuation", fg.get("edge_attenuation", 0.80)))
        self.texture_thresh = float(fg.get("texture_thresh", 0.20))
        self.slope = float(fg.get("slope", 30.0))
        self.boundary_kernel = int(fg.get("boundary_kernel", 5))
        self.pos_weight = float(fg.get("pos_weight", 5.0))
        self.fabric_weight = float(fg.get("fabric_loss_weight", 1.0))
        self.fold_weight = float(fg.get("fold_loss_weight", 0.5))
        self.global_weight = float(fg.get("global_loss_weight", 0.2))
        self.attention = LocalGlobalForegroundAttention(cfg, channels)

    def forward_parts(self, feats: dict) -> dict:
        return self.attention(feats)

    def context_only(self, x, feats):
        parts = self.forward_parts(feats)
        return parts["context"], parts["defect_fg"]

    def forward(self, x, feats, anomaly):
        parts = self.forward_parts(feats)
        context = parts["context"]
        defect_fg, fabric_mask, fold_band = (
            parts["defect_fg"], parts["fabric_mask"], parts["fold_band"])
        eff = anomaly * fabric_mask * (
            1.0 - self.fold_attenuation * fold_band * (1.0 - defect_fg))
        return eff.clamp(0.0, 1.0), context, defect_fg

    # ---- training-only helpers -------------------------------------------------
    def build_targets(self, x, defect_target):
        return build_fabric_context_targets(
            x, self.out_grid, defect_target,
            texture_thresh=self.texture_thresh, slope=self.slope,
            boundary_kernel=self.boundary_kernel)

    def loss(self, parts, tgt_fg, tgt_fabric, tgt_fold, tgt_global):
        return foreground_context_loss(
            parts, tgt_fg, tgt_fabric, tgt_fold, tgt_global,
            pos_weight=self.pos_weight, fabric_weight=self.fabric_weight,
            fold_weight=self.fold_weight, global_weight=self.global_weight)


# --------------------------------------------------------------------------------------
# Weak box supervision for the learned defect-foreground head
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


# --------------------------------------------------------------------------------------
# Dynamic pseudo targets for fabric_mask / fold_band / global cloth prior
# --------------------------------------------------------------------------------------
@torch.no_grad()
def build_fabric_context_targets(x: torch.Tensor, out_grid: int,
                                 defect_target: torch.Tensor | None = None,
                                 texture_thresh: float = 0.20,
                                 slope: float = 30.0,
                                 boundary_kernel: int = 5,
                                 pool_size: int = 320):
    """Per-frame WEAK targets; NOT part of the deployed graph.

    The cloth mask is derived from brightness-normalised local weave energy,
    so it follows 布幅变化 and is not tied to any fixed image border.  The fold
    band is the morphological boundary of THAT dynamic mask, again following
    the cloth edge wherever it is.  ``boundary_kernel`` only smooths the
    training label in grid cells; the deployed ``fold_band`` is learned by the
    attention head.
    """
    G = int(out_grid)
    B = x.shape[0]
    rgb = F.interpolate(x.detach().float(), size=(pool_size, pool_size),
                        mode="bilinear", align_corners=False)
    gray = (0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3])
    low = F.avg_pool2d(gray, kernel_size=9, stride=1, padding=4,
                       count_include_pad=False)
    # Brightness-normalised high-pass energy: a dark fold-in shadow keeps the
    # same relative weave contrast and must NOT be classified as background.
    energy = ((gray - low) ** 2) / (low ** 2 + 0.02)
    e = F.adaptive_avg_pool2d(energy, (G, G))
    mu = e.mean(dim=(2, 3), keepdim=True)
    sigma = e.std(dim=(2, 3), keepdim=True, unbiased=False)
    e_norm = e / (mu + 2.0 * sigma + 1e-6)
    sm = F.avg_pool2d(e_norm, kernel_size=7, stride=1, padding=3,
                      count_include_pad=False)
    fabric_target = torch.sigmoid((sm - texture_thresh) * slope)

    k = max(3, int(boundary_kernel) | 1)                  # odd kernel
    dil = F.max_pool2d(fabric_target, kernel_size=k, stride=1, padding=k // 2)
    ero = -F.max_pool2d(-fabric_target, kernel_size=k, stride=1, padding=k // 2)
    boundary = (dil - ero).clamp(0.0, 1.0) * fabric_target   # inner cloth side
    if defect_target is not None:
        boundary = boundary * (1.0 - defect_target.clamp(0.0, 1.0))
    fold_target = boundary
    global_target = fabric_target.mean(dim=(2, 3), keepdim=True)
    return (fabric_target.contiguous(), fold_target.contiguous(),
            global_target.contiguous())


# --------------------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------------------
def _bce_dice(pred: torch.Tensor, target: torch.Tensor,
              pos_weight: float = 1.0) -> torch.Tensor:
    # BCE on probabilities is not autocast-safe; do this small map in fp32.
    pred = pred.float().clamp(1e-6, 1.0 - 1e-6)
    target = target.float()
    w = 1.0 + (float(pos_weight) - 1.0) * target
    with torch.amp.autocast(device_type=pred.device.type, enabled=False):
        bce = F.binary_cross_entropy(pred, target, weight=w)
    inter = (pred * target).sum()
    dice = 1.0 - (2.0 * inter + 1.0) / (pred.sum() + target.sum() + 1.0)
    return bce + 0.5 * dice


def foreground_loss(pred: torch.Tensor, target: torch.Tensor,
                    pos_weight: float = 5.0) -> torch.Tensor:
    """Backward-compatible single-map BCE+Dice loss."""
    return _bce_dice(pred, target, pos_weight)


def foreground_context_loss(parts: dict, tgt_fg: torch.Tensor,
                            tgt_fabric: torch.Tensor, tgt_fold: torch.Tensor,
                            tgt_global: torch.Tensor,
                            pos_weight: float = 5.0,
                            fabric_weight: float = 1.0,
                            fold_weight: float = 0.5,
                            global_weight: float = 0.2):
    """Joint weak supervision for the three attention heads + global prior."""
    defect_fg = parts["defect_fg"]
    fabric_mask = parts["fabric_mask"]
    fold_band = parts["fold_band"]
    global_logit = parts["global_logit"]

    l_defect = _bce_dice(defect_fg, tgt_fg, pos_weight)
    l_fabric = _bce_dice(fabric_mask, tgt_fabric, 1.0)
    l_fold = _bce_dice(fold_band, tgt_fold, 1.0)
    g = global_logit.view(-1, 1)
    gt = tgt_global.detach().view(-1, 1).clamp(0.0, 1.0)
    l_global = F.binary_cross_entropy_with_logits(g, gt)

    total = (l_defect + float(fabric_weight) * l_fabric +
             float(fold_weight) * l_fold + float(global_weight) * l_global)
    sub = {"defect": l_defect, "fabric": l_fabric,
           "fold": l_fold, "global": l_global}
    return total, sub


# Keep the public symbol imported by older notebooks / scripts.  It is a thin
# alias of the attention module and is NOT a fixed edge-band estimator.
DefectForegroundHead = LocalGlobalForegroundAttention
