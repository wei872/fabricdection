"""Shared tensor / box utilities and a small dependency-free MACs profiler."""
from __future__ import annotations

import math
import pickle
import warnings
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def torch_load(file, map_location="cpu", **kwargs):
    """Load a checkpoint with ``weights_only=True`` by default (torch >= 2.0).

    Silences the PyTorch ``FutureWarning`` about ``weights_only=False`` for our
    own checkpoints, which contain only tensors / primitives / containers.
    Third-party DINOv1 weights sometimes store an ``argparse.Namespace``; for
    those we fall back to ``weights_only=False`` with a warning — only load
    checkpoints you trust.
    """
    kwargs.setdefault("weights_only", True)
    try:
        return torch.load(file, map_location=map_location, **kwargs)
    except TypeError:
        # torch < 2.0 has no weights_only argument.
        kwargs.pop("weights_only", None)
        return torch.load(file, map_location=map_location, **kwargs)
    except (pickle.UnpicklingError, RuntimeError, AttributeError) as exc:
        kwargs.pop("weights_only", None)
        warnings.warn(
            f"torch.load(weights_only=True) failed ({exc}); "
            "retrying with weights_only=False. Only load trusted checkpoints.",
            RuntimeWarning,
        )
        return torch.load(file, map_location=map_location, **kwargs)


# --------------------------------------------------------------------------------------
# Boxes
# --------------------------------------------------------------------------------------
def boxes_from_distances(points: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
    """Decode (l,t,r,b) distances into xyxy boxes.

    Args:
        points: (N, 2) centre coordinates in pixels.
        dist:   (N, 4) non-negative distances in pixels.
    """
    x, y = points[:, 0:1], points[:, 1:2]
    l, t, r, b = dist.chunk(4, -1)
    return torch.cat([x - l, y - t, x + r, y + b], dim=-1)


def boxes_to_distances(points: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    x, y = points[:, 0:1], points[:, 1:2]
    return torch.cat([x - boxes[:, 0:1], y - boxes[:, 1:2],
                      boxes[:, 2:3] - x, boxes[:, 3:4] - y], dim=-1)


def box_iou(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Pairwise IoU between two sets of xyxy boxes. Returns (N, M)."""
    inter = (torch.min(a[:, None, 2:], b[None, :, 2:]) -
             torch.max(a[:, None, :2], b[None, :, :2])).clamp(min=0)
    inter = inter[..., 0] * inter[..., 1]
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    return inter / (area_a[:, None] + area_b[None, :] - inter + eps)


def box_ciou(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Complete-IoU for two *aligned* (N,4) sets of boxes."""
    inter = (torch.min(a[:, 2:], b[:, 2:]) - torch.max(a[:, :2], b[:, :2])).clamp(min=0)
    inter = inter[:, 0] * inter[:, 1]
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    union = area_a + area_b - inter + eps
    iou = inter / union

    cw = torch.max(a[:, 2], b[:, 2]) - torch.min(a[:, 0], b[:, 0])
    ch = torch.max(a[:, 3], b[:, 3]) - torch.min(a[:, 1], b[:, 1])
    c2 = cw ** 2 + ch ** 2 + eps
    rho2 = ((a[:, 0] + a[:, 2] - b[:, 0] - b[:, 2]) ** 2 +
            (a[:, 1] + a[:, 3] - b[:, 1] - b[:, 3]) ** 2) / 4.0
    v = (4.0 / math.pi ** 2) * torch.pow(
        torch.atan((b[:, 2] - b[:, 0]) / (b[:, 3] - b[:, 1] + eps)) -
        torch.atan((a[:, 2] - a[:, 0]) / (a[:, 3] - a[:, 1] + eps)), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    return iou - (rho2 / c2 + alpha * v)


def scale_boxes(boxes: torch.Tensor, from_wh, to_wh) -> torch.Tensor:
    """Scale xyxy boxes from one image size to another (w, h tuples)."""
    sx, sy = to_wh[0] / from_wh[0], to_wh[1] / from_wh[1]
    out = boxes.clone().float()
    out[..., 0::2] *= sx
    out[..., 1::2] *= sy
    return out


# --------------------------------------------------------------------------------------
# MACs / parameter profiler
# --------------------------------------------------------------------------------------
def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters()
               if (p.requires_grad or not trainable_only))


def profile_macs(model: nn.Module, inputs: Iterable[torch.Tensor]) -> dict:
    """Count multiply-accumulates for Conv2d / Linear plus custom-attention hooks.

    This is an *estimate*: it ignores element-wise ops, normalisation, softmax and
    pooling (all <1% of the total here). It is deliberately dependency-free so the
    design's compute budget can be reproduced without thop/fvcore/torchprofile.
    """
    inputs = tuple(inputs)
    breakdown: dict[str, int] = {}

    def conv_hook(module: nn.Conv2d, inp, out):
        # out: (B, Cout, *spatial)
        macs = out.shape[0] * out.shape[1] * math.prod(out.shape[2:]) * \
            (module.in_channels // module.groups) * math.prod(module.kernel_size)
        breakdown["conv"] = breakdown.get("conv", 0) + macs

    def linear_hook(module: nn.Linear, inp, out):
        macs = out.numel() * module.in_features
        breakdown["linear"] = breakdown.get("linear", 0) + macs

    def custom_hook(module, inp, out):
        # modules may expose macs(input, output) -> int
        breakdown["attention"] = breakdown.get("attention", 0) + module.macs(inp[0], out)

    handles = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))
        elif hasattr(m, "macs") and callable(m.macs) and not isinstance(m, (nn.Conv2d, nn.Linear)):
            handles.append(m.register_forward_hook(custom_hook))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(*inputs)
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)

    breakdown["total"] = sum(breakdown.values())
    return breakdown


def human_macs(x: float) -> str:
    if x >= 1e9:
        return f"{x / 1e9:.2f} G"
    if x >= 1e6:
        return f"{x / 1e6:.1f} M"
    return f"{x:.0f}"


# --------------------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------------------
def gaussian2d(kernel_size: int, sigma: float, device=None, dtype=None) -> torch.Tensor:
    ax = torch.arange(kernel_size, device=device, dtype=dtype or torch.float32) - (kernel_size - 1) / 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    k = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def smooth_anomaly(score: torch.Tensor, k: int = 7, sigma: float = 2.0) -> torch.Tensor:
    """Gaussian smoothing of anomaly maps (PatchCore-style reweighting).

    Functional convenience version. For export use :class:`GaussianSmooth`,
    whose kernel is a registered buffer (static ONNX shape inference).
    """
    w = gaussian2d(k, sigma, score.device, score.dtype).reshape(1, 1, k, k)
    pad = k // 2
    return F.conv2d(F.pad(score, (pad, pad, pad, pad), mode="reflect"), w, padding=0)


class GaussianSmooth(nn.Module):
    """Deployment-friendly Gaussian smoother with a constant (buffer) kernel."""

    def __init__(self, k: int = 7, sigma: float = 2.0):
        super().__init__()
        self.pad = k // 2
        self.register_buffer("weight", gaussian2d(k, sigma).reshape(1, 1, k, k))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # plain conv padding (no F.pad): ONNX shape inference cannot propagate
        # through the opset-11+ Pad node, which would leave the exported
        # anomaly tensor with unknown dimensions for TensorRT.
        return F.conv2d(x, self.weight.to(x.dtype), padding=self.pad)
