"""Checkpoint, box geometry and class-aware NMS utilities."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def torch_load(file, map_location="cpu", **kwargs):
    """Load tensors / primitive checkpoint state without unsafe pickle fallback.

    Both official backbone state dicts and this project's detector checkpoints
    use weights_only-safe data. Arbitrary Python-object checkpoints are not
    part of the supported format.
    """
    kwargs.setdefault("weights_only", True)
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
# Parameter count
# --------------------------------------------------------------------------------------
def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters()
               if (p.requires_grad or not trainable_only))


@torch.no_grad()
def classwise_nms(boxes, scores, labels, iou_threshold=0.5):
    """Greedy class-aware NMS without a torchvision binary dependency.

    Optional class-aware detector postprocessing.
    Inputs are already bounded by detector top_k. Returns kept tensor indices.
    """
    order = scores.argsort(descending=True)
    keep = []
    while order.numel():
        i = order[0]
        keep.append(i)
        rest = order[1:]
        overlap = box_iou(boxes[i].unsqueeze(0), boxes[rest])[0]
        suppress = (overlap > iou_threshold) & (labels[rest] == labels[i])
        order = rest[~suppress]
    return torch.stack(keep) if keep else labels.new_empty((0,))
