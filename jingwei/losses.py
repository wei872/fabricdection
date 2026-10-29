"""Losses: supervised dual-assignment detection and box alignment.

Detection training uses two assignments:
  - one-to-many dense assignment (balanced hard-label focal targets) supervises the
    shared cls/reg features;
  - one-to-one Hungarian matching supervises the deployed objectness branch,
    inference may additionally apply configurable class-aware NMS.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import box_ciou, box_iou, boxes_from_distances, boxes_to_distances


def _grid_points(h, w, stride, device, dtype):
    ys, xs = torch.meshgrid(torch.arange(h, device=device, dtype=dtype),
                            torch.arange(w, device=device, dtype=dtype), indexing="ij")
    pts = torch.stack([xs, ys], -1).reshape(-1, 2)
    return (pts + 0.5) * stride


@torch.no_grad()
def build_dense_targets(pred, targets, stride, nc, reg_scale, radius=1.5, matches=None):
    """FCOS-style centre sampling with an IoU-aware quality target."""
    B, _, H, W = pred["cls"].shape
    device = pred["cls"].device
    dtype = torch.float32
    HW = H * W
    pts = _grid_points(H, W, stride, device, dtype)

    t_cls = torch.zeros(B, nc, HW, device=device, dtype=dtype)
    t_reg = torch.zeros(B, 4, HW, device=device, dtype=dtype)
    t_qual = torch.zeros(B, 1, HW, device=device, dtype=dtype)
    t_fg = torch.zeros(B, 1, HW, device=device, dtype=dtype)
    pos = torch.zeros(B, 1, HW, device=device, dtype=torch.bool)

    pred_dist = pred["reg"].reshape(B, 4, HW).float() * reg_scale * stride

    for b in range(B):
        boxes, labels = targets[b]["boxes"], targets[b]["labels"]
        boxes = boxes.to(device=device, dtype=dtype)
        labels = labels.to(device=device, dtype=torch.long)
        if boxes.numel() == 0:
            continue
        G = boxes.shape[0]

        l = pts[:, 0:1] - boxes[None, :, 0]
        t = pts[:, 1:2] - boxes[None, :, 1]
        r = boxes[None, :, 2] - pts[:, 0:1]
        bo = boxes[None, :, 3] - pts[:, 1:2]
        d = torch.stack([l, t, r, bo], -1)                     # (HW, G, 4)
        inside = d.min(-1).values >= 0
        cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
        cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
        centre = ((pts[:, 0:1] - cx[None, :]).abs() <= radius * stride) & \
                 ((pts[:, 1:2] - cy[None, :]).abs() <= radius * stride)
        valid = inside & centre
        area = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
        cost = torch.where(valid, area[None, :].expand(HW, G),
                           torch.full((HW, G), float("inf"), device=device, dtype=dtype))
        min_cost, gi = cost.min(1)
        idx = torch.nonzero(min_cost < float("inf")).flatten()
        if idx.numel() == 0:
            # A nonnegative-distance head cannot exactly represent a box with
            # no in-box grid point. Never manufacture negative l/t/r/b targets.
            continue
        g = gi[idx]

        t_cls[b, labels[g], idx] = 1.0
        t_reg[b, :, idx] = d[idx, g].t()
        pd = pred_dist[b, :, idx].t()
        pb = boxes_from_distances(pts[idx], pd)
        t_qual[b, 0, idx] = box_ciou(pb, boxes[g]).clamp(0, 1)
        pos[b, 0, idx] = True

        # foreground-alignment auxiliary target: a disc around each GT centre
        near = (((pts[:, 0:1] - cx[None, :]) ** 2 +
                 (pts[:, 1:2] - cy[None, :]) ** 2) <= (2.0 * stride) ** 2).any(1)
        t_fg[b, 0, near] = 1.0

    # Shared cls/reg tensors must not receive a dense negative or a different
    # GT at a cell chosen as a deployed positive (including outside the dense
    # centre-sampling window). Override that cell consistently for both losses.
    for b, flat, g in matches or ():
        box = targets[b]["boxes"][g:g + 1].to(device=device, dtype=dtype)
        label = targets[b]["labels"][g].to(device=device, dtype=torch.long)
        t_cls[b, :, flat] = 0
        t_cls[b, label, flat] = 1
        t_reg[b, :, flat] = boxes_to_distances(pts[flat:flat + 1], box)[0]
        pb = boxes_from_distances(pts[flat:flat + 1], pred_dist[b, :, flat][None])
        t_qual[b, 0, flat] = box_ciou(pb, box)[0].clamp(0, 1)
        pos[b, 0, flat] = True
        t_fg[b, 0, flat] = 1

    return t_cls, t_reg, t_qual, t_fg, pos, pts


# --------------------------------------------------------------------------------------
# One-to-one Hungarian matching
# --------------------------------------------------------------------------------------
@torch.no_grad()
def hungarian_match(preds, targets, strides, nc, reg_scale, max_candidates=1600):
    """Geometry-feasible one-to-one matches: [(batch, flat_index, gt_index)].

    Reserve in-box candidates per GT, independent of predicted confidence.
    Invalid point/GT pairs are never positives for a nonnegative-distance head.
    Dummy columns allow unrepresentable/crowded GTs to remain unmatched rather
    than forcing an impossible pair or crashing scipy on an infeasible matrix.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    B = preds[0]["cls"].shape[0]
    device = preds[0]["cls"].device
    per_scale: list[list] = [[] for _ in preds]
    height = preds[0]["cls"].shape[-2] * strides[0]
    width = preds[0]["cls"].shape[-1] * strides[0]
    normalizer = torch.tensor([width, height, width, height], device=device, dtype=torch.float32)

    for b in range(B):
        boxes_gt = targets[b]["boxes"].to(device=device, dtype=torch.float32)
        labels = targets[b]["labels"].to(device=device, dtype=torch.long)
        if boxes_gt.numel() == 0:
            continue  # no candidate sorting / CPU transfer on pure normal images
        G = len(boxes_gt)
        centers_gt = (boxes_gt[:, :2] + boxes_gt[:, 2:]) / 2
        all_cls, all_obj, all_boxes, all_points, meta = [], [], [], [], []
        for si, p in enumerate(preds):
            _, _, H, W = p["cls"].shape
            stride = strides[si]
            pts = _grid_points(H, W, stride, device, torch.float32)
            cls = p["cls"][b].float().reshape(nc, -1).t().sigmoid()
            obj = p["obj_o2o"][b].float().reshape(-1).sigmoid()
            dist = p["reg"][b].float().reshape(4, -1).t() * reg_scale * stride
            boxes = boxes_from_distances(pts, dist)
            score = obj * cls.max(-1).values
            k = min(max(1, max_candidates // len(preds)), score.numel())
            top = torch.topk(score, k).indices

            inside = ((pts[:, None] >= boxes_gt[None, :, :2]) &
                      (pts[:, None] <= boxes_gt[None, :, 2:])).all(-1)
            center_dist = ((pts[:, None] - centers_gt[None]) / stride).square().sum(-1)
            center_dist = center_dist.masked_fill(~inside, float("inf"))
            # The budget caps confidence-selected cells, not GT coverage. At
            # least G local candidates also reduce conflicts between nearby GTs.
            nearest = center_dist.topk(min(max(9, G), len(pts)), dim=0, largest=False).indices
            reserved = nearest[torch.isfinite(center_dist.gather(0, nearest))]
            top = torch.unique(torch.cat((top, reserved)))
            all_cls.append(cls[top])
            all_obj.append(obj[top])
            all_boxes.append(boxes[top])
            all_points.append(pts[top])
            meta.extend((si, int(flat)) for flat in top.cpu().tolist())

        cls_all = torch.cat(all_cls)
        obj_all = torch.cat(all_obj)
        box_all = torch.cat(all_boxes)  # same unclipped geometry as regression loss
        points = torch.cat(all_points)
        feasible = ((points[None] >= boxes_gt[:, None, :2]) &
                    (points[None] <= boxes_gt[:, None, 2:])).all(-1)
        cls_gt = cls_all[:, labels].t()
        iou = box_iou(box_all, boxes_gt).t()
        l1 = ((box_all[None] - boxes_gt[:, None]).abs() / normalizer).mean(-1)
        cost = (-cls_gt - .5 * obj_all[None] - 2 * iou + 5 * l1).cpu().numpy()
        valid = feasible.cpu().numpy()
        if not np.isfinite(cost).all():
            raise FloatingPointError("Non-finite Hungarian matching cost")
        if not valid.any():
            continue
        # Scale all feasible costs together to [0,1] (same linear objective
        # at fixed cardinality). A dummy costs G+1, so maximize the number of
        # feasible matches BEFORE comparing classification/localization costs.
        lo, hi = cost[valid].min(), cost[valid].max()
        normalized = (cost.astype(np.float64) - lo) / max(float(hi) - float(lo), 1e-12)
        real_cost = np.where(valid, normalized, (G + 1) * (G + 2))
        padded = np.concatenate((real_cost, np.full((G, G), G + 1.)), axis=1)
        rows, cols = linear_sum_assignment(padded)
        for g, c in zip(rows, cols):
            if c < len(meta) and valid[g, c]:
                si, flat = meta[c]
                per_scale[si].append((b, flat, int(g)))
    return per_scale


def o2o_negative_count(total_cells: int, n_pos: int, ratio: float,
                       min_neg: int) -> int:
    """Hard-negative count for the deployed one-to-one objectness branch.

    ``3 * n_pos`` gives only three negatives for a scale that receives no
    match in a batch (common for P5 early in training), so that scale's
    objectness is barely constrained and the deployed branch emits duplicate
    boxes.  A floor based on the cell count fixes this without changing the
    positive matching or the one-to-many branch.
    """
    n_pos = max(0, int(n_pos))
    total_cells = max(0, int(total_cells))
    available = max(0, total_cells - n_pos)
    if available == 0:
        return 0
    floor = max(3 * max(1, n_pos),
                int(float(ratio) * total_cells),
                max(0, int(min_neg)))
    return max(1, min(available, floor))


def balanced_class_focal_loss(logits, target, class_weights, min_neg=256, neg_ratio=3.):
    """Hard-label focal classification, balanced separately for each image.

    A correct class remains positive even when the box is initially inaccurate.
    Average positives and hard negatives separately; a normal image must not
    contribute an unnormalized sum over every background cell/class.
    ``target`` is binary (B,C,N), independent of predicted box quality.
    """
    logits = logits.float()
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weighted = ce * (logits.sigmoid() - target).square() * class_weights[None, :, None]
    per_image = []
    for b in range(len(logits)):
        positive = target[b].bool()
        n_pos = int(positive.sum())
        pos_loss = weighted[b][positive].sum() / max(1, n_pos)
        negatives = weighted[b][~positive]
        k = min(negatives.numel(), max(int(min_neg), int(neg_ratio * n_pos)))
        neg_loss = negatives.topk(k).values.mean() if k else negatives.sum()
        per_image.append(pos_loss + neg_loss)
    return torch.stack(per_image).mean()


@torch.no_grad()
def build_o2o_quality_targets(pred, matches, targets, stride, reg_scale, quality_floor=.05):
    """Detached IoU targets and a SEPARATE binary assignment mask.

    A low-quality match is still a positive assignment, never a hard negative.
    Use the same unclipped box geometry as matching/regression. A small floor
    bootstraps objectness for very inaccurate positive boxes, without training
    all matched boxes to confidence one regardless of localization quality.
    """
    obj = pred["obj_o2o"].flatten(1)
    quality = torch.zeros_like(obj, dtype=torch.float32)
    positive = torch.zeros_like(obj, dtype=torch.bool)
    h, w = pred["obj_o2o"].shape[-2:]
    points = _grid_points(h, w, stride, obj.device, torch.float32)
    reg = pred["reg"].float().flatten(2)
    for b, flat, g in matches:
        distance = reg[b, :, flat] * reg_scale * stride
        predicted_box = boxes_from_distances(points[flat:flat + 1], distance[None])
        gt = targets[b]["boxes"][g:g + 1].to(device=obj.device, dtype=torch.float32)
        quality[b, flat] = box_iou(predicted_box, gt)[0, 0].clamp(min=quality_floor, max=1.)
        positive[b, flat] = True
    return quality, positive


# --------------------------------------------------------------------------------------
# Combined detection loss
# --------------------------------------------------------------------------------------
class DetectionLoss(nn.Module):
    assignment_version = "geometry_feasible_v1"
    objective_version = "balanced_cls_iou_obj_v2"

    def __init__(self, cfg):
        super().__init__()
        self.nc = int(cfg.data.nc)
        self.strides = (8, 32)
        self.reg_scale = float(cfg.model.reg_scale)
        self.w = cfg.train.loss_weights
        cw = cfg.train.get("class_weights", None)
        self.register_buffer("class_weights",
                             torch.tensor(cw, dtype=torch.float32) if cw else torch.ones(self.nc))
        self.o2o_candidates = int(cfg.train.get("o2o_candidates", 1600))
        self.o2o_neg_ratio = float(cfg.train.get("o2o_neg_ratio", 0.02))
        self.o2o_min_neg = int(cfg.train.get("o2o_min_neg", 2048))
        self.cls_min_neg = int(cfg.train.get("cls_min_neg", 256))
        self.cls_neg_ratio = float(cfg.train.get("cls_neg_ratio", 3.))
        self.o2o_quality_floor = float(cfg.train.get("o2o_quality_floor", .05))
        if self.cls_min_neg < 1 or not math.isfinite(self.cls_neg_ratio) or self.cls_neg_ratio < 0:
            raise ValueError("cls_min_neg must be positive and cls_neg_ratio finite/nonnegative")
        if not 0 <= self.o2o_quality_floor < 1:
            raise ValueError("o2o_quality_floor must be in [0,1)")

    def forward(self, preds, targets, img_size):
        device = preds[0]["cls"].device
        losses = {k: torch.zeros((), device=device) for k in
                  ("cls_o2m", "box_o2m", "obj_o2m", "cls_o2o", "box_o2o", "obj_o2o", "fg_align")}

        matches = hungarian_match(preds, targets, self.strides, self.nc,
                                  self.reg_scale, self.o2o_candidates)
        n_gt = sum(len(target["boxes"]) for target in targets)
        n_assigned = sum(len(group) for group in matches)
        self.last_assignment = {"matched_gt": n_assigned, "total_gt": n_gt,
                                "unmatched_gt": n_gt - n_assigned}

        # ---------- one-to-many dense branch ----------
        for si, p in enumerate(preds):
            stride = self.strides[si]
            t_cls, t_reg, t_qual, t_fg, pos, pts = build_dense_targets(
                p, targets, stride, self.nc, self.reg_scale, matches=matches[si])
            B, _, H, W = p["cls"].shape
            HW = H * W

            cls = p["cls"].reshape(B, self.nc, HW)
            losses["cls_o2m"] = losses["cls_o2m"] + balanced_class_focal_loss(
                cls, t_cls, self.class_weights, self.cls_min_neg, self.cls_neg_ratio)

            bce_obj = F.binary_cross_entropy_with_logits(p["obj"].reshape(B, 1, HW), t_qual,
                                                         reduction="none")
            losses["obj_o2m"] = losses["obj_o2m"] + (bce_obj * (1.0 + 3.0 * pos.float())).mean()

            # Select cells in (batch, cell, coordinate) order, matching p_sel.
            # Masking (B, 4, HW) first flattens channel-major: reshape(-1, 4)
            # would mix l/t/r/b values from DIFFERENT cells whenever >1 is positive.
            cell_pos = pos.squeeze(1)
            pred_d = (p["reg"].float() * self.reg_scale * stride).reshape(
                B, 4, HW).transpose(1, 2)[cell_pos]
            tgt_d = t_reg.transpose(1, 2)[cell_pos]
            p_sel = pts[None].expand(B, HW, 2)[cell_pos]
            if pred_d.numel():
                pb = boxes_from_distances(p_sel, pred_d)
                gb = boxes_from_distances(p_sel, tgt_d)
                losses["box_o2m"] = losses["box_o2m"] + \
                    (1.0 - box_ciou(pb, gb)).mean() + 0.1 * F.l1_loss(pred_d, tgt_d)

            if p["fg"] is not None:
                losses["fg_align"] = losses["fg_align"] + \
                    F.binary_cross_entropy_with_logits(p["fg"].reshape(B, 1, HW), t_fg)

        # ---------- one-to-one deployed branch ----------
        for si, p in enumerate(preds):
            stride = self.strides[si]
            B, _, H, W = p["cls"].shape
            HW = H * W
            obj = p["obj_o2o"].reshape(B, -1)
            t_obj, obj_positive = build_o2o_quality_targets(
                p, matches[si], targets, stride, self.reg_scale, self.o2o_quality_floor)
            bce = F.binary_cross_entropy_with_logits(obj, t_obj, reduction="none")
            n_pos = len(matches[si])
            n_neg = o2o_negative_count(B * HW, n_pos, self.o2o_neg_ratio,
                                       self.o2o_min_neg)
            pos_loss = bce[obj_positive].sum() / float(max(1, n_pos))
            if n_neg:
                neg_loss = bce[~obj_positive].topk(n_neg).values.mean()
            else:
                neg_loss = obj.new_zeros(())
            losses["obj_o2o"] = losses["obj_o2o"] + pos_loss + neg_loss

            cls = p["cls"].reshape(B, self.nc, HW)
            reg = p["reg"].float().reshape(B, 4, HW)
            grid = _grid_points(H, W, stride, device, torch.float32)
            for b, flat, g in matches[si]:
                label = targets[b]["labels"][g].to(device).long()
                onehot = F.one_hot(label, self.nc).float()
                losses["cls_o2o"] = losses["cls_o2o"] + \
                    F.binary_cross_entropy_with_logits(cls[b, :, flat], onehot)
                gt_box = targets[b]["boxes"][g].to(device=device, dtype=torch.float32)
                pd = reg[b, :, flat] * self.reg_scale * stride
                pb = boxes_from_distances(grid[flat:flat + 1], pd[None])
                losses["box_o2o"] = losses["box_o2o"] + \
                    (1.0 - box_ciou(pb, gt_box[None])).mean() + \
                    0.1 * F.l1_loss(pd, boxes_to_distances(grid[flat:flat + 1], gt_box[None])[0])
        n_match = max(1, sum(len(m) for m in matches))
        losses["cls_o2o"] = losses["cls_o2o"] / n_match
        losses["box_o2o"] = losses["box_o2o"] / n_match

        total = (self.w.cls_o2m * losses["cls_o2m"] + self.w.box_o2m * losses["box_o2m"] +
                 self.w.obj_o2m * losses["obj_o2m"] + self.w.cls_o2o * losses["cls_o2o"] +
                 self.w.box_o2o * losses["box_o2o"] + self.w.obj_o2o * losses["obj_o2o"] +
                 self.w.fg_align * losses["fg_align"])
        losses["total"] = total
        return losses
