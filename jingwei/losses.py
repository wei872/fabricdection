"""Losses: supervised detection + foreground alignment + loose reconstruction.

Detection training uses a **dual assignment** (YOLOv10 / YOLO27 lineage):
  - one-to-many dense assignment (IoU-aware varifocal targets) supervises the
    shared cls/reg features;
  - one-to-one Hungarian matching supervises the deployed objectness branch,
    which is what makes inference genuinely NMS-free.
"""
from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import box_ciou, box_iou, boxes_from_distances


_SANITIZE_WARNED: set[str] = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _SANITIZE_WARNED:
        _SANITIZE_WARNED.add(key)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)


def resolve_label_offset(labels: torch.Tensor, offset, nc: int) -> int:
    """Resolve a YOLO class-id offset without corrupting already-0-based data.

    ``offset=0``      : no shift (default, YOLO standard 0..nc-1).
    ``offset=1``      : shift 1-based labels (1..nc) to 0..nc-1.  The shift is
                        applied only when the labels actually look 1-based, so
                        calling this twice is safe.
    ``offset='auto'`` : shift only when a class id >= nc is present AND all
                        ids are within 1..nc (a clear 1-based dataset).
    """
    mode = str(offset).strip().lower()
    if mode in ("0", "none", "false", ""):
        return 0
    if labels.numel() == 0:
        return 0
    lo, hi = int(labels.min()), int(labels.max())
    # auto: only infer 1-based when an out-of-range id proves it
    if mode == "auto":
        return 1 if lo >= 1 and hi >= nc and hi <= nc else 0
    explicit = 1 if mode in ("1", "one", "one_based", "true") else None
    if explicit is None:
        try:
            explicit = int(mode)
        except ValueError:
            explicit = 0
    if not explicit:
        return 0
    # explicit 1: shift only when the labels really look 1-based, so calling
    # this twice (model.forward -> DetectionLoss.forward) is idempotent.
    return 1 if explicit == 1 and lo >= 1 and hi <= nc else 0


@torch.no_grad()
def sanitize_detection_targets(targets: Sequence[dict], nc: int, img_size,
                               device, label_offset=0) -> list[dict]:
    """Filter invalid boxes / class ids BEFORE any CUDA indexing.

    A single YOLO label with ``cls >= nc`` (or a 1-based dataset without an
    offset) used to trigger a device-side assert deep in ``build_dense_targets``
    on CUDA.  This function drops malformed targets on the host-side logic and
    emits ONE clear warning with counts, so training continues and the user can
    fix the data with ``tools/validate_labels.py``.
    """
    if isinstance(img_size, (int, float)):
        H = W = int(img_size)
    else:
        # PyTorch convention used everywhere in JingWei: (H, W).
        H, W = (int(img_size[0]), int(img_size[1]))
    out: list[dict] = []
    n_bad_label = n_bad_box = n_mismatch = n_shifted = 0

    for t in targets:
        boxes = t.get("boxes")
        labels = t.get("labels")
        if boxes is None or labels is None:
            out.append({"boxes": torch.zeros((0, 4), device=device),
                        "labels": torch.zeros((0,), dtype=torch.long, device=device)})
            continue
        boxes = boxes.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device, dtype=torch.long).reshape(-1)
        if boxes.numel() == 0:
            boxes = boxes.reshape(-1, 4)
            labels = labels[:0]
        else:
            boxes = boxes.reshape(-1, 4)
            g = min(boxes.shape[0], labels.shape[0])
            if boxes.shape[0] != labels.shape[0]:
                n_mismatch += abs(boxes.shape[0] - labels.shape[0])
                boxes, labels = boxes[:g], labels[:g]
            # Ignore absurd ids (e.g. 99) when inferring auto/1 offset; they
            # are dropped below as bad_class anyway.
            offset_candidates = labels[(labels >= 0) & (labels <= nc)]
            off = resolve_label_offset(offset_candidates, label_offset, nc)
            if off:
                labels = labels - off
                n_shifted += labels.numel()
            good_label = (labels >= 0) & (labels < nc)
            finite = torch.isfinite(boxes).all(1)
            positive = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            in_bounds = ((boxes[:, 0] >= -1) & (boxes[:, 1] >= -1) &
                         (boxes[:, 2] <= W + 1) & (boxes[:, 3] <= H + 1))
            keep = good_label & finite & positive & in_bounds
            n_bad_label += int((~good_label).sum())
            n_bad_box += int((~(finite & positive & in_bounds)).sum())
            boxes, labels = boxes[keep], labels[keep]
            if boxes.numel():
                boxes[:, 0::2] = boxes[:, 0::2].clamp(0, W)
                boxes[:, 1::2] = boxes[:, 1::2].clamp(0, H)
        out.append({"boxes": boxes, "labels": labels})

    if n_bad_label or n_bad_box or n_mismatch:
        _warn_once(
            "invalid_targets",
            f"sanitize_detection_targets dropped invalid YOLO targets: "
            f"bad_label={n_bad_label}, bad_box={n_bad_box}, "
            f"box/label mismatch={n_mismatch}, shifted_1based={n_shifted}. "
            f"Expected class ids 0..{nc-1} and xyxy boxes inside {W}x{H}. "
            f"If your labels are 1..{nc}, use --label-offset 1 or run "
            f"tools/validate_labels.py --fix-one-based.")
    return out


# --------------------------------------------------------------------------------------
# 1. Dinomaly loose reconstruction loss
# --------------------------------------------------------------------------------------
class HardMiningCosineLoss(nn.Module):
    """Global cosine loss on the hardest ``p`` fraction of patches.

    Dinomaly's "Loose Reconstruction": no layer-to-layer / point-by-point MSE,
    just a cosine dissimilarity concentrated on hard patches. ``p`` is ramped
    from 0 -> p_final over ``warmup`` iterations so training starts stable.
    """

    def __init__(self, p_final: float = 0.9, warmup: int = 1000, factor: float = 0.1):
        super().__init__()
        self.p_final, self.warmup, self.factor = p_final, max(1, warmup), factor

    def forward(self, target: torch.Tensor, reconstructed: torch.Tensor, it: int = 0):
        t = F.normalize(target.detach(), dim=-1)
        r = F.normalize(reconstructed, dim=-1)
        loss_map = 1.0 - (t * r).sum(-1)                # (B, N)
        flat = loss_map.reshape(loss_map.shape[0], -1)
        p = min(self.p_final, self.p_final * it / self.warmup)
        k = max(1, int(flat.shape[1] * p))
        hard = torch.topk(flat, k, dim=1).values.mean()
        return hard + self.factor * flat.mean()


# --------------------------------------------------------------------------------------
# 2. Dense (one-to-many) target builder
# --------------------------------------------------------------------------------------
def _grid_points(h, w, stride, device, dtype):
    ys, xs = torch.meshgrid(torch.arange(h, device=device, dtype=dtype),
                            torch.arange(w, device=device, dtype=dtype), indexing="ij")
    pts = torch.stack([xs, ys], -1).reshape(-1, 2)
    return (pts + 0.5) * stride


@torch.no_grad()
def build_dense_targets(pred, targets, stride, nc, reg_scale, radius=1.5):
    """FCOS-style centre sampling with an IoU-aware quality target."""
    B, _, H, W = pred["cls"].shape
    device, dtype = pred["cls"].device, pred["cls"].dtype
    HW = H * W
    pts = _grid_points(H, W, stride, device, dtype)

    t_cls = torch.zeros(B, nc, HW, device=device, dtype=dtype)
    t_reg = torch.zeros(B, 4, HW, device=device, dtype=dtype)
    t_qual = torch.zeros(B, 1, HW, device=device, dtype=dtype)
    t_fg = torch.zeros(B, 1, HW, device=device, dtype=dtype)
    pos = torch.zeros(B, 1, HW, device=device, dtype=torch.bool)

    pred_dist = pred["reg"].reshape(B, 4, HW) * reg_scale * stride

    for b in range(B):
        boxes, labels = targets[b]["boxes"], targets[b]["labels"]
        boxes = boxes.to(device=device, dtype=dtype)
        labels = labels.to(device=device, dtype=torch.long)
        if boxes.numel() == 0:
            continue
        G = boxes.shape[0]
        if G == 0:
            continue
        if labels.numel() != G:
            raise ValueError(f"build_dense_targets: {G} boxes but "
                             f"{labels.numel()} labels after sanitization")
        if bool(((labels < 0) | (labels >= nc)).any()):
            raise ValueError(
                f"build_dense_targets: class id out of range 0..{nc-1}; "
                f"got {labels.min().item()}..{labels.max().item()}. "
                f"If labels are 1-based, use --label-offset 1 or run "
                f"tools/validate_labels.py --fix-one-based.")

        l = pts[:, 0:1] - boxes[None, :, 0]
        t = pts[:, 1:2] - boxes[None, :, 1]
        r = boxes[None, :, 2] - pts[:, 0:1]
        bo = boxes[None, :, 3] - pts[:, 1:2]
        d = torch.stack([l, t, r, bo], -1)                     # (HW, G, 4)
        inside = d.min(-1).values > 0
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
            # Fallback: no centre-sampling cell was valid (tiny / edge boxes).
            # ``nearest`` is a FLAT GRID index, not a GT index.  The old code
            # then did gi[idx] with gi=arange(G), which indexed a length-G
            # tensor with grid ids -> CUDA "index out of bounds" assert.
            cxy = torch.stack([cx, cy], -1)
            nearest = ((pts[:, None, :] - cxy[None, :, :]).abs().sum(-1)).argmin(0)
            idx = nearest.clamp(0, HW - 1)
            g = torch.arange(G, device=device)
        else:
            g = gi[idx].clamp(0, G - 1)

        t_cls[b, labels[g], idx] = 1.0
        t_reg[b, :, idx] = d[idx, g].t()
        pd = pred_dist[b, :, idx].t()
        pb = boxes_from_distances(pts[idx], pd)
        t_qual[b, 0, idx] = box_ciou(pb, boxes[g]).clamp(0, 1)
        pos[b, 0, idx] = True

        # foreground-alignment auxiliary target: a disc around each GT centre
        for j in range(G):
            near = ((pts[:, 0] - cx[j]) ** 2 + (pts[:, 1] - cy[j]) ** 2) <= (2.0 * stride) ** 2
            t_fg[b, 0, near] = 1.0

    return t_cls, t_reg, t_qual, t_fg, pos, pts


# --------------------------------------------------------------------------------------
# 3. One-to-one Hungarian matching
# --------------------------------------------------------------------------------------
@torch.no_grad()
def hungarian_match(preds, targets, strides, nc, reg_scale, max_candidates=1600):
    """Return matches[si] = list of (batch, flat_index, gt_index)."""
    from scipy.optimize import linear_sum_assignment

    B = preds[0]["cls"].shape[0]
    device = preds[0]["cls"].device
    per_scale: list[list] = [[] for _ in preds]
    candidate_meta = [[] for _ in preds]      # (si, local flat idx) per global slot order

    for b in range(B):
        all_cls, all_obj, all_boxes, meta = [], [], [], []
        for si, p in enumerate(preds):
            Bn, _, H, W = p["cls"].shape
            stride = strides[si]
            pts = _grid_points(H, W, stride, device, p["cls"].dtype)
            cls = p["cls"][b].reshape(nc, -1).t().sigmoid()             # (HW, nc)
            obj = p["obj_o2o"][b].reshape(-1).sigmoid()
            dist = p["reg"][b].reshape(4, -1).t() * reg_scale * stride
            boxes = boxes_from_distances(pts, dist)
            score = obj * cls.max(-1).values
            k = min(max(1, max_candidates // len(preds)), score.numel())
            top = torch.topk(score, k).indices
            all_cls.append(cls[top])
            all_obj.append(obj[top])
            all_boxes.append(boxes[top])
            meta.extend([(si, int(flat)) for flat in top.cpu().tolist()])

        boxes_gt = targets[b]["boxes"].to(device=device, dtype=all_boxes[0].dtype)
        labels = targets[b]["labels"].to(device=device, dtype=torch.long)
        if boxes_gt.numel() == 0 or labels.numel() == 0:
            continue
        g = min(boxes_gt.shape[0], labels.shape[0])
        boxes_gt, labels = boxes_gt[:g], labels[:g]
        if bool(((labels < 0) | (labels >= nc)).any()):
            raise ValueError(
                f"hungarian_match: class id out of range 0..{nc-1}; "
                f"got {labels.min().item()}..{labels.max().item()}. "
                f"Use --label-offset 1 for 1-based YOLO labels.")
        cls_all = torch.cat(all_cls, 0)                # (C, nc)
        obj_all = torch.cat(all_obj, 0)
        box_all = torch.cat(all_boxes, 0).clamp(min=0)

        cls_gt = cls_all[:, labels].t()                # (G, C)
        iou = box_iou(box_all, boxes_gt).t()           # (G, C)
        cost = (-cls_gt - 0.5 * obj_all[None, :] - 2.0 * iou).detach().cpu().numpy()

        rows, cols = linear_sum_assignment(cost)
        for g, c in zip(rows, cols):
            si, flat = meta[c]
            per_scale[si].append((b, flat, int(g)))
    return per_scale


# --------------------------------------------------------------------------------------
# 4. Combined detection loss
# --------------------------------------------------------------------------------------
class DetectionLoss(nn.Module):
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
        self.label_offset = cfg.data.get("label_offset", 0)

    def forward(self, preds, targets, img_size):
        device = preds[0]["cls"].device
        # Host-side validation BEFORE any CUDA indexing.  Idempotent, so
        # model.forward() may also sanitize before the foreground target.
        targets = sanitize_detection_targets(
            targets, self.nc, img_size, device, self.label_offset)
        losses = {k: torch.zeros((), device=device) for k in
                  ("cls_o2m", "box_o2m", "obj_o2m", "cls_o2o", "box_o2o", "obj_o2o", "fg_align")}

        # ---------- one-to-many dense branch ----------
        for si, p in enumerate(preds):
            stride = self.strides[si]
            t_cls, t_reg, t_qual, t_fg, pos, pts = build_dense_targets(
                p, targets, stride, self.nc, self.reg_scale)
            B, _, H, W = p["cls"].shape
            HW = H * W

            cls = p["cls"].reshape(B, self.nc, HW)
            target = t_cls * t_qual
            ce = F.binary_cross_entropy_with_logits(cls, target, reduction="none")
            mod = (cls.sigmoid() - target).abs().pow(2)
            cw = self.class_weights[None, :, None]
            losses["cls_o2m"] = losses["cls_o2m"] + (ce * mod * cw).sum() / (pos.sum() + 1.0)

            bce_obj = F.binary_cross_entropy_with_logits(p["obj"].reshape(B, 1, HW), t_qual,
                                                         reduction="none")
            losses["obj_o2m"] = losses["obj_o2m"] + (bce_obj * (1.0 + 3.0 * pos.float())).mean()

            pm = pos.expand(B, 4, HW)
            pred_d = (p["reg"] * self.reg_scale * stride).reshape(B, 4, HW)[pm].reshape(-1, 4)
            tgt_d = t_reg[pm].reshape(-1, 4)
            p_sel = pts[None].expand(B, HW, 2)[pos.squeeze(1)].reshape(-1, 2)
            if pred_d.numel():
                pb = boxes_from_distances(p_sel, pred_d)
                gb = boxes_from_distances(p_sel, tgt_d)
                losses["box_o2m"] = losses["box_o2m"] + \
                    (1.0 - box_ciou(pb, gb)).mean() + 0.1 * F.l1_loss(pred_d, tgt_d)

            if p["fg"] is not None:
                losses["fg_align"] = losses["fg_align"] + \
                    F.binary_cross_entropy_with_logits(p["fg"].reshape(B, 1, HW), t_fg)

        # ---------- one-to-one deployed branch ----------
        matches = hungarian_match(preds, targets, self.strides, self.nc,
                                  self.reg_scale, self.o2o_candidates)
        for si, p in enumerate(preds):
            stride = self.strides[si]
            B, _, H, W = p["cls"].shape
            HW = H * W
            obj = p["obj_o2o"].reshape(B, -1)
            t_obj = torch.zeros_like(obj)
            for b, flat, _ in matches[si]:
                t_obj[b, flat] = 1.0
            bce = F.binary_cross_entropy_with_logits(obj, t_obj, reduction="none")
            n_pos = t_obj.sum().clamp(min=1.0)
            n_neg = min((1.0 - t_obj).sum(), 3.0 * n_pos)
            pos_loss = (bce * t_obj).sum() / n_pos
            neg_loss = torch.topk((bce * (1 - t_obj)).reshape(-1),
                                  k=max(1, int(n_neg.item()))).values.mean()
            losses["obj_o2o"] = losses["obj_o2o"] + pos_loss + neg_loss

            cls = p["cls"].reshape(B, self.nc, HW)
            reg = p["reg"].reshape(B, 4, HW)
            grid = _grid_points(H, W, stride, device, cls.dtype)
            for b, flat, g in matches[si]:
                label = targets[b]["labels"][g].to(device).long()
                onehot = F.one_hot(label, self.nc).float()
                losses["cls_o2o"] = losses["cls_o2o"] + \
                    F.binary_cross_entropy_with_logits(cls[b, :, flat], onehot)
                gt_box = targets[b]["boxes"][g].to(device=device, dtype=cls.dtype)
                pd = reg[b, :, flat] * self.reg_scale * stride
                pb = boxes_from_distances(grid[flat:flat + 1], pd[None])
                losses["box_o2o"] = losses["box_o2o"] + \
                    (1.0 - box_ciou(pb, gt_box[None])).mean() + \
                    0.1 * F.l1_loss(pd, torch.stack([
                        grid[flat, 0] - gt_box[0], grid[flat, 1] - gt_box[1],
                        gt_box[2] - grid[flat, 0], gt_box[3] - grid[flat, 1]]).clamp(min=0))
        n_match = max(1, sum(len(m) for m in matches))
        losses["cls_o2o"] = losses["cls_o2o"] / n_match
        losses["box_o2o"] = losses["box_o2o"] / n_match

        total = (self.w.cls_o2m * losses["cls_o2m"] + self.w.box_o2m * losses["box_o2m"] +
                 self.w.obj_o2m * losses["obj_o2m"] + self.w.cls_o2o * losses["cls_o2o"] +
                 self.w.box_o2o * losses["box_o2o"] + self.w.obj_o2o * losses["obj_o2o"] +
                 self.w.fg_align * losses["fg_align"])
        losses["total"] = total
        return losses
