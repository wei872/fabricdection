"""Losses: supervised detection + foreground alignment + loose reconstruction.

Detection training uses a **dual assignment** (YOLOv10 / YOLO27 lineage):
  - one-to-many dense assignment (IoU-aware varifocal targets) supervises the
    shared cls/reg features;
  - one-to-one Hungarian matching supervises the deployed objectness branch,
    which is what makes inference genuinely NMS-free.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import box_ciou, box_iou, boxes_from_distances


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


def _balanced_bce(bce: torch.Tensor, is_pos: torch.Tensor, pos_weight: float = 1.0,
                  neg_weight: float = 1.0):
    """Mean-over-positives + mean-over-negatives of an elementwise BC(E) map.

    A dense detector has ~3 orders of magnitude more background cells than
    positive cells (1700 vs 2-8 even at 320 px).  Summing the elementwise loss
    and dividing by the *positive* count -- or taking a plain ``.mean()`` over
    every cell -- makes the value (and the gradient) of the positive term an
    arbitrary function of the map size:

    * ``sum / n_pos``  blows the loss up (cls_o2m reached 46.9 in a real run
      while the deployed objectness term sat at 1.2), so the share of the
      *useful* term collapses and the log's ``loss share`` swings 5% -> 66%
      between epochs for no modelling reason;
    * ``.mean()`` over all cells (obj_o2m) drives positives towards 0 because
      every positive cell is outvoted 100:1.

    Balanced means give each class a fixed share of the term (50/50 by
    default, tunable through ``pos_weight``/``neg_weight``) independently of
    the map size, which is what makes the reported loss share comparable
    across scales and epochs.
    """
    pos = is_pos.to(bce.dtype)
    n_pos = pos.sum()
    n_neg = (1.0 - pos).sum()
    out = bce.new_zeros(())
    if float(n_pos) > 0:
        out = out + pos_weight * (bce * pos).sum() / n_pos
    if float(n_neg) > 0:
        out = out + neg_weight * (bce * (1.0 - pos)).sum() / n_neg
    return out


@torch.no_grad()
def build_dense_targets(pred, targets, stride, nc, reg_scale, radius=1.5,
                        quality_floor=0.5):
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
            # fallback: at least the cells containing a GT centre
            cxy = torch.stack([cx, cy], -1)
            nearest = ((pts[:, None, :] - cxy[None, :, :]).abs().sum(-1)).argmin(0)
            idx = nearest
            g = torch.arange(G, device=device)
        else:
            g = gi[idx]

        t_cls[b, labels[g], idx] = 1.0
        t_reg[b, :, idx] = d[idx, g].t()
        pd = pred_dist[b, :, idx].t()
        pb = boxes_from_distances(pts[idx], pd)
        qual = box_ciou(pb, boxes[g]).clamp(0, 1)
        # COLD-START FIX (round 3): the objectness/cls target used to be the
        # *predicted* IoU of the matched cell.  On a fresh model every box is a
        # degenerate dot, so that IoU is ~0 -- which means the branch was being
        # trained towards ``target = 0`` **on the positive cells themselves**:
        # ``obj_o2m`` sat at 0.025 (== "predict 0 everywhere, it is nearly
        # right") for the whole run and ``cls`` was taught to predict 0.02 on a
        # defect, so the deployed score ``sigmoid(obj) * sigmoid(cls)`` could
        # never exceed the 0.05/0.15 gates -> 0 boxes per image in the log.
        #
        # Normalise by the best positive of the scale (TAL / YOLOv8 do this for
        # the alignment metric) and clamp from below, so the branch always has a
        # real positive signal -- at cold start it defaults to the uniform 1.0 --
        # while still ranking positives by quality once the regression works.
        qmax = float(qual.max()) if qual.numel() else 0.0
        if qmax > 1e-6:
            # Blend "binary target" and "IoU-aware target" by the scale's own
            # best IoU (v0.2.2).
            #
            # ``qual / qmax`` alone is *noise amplification* at cold start:
            # every IoU is ~0.02, they differ only by noise, yet the largest
            # of them is mapped to 1.0 and the rest clamped -- so an arbitrary
            # positive cell is taught "objectness 1.0" while its neighbour is
            # taught 0.5, and which is which flips every batch.  Using the
            # best IoU itself as the blend weight removes that: if the
            # regression cannot localise (qmax ~ 0) the target is the uniform
            # 1.0 -- exactly the "there is an object here" signal objectness
            # needs -- and the quality ranking fades in only once the boxes
            # start matching (qmax -> 1).
            rel = qual / qmax
            w = min(1.0, max(0.0, qmax))
            t_qual[b, 0, idx] = (w * rel + (1.0 - w)).clamp(min=float(quality_floor))
        else:
            t_qual[b, 0, idx] = 1.0
        pos[b, 0, idx] = True

        # foreground-alignment auxiliary target: a disc around each GT centre
        near = (((pts[:, 0:1] - cx[None, :]) ** 2 +
                 (pts[:, 1:2] - cy[None, :]) ** 2) <= (2.0 * stride) ** 2).any(1)
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
            pts = _grid_points(H, W, stride, device, torch.float32)
            cls = p["cls"][b].float().reshape(nc, -1).t().sigmoid()    # (HW, nc)
            obj = p["obj_o2o"][b].float().reshape(-1).sigmoid()
            dist = p["reg"][b].float().reshape(4, -1).t() * reg_scale * stride
            boxes = boxes_from_distances(pts, dist)
            score = obj * cls.max(-1).values
            k = min(max(1, max_candidates // len(preds)), score.numel())
            top = torch.topk(score, k).indices
            all_cls.append(cls[top])
            all_obj.append(obj[top])
            all_boxes.append(boxes[top])
            meta.extend([(si, int(flat)) for flat in top.cpu().tolist()])

        boxes_gt = targets[b]["boxes"].to(device=device, dtype=torch.float32)
        labels = targets[b]["labels"].to(device=device, dtype=torch.long)
        if boxes_gt.numel() == 0:
            continue
        cls_all = torch.cat(all_cls, 0)                # (C, nc)
        obj_all = torch.cat(all_obj, 0)
        box_all = torch.cat(all_boxes, 0).clamp(min=0)

        cls_gt = cls_all[:, labels].t()                # (G, C)
        iou = box_iou(box_all, boxes_gt).t()           # (G, C)
        cost = (-cls_gt - 0.5 * obj_all[None, :] - 2.0 * iou).detach().float().cpu().numpy()

        rows, cols = linear_sum_assignment(cost)
        for g, c in zip(rows, cols):
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
        self.o2o_neg_ratio = float(cfg.train.get("o2o_neg_ratio", 0.02))
        self.o2o_min_neg = int(cfg.train.get("o2o_min_neg", 2048))
        # How the DEPLOYED objectness branch is supervised.
        #
        # ``ohem`` (old): positives + only the top ``o2o_neg_ratio`` ~ 8% hardest
        #   negative cells.  The other ~92% of the map never gets an explicit
        #   "background" gradient, so a cell the teacher never looked at can
        #   keep a high objectness for ever -> *background* false positives that
        #   no score threshold removes.  It also makes precision depend on which
        #   cells happened to be sampled.
        # ``focal`` (default): sigmoid focal loss over EVERY cell (alpha/gamma),
        #   normalised by the positive count -- the standard dense-detector
        #   treatment, still bounded and cheap (one extra elementwise pass).
        self.o2o_mode = str(cfg.train.get("obj_o2o_mode", "focal")).lower()
        self.o2o_gamma = float(cfg.train.get("obj_o2o_gamma", 2.0))
        self.o2o_alpha = float(cfg.train.get("obj_o2o_alpha", 0.5))
        self.o2o_neg_weight = float(cfg.train.get("obj_o2o_neg_weight", 1.0))
        # cold-start knobs (round 3): see build_dense_targets / _balanced_bce
        self.quality_floor = float(cfg.train.get("obj_quality_floor", 0.5))
        self.cls_quality_target = bool(cfg.train.get("cls_quality_target", False))
        self.o2m_pos_weight = float(cfg.train.get("obj_o2m_pos_weight", 1.0))
        self.cls_o2m_pos_weight = float(cfg.train.get("cls_o2m_pos_weight", 1.0))
        # Background pressure (v0.2.2).  A balanced 50/50 term lets the
        # objectness/classification field settle on a *flat* mid value: with
        # ``obj_o2o`` = 0.086 (a real run) the implied mean background
        # objectness is ~0.5, and at ``score = obj*cls`` every 3x3 peak then
        # clears the 0.15 gate -> 55 boxes/img, precision 0.3%.  >1 pushes the
        # background side down so the field becomes selective.
        self.o2m_neg_weight = float(cfg.train.get("obj_o2m_neg_weight", 1.0))
        self.cls_o2m_neg_weight = float(cfg.train.get("cls_o2m_neg_weight", 1.0))

    def forward(self, preds, targets, img_size):
        device = preds[0]["cls"].device
        # The auxiliary L1 term below is computed on *pixel* distances.  Without
        # a reference length its magnitude is proportional to the input
        # resolution: a real run logged ``box_o2m = 28.75`` although
        # ``1 - CIoU <= 2`` per scale (<= 4 for the two scales), i.e. ~25 of it
        # was ``0.1 * L1`` with L1 ~ 250 px -- 66% of the whole loss while the
        # deployed objectness branch sat at 0.7%.  Normalising by the image side
        # makes every loss term O(1), resolution-independent, and makes the
        # ``loss share`` line comparable across input sizes.
        img_ref = float(max(float(img_size[0]), float(img_size[1])))
        losses = {k: torch.zeros((), device=device) for k in
                  ("cls_o2m", "box_o2m", "obj_o2m", "cls_o2o", "box_o2o", "obj_o2o", "fg_align")}
        # ---------- one-to-many dense branch ----------
        for si, p in enumerate(preds):
            stride = self.strides[si]
            t_cls, t_reg, t_qual, t_fg, pos, pts = build_dense_targets(
                p, targets, stride, self.nc, self.reg_scale,
                quality_floor=self.quality_floor)
            B, _, H, W = p["cls"].shape
            HW = H * W

            cls = p["cls"].reshape(B, self.nc, HW)
            # the classification target is the class label, NOT the (cold-start
            # zero) regression quality -- multiplying them taught the deployed
            # cls head to predict ~0.02 on a real defect.
            target = t_cls * t_qual if self.cls_quality_target else t_cls
            ce = F.binary_cross_entropy_with_logits(cls, target, reduction="none")
            mod = (cls.sigmoid() - target).abs().pow(2)
            cw = self.class_weights[None, :, None]
            # "positive" means "this (cell, class) entry carries a label".
            # Deliberately NOT ``pos.expand(B, nc, HW)`` (that counts the 19
            # classes of every positive cell as positive and dilutes the
            # positive mean 19x, so the term silently drifts back to
            # background-dominated), and not ``target > 0.5`` either (with
            # ``cls_quality_target`` the target is soft -- a well-localised box
            # would read 0.6 and its class would be counted as background).
            losses["cls_o2m"] = losses["cls_o2m"] + _balanced_bce(
                ce * mod * cw, t_cls > 0.5,
                pos_weight=self.cls_o2m_pos_weight,
                neg_weight=self.cls_o2m_neg_weight)

            bce_obj = F.binary_cross_entropy_with_logits(p["obj"].reshape(B, 1, HW), t_qual,
                                                         reduction="none")
            losses["obj_o2m"] = losses["obj_o2m"] + _balanced_bce(
                bce_obj, pos, pos_weight=self.o2m_pos_weight,
                neg_weight=self.o2m_neg_weight)

            pm = pos.expand(B, 4, HW)
            pred_d = (p["reg"].float() * self.reg_scale * stride).reshape(B, 4, HW)[pm].reshape(-1, 4)
            tgt_d = t_reg[pm].reshape(-1, 4)
            p_sel = pts[None].expand(B, HW, 2)[pos.squeeze(1)].reshape(-1, 2)
            if pred_d.numel():
                pb = boxes_from_distances(p_sel, pred_d)
                gb = boxes_from_distances(p_sel, tgt_d)
                # CIoU is already scale-free; the L1 auxiliary is on pixel
                # distances and therefore divided by the image side.
                losses["box_o2m"] = losses["box_o2m"] + \
                    (1.0 - box_ciou(pb, gb)).mean() + \
                    0.1 * F.l1_loss(pred_d / img_ref, tgt_d / img_ref)

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
            n_pos = len(matches[si])
            norm = float(max(1, n_pos))
            if self.o2o_mode == "focal":
                # BALANCED focal: mean over positives + mean over negatives.
                #
                # The previous form (``foc.sum() / n_pos``) divides a sum that
                # contains ~25k negative cells by ~8 positive cells, so the
                # positive term is a rounding error of the objective: measured
                # on a real run obj_o2o was 0.7% of the total loss.  Balanced
                # means give the positives a fixed ~50% share regardless of
                # how many cells the map has.
                #
                # NOTE: do not name this ``p`` -- ``p`` is the prediction dict
                # of this scale (shadowing it broke the cls/reg losses).
                prob = obj.sigmoid()
                p_t = prob * t_obj + (1.0 - prob) * (1.0 - t_obj)
                a_t = self.o2o_alpha * t_obj + (1.0 - self.o2o_alpha) * (1.0 - t_obj)
                foc = a_t * (1.0 - p_t).pow(self.o2o_gamma) * bce
                is_pos = t_obj > 0.5
                if bool(is_pos.any()):
                    losses["obj_o2o"] = losses["obj_o2o"] + foc[is_pos].mean()
                is_neg = ~is_pos
                if bool(is_neg.any()):
                    losses["obj_o2o"] = losses["obj_o2o"] + \
                        self.o2o_neg_weight * foc[is_neg].mean()
            else:
                n_neg = o2o_negative_count(B * HW, n_pos, self.o2o_neg_ratio,
                                           self.o2o_min_neg)
                pos_loss = (bce * t_obj).sum() / norm
                if n_neg:
                    neg_loss = torch.topk((bce * (1 - t_obj)).reshape(-1),
                                          k=n_neg).values.mean()
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
                tgt_d = torch.stack([
                    grid[flat, 0] - gt_box[0], grid[flat, 1] - gt_box[1],
                    gt_box[2] - grid[flat, 0], gt_box[3] - grid[flat, 1]]).clamp(min=0)
                losses["box_o2o"] = losses["box_o2o"] + \
                    (1.0 - box_ciou(pb, gt_box[None])).mean() + \
                    0.1 * F.l1_loss(pd / img_ref, tgt_d / img_ref)
        n_match = max(1, sum(len(m) for m in matches))
        losses["cls_o2o"] = losses["cls_o2o"] / n_match
        losses["box_o2o"] = losses["box_o2o"] / n_match

        total = (self.w.cls_o2m * losses["cls_o2m"] + self.w.box_o2m * losses["box_o2m"] +
                 self.w.obj_o2m * losses["obj_o2m"] + self.w.cls_o2o * losses["cls_o2o"] +
                 self.w.box_o2o * losses["box_o2o"] + self.w.obj_o2o * losses["obj_o2o"] +
                 self.w.fg_align * losses["fg_align"])
        losses["total"] = total
        return losses
