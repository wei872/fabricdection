"""Dual-scale, anchor-free, NMS-free detection head.

The head is decoupled (cls / reg / obj), DFL-free (YOLO26/27 lineage) and has
**two objectness branches that share the same cls/reg predictions**:

* ``obj``     — one-to-many, trained with dense IoU-aware targets. Rich dense
                supervision for the shared features. TRAIN ONLY.
* ``obj_o2o`` — one-to-one, trained with Hungarian matching. This is the branch
                used at inference: one peak per object -> no NMS, deterministic
                latency. (Same spirit as YOLOv10 / YOLO27's "foreground
                alignment supervision", but implemented as an explicit branch
                so the deployed graph is trivially exportable.)
* ``fg``      — train-only foreground-alignment auxiliary head, removed on
                export. Costs nothing at deployment.

Only P3 (stride 8, tiny defects) and P5 (stride 32, large defects) carry a
head; P4 is fusion-only (YOLO27 n/s "dual-scale detection").
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import DWConv
from .utils import boxes_from_distances


class DecoupledHead(nn.Module):
    def __init__(self, c: int, nc: int, width: float, training_branch: bool = True):
        super().__init__()
        h = max(16, int(c * width))
        stem = lambda: nn.Sequential(DWConv(c, h, 3), DWConv(h, h, 3))
        self.cls_stem, self.reg_stem = stem(), stem()
        self.cls_pred = nn.Conv2d(h, nc, 1)
        self.reg_pred = nn.Conv2d(h, 4, 1)
        self.obj_pred = nn.Conv2d(c, 1, 1)          # one-to-many (train only)
        self.obj_o2o = nn.Conv2d(c, 1, 1)           # one-to-one  (deployed)
        self.fg_pred = nn.Conv2d(h, 1, 1) if training_branch else None
        self.fg_stem = stem() if training_branch else None

        nn.init.constant_(self.obj_pred.bias, -4.0)
        nn.init.constant_(self.obj_o2o.bias, -4.0)
        nn.init.constant_(self.cls_pred.bias, -2.0)

    def forward(self, x, training: bool = False):
        f_cls, f_reg = self.cls_stem(x), self.reg_stem(x)
        out = {
            "cls": self.cls_pred(f_cls),
            "reg": F.relu(self.reg_pred(f_reg)),
            "obj": self.obj_pred(x) if training else None,
            "obj_o2o": self.obj_o2o(x),
        }
        out["fg"] = self.fg_pred(self.fg_stem(x)) if (training and self.fg_stem is not None) else None
        return out


class NMSFreeDetect(nn.Module):
    def __init__(self, cfg, channels):
        super().__init__()
        m = cfg.model
        self.nc = int(cfg.data.nc)
        self.strides = (8, 32)
        self.top_k = int(m.top_k)
        self.score_thresh = float(m.score_thresh)
        self.reg_scale = float(m.reg_scale)
        # One-to-one (NMS-free) decoding still has to be *peak* decoding.
        # Without a local-maximum filter every cell of a high-objectness
        # plateau becomes its own box: a single 12x12 px blob on the P3 map
        # decoded to 144 overlapping boxes at the same score, i.e. 1 TP and
        # 143 FP.  That is the structural reason validation precision stays
        # near 0.1 no matter how well the detector localises.
        self.peak_filter = bool(m.get("peak_filter", True))
        self.peak_kernel = max(1, int(m.get("peak_kernel", 3)))
        # Peak filtering is per scale; the SAME defect is still found by both
        # P3 and P5 (a small box inside a big one -> low IoU, so an IoU-only
        # NMS would keep both).  Deduplicate across scales with IoU plus a
        # containment test before the boxes reach the evaluator.
        self.dedup = bool(m.get("dedup", True))
        self.dedup_iou = float(m.get("dedup_iou", 0.70))
        self.dedup_contain = float(m.get("dedup_contain", 0.85))
        self.heads = nn.ModuleList([
            DecoupledHead(channels[1], self.nc, m.head_width, m.fg_align),
            DecoupledHead(channels[3], self.nc, m.head_width, m.fg_align),
        ])
        self._points_cache: dict = {}

    def _points(self, h, w, stride, device):
        key = (h, w, stride, device)
        if key not in self._points_cache:
            ys, xs = torch.meshgrid(
                torch.arange(h, device=device, dtype=torch.float32),
                torch.arange(w, device=device, dtype=torch.float32), indexing="ij")
            grid = torch.stack([xs, ys], -1).reshape(-1, 2)
            self._points_cache[key] = (grid + 0.5) * stride
        return self._points_cache[key]

    def forward(self, feats: dict, training: bool = False):
        return [head(feats[name], training=training)
                for head, name in zip(self.heads, ("n3", "n5"))]

    @torch.no_grad()
    def _dedup_mask(self, boxes: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Keep-mask that removes duplicates (incl. cross-scale, nested) boxes.

        ``boxes``/``scores`` must be sorted by descending score, so the greedy
        pass keeps the most confident of each group.  Two boxes are duplicates
        when

        * ``IoU > dedup_iou``            (same object, similar size), or
        * ``inter / min(area) > contain`` (small box nested in a big one, which
          is exactly what P3-inside-P5 looks like and has a *low* IoU).

        The loop runs over at most ``top_k`` boxes, i.e. a fixed, small bound,
        and it is executed outside the exported graph (like the rest of
        ``decode``), so latency stays deterministic.
        """
        n = boxes.shape[0]
        keep = torch.ones(n, dtype=torch.bool, device=boxes.device)
        if n <= 1:
            return keep
        area = ((boxes[:, 2] - boxes[:, 0]).clamp(min=0) *
                (boxes[:, 3] - boxes[:, 1]).clamp(min=0))
        for i in range(n):
            if not bool(keep[i]):
                continue
            rest = keep.clone()
            rest[:i + 1] = False
            if not bool(rest.any()):
                break
            b = boxes[rest]
            inter = ((torch.min(boxes[i, 2], b[:, 2]) - torch.max(boxes[i, 0], b[:, 0])).clamp(min=0) *
                     (torch.min(boxes[i, 3], b[:, 3]) - torch.max(boxes[i, 1], b[:, 1])).clamp(min=0))
            union = area[i] + area[rest] - inter
            iou = inter / union.clamp(min=1e-7)
            contain = inter / torch.minimum(area[i], area[rest]).clamp(min=1e-7)
            dup = (iou > self.dedup_iou) | (contain > self.dedup_contain)
            keep[rest] = ~dup
        return keep

    # ----------------------------------------------------------------------------------
    @torch.no_grad()
    def decode(self, preds, img_size, anomaly_boost: torch.Tensor | None = None):
        """local-peak + top-k + threshold over the one-to-one branch.

        No NMS, no post-hoc data-dependent loops -> fixed, bounded latency.
        ``model.peak_filter`` (default on) keeps only 3x3 objectness maxima,
        which is what makes "one peak per object" actually hold at inference;
        with ``top_k`` alone a single defect yields dozens of duplicate boxes.

        Returns list[B] of dict(boxes(N,4) xyxy px, scores(N,), labels(N,),
                                obj(N,), points(N,2), scale_idx(N,)).
        """
        B = preds[0]["cls"].shape[0]
        device = preds[0]["cls"].device
        results = []

        for b in range(B):
            cand_boxes, cand_scores, cand_labels, cand_obj, cand_pts, cand_si = [], [], [], [], [], []
            for si, (p, stride) in enumerate(zip(preds, self.strides)):
                cls, reg, obj = p["cls"][b], p["reg"][b], p["obj_o2o"][b]
                _, H, W = cls.shape
                cls = cls.reshape(self.nc, -1).T          # (HW, nc)
                reg = reg.reshape(4, -1).T               # (HW, 4)
                obj = obj.reshape(-1)

                if anomaly_boost is not None:
                    a = F.interpolate(anomaly_boost[b:b + 1], size=(H, W),
                                      mode="bilinear", align_corners=False).reshape(-1)
                    obj = obj + float(self.anomaly_logit_gain) * a

                # ---- keep only local objectness maxima (one peak per object) --
                # The o2o branch is trained with a single positive cell per GT,
                # but objectness in the neighbourhood of a defect is a smooth
                # plateau, so thresholding alone keeps every cell of it.  A
                # 3x3 max-pool comparison removes those duplicates *inside*
                # the graph-external decode (no data-dependent loops, static
                # latency, still NMS-free).
                peak_mask = None
                if self.peak_filter and self.peak_kernel > 1:
                    o2d = obj.reshape(1, 1, H, W)
                    # tie-break ramp: 1e-6 * flat index, far below any real
                    # logit difference, but makes saturated/constant plateaus
                    # collapse to exactly one peak instead of surviving intact.
                    ramp = torch.arange(H * W, device=device,
                                        dtype=torch.float32).reshape(1, 1, H, W)
                    ranked = o2d - 1e-6 * ramp
                    pooled = F.max_pool2d(ranked, self.peak_kernel, 1,
                                          self.peak_kernel // 2)
                    peak_mask = (ranked >= pooled).reshape(-1)

                cls_prob = cls.sigmoid()
                cscore, clabel = cls_prob.max(dim=-1)
                score = obj.sigmoid() * cscore
                if peak_mask is not None:
                    score = score.masked_fill(~peak_mask, -1.0)

                k = min(self.top_k, score.numel())
                top = torch.topk(score, k)
                keep = top.values > self.score_thresh
                if not keep.any():
                    continue
                idx = top.indices[keep]

                pts = self._points(H, W, stride, device)[idx]
                dist = reg[idx] * self.reg_scale * stride
                cand_boxes.append(boxes_from_distances(pts, dist))
                cand_scores.append(score[idx])
                cand_labels.append(clabel[idx])
                cand_obj.append(obj[idx].sigmoid())
                cand_pts.append(pts)
                cand_si.append(torch.full_like(clabel[idx], si))

            if not cand_boxes:
                empty = torch.zeros((0, 4), device=device)
                results.append(dict(boxes=empty, scores=torch.zeros(0, device=device),
                                    labels=torch.zeros(0, dtype=torch.long, device=device),
                                    obj=torch.zeros(0, device=device), points=empty[:, :2],
                                    scale_idx=torch.zeros(0, dtype=torch.long, device=device)))
                continue

            scores = torch.cat(cand_scores)
            k = min(self.top_k, scores.numel())
            sel = torch.topk(scores, k).indices
            boxes = torch.cat(cand_boxes)[sel]
            labels = torch.cat(cand_labels)[sel]
            obj = torch.cat(cand_obj)[sel]
            pts = torch.cat(cand_pts)[sel]
            si = torch.cat(cand_si)[sel]
            boxes[..., 0::2] = boxes[..., 0::2].clamp(0, img_size[1])
            boxes[..., 1::2] = boxes[..., 1::2].clamp(0, img_size[0])
            scores = scores[sel]
            box_keep = self._dedup_mask(boxes, scores) if self.dedup else \
                torch.ones(boxes.shape[0], dtype=torch.bool, device=device)
            results.append(dict(boxes=boxes[box_keep], scores=scores[box_keep],
                                labels=labels[box_keep], obj=obj[box_keep],
                                points=pts[box_keep], scale_idx=si[box_keep]))
        return results

    @property
    def anomaly_logit_gain(self):
        return getattr(self, "_anomaly_logit_gain", 1.0)

    def set_anomaly_logit_gain(self, value: float):
        self._anomaly_logit_gain = float(value)
