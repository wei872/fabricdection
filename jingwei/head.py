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
    def __init__(self, c: int, nc: int, width: float, training_branch: bool = True,
                 obj_bias_init: float = -4.0, cls_bias_init: float = -2.0,
                 obj_from: str = "reg", obj_pred_bias_init: float = -4.0):
        super().__init__()
        h = max(16, int(c * width))
        stem = lambda: nn.Sequential(DWConv(c, h, 3), DWConv(h, h, 3))
        self.cls_stem, self.reg_stem = stem(), stem()
        self.cls_pred = nn.Conv2d(h, nc, 1)
        self.reg_pred = nn.Conv2d(h, 4, 1)
        # WHERE the objectness heads read from (v0.2.2).
        #
        # They used to be a bare 1x1 conv on the *raw neck feature* ``x`` while
        # cls / reg each get two 3x3 DWConvs.  A 1x1 conv cannot turn a smooth
        # neck map into a one-cell peak, so the objectness field settled on a
        # flat compromise -- measured from a real run: ``obj_o2o`` ~0.5 and
        # ``obj_o2m`` ~0.78 *on background*, which at ``score = obj*cls`` makes
        # every 3x3 peak clear the 0.15 gate => 55 boxes/img, precision 0.3%,
        # normal-image FPR 1.0, and the flood is what caps precision.
        #
        # ``obj_from: "reg"`` (default) reads the regression stem, which is
        # localisation-aware and *already computed* (the 1x1 convs even get
        # narrower, h < c); the extra objectness stem below costs +0.17 G MACs
        # at 1280 px on pico, i.e. +2.1% (8.086 -> 8.257 G).  Ablation with the
        # shipped knobs and this stem removed (`obj_head_from: neck`): the
        # deployed score is still 0.033 after 300 Adam steps -- the gate never
        # opens at all.  ``"neck"`` therefore stays only for A/B.
        self.obj_from = str(obj_from).lower()
        if self.obj_from == "reg":
            # A defect is a *local contrast* event ("this patch does not look
            # like the fabric around it"), so the objectness head needs a
            # receptive field of several cells, not one.  3x3 then 5x5 depthwise
            # convs take it to 9 cells ~ 72 px at stride 8, for ~0.1 G MACs at
            # 1280 px (< 1.5% of the pico budget).  This is the head that was a
            # bare 1x1 conv on the neck -- the layer that produced a flat
            # objectness field (obj ~ 0.5 on background).
            self.obj_stem = nn.Sequential(DWConv(h, h, 3), DWConv(h, h, 5))
            src = h
        else:
            self.obj_stem = nn.Identity()
            src = c
        self.obj_pred = nn.Conv2d(src, 1, 1)        # one-to-many (train only)
        self.obj_o2o = nn.Conv2d(src, 1, 1)         # one-to-one  (deployed)
        self.fg_pred = nn.Conv2d(h, 1, 1) if training_branch else None
        self.fg_stem = stem() if training_branch else None

        # Objectness / classification priors.
        #
        # ``obj_bias_init`` is NOT the cold-start bottleneck -- do not raise it
        # to "get boxes sooner".  Measured (head-only probe, frozen real
        # backbone features, 320 px, AdamW lr 2e-3): with ``-4`` the deployed
        # branch's positive objectness goes 0.06 -> 0.98 in **9 steps**, because
        # the 1x1 conv raises the logit through its *weights*, not through the
        # bias.  What a high prior does buy is a *higher background level* that
        # then has to be pressed back down (background p99 after 200 steps):
        # ``-4 -> 0.063``, ``-2 -> 0.119``, ``-1 -> 0.146`` (and occasionally a
        # positive cell collapses).  Since ``score = obj * cls`` and a flat
        # background is exactly the v0.2.1 failure (55 boxes/img, precision
        # 0.3%, normal_fpr 1.0), the low prior is the right default.
        #
        # The cold start is fixed structurally instead: the objectness heads
        # read the regression stem (``obj_head_from``), carry a real share of
        # the loss (``train.loss_weights.obj_o2o``) and get explicit background
        # pressure (``train.obj_o2o_neg_weight``).
        nn.init.constant_(self.obj_pred.bias, float(obj_pred_bias_init))
        nn.init.constant_(self.obj_o2o.bias, float(obj_bias_init))
        nn.init.constant_(self.cls_pred.bias, float(cls_bias_init))

    def forward(self, x, training: bool = False):
        f_cls, f_reg = self.cls_stem(x), self.reg_stem(x)
        obj_src = self.obj_stem(f_reg) if self.obj_from == "reg" else x
        out = {
            "cls": self.cls_pred(f_cls),
            "reg": F.relu(self.reg_pred(f_reg)),
            "obj": self.obj_pred(obj_src) if training else None,
            "obj_o2o": self.obj_o2o(obj_src),
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
        obj_bias_init = float(m.get("obj_bias_init", -4.0))
        cls_bias_init = float(m.get("cls_bias_init", -2.0))
        obj_from = str(m.get("obj_head_from", "reg"))
        obj_pred_bias_init = float(m.get("obj_pred_bias_init", -4.0))
        self.obj_bias_init = obj_bias_init
        self.cls_bias_init = cls_bias_init
        self.obj_from = obj_from
        self.heads = nn.ModuleList([
            DecoupledHead(channels[1], self.nc, m.head_width, m.fg_align,
                          obj_bias_init=obj_bias_init, cls_bias_init=cls_bias_init,
                          obj_from=obj_from, obj_pred_bias_init=obj_pred_bias_init),
            DecoupledHead(channels[3], self.nc, m.head_width, m.fg_align,
                          obj_bias_init=obj_bias_init, cls_bias_init=cls_bias_init,
                          obj_from=obj_from, obj_pred_bias_init=obj_pred_bias_init),
        ])
        # Diagnostics for "is the deployed gate open?" (see train_detector's
        # val line).  decode() is graph-external, so bookkeeping here is free.
        self.last_stats: dict = {}
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
        # ``score_p99`` is the *background* level: the 99th percentile of
        # ``obj*cls`` over EVERY cell of the map.  ``score_max`` alone cannot
        # tell "the detector found a defect" from "the field is flat but one
        # cell won the lottery": in the v0.2.1 run ``score_max`` sat at 0.9997
        # (one lucky cell per frame, logits are unbounded) while the median
        # cell sat at ~0.5 -> every 3x3 peak cleared the gate -> 55 boxes/img
        # and normal_fpr = 1.0.  A healthy frame has p99 near the gate's
        # *background* side (<= 0.02); a flat field shows p99 ~ obj_p99.
        stats = dict(score_max=0.0, score_p99=0.0, obj_max=0.0, obj_mean=0.0,
                     cls_max=0.0, n_peak=0, n_above=0, n_out=0,
                     score_thresh=self.score_thresh)

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
                prob_obj = obj.sigmoid()
                score = prob_obj * cscore
                if peak_mask is not None:
                    score = score.masked_fill(~peak_mask, -1.0)

                k = min(self.top_k, score.numel())
                top = torch.topk(score, k)

                # ---- liveness bookkeeping (measured BEFORE the gate) --------
                # The whole "precision never rises" report reduces to these
                # numbers being tiny: if ``score_max`` never exceeds the gate,
                # every epoch reports 0 boxes and mAP/precision/F1 stay
                # identically 0 whatever else is tuned.
                stats["score_max"] = max(stats["score_max"],
                                        float(top.values.max()) if top.values.numel() else 0.0)
                # ``quantile`` has no half/bfloat16 kernel and validation
                # may run under autocast -> always quantile in float32.
                sf = score.float()
                sel_q = peak_mask if peak_mask is not None and bool(peak_mask.any()) else None
                stats["score_p99"] = max(stats["score_p99"],
                                         float((sf[sel_q] if sel_q is not None
                                                else sf).quantile(0.99)))
                stats["obj_max"] = max(stats["obj_max"], float(prob_obj.max()))
                # accumulated, then divided by (#scales * B) before returning
                stats["obj_mean"] = stats["obj_mean"] + float(prob_obj.mean())
                stats["_obj_n"] = stats.get("_obj_n", 0) + 1
                stats["cls_max"] = max(stats["cls_max"], float(cscore.max()))
                stats["n_peak"] += int(peak_mask.sum()) if peak_mask is not None else score.numel()
                stats["n_above"] += int((top.values > self.score_thresh).sum())

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
            stats["n_out"] += int(box_keep.sum())
        stats["obj_mean"] = stats["obj_mean"] / max(1, stats.pop("_obj_n", 1))
        self.last_stats = stats
        return results

    @property
    def anomaly_logit_gain(self):
        return getattr(self, "_anomaly_logit_gain", 1.0)

    def set_anomaly_logit_gain(self, value: float):
        self._anomaly_logit_gain = float(value)
