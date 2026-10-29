"""Dual-scale supervised detector with shared class/box predictions.

Dense objectness and Hungarian one-to-one objectness are trained together;
inference uses the latter plus configurable class-aware NMS. A box-derived
alignment auxiliary head is training-only. P3/P5 predict; P4 only fuses features.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import detector_strides
from .layers import DWConv
from .utils import boxes_from_distances, classwise_nms


class DecoupledHead(nn.Module):
    def __init__(self, c: int, nc: int, width: float, training_branch: bool = True,
                 reg_activation: str = "softplus"):
        super().__init__()
        h = max(16, int(c * width))
        stem = lambda: nn.Sequential(DWConv(c, h, 3), DWConv(h, h, 3))
        self.cls_stem, self.reg_stem = stem(), stem()
        if reg_activation not in ("relu", "softplus"):
            raise ValueError("reg_activation must be relu or softplus")
        self.reg_activation = reg_activation
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
            "reg": (F.softplus(self.reg_pred(f_reg)) if self.reg_activation == "softplus"
                    else F.relu(self.reg_pred(f_reg))),
            "obj": self.obj_pred(x) if training else None,
            "obj_o2o": self.obj_o2o(x),
        }
        out["fg"] = self.fg_pred(self.fg_stem(x)) if (training and self.fg_stem is not None) else None
        return out


class DualScaleDetect(nn.Module):
    def __init__(self, cfg, channels):
        super().__init__()
        m = cfg.model
        self.nc = int(cfg.data.nc)
        self.strides = detector_strides(cfg)
        self.top_k = int(m.top_k)
        self.score_thresh = float(m.score_thresh)
        self.reg_scale = float(m.reg_scale)
        self.heads = nn.ModuleList([
            DecoupledHead(channels[1], self.nc, m.head_width, m.fg_align,
                          m.get("reg_activation", "softplus")),
            DecoupledHead(channels[3], self.nc, m.head_width, m.fg_align,
                          m.get("reg_activation", "softplus")),
        ])
        self.nms_iou = float(m.get("detector_nms_iou", 0.0))
        if not 0.0 <= self.nms_iou <= 1.0:
            raise ValueError("detector_nms_iou must be between 0 and 1 (0 disables NMS)")
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

    # ----------------------------------------------------------------------------------
    @torch.no_grad()
    def decode(self, preds, img_size):
        """top-k + threshold over the one-to-one branch; optional class-aware NMS.
        Set detector_nms_iou=0 to skip NMS and its data-dependent work.

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

                cls_prob = cls.sigmoid()
                cscore, clabel = cls_prob.max(dim=-1)
                score = obj.sigmoid() * cscore

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
            result = dict(boxes=boxes, scores=scores[sel], labels=labels,
                          obj=obj, points=pts, scale_idx=si)
            if self.nms_iou > 0:
                keep = classwise_nms(boxes, result["scores"], labels, self.nms_iou)
                result = {key: value[keep] for key, value in result.items()}
            results.append(result)
        return results
