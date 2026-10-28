"""JingWei (经纬) — top-level model.

                     1280 x 1280 x 3
                            |
                 [LocalContrastNorm]  reflectance / 反光 removal
                            |
     +----------------------+-----------------------+
     |            shared WeaveCSP backbone           |   (one pass only)
     +----------------------+-----------------------+
              P3 / P4 / P5  |  P3 / P4 / P5
              +-------------+-------------+
              |                           |
     DualScalePAN + NMS-free head    NormalityBranch
     -> 19-class boxes/classes      B1 compressed PatchCore bank
        (one-to-one, no NMS)        B2 Dinomaly reconstruction residual
              |                           |  anomaly map (160x160)
              +-------------+-------------+
                            |
              fusion + deterministic safety net + ROI refiner
                            |
              19 known classes + 1 未知异常, guaranteed large-defect recall
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .backbone import WeaveCSP
from .config import Cfg, load_config
from .foreground import FabricContext, build_foreground_target
from .fusion import ROIRefiner, fuse_detections
from .head import NMSFreeDetect
from .losses import (DetectionLoss, HardMiningCosineLoss,
                     sanitize_detection_targets)
from .neck import DualScalePAN
from .normality import NormalityBranch


class JingWei(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        self.img_size = tuple(int(v) for v in cfg.input_size)
        self.unknown_label = int(cfg.data.nc)          # extra class id = 19

        self.backbone = WeaveCSP(cfg)
        channels = self.backbone.out_channels          # (cd, c3, c4, c5)
        self.neck = DualScalePAN(cfg, channels)
        self.head = NMSFreeDetect(cfg, channels)
        self.normality = NormalityBranch(cfg, channels) if cfg.normality.enabled else None
        self.roi = ROIRefiner(cfg, channels) if cfg.fusion.roi_head else None
        self.foreground_enabled = bool(cfg.get("foreground", {}).get("enabled", False))
        self.foreground = (FabricContext(cfg, channels)
                           if self.foreground_enabled else None)
        self.fg_loss_weight = float(cfg.foreground.loss_weight) if self.foreground_enabled else 0.0

        self.det_loss = DetectionLoss(cfg)
        self.rec_loss = HardMiningCosineLoss(cfg.train.rec_p_final,
                                             cfg.train.rec_warmup,
                                             cfg.train.rec_factor)
        self.rec_weight = float(cfg.train.rec_weight)
        self.head.set_anomaly_logit_gain(float(cfg.fusion.anomaly_logit_gain))
        self._normality_active = True

    def set_stage(self, stage: str):
        """'detector' skips the normality forward in training (saves ~5 G MACs);
        'normality' / 'full' keeps it."""
        self._normality_active = (stage != "detector")
        return self

    # ----------------------------------------------------------------------------------
    def forward(self, x, targets=None, proto_id=0, it: int = 0):
        """Training returns a dict of losses; eval returns list[B] of detections."""
        feats_bb = self.backbone(x)
        feats = self.neck(feats_bb)

        if self.training:
            preds = self.head(feats, training=True)
            out = {"preds": preds, "feats": feats}
            if targets is not None:
                # Drop malformed boxes / class ids (and apply a 1-based label
                # offset if configured) BEFORE both the detection loss and the
                # foreground pseudo-target builder touch them on CUDA.
                targets = sanitize_detection_targets(
                    targets, self.unknown_label, self.img_size, x.device,
                    self.det_loss.label_offset)
            losses = self.det_loss(preds, targets, self.img_size) if targets is not None else None
            if self.normality is not None and self._normality_active:
                nrm = self.normality(feats, proto_id, return_parts=True)
                out["normality"] = nrm
                if losses is not None:
                    rec = self.rec_loss(nrm["target"], nrm["reconstructed"], it)
                    losses["rec"] = rec
                    losses["total"] = losses["total"] + self.rec_weight * rec
            # Weakly-supervised LOCAL + GLOBAL attention foreground context:
            # YOLO boxes supervise defect_fg; per-frame dynamic pseudo targets
            # supervise fabric_mask / fold_band / global cloth prior.  Normal
            # fold-in has no boxes, so it is an explicit negative for defect_fg.
            if self.foreground is not None and targets is not None:
                parts = self.foreground.forward_parts(feats)
                tgt_fg = build_foreground_target(
                    targets, self.img_size, self.foreground.out_grid,
                    x.device, x.dtype,
                    shrink=float(self.cfg.foreground.target_shrink))
                tgt_fabric, tgt_fold, tgt_global = self.foreground.build_targets(
                    x, tgt_fg)
                fgl, fgl_parts = self.foreground.loss(
                    parts, tgt_fg, tgt_fabric, tgt_fold, tgt_global)
                if losses is None:
                    losses = {"total": fgl * self.fg_loss_weight}
                else:
                    losses["fg_context"] = fgl
                    losses["fg_defect"] = fgl_parts["defect"]
                    losses["fg_fabric"] = fgl_parts["fabric"]
                    losses["fg_fold"] = fgl_parts["fold"]
                    losses["fg_global"] = fgl_parts["global"]
                    losses["total"] = losses["total"] + self.fg_loss_weight * fgl
                out["foreground"] = parts["defect_fg"]
                out["fabric_mask"] = parts["fabric_mask"]
                out["fold_band"] = parts["fold_band"]
            if losses is not None:
                out["losses"] = losses
            return out

        return self.inference_from_features(feats, x.shape[-2:], proto_id, x=x)

    # ----------------------------------------------------------------------------------
    @torch.no_grad()
    def inference_from_features(self, feats, img_size, proto_id=0, x=None):
        preds = self.head(feats, training=False)
        anomaly, context = None, None
        if self.normality is not None:
            raw = self.normality(feats, proto_id)["anomaly"]        # (B,1,G,G)
            if self.foreground is not None and x is not None:
                anomaly, context, _ = self.foreground(x, feats, raw)
            else:
                anomaly = raw
        dets = self.head.decode(preds, img_size, anomaly_boost=anomaly)

        B = preds[0]["cls"].shape[0]
        results = []
        for b in range(B):
            d = dets[b]
            if anomaly is not None:
                fb = {k: v[b:b + 1] for k, v in feats.items()}
                ctx = context[b:b + 1] if context is not None else None
                d = fuse_detections(d, anomaly[b:b + 1], img_size, self.cfg,
                                    self.unknown_label, feats=fb,
                                    roi_refiner=self.roi if self.cfg.fusion.roi_head else None,
                                    context=ctx)
            else:
                keep = d["scores"] >= float(self.cfg.fusion.final_score_thresh)
                d = {k: v[keep] for k, v in d.items()}
            results.append(d)
        return results

    # ----------------------------------------------------------------------------------
    def export_forward(self, x, proto_id: int = 0, return_feats: bool = False):
        """Tensor-only forward for ONNX/TensorRT.

        ``decode`` + the connected-component safety net run OUTSIDE the graph
        via :meth:`postprocess_from_tensors`. With ``foreground.enabled`` an
        8th tensor ``context (1,3,G,G) = [defect_fg, fabric_mask, fold_band]``
        is exported.  All three maps come from the local+global attention
        context head; ``fold_band`` is dynamic, not a fixed pixel edge strip.
        When ``roi_head`` is enabled n3/n5 are appended last.

        Returns: cls3, reg3, obj3, cls5, reg5, obj5, anomaly [, context]
                 [, n3, n5]
        """
        assert not self.training
        feats = self.neck(self.backbone(x))
        preds = self.head(feats, training=False)
        raw = self.normality(feats, proto_id)["anomaly"] if self.normality is not None \
            else torch.zeros((x.shape[0], 1, self.cfg.normality.out_grid,
                              self.cfg.normality.out_grid), device=x.device, dtype=x.dtype)
        anomaly, context = raw, None
        if self.foreground is not None:
            anomaly, context, _ = self.foreground(x, feats, raw)
        outs = (preds[0]["cls"], preds[0]["reg"], preds[0]["obj_o2o"],
                preds[1]["cls"], preds[1]["reg"], preds[1]["obj_o2o"], anomaly)
        if context is not None:
            outs = outs + (context,)
        if return_feats:
            outs = outs + (feats["n3"], feats["n5"])
        return outs

    # ----------------------------------------------------------------------------------
    @torch.no_grad()
    def postprocess_from_tensors(self, outs, img_size, proto_id=0):
        """Graph-external decode + safety net (+ optional ROI refiner).

        Counterpart of :meth:`export_forward`.  The optional ``context`` tensor
        (learned local+global attention maps) is what lets the safety net reject
        background and normal edge fold-in without fixed image-border pixels.
        """
        cls3, reg3, obj3, cls5, reg5, obj5, anomaly = outs[:7]
        idx = 7
        context = None
        if self.foreground_enabled and len(outs) > idx:
            context, idx = outs[idx], idx + 1
        feats = ({"n3": outs[idx], "n5": outs[idx + 1]}
                 if len(outs) >= idx + 2 else None)
        preds = [dict(cls=cls3, reg=reg3, obj_o2o=obj3),
                 dict(cls=cls5, reg=reg5, obj_o2o=obj5)]
        raw = self.head.decode(preds, img_size, anomaly_boost=anomaly)

        B = cls3.shape[0]
        results = []
        for b in range(B):
            d = raw[b]
            fb = {k: v[b:b + 1] for k, v in feats.items()} if feats is not None else None
            ctx = context[b:b + 1] if context is not None else None
            d = fuse_detections(d, anomaly[b:b + 1], img_size, self.cfg,
                                self.unknown_label, feats=fb,
                                roi_refiner=self.roi if (self.cfg.fusion.roi_head and fb) else None,
                                context=ctx)
            results.append(d)
        return results

    # ----------------------------------------------------------------------------------
    @torch.no_grad()
    def strip_train_only(self):
        """Remove o2m / train-only fg-alignment branches before export.

        The deployed ``FabricContext`` local+global attention head is
        deliberately kept; it is part of the inference graph.
        """
        for h in self.head.heads:
            h.obj_pred = None
            h.fg_pred = None
            h.fg_stem = None
        return self

    def parameter_groups(self, stage: str):
        """Return the parameter groups used by each training stage."""
        if stage == "detector":
            fg = list(self.foreground.parameters()) if self.foreground is not None else []
            return list(self.backbone.parameters()) + list(self.neck.parameters()) + \
                   list(self.head.parameters()) + fg + \
                   (list(self.roi.parameters()) if self.roi else [])
        if stage == "normality":
            return list(self.normality.normality_parameters())
        if stage == "roi":
            return list(self.roi.parameters()) if self.roi else []
        raise ValueError(stage)


def build_model(cfg) -> JingWei:
    if isinstance(cfg, (str, Path)):
        cfg = load_config(cfg)
    return JingWei(cfg)
