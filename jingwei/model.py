"""Pretrained DINOv2 → feature pyramid → PAN → supervised defect detector."""
from pathlib import Path

import torch
import torch.nn as nn

from .config import Cfg, load_config
from .dinov2_backbone import DINOv2Backbone
from .head import DualScaleDetect
from .losses import DetectionLoss
from .neck import DualScalePAN
from .novelty import NormalPrototypes


class JingWei(nn.Module):
    """One model and one training path: direct pretrained DINOv2 detection."""

    def __init__(self, cfg: Cfg):
        super().__init__()
        if cfg.model.get("backbone", "dinov2") != "dinov2":
            raise ValueError("Only the direct DINOv2 backbone is supported")
        self.cfg = cfg
        self.known_only = False
        self.img_size = tuple(int(v) for v in cfg.input_size)
        self.backbone = DINOv2Backbone(cfg)
        channels = self.backbone.out_channels
        self.neck = DualScalePAN(cfg, channels)
        self.head = DualScaleDetect(cfg, channels)
        self.det_loss = DetectionLoss(cfg)
        u = cfg.get("unknown", {})
        self.novelty = (NormalPrototypes(self.backbone.encoder.embed_dim,
                                        u.get("prototypes", 32), u.get("momentum", .95),
                                        u.get("min_patches", 1), u.get("max_regions", 8))
                        if u.get("enabled", False) else None)

    def forward(self, x, targets=None, normal_mask=None):
        backbone = self.backbone(x, return_raw=self.novelty is not None and (self.training or not self.known_only), include_detail=False)
        if self.training and self.novelty is not None and normal_mask is not None:
            self.novelty.update(backbone["raw"][normal_mask.to(x.device).bool()])
        feats = self.neck(backbone)
        if self.training:
            preds = self.head(feats, training=True)
            out = {"preds": preds, "feats": feats}
            if targets is not None:
                out["losses"] = self.det_loss(preds, targets, x.shape[-2:])
                out["assignment"] = dict(self.det_loss.last_assignment)
            return out
        detections = self._decode(self.head(feats, training=False), x.shape[-2:])
        if self.novelty is not None and not self.known_only and bool(self.novelty.calibrated):
            for detection, regions in zip(detections, self.novelty.regions(backbone["raw"], x.shape[-2:])):
                detection.update(regions)
        return detections

    @torch.no_grad()
    def _decode(self, preds, img_size):
        detections = self.head.decode(preds, img_size)
        threshold = float(self.cfg.inference.final_score_thresh)
        return [{k: v[d["scores"] >= threshold] for k, v in d.items()}
                for d in detections]

    def export_forward(self, x):
        """Six prediction tensors; thresholding and NMS run outside ONNX."""
        if self.training:
            raise RuntimeError("Call eval() before export_forward")
        preds = self.head(self.neck(self.backbone(x, include_detail=False)), training=False)
        return tuple(p[key] for p in preds for key in ("cls", "reg", "obj_o2o"))

    @torch.no_grad()
    def postprocess_from_tensors(self, outs, img_size):
        if len(outs) != 6:
            raise ValueError("Expected six detector tensors (cls/reg/obj for each scale)")
        preds = [dict(cls=outs[i], reg=outs[i + 1], obj_o2o=outs[i + 2])
                 for i in (0, 3)]
        return self._decode(preds, img_size)


def build_model(cfg) -> JingWei:
    if isinstance(cfg, (str, Path)):
        cfg = load_config(cfg)
    return JingWei(cfg)
