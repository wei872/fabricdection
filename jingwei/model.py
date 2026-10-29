"""Pretrained DINOv2 → feature pyramid → PAN → supervised defect detector."""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Cfg, load_config
from .dinov2_backbone import DINOv2Backbone
from .head import DualScaleDetect
from .losses import DetectionLoss
from .layers import ConvBNAct, DWConv
from .neck import DualScalePAN
from .novelty import NormalPrototypes, texture_descriptor


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
                                        u.get("min_patches", 1), u.get("max_regions", 8),
                                        texture_detail=u.get("texture_detail", True))
                        if u.get("enabled", False) else None)

    @property
    def architecture_signature(self):
        return dict(local_detail=self.backbone.local_detail is not None,
                    texture_detail=self.novelty is not None and self.novelty.texture is not None,
                    detector_input=int(self.img_size[0]),
                    pyramid_size=int(self.img_size[0]) // self.backbone.pyramid_ratio,
                    strides=list(self.head.strides))

    def architecture_compatible(self, saved):
        if saved == self.architecture_signature:
            return True
        # Old 640 v2 checkpoints predate explicit geometry metadata; their
        # actual computation is unchanged. Never infer legacy high-res geometry.
        legacy = {k: self.architecture_signature[k] for k in ("local_detail", "texture_detail")}
        return (saved == legacy and self.img_size == (640, 640) and
                self.backbone.pyramid_ratio == 1)

    def _features(self, backbone):
        feats = self.neck(backbone)
        if "native_residual" in backbone:
            residual = backbone["native_residual"]
            feats["n3"] = F.interpolate(feats["n3"], size=residual.shape[-2:],
                                         mode="bilinear", align_corners=False) + residual
        return feats

    def train(self, mode=True):
        if mode and getattr(self, "_inference_fused", False):
            raise RuntimeError("Fused model is inference-only; reload the unfused checkpoint for training")
        return super().train(mode)

    def fuse_for_inference(self):
        """Fold Conv-BN / pointwise-BN in eval mode; no saved checkpoint rewrite."""
        if self.training:
            raise RuntimeError("Call eval() before inference fusion")
        if getattr(self, "_inference_fused", False):
            return self
        from torch.nn.utils.fusion import fuse_conv_bn_eval
        count = 0
        for module in list(self.modules()):
            if isinstance(module, (ConvBNAct, DWConv)) and isinstance(module.bn, nn.BatchNorm2d):
                attribute = "conv" if isinstance(module, ConvBNAct) else "pw"
                setattr(module, attribute, fuse_conv_bn_eval(getattr(module, attribute), module.bn))
                module.bn = nn.Identity()
                count += 1
        self._inference_fused = True
        self.fused_bn_count = count
        return self

    def forward(self, x, targets=None, normal_mask=None):
        backbone = self.backbone(x, return_raw=self.novelty is not None and (self.training or not self.known_only), include_detail=False)
        if self.training and self.novelty is not None and normal_mask is not None:
            mask = normal_mask.to(x.device).bool()
            texture = texture_descriptor(x[mask]) if self.novelty.texture is not None else None
            self.novelty.update(backbone["raw"][mask], texture)
        feats = self._features(backbone)
        if self.training:
            preds = self.head(feats, training=True)
            out = {"preds": preds, "feats": feats}
            if targets is not None:
                out["losses"] = self.det_loss(preds, targets, x.shape[-2:])
                out["assignment"] = dict(self.det_loss.last_assignment)
            return out
        detections = self._decode(self.head(feats, training=False), x.shape[-2:])
        if self.novelty is not None and not self.known_only and bool(self.novelty.calibrated):
            texture = texture_descriptor(x) if self.novelty.texture is not None else None
            for detection, regions in zip(detections, self.novelty.regions(
                    backbone["raw"], x.shape[-2:], texture)):
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
        preds = self.head(self._features(self.backbone(x, include_detail=False)), training=False)
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
