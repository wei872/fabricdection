"""Direct, pretrained DINOv2 detection backbone (NOT offline distillation).

A frozen ViT supplies four intermediate maps to trainable pyramid adapters.
The encoder remains in the inference graph. Resize interpolates features; it
cannot recover small details lost at the configured encoder input resolution.
"""
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ConvBNAct
from .utils import torch_load


class DINOv2Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg.model
        self.name = str(m.get("dino_name", "dinov2_vitb14"))
        if not self.name.startswith("dinov2_vit"):
            raise ValueError("Direct backbone requires a DINOv2 ViT hub model")
        repo = str(m.get("dino_repo", "facebookresearch/dinov2"))
        source = "local" if Path(repo).is_dir() else "github"
        initialize = bool(m.get("dino_pretrained", True))
        weights = m.get("dino_weights")
        try:
            self.encoder = torch.hub.load(repo, self.name, source=source,
                                          pretrained=initialize and not bool(weights))
            if initialize and weights:
                # Require official backbone-only weights. Never silently leave
                # unmatched parameters random when requesting pretrained features.
                self.encoder.load_state_dict(torch_load(weights, map_location="cpu"), strict=True)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load direct DINOv2 backbone {self.name}. "
                "Set model.dino_repo to a local DINOv2 checkout and "
                "model.dino_weights to official backbone-only pretrained .pth weights. "
                "No random-backbone fallback is used."
            ) from exc
        if not hasattr(self.encoder, "get_intermediate_layers"):
            raise ValueError("DINOv2 backbone must expose get_intermediate_layers")
        self.frozen = bool(m.get("dino_freeze", True))
        self.encoder.requires_grad_(not self.frozen)
        if self.frozen:
            self.encoder.eval()
        self.encoder_size = int(m.get("dino_input_size", 448))
        if self.encoder_size <= 0 or self.encoder_size % 14:
            raise ValueError("model.dino_input_size must be a positive multiple of 14")
        # Four distributed layers, not four copies of the last feature map.
        depth = len(self.encoder.blocks)
        self.layers = tuple(max(0, (depth * i // 4) - 1) for i in range(1, 5))
        if len(set(self.layers)) != 4:
            raise ValueError("DINOv2 encoder needs at least four transformer blocks")
        dim = int(self.encoder.embed_dim)
        self.out_channels = tuple(int(m[k] * m.width)
                                  for k in ("detail_ch", "p3_ch", "p4_ch", "p5_ch"))
        self.strides = (4, 8, 16, 32)
        self.adapters = nn.ModuleList(ConvBNAct(dim, c, 1) for c in self.out_channels)
        self.register_buffer("mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        # model.train() must not switch a frozen foundation encoder to train mode.
        if self.frozen:
            self.encoder.eval()
        return self

    def forward(self, x):
        h, w = x.shape[-2:]
        if h % 32 or w % 32:
            raise ValueError("Detector image dimensions must be multiples of 32")
        z = F.interpolate(x, (self.encoder_size, self.encoder_size),
                          mode="bilinear", align_corners=False)
        z = (z - self.mean) / self.std
        # no_grad, NOT inference_mode: adapters need to save their input for backward.
        with torch.no_grad() if self.frozen else nullcontext():
            maps = self.encoder.get_intermediate_layers(z, n=self.layers, reshape=True)
        return {name: F.interpolate(adapter(feat), (h // stride, w // stride),
                                    mode="bilinear", align_corners=False)
                for name, adapter, feat, stride in
                zip(("d", "p3", "p4", "p5"), self.adapters, maps, self.strides)}
