"""Optional DINO / DINOv2 teacher utilities for **offline** stage-0 distillation.

This module is imported only by ``tools/pretrain_dino.py``.  Nothing here is
part of the deployed ONNX/TensorRT graph: the (large) ViT teacher runs on the
training machine, the lightweight CNN student keeps its 7.59 / 15.65 G MACs.

Why it exists
-------------
Training the detector + normality branch from random weights needs a lot of
labelled defects.  DINOv2 is self-supervised and produces strong dense patch
features without labels, so we use it as a frozen teacher on ``normaltrain``:

    teacher patch tokens ──cosine──> student n3 / n5 / target_proj / q

After stage 0 the teacher is discarded; ``target_proj`` is re-frozen exactly as
Dinomaly's frozen foundation encoder requires (docs/DESIGN.md §6 stage 0).

Supported teachers
------------------
* ``dinov2_vits14`` / ``vitb14`` / ``vitl14`` / ``vitg14`` (+ ``_reg`` variants)
  from ``facebookresearch/dinov2`` via ``torch.hub`` (standard weights are
  Apache-2.0; specialized XRay/Cell/Channel-Adaptive variants are NOT).
* ``dino_vits16`` / ``dino_vitb16`` / ``dino_vits8`` / ``dino_vitb8`` from the
  original ``facebookresearch/dino`` (Apache-2.0).
* ``synthetic`` — a tiny random conv teacher used for smoke tests / CI, so the
  whole distillation loop can be verified without downloading weights.

If the sandbox has no internet, clone the repo and pass ``--repo /path/to/repo``
plus ``--weights /path/to/backbone_only.pth`` to ``tools/pretrain_dino.py``.
"""
from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class TeacherSpec:
    name: str
    family: str
    patch_size: int
    embed_dim: int


def infer_spec(name: str, synthetic_dim: int = 384) -> TeacherSpec:
    """Infer (patch_size, embed_dim) from a torch.hub model name."""
    if name == "synthetic":
        return TeacherSpec(name, "synthetic", 14, int(synthetic_dim))
    if name.startswith("dinov3"):
        raise ValueError("DINOv3 uses a custom license / hub API; use a DINOv2 "
                         "ViT teacher here, or wire DINOv3 explicitly after a legal review.")
    if name.startswith("dino_resnet") or "resnet" in name:
        raise ValueError("DINO ResNet-50 has no dense patch tokens; choose a ViT "
                         "teacher (e.g. dinov2_vits14) for pixel-aligned distillation.")
    m = re.search(r"vit([sblg])(\d+)", name)
    if not m:
        raise ValueError(f"cannot infer patch size / dim from teacher {name!r}; "
                         "use a DINOv2/DINO ViT name or 'synthetic'.")
    dim = {"s": 384, "b": 768, "l": 1024, "g": 1536}[m.group(1)]
    family = "dinov2" if name.startswith("dinov2") else "dino"
    return TeacherSpec(name, family, int(m.group(2)), dim)


class SyntheticPatchTeacher(nn.Module):
    """Deterministic fake DINOv2: 14x14 non-overlapping conv → patch tokens.

    Used by ``--smoke`` and by the unit test.  It has the same interface as a
    real DINOv2 backbone (``forward_features`` → dict with patch tokens) but
    costs almost nothing and has no external dependency.
    """

    def __init__(self, dim: int = 384, patch: int = 14):
        super().__init__()
        self.dim, self.patch = int(dim), int(patch)
        self.proj = nn.Conv2d(3, dim, kernel_size=patch, stride=patch, bias=False)
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward_features(self, x: torch.Tensor) -> dict:
        y = self.proj(x)                          # (B, D, H/p, W/p)
        b, d, h, w = y.shape
        tokens = y.flatten(2).transpose(1, 2)     # (B, N, D)
        return {"x_norm_patchtokens": F.normalize(tokens, dim=-1)}


def _hub_repo(name: str, repo: str | None) -> tuple[str, str]:
    if repo:
        p = Path(repo)
        if p.is_dir():
            return str(p), "local"
        return repo, "github"
    if name.startswith("dino_"):                # DINOv1 lives in a different repo
        return "facebookresearch/dino", "github"
    return "facebookresearch/dinov2", "github"


def load_teacher(name: str, repo: str | None = None, weights: str | None = None,
                 device: str = "cpu", synthetic_dim: int = 384) -> "DINOTeacher":
    """Load a frozen teacher and return a wrapper with ``patch_map(x)``."""
    spec = infer_spec(name, synthetic_dim)
    if spec.family == "synthetic":
        model = SyntheticPatchTeacher(spec.embed_dim, spec.patch_size)
    else:
        hub_repo, source = _hub_repo(name, repo)
        try:
            if spec.family == "dino" and weights:
                # DINOv1 hub functions do not accept a `weights=` kwarg.
                model = torch.hub.load(hub_repo, name, source=source,
                                       pretrained=False)
                sd = torch.load(str(weights), map_location="cpu")
                sd = {k.replace("module.", ""): v for k, v in sd.items()}
                model.load_state_dict(sd, strict=False)
            else:
                kwargs = {"pretrained": True}
                if weights and spec.family == "dinov2":
                    kwargs["weights"] = str(weights)
                model = torch.hub.load(hub_repo, name, source=source, **kwargs)
        except Exception as exc:                       # pragma: no cover - needs net
            raise RuntimeError(
                f"Failed to load DINO teacher {name!r} from {hub_repo!r}.\n"
                "If this machine has no internet, clone the repo and pass:\n"
                "  --repo /path/to/dinov2 --weights /path/to/dinov2_vitb14_pretrain.pth\n"
                "Standard DINOv2 backbone weights are Apache-2.0; do NOT use the "
                "XRay/Cell/Channel-Adaptive variants for commercial deployment."
            ) from exc
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return DINOTeacher(model, spec)


def _extract_tokens(out) -> torch.Tensor:
    """Pull (B, N, C) patch tokens out of the various DINO forward APIs."""
    if isinstance(out, dict):
        for key in ("x_norm_patchtokens", "patch_tokens", "x"):
            v = out.get(key)
            if torch.is_tensor(v) and v.dim() == 3 and v.shape[1] > 1:
                return v
    if isinstance(out, (tuple, list)):
        for v in out:
            if torch.is_tensor(v) and v.dim() == 3 and v.shape[1] > 1:
                return v
    if torch.is_tensor(out) and out.dim() == 3:
        return out
    raise RuntimeError("DINO teacher did not return dense patch tokens; "
                       "use a ViT backbone, not a classifier-only head.")


class DINOTeacher(nn.Module):
    """Normalises RGB, runs the frozen ViT, returns a (B,C,Ht,Wt) feature map."""

    def __init__(self, model: nn.Module, spec: TeacherSpec):
        super().__init__()
        self.model = model
        self.spec = spec
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def patch_map(self, x_rgb_01: torch.Tensor) -> torch.Tensor:
        """x: float RGB in [0,1], shape (B,3,H,W). Returns L2-normalised map."""
        if x_rgb_01.dim() != 4 or x_rgb_01.shape[1] != 3:
            raise ValueError("teacher expects (B,3,H,W) RGB")
        mean = torch.tensor(IMAGENET_MEAN, device=x_rgb_01.device,
                            dtype=x_rgb_01.dtype).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=x_rgb_01.device,
                           dtype=x_rgb_01.dtype).view(1, 3, 1, 1)
        z = (x_rgb_01 - mean) / std
        if hasattr(self.model, "forward_features"):
            out = self.model.forward_features(z)
        elif hasattr(self.model, "get_intermediate_layers"):   # DINOv1 ViT
            out = self.model.get_intermediate_layers(z, n=1)[0]
        else:                                                  # pragma: no cover
            out = self.model(z)
        tokens = _extract_tokens(out).float()
        b, n, c = tokens.shape
        h = x_rgb_01.shape[-2] // self.spec.patch_size
        w = x_rgb_01.shape[-1] // self.spec.patch_size
        if h * w != n:
            s = int(math.isqrt(n))
            if s * s != n:
                raise RuntimeError(f"teacher token count {n} is not a square grid "
                                   f"for input {tuple(x_rgb_01.shape[-2:])}")
            h = w = s
        tokens = F.normalize(tokens, dim=-1)
        return tokens.transpose(1, 2).reshape(b, c, h, w).contiguous()


# --------------------------------------------------------------------------------------
# Views: same geometry for teacher + student, strong photometric only on student
# --------------------------------------------------------------------------------------
def _random_crop(x: torch.Tensor, size: int) -> torch.Tensor:
    b, _, h, w = x.shape
    if h < size or w < size:
        raise ValueError(f"crop {size} larger than image {(h, w)}")
    ys = torch.randint(0, h - size + 1, (b,), device=x.device)
    xs = torch.randint(0, w - size + 1, (b,), device=x.device)
    return torch.stack([x[i, :, ys[i]:ys[i] + size, xs[i]:xs[i] + size]
                        for i in range(b)], dim=0)


def _shared_geometry(x: torch.Tensor) -> torch.Tensor:
    if random.random() < 0.5:
        x = x.flip(-1)
    if random.random() < 0.5:
        x = x.flip(-2)
    k = random.choice((0, 0, 0, 0, 1, 2, 3))
    if k:
        x = torch.rot90(x, k, (-2, -1))
    return x


def strong_photometric(x: torch.Tensor) -> torch.Tensor:
    """Fabric-specific corruption: exposure, reflection, yarn blur, erase."""
    x = x.clone()
    b, _, h, w = x.shape
    contrast = torch.empty(b, 1, 1, 1, device=x.device).uniform_(0.6, 1.4)
    bias = torch.empty(b, 1, 1, 1, device=x.device).uniform_(-0.15, 0.15)
    x = (x - 0.5) * contrast + 0.5 + bias
    gray = x.mean(1, keepdim=True)
    sat = torch.empty(b, 1, 1, 1, device=x.device).uniform_(0.5, 1.5)
    x = gray + (x - gray) * sat
    grayscale = torch.rand(b, 1, 1, 1, device=x.device) < 0.2
    x = torch.where(grayscale, gray.expand_as(x), x)
    if random.random() < 0.5:                     # low-frequency 反光
        gh, gw = max(2, h // 16), max(2, w // 16)
        field = torch.randn(b, 1, gh, gw, device=x.device)
        field = F.interpolate(field, size=(h, w), mode="bilinear", align_corners=False)
        x = x + 0.08 * field
    if random.random() < 0.3:                     # yarn width / slight defocus
        x = F.avg_pool2d(x, 3, 1, 1)
    if random.random() < 0.5:                     # random erasing
        for i in range(b):
            eh = random.randint(max(1, h // 16), max(2, h // 4))
            ew = random.randint(max(1, w // 16), max(2, w // 4))
            y0 = random.randint(0, h - eh)
            x0 = random.randint(0, w - ew)
            x[i, :, y0:y0 + eh, x0:x0 + ew] = 0.0
    return x.clamp_(0.0, 1.0)


def make_views(x: torch.Tensor, crop: int, strong: bool = True):
    """Return (teacher_view, student_view), both (B,3,crop,crop) in [0,1]."""
    patch = _shared_geometry(_random_crop(x, crop))
    teacher_view = patch
    student_view = strong_photometric(patch) if strong else patch.clone()
    return teacher_view, student_view


# --------------------------------------------------------------------------------------
# Student adaptor + cosine distillation loss
# --------------------------------------------------------------------------------------
class DINOStudentAdaptor(nn.Module):
    """Auxiliary 1x1 projections from student features to teacher dim.

    These are training-only.  They are NOT saved into the deployed model state
    dict (the real model already has its own heads), but are checkpointed under
    ``dino_adaptor`` for resume/debugging.
    """

    def __init__(self, model, teacher_dim: int):
        super().__init__()
        ch = model.backbone.out_channels           # (cd, c3, c4, c5)
        self.proj3 = nn.Conv2d(ch[1], teacher_dim, 1)
        self.proj5 = nn.Conv2d(ch[3], teacher_dim, 1)
        self.proj_rec = nn.Conv2d(int(model.cfg.normality.recon_dim), teacher_dim, 1)
        self.proj_patch = nn.Conv2d(int(model.cfg.normality.patch_dim), teacher_dim, 1)

    def forward(self, feats, q, rec_target):
        return {
            "n3": self.proj3(feats["n3"]),
            "n5": self.proj5(feats["n5"]),
            "rec": self.proj_rec(F.normalize(rec_target, dim=1)),
            "patch": self.proj_patch(q),
        }


class DINODistillLoss(nn.Module):
    """Sum of (1 - cosine) after bilinear-resizing teacher map to each scale."""

    def __init__(self, weights: dict | None = None):
        super().__init__()
        self.weights = dict(weights or {"n3": 1.0, "n5": 1.0, "rec": 1.0, "patch": 0.5})

    def forward(self, student_maps: dict, teacher_map: torch.Tensor) -> torch.Tensor:
        loss = teacher_map.new_zeros(())
        for name, s in student_maps.items():
            w = float(self.weights.get(name, 0.0))
            if w <= 0.0:
                continue
            t = F.interpolate(teacher_map, size=s.shape[-2:], mode="bilinear",
                              align_corners=False)
            s = F.normalize(s, dim=1)
            t = F.normalize(t, dim=1)
            loss = loss + w * (1.0 - (s * t).sum(1, keepdim=True)).mean()
        return loss
