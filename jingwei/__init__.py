"""JingWei (经纬) — real-time fabric defect detection.

A dual-branch architecture that combines

* a YOLO27-inspired real-time, dual-scale, NMS-free **supervised detector**,
* a PatchCore-inspired **fabric-conditioned normality memory bank**, and
* a Dinomaly-inspired **lightweight reconstruction residual** branch,

fused by a deterministic **safety net** that guarantees every large anomalous
region yields at least one bounding box.

See ``docs/DESIGN.md`` for the full design rationale.
"""

from .config import Cfg, load_config
from .image_io import (VideoCaptureUnicode, ascii_safe, console_safe,
                       imread_unicode, imwrite_unicode)
from .model import JingWei, build_model
from .utils import torch_load

__all__ = [
    "Cfg", "load_config", "JingWei", "build_model",
    "imread_unicode", "imwrite_unicode", "VideoCaptureUnicode",
    "ascii_safe", "console_safe", "torch_load",
]
__version__ = "0.2.0"
