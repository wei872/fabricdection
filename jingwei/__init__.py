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
from .model import JingWei, build_model

__all__ = ["Cfg", "load_config", "JingWei", "build_model"]
__version__ = "0.1.0"
