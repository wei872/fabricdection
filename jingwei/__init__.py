"""Fabric defect detection using an official pretrained DINOv2 backbone."""

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
