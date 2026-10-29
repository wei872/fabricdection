"""Lightweight nested-dict config with attribute access."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Cfg(dict):
    """A dict whose keys are also reachable as attributes (recursively).

    >>> c = Cfg({"model": {"width": 0.5}})
    >>> c.model.width
    0.5
    """

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:  # fail loudly on typos
            raise AttributeError(f"config has no key {key!r}") from exc
        if isinstance(value, dict) and not isinstance(value, Cfg):
            value = Cfg(value)
            self[key] = value
        return value

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def copy(self) -> "Cfg":
        return Cfg({k: (Cfg(v) if isinstance(v, dict) else v) for k, v in self.items()})


def load_config(path: str | Path) -> Cfg:
    """Load and validate the single supported DINOv2 detector configuration."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} does not contain a YAML mapping")
    retired = {"normality", "foreground", "fusion", "dino"}.intersection(raw)
    model = raw.get("model")
    if retired or (isinstance(model, dict) and model.get("backbone", "dinov2") != "dinov2"):
        raise ValueError("Unsupported legacy configuration. Use configs/jingwei_dinov2.yaml; "
                         "only direct pretrained DINOv2 detection is supported.")
    for key in ("model", "train", "data", "inference"):
        if not isinstance(raw.get(key), dict):
            raise ValueError(f"missing configuration section: {key}")
    size = raw.get("input_size", [])
    if (not isinstance(size, (list, tuple)) or len(size) != 2 or any(not isinstance(v, int) or v <= 0 or v % 32 for v in size)
            or size[0] != size[1]):
        raise ValueError("input_size must be square, positive, and a multiple of 32")
    nc, names = raw["data"].get("nc"), raw["data"].get("names")
    if (not isinstance(nc, int) or nc < 1 or not isinstance(names, list)
            or nc != len(names) or not all(isinstance(name, str) for name in names)):
        raise ValueError("data.nc must be positive and match the list of class names")
    if raw["train"].get("best_metric", "best_f1") not in ("mAP50", "precision", "f1", "best_f1"):
        raise ValueError("train.best_metric must be mAP50, precision, f1 or best_f1")
    if float(raw["train"].get("warmup_epochs", 0)) < 0:
        raise ValueError("train.warmup_epochs must be nonnegative")
    for section, key in (("model", "score_thresh"), ("inference", "final_score_thresh")):
        if not 0 <= float(raw[section][key]) <= 1:
            raise ValueError(f"{section}.{key} must be between 0 and 1")
    u = raw.get("unknown", {})
    if not isinstance(u, dict):
        raise ValueError("unknown must be a mapping")
    if u.get("enabled", False):
        if not raw["model"].get("dino_freeze", True):
            raise ValueError("Shared normal prototypes require dino_freeze=true")
        for key, default in (("prototypes", 32), ("min_patches", 1), ("max_regions", 8)):
            value = u.get(key, default)
            if not isinstance(value, int) or not 1 <= value <= 256:
                raise ValueError(f"unknown.{key} must be an integer between 1 and 256")
        if not 0 <= float(u.get("momentum", .95)) < 1:
            raise ValueError("unknown.momentum must be in [0,1)")
    if not 0 < float(raw["train"].get("normal_ratio", .5)) < 1:
        raise ValueError("train.normal_ratio must be between 0 and 1")
    if not isinstance(raw["model"].get("local_detail", True), bool):
        raise ValueError("model.local_detail must be a boolean")
    if not isinstance(u.get("texture_detail", True), bool):
        raise ValueError("unknown.texture_detail must be a boolean")
    cfg = Cfg(raw)
    cfg["_config_path"] = str(path)
    pyramid_ratio(cfg)
    return cfg


def pyramid_ratio(cfg):
    """Integer native-input to semantic-pyramid ratio (one means legacy)."""
    native = int(cfg.input_size[0])
    size = cfg.model.get("pyramid_size", None)
    if size is None:
        return 1
    if (not isinstance(size, int) or isinstance(size, bool) or size <= 0 or
            size % 32 or size > native or native % size):
        raise ValueError("model.pyramid_size must divide input_size, be <= it, and be a positive multiple of 32")
    if size < native and not cfg.model.get("local_detail", True):
        raise ValueError("Capped semantic pyramids require model.local_detail=true for native details")
    return native // size


def detector_strides(cfg):
    return (8, 32 * pyramid_ratio(cfg))
