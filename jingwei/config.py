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
    """Load a YAML config. Unknown/missing keys are the caller's responsibility."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} does not contain a YAML mapping")
    cfg = Cfg(raw)
    cfg["_config_path"] = str(path)
    return cfg


def merge(defaults: dict, overrides: dict) -> Cfg:
    """Recursive merge used by CLI overrides (``--opts model.width=0.75``)."""
    out = Cfg({k: (Cfg(v) if isinstance(v, dict) else v) for k, v in defaults.items()})
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = value
    return out
