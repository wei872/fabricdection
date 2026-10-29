"""Tiny local hub fixture: tests never download or claim real pretrained accuracy."""
from pathlib import Path

import torch
from torch import nn
import yaml


class FakeEncoder(nn.Module):
    embed_dim = 12

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(nn.Linear(12, 12) for _ in range(12))
        self.proj = nn.Conv2d(3, 12, 14, 14)

    def get_intermediate_layers(self, x, n, reshape):
        assert tuple(n) == (2, 5, 8, 11) and reshape
        f = self.proj(x)
        return tuple(f * (1 + i / 12) for i in n)


def local_config(root):
    import inspect
    root = Path(root)
    hub = root / "hub"
    hub.mkdir(exist_ok=True)
    (hub / "hubconf.py").write_text(
        "from torch import nn\n" + inspect.getsource(FakeEncoder) +
        "\ndef dinov2_vitb14(pretrained=True):\n    return FakeEncoder()\n", encoding="utf-8")
    repo = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((repo / "configs/jingwei_dinov2.yaml").read_text(encoding="utf-8"))
    cfg["input_size"] = [128, 128]
    cfg["model"].update(dino_repo=str(hub), dino_input_size=56)
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return path
