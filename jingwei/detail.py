"""Small native-resolution texture path; no second foundation encoder."""
import torch
from torch import nn
from .layers import ConvBNAct, DWConv


class LocalDetail(nn.Module):
    """Read every input pixel before learned downsampling to stride eight.

    PixelUnshuffle(2) is lossless; unlike resizing the image to 224 it exposes
    native 2x2 RGB neighborhoods as channels. Later learned downsampling is not
    lossless and does not guarantee recall for arbitrary sub-grid defects.
    """
    def __init__(self, out_channels):
        super().__init__()
        self.pack = nn.PixelUnshuffle(2)
        self.layers = nn.Sequential(
            ConvBNAct(12, 16, 1),
            DWConv(16, 24, 3, 2),
            DWConv(24, 32, 3, 2),
            DWConv(32, 32, 3),
            ConvBNAct(32, out_channels, 1, act=False),
        )
        self.gate = nn.Parameter(torch.tensor(.1))

    def forward(self, x):
        return self.gate.tanh() * self.layers(self.pack(x))
