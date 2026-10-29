"""Experimental fixed-shape ONNX export of six DINOv2 detector tensors.

Decoding, thresholding and NMS are external. No TensorRT/INT8 support is claimed.
Actual DINOv2 ONNX operator compatibility must be verified in the target environment.
"""
import argparse

import torch
from torch import nn

from _common import load_model, resolve


class ExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.export_forward(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cfg", default="configs/jingwei_dinov2.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="runs/export/dinov2_detector.onnx")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    try:
        import onnx
    except ImportError as exc:
        raise SystemExit("ONNX export requires: pip install onnx") from exc
    model, cfg = load_model(args.cfg, args.ckpt, args.device)
    model.eval()
    path = resolve(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(ExportWrapper(model), torch.zeros(1, 3, *cfg.input_size, device=args.device),
                      str(path), opset_version=args.opset, input_names=["images"],
                      output_names=["cls3", "reg3", "obj3", "cls5", "reg5", "obj5"],
                      dynamic_axes=None, do_constant_folding=True)
    if args.check:
        onnx.checker.check_model(onnx.load(str(path)))
    print(f"exported -> {path}; six prediction tensors, NMS must run externally")


if __name__ == "__main__":
    main()
