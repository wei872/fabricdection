"""ONNX / TensorRT export.

    python tools/export.py --cfg configs/jingwei_n.yaml --ckpt runs/bank/memory_bank.pt \
        --proto 0 --out runs/export/jingwei_n.onnx

Only the *tensor* part of the graph is exported (backbone + neck + one-to-one
head + normality branch). ``decode`` and the connected-component safety net run
as fixed-shape, bounded CPU/CUDA post-processing outside the graph — this keeps
latency deterministic and avoids exporting data-dependent control flow.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from _common import ROOT, load_model
from jingwei.image_io import console_safe

try:
    import onnx as _onnx_pkg  # torch.onnx.export needs it to materialise the file
except ImportError:
    _onnx_pkg = None
HAS_ONNX = _onnx_pkg is not None


class ExportWrapper(nn.Module):
    def __init__(self, model, proto_id):
        super().__init__()
        self.model = model
        self.proto_id = int(proto_id)
        # ROI refiner needs n3/n5; append them to the engine outputs when enabled
        self.return_feats = bool(model.cfg.fusion.roi_head)

    def forward(self, x):
        return self.model.export_forward(x, self.proto_id, self.return_feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/jingwei_n.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--proto", type=int, default=0, help="fabric prototype baked into the bank conv")
    ap.add_argument("--out", default="runs/export/jingwei_n.onnx")
    ap.add_argument("--opset", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--check", action="store_true", help="validate with onnx.checker if available")
    args = ap.parse_args()

    if not HAS_ONNX:
        raise SystemExit("`pip install onnx` is required for export (torch.onnx needs it)")

    model, cfg = load_model(args.cfg, args.ckpt, args.device)
    model.eval().strip_train_only()
    if model.normality is not None:
        model.normality.bank.set_active(args.proto)
    opset = args.opset or int(cfg.deploy.opset)
    size = int(cfg.input_size[0])
    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    wrapper = ExportWrapper(model, args.proto).eval()
    dummy = torch.zeros(1, 3, size, size, device=args.device)
    names = ["cls3", "reg3", "obj3", "cls5", "reg5", "obj5", "anomaly"]
    G = int(cfg.normality.out_grid)
    shape_note = ["(1,19,160,160)", "(1,4,160,160)", "(1,1,160,160)",
                  "(1,19,40,40)", "(1,4,40,40)", "(1,1,40,40)",
                  f"(1,1,{G},{G})"]
    if model.foreground is not None:
        names += ["context"]
        shape_note += [f"(1,3,{G},{G}) = [defect_fg, fabric_mask, fold_band]"]
    if wrapper.return_feats:
        names += ["n3", "n5"]
        shape_note += ["(1,96,160,160)", "(1,384,40,40)"]
        print("roi_head is ON -> engine also exports n3/n5 for the ROI refiner. "
              "On GTX 1660 SUPER prefer configs/jingwei_p.yaml (roi_head: false).")
    torch.onnx.export(
        wrapper, dummy, str(out), opset_version=opset,
        input_names=["images"], output_names=names,
        dynamic_axes=None, do_constant_folding=True)
    print(console_safe(f"exported -> {out}"))
    print("outputs:", names)
    print("shapes at 1280:", ", ".join(shape_note))

    if args.check and HAS_ONNX:
        m = _onnx_pkg.load(str(out))
        _onnx_pkg.checker.check_model(m)
        print("onnx checker: OK")

    print("\nTensorRT (on the target device; GTX 1660 SUPER can use TensorRT, "
          "but has NO Tensor Cores):")
    shapes = (f"--minShapes=images:1x3x{size}x{size} "
              f"--optShapes=images:1x3x{size}x{size} "
              f"--maxShapes=images:1x3x{size}x{size}")
    print("  # FP32 first - recommended on TU116/TU117 (no Tensor Cores)")
    print(console_safe(
        f"  trtexec --onnx={out} --saveEngine={out.with_suffix('.engine')} "
        f"--workspace=4096 {shapes}"))
    print("  # FP16 A/B - NOT guaranteed faster on GTX 16-series; measure P99")
    print(console_safe(
        f"  trtexec --onnx={out} --saveEngine={out.with_suffix('.fp16.engine')} "
        f"--fp16 --workspace=4096 {shapes}"))
    print("  # INT8: not recommended on GTX 16-series (no INT8 Tensor Cores); "
          "if used, provide a calibration cache from ~500 normal images "
          "(one per fabric prototype).")


if __name__ == "__main__":
    main()
