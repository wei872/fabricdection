"""Experimental fixed-shape ONNX export of six DINOv2 detector tensors.

Decoding, thresholding and NMS are external. No TensorRT/INT8 support is claimed.
Actual DINOv2 ONNX operator compatibility must be verified in the target environment.
"""
import argparse
import json

import torch
from torch import nn

from _common import load_model, resolve


class ExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.export_forward(x)


def export_metadata(model, cfg):
    return dict(input_shape=[1, 3, *cfg.input_size], architecture=model.architecture_signature,
                prediction_strides=list(model.head.strides), reg_scale=model.head.reg_scale,
                nc=model.head.nc, names=list(cfg.data.names),
                score="sigmoid(obj_o2o) * max(sigmoid(cls))",
                box_distances="reg * reg_scale * prediction_stride",
                top_k=model.head.top_k, head_score_threshold=model.head.score_thresh,
                final_score_threshold=float(cfg.inference.final_score_thresh),
                nms_iou=model.head.nms_iou, known_only=True,
                outputs=["cls3", "reg3", "obj3", "cls5", "reg5", "obj5"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cfg", default="configs/jingwei_dinov2.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="runs/export/dinov2_detector.onnx")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--known-only", action="store_true", help="acknowledge export excludes novelty prototypes and regions")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    try:
        import onnx
    except ImportError as exc:
        raise SystemExit('ONNX export requires: pip install "numpy<2" "ml_dtypes<0.6" onnx') from exc
    model, cfg = load_model(args.cfg, args.ckpt, args.device)
    if model.novelty is not None and not args.known_only:
        ap.error("Six-tensor ONNX contains known detection only; explicitly pass --known-only")
    model.eval()
    path = resolve(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(ExportWrapper(model), torch.zeros(1, 3, *cfg.input_size, device=args.device),
                      str(path), opset_version=args.opset, input_names=["images"],
                      output_names=["cls3", "reg3", "obj3", "cls5", "reg5", "obj5"],
                      dynamic_axes=None, do_constant_folding=True)
    if args.check:
        onnx.checker.check_model(onnx.load(str(path)))
    metadata_path = path.with_suffix(".json")
    metadata_path.write_text(json.dumps(export_metadata(model, cfg), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"exported -> {path}; decoder metadata -> {metadata_path}; NMS must run externally")


if __name__ == "__main__":
    main()
