"""Calibrate shared-feature novelty on independent clean normal images only."""
import argparse
import math

import torch
from torch.utils.data import DataLoader

from _common import data_paths, load_model, resolve, save_checkpoint
from jingwei.data import NormalImageDataset, collate
from jingwei.novelty import texture_descriptor


def conformal_threshold(scores, target_fpr):
    if not 0 < target_fpr < 1 or not scores:
        raise ValueError("Need scores and 0 < target_fpr < 1")
    # Image-max score controls an image-level unknown alert, not patch FPR.
    rank = math.ceil((len(scores) + 1) * (1 - target_fpr))
    if rank > len(scores):
        raise ValueError(f"Too few normal calibration images for FPR {target_fpr}; "
                         f"need at least {math.ceil(1 / target_fpr) - 1}")
    return sorted(scores)[rank - 1]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cfg", default="configs/jingwei_dinov2.yaml")
    ap.add_argument("--data")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="runs/detector_dinov2/calibrated.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--target-fpr", type=float, default=.01)
    ap.add_argument("--half", action="store_true", help="CUDA FP16; inference must use the same precision")
    args = ap.parse_args()
    model, cfg = load_model(args.cfg, args.ckpt, args.device)
    if model.novelty is None or model.novelty.updates.item() == 0:
        ap.error("Need unknown.enabled and prototypes trained on normaltrain")
    if model.novelty.texture is not None and int(model.novelty.texture.updates) == 0:
        ap.error("Native texture prototypes are untrained; first train with normaltrain, not normalcal")
    if args.half and not args.device.startswith("cuda"):
        ap.error("--half requires CUDA")
    model.eval()
    if args.half:
        model.half()
    paths = data_paths(cfg, args.data)
    if not paths.get("normalcal"):
        ap.error("Set normalcal to independent clean calibration images (not normaltrain/normalval)")
    calibration_dir = resolve(paths["normalcal"]).resolve()
    for key in ("normaltrain", "normalval", "train", "val"):
        if paths.get(key):
            other = resolve(paths[key]).resolve()
            if calibration_dir == other or other in calibration_dir.parents or calibration_dir in other.parents:
                ap.error(f"normalcal overlaps directory {key}; use disjoint data splits")
    dataset = NormalImageDataset(calibration_dir, int(cfg.input_size[0]), augment=False)
    scores = []
    for batch in DataLoader(dataset, batch_size=args.batch, collate_fn=collate):
        x = batch["image"].to(args.device)
        if args.half:
            x = x.half()
        features = model.backbone(x, return_raw=True, include_detail=False)["raw"]
        texture = texture_descriptor(x) if model.novelty.texture is not None else None
        scores.extend(model.novelty.score(features, texture).flatten(1).amax(1).cpu().tolist())
    threshold = conformal_threshold(scores, args.target_fpr)
    model.novelty.threshold.fill_(threshold)
    model.novelty.calibration_shape.copy_(torch.tensor([model.backbone.encoder_size, int(cfg.input_size[0])], device=model.novelty.calibration_shape.device))
    model.novelty.calibrated.fill_(True)
    model.novelty.precision_bits.fill_(16 if args.half else 32)
    # Inference/weight-only artifact: never confuse it with a full resume state.
    save_checkpoint({"model": model.state_dict(), "calibration": {
        "normalcal": str(calibration_dir), "images": len(scores),
        "target_unknown_image_fpr": args.target_fpr, "threshold": float(model.novelty.threshold),
        "precision": "fp16" if args.half else "fp32",
        "warning": "Exchangeability assumption only; not a guarantee under fabric/lighting domain shift"
    }}, resolve(args.out))
    print(f"Saved {args.out}; unknown threshold={float(model.novelty.threshold):.6f}, n={len(scores)}")
    print("Validate FPR and held-out unknown recall independently; this is not an accuracy guarantee.")


if __name__ == "__main__":
    main()
