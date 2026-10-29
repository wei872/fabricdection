"""Batch-1 latency acceptance, optionally including real-frame preprocessing/H2D.

Camera waiting, image decoding from disk, display and disk output are excluded.
A synthetic model-only measurement is NEVER marked as full 1660 acceptance.
"""
import argparse
import json
import time

import numpy as np
import torch

from _common import configure_inference, load_model, resolve
from predict import predict_image, source_images
from jingwei.utils import count_parameters


@torch.inference_mode()
def measure(model, cfg, device, warmup=10, iters=100, frames=None, half=False):
    if warmup < 0 or iters < 1 or frames == []:
        raise ValueError("warmup >= 0, iters > 0 and nonempty frames required")
    x = torch.rand(1, 3, *cfg.input_size, device=device)
    if half:
        x = x.half()
    def sync():
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
    known, unknown = [], []
    def step(index):
        if frames is None:
            return model(x)[0]
        det = predict_image(model, frames[index % len(frames)], int(cfg.input_size[0]), device, half)
        # Include host availability of boxes/scores for downstream controls.
        return {k: v.cpu() if torch.is_tensor(v) else v for k, v in det.items()}
    for i in range(warmup):
        step(i)
    sync()
    elapsed = []
    for i in range(iters):
        sync()
        t0 = time.perf_counter()
        det = step(i)
        sync()
        elapsed.append((time.perf_counter() - t0) * 1000.)
        known.append(len(det["boxes"]))
        unknown.append(len(det.get("unknown_boxes", [])))
    return dict(mean_ms=float(np.mean(elapsed)), p50_ms=float(np.percentile(elapsed, 50)),
                p95_ms=float(np.percentile(elapsed, 95)), p99_ms=float(np.percentile(elapsed, 99)),
                fps_mean=1000. / max(float(np.mean(elapsed)), 1e-6),
                fps_p50=1000. / max(float(np.percentile(elapsed, 50)), 1e-6),
                max_known_detections=max(known), max_unknown_regions=max(unknown))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cfg", default="configs/jingwei_dinov2.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--source", help="representative known/unknown/normal images, folder or video")
    ap.add_argument("--max-frames", type=int, default=32)
    ap.add_argument("--half", action="store_true", help="CUDA only; may NOT be faster on GTX 1660")
    ap.add_argument("--known-only", action="store_true")
    ap.add_argument("--target-fps", type=float, default=30.)
    ap.add_argument("--require-target", action="store_true", help="exit 2 unless full real-frame GTX1660 acceptance passes")
    ap.add_argument("--out", default="runs/benchmark.json")
    args = ap.parse_args()
    if args.target_fps <= 0 or args.max_frames < 1:
        ap.error("target-fps and max-frames must be positive")
    model, cfg = load_model(args.cfg, args.ckpt, args.device)
    configure_inference(model, args.known_only, args.half)
    frames = None
    if args.source:
        frames = []
        for _, frame in source_images(resolve(args.source)):
            if frame is None:
                ap.error("source contains unreadable images")
            frames.append(frame)
            if len(frames) >= args.max_frames:
                break
    name = torch.cuda.get_device_name(args.device) if torch.device(args.device).type == "cuda" else "CPU"
    result = {"device": args.device, "hardware": name, "params": count_parameters(model),
              "encoder": model.backbone.name, "encoder_input": model.backbone.encoder_size,
              "detector_input": list(cfg.input_size), "precision": "fp16" if args.half else "fp32",
              "scope": "real_frame_preprocess_h2d_model_postprocess_d2h" if frames else "synthetic_model_only",
              "unknown_enabled": model.novelty is not None and not args.known_only,
              "frames": len(frames) if frames else 0,
              **measure(model, cfg, args.device, args.warmup, args.iters, frames, args.half)}
    budget = 1000 / args.target_fps
    result["target_fps"] = args.target_fps
    result["latency_pass_this_run"] = result["fps_mean"] >= args.target_fps and result["p95_ms"] <= budget
    result["measurement_sufficient"] = bool(frames and len(frames) >= 10 and args.warmup >= 10 and args.iters >= 100)
    result["source_sizes_wh"] = sorted({(frame.shape[1], frame.shape[0]) for frame in frames}) if frames else []
    result["gtx1660_full_acceptance"] = bool(
        "1660" in name and result["measurement_sufficient"] and result["unknown_enabled"] and result["latency_pass_this_run"])
    result["quality_acceptance"] = "NOT measured: require independent known/unknown recall and normal FPR"
    path = resolve(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print("Excludes camera wait, disk decoding, display and writing. No accuracy guarantee.")
    if args.require_target and not result["gtx1660_full_acceptance"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
