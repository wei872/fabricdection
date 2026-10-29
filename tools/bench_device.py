"""Measure the actual DINOv2 detector's latency, including decoding and NMS.

No CNN substitution, hardware-specific configuration switching or FPS promises.
"""
import argparse
import json
import time

import numpy as np
import torch

from _common import load_model, resolve
from jingwei.utils import count_parameters


@torch.no_grad()
def measure(model, cfg, device, warmup=3, iters=20):
    if warmup < 0 or iters < 1:
        raise ValueError("warmup must be nonnegative and iters must be positive")
    x = torch.rand(1, 3, *cfg.input_size, device=device)
    def sync():
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
    for _ in range(warmup):
        model(x)
    sync()
    elapsed = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model(x)
        sync()
        elapsed.append((time.perf_counter() - t0) * 1000.)
    return dict(p50_ms=float(np.percentile(elapsed, 50)),
                p95_ms=float(np.percentile(elapsed, 95)),
                p99_ms=float(np.percentile(elapsed, 99)),
                fps_p50=1000. / max(float(np.percentile(elapsed, 50)), 1e-6))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cfg", default="configs/jingwei_dinov2.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", default="runs/benchmark.json")
    args = ap.parse_args()
    model, cfg = load_model(args.cfg, args.ckpt, args.device)
    model.eval()
    result = {"device": args.device, "params": count_parameters(model),
              **measure(model, cfg, args.device, args.warmup, args.iters)}
    path = resolve(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print("Includes decode/NMS; excludes image loading, resizing and H2D transfer.")


if __name__ == "__main__":
    main()
