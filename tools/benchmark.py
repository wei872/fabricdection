"""Compute budget / latency model for the 1280x1280 real-time requirement.

    python tools/benchmark.py --both
    python tools/benchmark.py --cfg configs/jingwei_p.yaml
    python tools/benchmark.py --measure        # also time real CPU forwards

Two SEPARATE latency models are reported, because mixing them up is the single
biggest source of wrong answers:

(A) Tensor-Core GPUs (RTX 20/30/40/50, RTX PRO, Jetson Orin): calibrated from
    Ultralytics' published YOLO27n figure (7.2 G MACs -> 0.44 ms, TRT11 FP16).

(B) GPUs WITHOUT Tensor Cores (GTX 1660 SUPER / 1660 Ti / 1650, TU116/TU117):
    the YOLO27 anchor is INVALID here. These cards have FP32 5.03 TFLOPS and
    FP16 10.05 TFLOPS at the *shader* rate (2:1), not an 8x Tensor-Core rate.
    We instead use a peak-FLOPS x achieved-efficiency model. Because small
    convs, depthwise convs and small GEMMs are poorly served by CUDA cores,
    realistic achieved efficiency is only ~20-35%.

ALL numbers here are estimates. ALWAYS verify with ``tools/bench_device.py``
and ``trtexec`` on the actual device.
"""
from __future__ import annotations

import argparse
import time

import torch

from _common import load_model
from jingwei.utils import count_parameters, profile_macs
from jingwei.image_io import console_safe

# ---------------------------------------------------------------- (A) Tensor Core
REF_MACS = 7.2e9          # YOLO27n @640, RTX PRO 6000, TRT11 FP16
REF_MS = 0.44
NONIDEAL = 1.6
OVERHEAD_MS = 1.2
GPUS_TC = {
    "RTX PRO 6000 (reference)": 1.0,
    "RTX 4090": 1.5,
    "RTX 3060": 4.0,
    "Jetson AGX Orin": 8.0,
}

# ---------------------------------------------------------------- (B) no Tensor Core
# GTX 1660 SUPER: TU116, 1408 CUDA cores, 1785 MHz, NO Tensor Cores,
#   5.03 TFLOPS FP32 / 10.05 TFLOPS FP16 (2:1 shader), 6 GB GDDR6, 336 GB/s.
# FP32 peak FLOP/s -> MAC/s is peak/2 (1 MAC = 2 FLOPs).
GPUS_NO_TC = {
    "GTX 1660 SUPER (FP32, 5.03 TFLOPS)": (5.03e12, 1.0),
    "GTX 1660 SUPER (FP16 shader, ~1.3x)": (5.03e12, 1.3),
    "GTX 1660 Ti / 1660 (FP32)": (5.03e12, 0.9),
}
EFF_LOW, EFF_HIGH = 0.20, 0.35      # achieved fraction of FP32 peak
OVERHEAD_NO_TC_MS = 4.0             # LCN memory passes, H2D, connected comps, launches


@torch.no_grad()
def measure_cpu(model, size, iters=2):
    x = torch.randn(1, 3, size, size)
    model(x)
    t0 = time.perf_counter()
    for _ in range(iters):
        model(x)
    return (time.perf_counter() - t0) / iters * 1000.0


def report(cfg_path, sizes=(640, 1280), measure=False):
    model, cfg = load_model(cfg_path, device="cpu")
    model.eval()
    print(console_safe(f"\n=== {cfg.name} ({cfg_path}) ==="))
    print(f"parameters: {count_parameters(model) / 1e6:.2f} M")
    rows = []
    for s in sizes:
        macs = profile_macs(model, [torch.zeros(1, 3, s, s)])
        rows.append((s, macs))
        print(f"  {s}x{s}: {macs['total'] / 1e9:6.2f} G MACs "
              f"[conv {macs.get('conv', 0)/1e9:.2f}, linear {macs.get('linear', 0)/1e9:.2f}, "
              f"attention {macs.get('attention', 0)/1e9:.2f}]")
    m1280 = rows[-1][1]["total"]

    # ---- (A) Tensor Core model ----
    base = REF_MS * (m1280 / REF_MACS) * NONIDEAL
    print("  [A] Tensor-Core GPUs, FP16 TRT (reference-calibrated):")
    for name, f in GPUS_TC.items():
        ms = base * f + OVERHEAD_MS
        print(f"      {name:<40s}: {ms:6.2f} ms -> {1000 / ms:6.1f} FPS")

    # ---- (B) no Tensor Core model ----
    flops = m1280 * 2.0
    print("  [B] GPUs WITHOUT Tensor Cores (e.g. GTX 1660 SUPER):")
    for name, (peak, speedup) in GPUS_NO_TC.items():
        lo = (flops / (peak * speedup * EFF_HIGH)) * 1000.0 + OVERHEAD_NO_TC_MS
        hi = (flops / (peak * speedup * EFF_LOW)) * 1000.0 + OVERHEAD_NO_TC_MS
        print(f"      {name:<40s}: {lo:5.1f} - {hi:5.1f} ms "
              f"-> {1000/hi:5.1f} - {1000/lo:5.1f} FPS")
    print(f"      (efficiency {EFF_LOW:.0%}-{EFF_HIGH:.0%} of FP32 peak + "
          f"{OVERHEAD_NO_TC_MS:.1f} ms overhead; verify with tools/bench_device.py)")

    if measure:
        for s in sizes:
            print(f"  measured CPU (this sandbox, FP32, no TRT): "
                  f"{s}x{s} = {measure_cpu(model, s):.0f} ms")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/jingwei_n.yaml")
    ap.add_argument("--both", action="store_true", help="also report jingwei_s")
    ap.add_argument("--all", action="store_true", help="report p / n / s")
    ap.add_argument("--measure", action="store_true")
    args = ap.parse_args()
    if args.all:
        for c in ("configs/jingwei_p.yaml", "configs/jingwei_n.yaml", "configs/jingwei_s.yaml"):
            report(c, measure=args.measure)
    else:
        report(args.cfg, measure=args.measure)
        if args.both:
            report("configs/jingwei_s.yaml", measure=args.measure)
