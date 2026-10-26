"""Measure REAL latency on YOUR GPU (e.g. GTX 1660 SUPER) — no guessing.

    # 在有 GPU 的机器上（Windows / Linux 都行）
    python tools/bench_device.py --cfg configs/jingwei_p.yaml
    python tools/bench_device.py --cfg configs/jingwei_p.yaml --compare
    python tools/bench_device.py --cfg configs/jingwei_p.yaml --precision both

Reports P50/P95/P99 for: H2D copy, network-only (CUDA events), external
post-process (decode + safety net), and end-to-end. It also A/B tests FP32 vs
FP16 because on GTX 16-series (sm_75 WITHOUT Tensor Cores) FP16 is sometimes
SLOWER than FP32 — see docs/DEPLOYMENT.md §4.1.

NOTE: this measures the PyTorch graph, which is an UPPER bound. A TensorRT
engine is typically 1.5–3x faster. After measuring here, build the engine with
tools/export.py + trtexec and re-measure with `trtexec --duration=60`.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from _common import ROOT, load_model
from jingwei.utils import count_parameters, profile_macs

TARGET_MS = 25.0


def gpu_info():
    if not torch.cuda.is_available():
        return {"cuda": False}
    p = torch.cuda.get_device_properties(0)
    cap = torch.cuda.get_device_capability(0)
    return {
        "cuda": True,
        "name": p.name,
        "sm": f"{cap[0]}.{cap[1]}",
        "cores": getattr(p, "multi_processor_count", None),
        "vram_GB": round(p.total_memory / 1e9, 2),
        "cuda_version": torch.version.cuda,
        "torch": torch.__version__,
        # sm_75 with < 8.0 and no Tensor Cores is the GTX 16-series trap
        "tensor_cores": cap[0] >= 7 and "GTX 16" not in p.name and "GTX 1650" not in p.name,
    }


def pct(x, q):
    return float(np.percentile(np.asarray(x, np.float64), q))


@torch.no_grad()
def measure(model, cfg, device, precision, warmup, iters):
    size = int(cfg.input_size[0])
    dtype = torch.float16 if precision == "fp16" else torch.float32
    m = model.half() if precision == "fp16" else model.float()
    m.eval()

    # uint8 host buffer (what a camera actually delivers) + FP32 tensor
    host_u8 = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    x = torch.zeros(1, 3, size, size, device=device, dtype=dtype)
    try:
        x.copy_(torch.from_numpy(host_u8).permute(2, 0, 1).unsqueeze(0).to(device).to(dtype) / 255.0)
    except RuntimeError:
        x = torch.zeros(1, 3, size, size, device=device, dtype=dtype)

    need_feats = bool(cfg.fusion.roi_head)
    # ---- warmup ----
    for _ in range(warmup):
        outs = m.export_forward(x, 0, return_feats=need_feats)
        m.postprocess_from_tensors(outs, (size, size))
    if device.type == "cuda":
        torch.cuda.synchronize()

    h2d, net, post, total = [], [], [], []
    for _ in range(iters):
        if device.type == "cuda":
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
        t0 = time.perf_counter()

        # (1) host -> device (uint8 -> float, on GPU)
        t = torch.from_numpy(host_u8).to(device, non_blocking=False)
        xi = t.permute(2, 0, 1).unsqueeze(0).to(dtype).div_(255.0)

        if device.type == "cuda":
            mid = torch.cuda.Event(enable_timing=True)
            mid.record()

        # (2) network
        outs = m.export_forward(xi, 0, return_feats=need_feats)
        if device.type == "cuda":
            end.record()
            torch.cuda.synchronize()
            h2d.append(start.elapsed_time(mid))
            net.append(mid.elapsed_time(end))

        # (3) graph-external post-process (includes D2H sync)
        tp = time.perf_counter()
        m.postprocess_from_tensors(outs, (size, size))
        post.append((time.perf_counter() - tp) * 1000.0)
        total.append((time.perf_counter() - t0) * 1000.0)

    res = {
        "precision": precision,
        "h2d_ms_p50": pct(h2d, 50) if h2d else None,
        "net_ms_p50": pct(net, 50) if net else None,
        "post_ms_p50": pct(post, 50),
        "total_ms_p50": pct(total, 50),
        "total_ms_p95": pct(total, 95),
        "total_ms_p99": pct(total, 99),
        "fps_p50": 1000.0 / max(pct(total, 50), 1e-6),
        "p99_over_p50": pct(total, 99) / max(pct(total, 50), 1e-6),
    }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/jingwei_p.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--precision", choices=["fp32", "fp16", "both"], default="fp32")
    ap.add_argument("--compare", action="store_true", help="also time jingwei_n / jingwei_s")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="runs/bench_device.json")
    args = ap.parse_args()

    info = gpu_info()
    print("=" * 72)
    print("GPU:", json.dumps(info, ensure_ascii=False))
    if not info["cuda"]:
        print("\n[!] 没有检测到 CUDA GPU。下面只是 CPU 上的功能自检，不能当作 1660S 的延迟。")
        print("    请在装有 1660 SUPER 的机器上运行本脚本。")
        print("    自动把 warmup/iters 降到 1/2 以免在 CPU 上跑几分钟。\n")
        args.warmup, args.iters = 1, 2
    elif not info["tensor_cores"]:
        print("[!] 该卡疑似无 Tensor Core（GTX 16 系列）。FP16/INT8 可能不加速甚至变慢，")
        print("    请务必用 --precision both 做 A/B 测试。\n")

    cfgs = [args.cfg]
    if args.compare:
        cfgs = ["configs/jingwei_p.yaml", "configs/jingwei_n.yaml", "configs/jingwei_s.yaml"]
        cfgs = list(dict.fromkeys(cfgs + [args.cfg]))

    precisions = ["fp32", "fp16"] if args.precision == "both" else [args.precision]
    device = torch.device(args.device)
    all_results = {"gpu": info, "configs": {}}

    for cfg_path in cfgs:
        model, cfg = load_model(cfg_path, args.ckpt, str(device))
        model.eval()
        macs = profile_macs(model, [torch.zeros(1, 3, int(cfg.input_size[0]),
                                                int(cfg.input_size[1]))])
        print(f"\n=== {cfg.name} ({cfg_path})  params={count_parameters(model)/1e6:.2f} M  "
              f"MACs={macs['total']/1e9:.2f} G ===")
        all_results["configs"][cfg.name] = {"params_M": count_parameters(model) / 1e6,
                                            "macs_G": macs["total"] / 1e9, "runs": {}}
        for prec in precisions:
            if prec == "fp16" and device.type == "cpu":
                print("  fp16 在 CPU 上跳过")
                continue
            try:
                r = measure(model, cfg, device, prec, args.warmup, args.iters)
            except RuntimeError as e:
                print(f"  {prec}: FAILED -> {e}")
                continue
            all_results["configs"][cfg.name]["runs"][prec] = r
            verdict = "PASS ✅" if r["total_ms_p99"] <= TARGET_MS else "FAIL ❌"
            print(f"  {prec:5s} net={fmt(r['net_ms_p50'])} ms  post={fmt(r['post_ms_p50'])} ms  "
                  f"P50={fmt(r['total_ms_p50'])} ms  P95={fmt(r['total_ms_p95'])} ms  "
                  f"P99={fmt(r['total_ms_p99'])} ms  -> {r['fps_p50']:.1f} FPS  "
                  f"P99<={TARGET_MS:.0f}ms: {verdict}")

    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(all_results, fh, ensure_ascii=False, indent=2)
    print(f"\n结果已保存 -> {out}")
    print("注意：这是 PyTorch 图的上界；TensorRT engine 通常快 1.5–3x。")
    print("      导出后用 `trtexec --loadEngine=... --duration=60` 复测 P99。")


def fmt(v):
    return "  n/a " if v is None else f"{v:6.2f}"


if __name__ == "__main__":
    main()
