#!/usr/bin/env python3
"""实时性实测：分阶段时延 + 端到端 FPS（design §6.1 预算表的实测对照）。

用法：
  python scripts/benchmark_latency.py --device cuda --warmup 30 --iters 200   # 目标机
  python scripts/benchmark_latency.py --device cpu  --warmup 5  --iters 20    # 冒烟
输出 runs/bench/<ts>/latency_report.json + 终端表。

注意：CPU 数值只用于验证管线与相对量级；验收以目标 GPU（FP16/TRT）为准。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fadnet.backbone.vit import build_backbone
from fadnet.fusion.arbitrate import DetBox, FHead
from fadnet.pipeline.flatfield import FlatFieldCorrector
from fadnet.pipeline.tiling import detect_fabric_roi, slice_tiles
from fadnet.ustream.dinomaly_branch import DinomalyReconBranch
from fadnet.ustream.heatmap import extract_regions, self_normalize
from fadnet.ustream.patchcore_memory import (MemoryBank, knn_anomaly_scores,
                                             neighborhood_aware, scores_to_heatmap,
                                             style_center)


def bench(fn, warmup: int, iters: int) -> tuple[float, float, float]:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    a = np.asarray(ts)
    return float(a.mean()), float(np.percentile(a, 50)), float(np.percentile(a, 95))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--size", type=int, default=896)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--strip", default="data/KBA26050009_20261014_101648_109095.bmp")
    args = ap.parse_args()
    dev = torch.device(args.device)
    out = Path(f"runs/bench/{int(time.time())}"); out.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        args.warmup, args.iters = 2, 3

    rep: dict[str, dict] = {}
    from PIL import Image
    strip = np.asarray(Image.open(args.strip).convert("L")).astype(np.float32) \
        if Path(args.strip).exists() else np.random.rand(846, 2440).astype(np.float32) * 255

    # 1) 预处理（平场 + ROI + 切块）
    ff = FlatFieldCorrector()
    ff.observe(strip)
    m, p50, p95 = bench(lambda: slice_tiles(ff.correct(strip), 1280, 0.15,
                                            roi=detect_fabric_roi(strip)),
                        args.warmup, args.iters)
    rep["preprocess"] = dict(mean=m, p50=p50, p95=p95)

    # 2) U 分支
    backbone = build_backbone(smoke=True, img_size=args.size).to(dev)
    dim = backbone.blocks[0].norm1.normalized_shape[0]
    branch = DinomalyReconBranch(backbone, feat_dim=dim, noise_p=0.0).to(dev).eval()
    bank = MemoryBank(dim=dim); bank.centers = torch.randn(8192, dim, device=dev)
    x = (torch.rand(1, 3, args.size, args.size, device=dev) - 0.5) * 2
    with torch.no_grad():
        m, p50, p95 = bench(lambda: branch.anomaly_map(x), args.warmup, args.iters)
    rep["u_recon"] = dict(mean=m, p50=p50, p95=p95)
    with torch.no_grad():
        mid = branch._mid_tokens(x)[1]
    g = int(mid.shape[1] ** 0.5)
    fmap = neighborhood_aware(style_center(mid.transpose(1, 2).reshape(1, dim, g, g)))
    q = fmap.flatten(2).permute(0, 2, 1).reshape(-1, dim)

    def knn_step():
        with torch.no_grad():
            s = knn_anomaly_scores(q, bank.centers)
            scores_to_heatmap(s, g, g, (args.size, args.size))
    m, p50, p95 = bench(knn_step, args.warmup, args.iters)
    rep["u_knn"] = dict(mean=m, p50=p50, p95=p95)

    # 3) 融合 + 区域提取 + F-Head
    with torch.no_grad():
        a = torch.maximum(self_normalize(branch.anomaly_map(x)),
                          self_normalize(scores_to_heatmap(
                              knn_anomaly_scores(q, bank.centers), g, g,
                              (args.size, args.size))))
    head = FHead()
    dets = [DetBox(100, 100, 300, 300, 0.6, "hole")]

    def fuse_step():
        with torch.no_grad():
            regs = extract_regions(a, tau_u=0.9, min_area=9)
        head.fuse(dets, regs)
    m, p50, p95 = bench(fuse_step, args.warmup, args.iters)
    rep["fusion_fhead"] = dict(mean=m, p50=p50, p95=p95)

    # 4) S 分支（若环境有 ultralytics）
    try:
        from ultralytics import YOLO
        det = YOLO("yolo26n.yaml").to(dev)
        rgb = np.dstack([np.uint8(np.random.rand(1280, 1280) * 255)] * 3)
        m, p50, p95 = bench(lambda: det.predict(rgb, verbose=False, imgsz=1280,
                                                device=dev), args.warmup, args.iters)
        rep["s_detect"] = dict(mean=m, p50=p50, p95=p95)
    except Exception as e:                                      # noqa: BLE001
        rep["s_detect"] = dict(mean=-1, note=str(e))

    total = sum(v["mean"] for v in rep.values() if v.get("mean", -1) >= 0)
    rep["total_per_tile"] = dict(mean=total, fps=1000.0 / max(total, 1e-9))
    (out / "latency_report.json").write_text(json.dumps(rep, indent=1))
    print(f"device={args.device} size={args.size} (warmup={args.warmup}, iters={args.iters})")
    for k, v in rep.items():
        extra = f" p50={v['p50']:8.2f} p95={v['p95']:8.2f}" if "p50" in v else ""
        print(f"{k:16s} mean={v['mean']:8.2f}{extra} ms")
    print(f"=> 单 tile 合计 {total:.1f} ms ≈ {1000/total:.1f} FPS（U+S 串行；"
          f"dual-stream 并行后可再省 ~20%，见 design §6.2）")
    print(f"[done] → {out/'latency_report.json'}")


if __name__ == "__main__":
    main()
