#!/usr/bin/env python3
"""端到端测试：条带 → 预处理 → U 分支(重构+kNN) [+ S 分支] → F-Head → 输出。

输出（runs/test/<ts>/）：
  verdicts.jsonl   每 tile 一行：全局坐标结果 + 分阶段耗时(ms)
  overlay/*.png    红=critical/必报，蓝=major/minor，黄=review
  latency.json     分阶段 mean/p50/p95

用法：
  # 完整（DINOv3 ckpt + 已训练解码器 + 记忆库 + 阈值表 + YOLO 权重）
  python scripts/test_e2e.py --src data --ckpt weights/dinov3_vitsp16.pth \
      --decoder runs/ustream/ustream_decoder.pt --bank runs/ustream/memory_bank.pt \
      --det runs/sstream/det/weights/best.pt --device cuda
  # 冒烟（无真实权重，仅验证管线与时延结构）
  python scripts/test_e2e.py --src data --smoke --max-strips 2
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

from PIL import Image, ImageDraw

from fadnet.backbone.vit import build_backbone
from fadnet.fusion.arbitrate import DetBox, FHead, Verdict
from fadnet.fusion.calibrate import PerClassCalibrator
from fadnet.pipeline.flatfield import FlatFieldCorrector
from fadnet.pipeline.tiling import detect_fabric_roi, slice_tiles
from fadnet.ustream.dinomaly_branch import DinomalyReconBranch
from fadnet.ustream.heatmap import extract_regions, self_normalize
from fadnet.ustream.patchcore_memory import (MemoryBank, knn_anomaly_scores,
                                             neighborhood_aware, scores_to_heatmap,
                                             style_center)

GRADE_COLOR = {"critical": (255, 60, 60), "major": (80, 160, 255),
               "minor": (80, 160, 255), "review": (255, 200, 0)}


def to_tensor(x: np.ndarray, size: int, dev: torch.device) -> torch.Tensor:
    im = Image.fromarray(np.clip(x, 0, 255).astype(np.uint8)).resize((size, size),
                                                                     Image.BILINEAR)
    a = np.asarray(im, np.float32) / 255.0
    t = torch.from_numpy(np.dstack([a] * 3)).permute(2, 0, 1)[None].to(dev)
    return (t - 0.5) / 0.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--backbone", default="dinov3_vitsp16")
    ap.add_argument("--decoder", default="runs/ustream/ustream_decoder.pt")
    ap.add_argument("--bank", default="runs/ustream/memory_bank.pt")
    ap.add_argument("--thresholds", default="configs/thresholds.yaml")
    ap.add_argument("--config", default="configs/fadnet.yaml")
    ap.add_argument("--det", default=None, help="YOLO 权重(.pt/.engine)；缺省仅 U 分支")
    ap.add_argument("--size", type=int, default=896)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-strips", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = torch.device(args.device)
    out = Path(args.out or f"runs/test/{int(time.time())}")
    (out / "overlay").mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(Path(args.config).read_text())
    thr = (yaml.safe_load(Path(args.thresholds).read_text()) or {}).get("*", {})
    tau_u = float(thr.get("tau_u", 4.0))
    bg_q = float(thr.get("bg_quantile", 90))
    tau_large = float(thr.get("tau_large") or 1e9)      # null → 仅面积规则必报
    calib = PerClassCalibrator.load(cfg["fhead"].get("calibration",
                                                     "configs/class_calibration.yaml"))
    head = FHead(tau_det_low=cfg["sstream"]["tau_det_low"],
                 tau_det_heavy=cfg["sstream"]["tau_det_heavy"],
                 large_area=cfg["fhead"]["large_area"], tau_large=tau_large,
                 calib=calib)

    # ---- U 分支 ----
    backbone = build_backbone(args.backbone, args.ckpt, img_size=args.size,
                              smoke=args.smoke).to(dev)
    dim = backbone.blocks[0].norm1.normalized_shape[0]
    branch = DinomalyReconBranch(backbone, feat_dim=dim, noise_p=0.0).to(dev).eval()
    if not args.smoke and Path(args.decoder).exists():
        sd = torch.load(args.decoder, map_location=dev)
        branch.load_state_dict(sd["decoder"], strict=False)
        print(f"[load] decoder ← {args.decoder}")
    bank = MemoryBank(dim=dim)
    if not args.smoke and Path(args.bank).exists():
        bank.load(args.bank, device=str(dev))
    if bank.centers is None:
        bank.centers = torch.randn(512, dim, device=dev)
        print("[warn] 无记忆库（--smoke 或缺文件）：随机记忆库，输出仅验证管线")
    mem_layer = min(5, len(backbone.blocks) - 1)

    # ---- S 分支（可选） ----
    det = None
    if args.det:
        from ultralytics import YOLO
        det = YOLO(args.det)
        print(f"[load] detector ← {args.det}")

    ff = FlatFieldCorrector()
    strips = sorted(Path(args.src).glob("*.bmp"))[: args.max_strips]
    assert strips, f"no BMP under {args.src}"
    jl = open(out / "verdicts.jsonl", "w", encoding="utf-8")
    lat: dict[str, list[float]] = {}
    tile_scale = 1280.0 / args.size                     # U 分支网格 → tile 局部坐标

    for bmp in strips:
        t_strip = time.perf_counter()
        gray = np.asarray(Image.open(bmp).convert("L")).astype(np.float32)
        ff.observe(gray)
        flat = ff.correct(gray)
        tiles = slice_tiles(flat, 1280, cfg["input"]["overlap"],
                            roi=detect_fabric_roi(flat), fabric_id=bmp.stem[:7])
        strip_verdicts: list[Verdict] = []
        for t in tiles:
            t_tile = time.perf_counter()
            tm: dict[str, float] = {}

            t0 = time.perf_counter()
            x = to_tensor(t.image, args.size, dev)
            tm["preprocess_ms"] = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            with torch.no_grad():
                a_rec = branch.anomaly_map(x)                       # [1,1,S,S]
            tm["u_recon_ms"] = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            with torch.no_grad():
                mid = branch._mid_tokens(x)[1]                      # 第 2 组层=block5
            g = int(mid.shape[1] ** 0.5)
            fmap = neighborhood_aware(style_center(
                mid.transpose(1, 2).reshape(1, dim, g, g)))          # 免登记：逐 tile 中心化
            q = fmap.flatten(2).permute(0, 2, 1).reshape(-1, dim)
            s = knn_anomaly_scores(q, bank.centers)
            a_knn = scores_to_heatmap(s, g, g, (args.size, args.size))
            tm["u_knn_ms"] = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            a = torch.maximum(self_normalize(a_rec, bg_q),          # 免登记：逐 tile 自归一
                              self_normalize(a_knn, bg_q))
            regions = extract_regions(a, tau_u=tau_u,
                                      min_area=cfg["fhead"]["min_anom_area"])
            for r in regions:                                       # S 空间 → tile 局部 1280
                r.x0, r.y0, r.x1, r.y1 = (r.x0 * tile_scale, r.y0 * tile_scale,
                                          r.x1 * tile_scale, r.y1 * tile_scale)
            tm["fusion_ms"] = (time.perf_counter() - t0) * 1000

            dets: list[DetBox] = []
            if det is not None:
                t0 = time.perf_counter()
                rgb = np.dstack([np.clip(t.image, 0, 255).astype(np.uint8)] * 3)
                r0 = det.predict(rgb, verbose=False, imgsz=1280,
                                 conf=cfg["sstream"]["tau_det_low"], device=dev)[0]
                for b in r0.boxes:
                    dets.append(DetBox(*map(float, b.xyxy[0]), float(b.conf[0]),
                                       r0.names[int(b.cls[0])]))
                tm["s_detect_ms"] = (time.perf_counter() - t0) * 1000
            else:
                tm["s_detect_ms"] = 0.0

            vs = head.fuse(dets, regions)                           # tile 局部 1280 坐标
            tm["fhead_ms"] = 0.0
            for v in vs:                                            # 全局坐标副本给台账
                gx0, gy0, gx1, gy1 = v.box
                v.evidence["global"] = [gx0 + t.x0, gy0 + t.y0, gx1 + t.x0, gy1 + t.y0]
            strip_verdicts += vs
            tm["tile_ms"] = (time.perf_counter() - t_tile) * 1000
            for k, v in tm.items():
                lat.setdefault(k, []).append(v)

            jl.write(json.dumps({"strip": bmp.name, "tile_xy": [t.x0, t.y0],
                                 "peak_score": float(a.max()),
                                 "verdicts": [v.to_json() for v in vs],
                                 "timing_ms": {k: round(v, 2) for k, v in tm.items()}},
                                ensure_ascii=False) + "\n")

            # ---- overlay（tile 局部坐标） ----
            vis = np.asarray(Image.fromarray(
                np.clip(t.image, 0, 255).astype(np.uint8)).resize((1280, 1280)))
            pim = ImageDraw.Draw(pim_img := Image.fromarray(np.dstack([vis] * 3)))
            for v in vs:
                x0, y0, x1, y1 = v.box
                c = GRADE_COLOR[v.grade]
                pim.rectangle([x0, y0, x1, y1], outline=c, width=3)
                pim.text((x0 + 4, y0 + 4), f"{v.cls}:{v.conf:.2f}", fill=c)
            pim_img.save(out / "overlay" / f"{bmp.stem}_t{t.x0}_{t.y0}.png")

        glob = [Verdict(box=tuple(v.evidence["global"]), cls=v.cls, conf=v.conf,
                        anomaly_score=v.anomaly_score, grade=v.grade,
                        evidence=v.evidence) for v in strip_verdicts]
        bg = FHead.stitch(glob, iou_th=cfg["fhead"]["cross_tile_iou"])
        print(f"[{bmp.name}] tiles={len(tiles)} verdicts={len(strip_verdicts)} "
              f"stitched={len(bg)} strip_wall_ms={(time.perf_counter()-t_strip)*1000:.0f}")
    jl.close()

    print("\n==== 时延（per tile；本机 CPU 仅验证管线，GPU 预算见 design §6.1） ====")
    rows = []
    for k, vs in lat.items():
        a = np.asarray(vs)
        rows.append({"stage": k, "mean": round(float(a.mean()), 2),
                     "p50": round(float(np.percentile(a, 50)), 2),
                     "p95": round(float(np.percentile(a, 95)), 2)})
        print(f"{k:16s} mean={rows[-1]['mean']:8.2f} p50={rows[-1]['p50']:8.2f} "
              f"p95={rows[-1]['p95']:8.2f} ms")
    (out / "latency.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print(f"[done] → {out}")


if __name__ == "__main__":
    main()
