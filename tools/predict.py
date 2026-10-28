"""Single-image / folder / video inference demo.

    python tools/predict.py --cfg configs/jingwei_n.yaml --ckpt runs/bank/memory_bank.pt \
        --source path/to/image.jpg --out runs/predict

Chinese / Unicode image, folder and video names are supported on Windows and
Linux; see :mod:`jingwei.image_io`.

GTX 1660 SUPER real-time path::

    python tools/predict.py --cfg configs/jingwei_p.yaml --realtime-1660 \
        --ckpt runs/detector/best.pt --source path/to/image.jpg

``--realtime-1660`` auto-detects a GTX 16-series / no-Tensor-Core GPU, forces
the pico config + FP32, benchmarks P99 and warns if it exceeds 25 ms.  Unknown
defect types are emitted as ``未知异常`` boxes by default.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from _common import ROOT, data_paths, load_model, log
from bench_device import TARGET_MS, gpu_info, measure
from jingwei.data import list_images
from jingwei.fabric import FabricPrototypes, fabric_descriptor
from jingwei.image_io import (VideoCaptureUnicode, ascii_safe, console_safe,
                              imread_unicode, imwrite_unicode)

COLORS = [(int(255 * (0.11 * i % 1)), int(255 * (0.37 * i % 1)),
           int(255 * (0.71 * i % 1))) for i in range(64)]


def load_proto(path, cfg):
    p = Path(path)
    if not p.exists():
        return None
    z = np.load(p)
    return FabricPrototypes(centers=z["centers"], mean=z["mean"], std=z["std"])


def boxes_to_image_pixels(boxes: np.ndarray, net_size: int,
                          w0: int, h0: int) -> np.ndarray:
    """Map xyxy boxes from the squashed ``net_size x net_size`` frame back to
    the original image resolution (and clip to it).

    ``boxes`` is (N, 4) float; returns a new (N, 4) float32 array.
    """
    b = np.asarray(boxes, np.float32).reshape(-1, 4).copy()
    if b.size == 0:
        return b
    sx, sy = w0 / float(net_size), h0 / float(net_size)
    b *= np.array([sx, sy, sx, sy], np.float32)
    b[:, 0::2] = b[:, 0::2].clip(0, w0)
    b[:, 1::2] = b[:, 1::2].clip(0, h0)
    return b


def iter_sources(src: Path, dump_frames: bool = False, out: Path | None = None):
    """Yield ``(name, bgr_frame)`` for a single image, a folder, or a video.

    Videos are decoded lazily: the old implementation first wrote *every*
    frame to ``out/_frames`` and then re-read them, which doubles the I/O and
    can fill the disk on a long 产线 video.  ``dump_frames`` brings the old
    behaviour back for debugging.
    """
    if src.is_dir():
        for f in list_images(src):
            yield Path(f).name, imread_unicode(f, cv2.IMREAD_COLOR)
        return
    if src.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
        with VideoCaptureUnicode(src) as cap:
            if not cap.isOpened():
                raise FileNotFoundError(f"cannot open video: {console_safe(src)}")
            i = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                name = f"{i:06d}.jpg"
                if dump_frames and out is not None:
                    imwrite_unicode(out / "_frames" / name, frame)
                yield name, frame
                i += 1
        return
    yield src.name, imread_unicode(src, cv2.IMREAD_COLOR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/jingwei_n.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--prototypes", default="runs/bank/fabric_prototypes.npz")
    ap.add_argument("--source", required=True, help="image, folder, or video file")
    ap.add_argument("--out", default="runs/predict")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-draw", action="store_true")
    ap.add_argument("--pico", action="store_true",
                    help="force configs/jingwei_p.yaml (1660S realtime variant)")
    ap.add_argument("--realtime-1660", action="store_true",
                    help="auto-force pico + FP32 on GTX 1660 / no-Tensor-Core GPU and benchmark P99")
    ap.add_argument("--allow-slow", action="store_true",
                    help="do NOT auto-switch config on --realtime-1660")
    ap.add_argument("--half", action="store_true",
                    help="FP16 inference (NOT recommended on GTX 16-series)")
    ap.add_argument("--bench", action="store_true",
                    help="measure real latency / P99 / FPS before inference")
    ap.add_argument("--bench-iters", type=int, default=30)
    ap.add_argument("--emit-unknown", action="store_true",
                    help="force unknown-defect boxes on")
    ap.add_argument("--no-unknown", action="store_true",
                    help="force unknown-defect boxes off")
    ap.add_argument("--dump-frames", action="store_true",
                    help="also write every decoded video frame to out/_frames "
                         "(debug only; predates streaming decode)")
    args = ap.parse_args()

    info = gpu_info() if (args.realtime_1660 or args.bench) else {"cuda": False}
    cfg_path = args.cfg
    pico_cfg = "configs/jingwei_p.yaml"
    if args.pico:
        cfg_path = pico_cfg
    if args.realtime_1660:
        gpu_name = str(info.get("name", ""))
        no_tc = bool(info.get("cuda")) and not bool(info.get("tensor_cores", True))
        is_16 = any(t in gpu_name for t in ("1660", "1650", "TU116", "TU117"))
        if info.get("cuda") and (is_16 or no_tc) and not args.allow_slow:
            if cfg_path != pico_cfg:
                log(f"[realtime-1660] detected {gpu_name}: switching "
                    f"{cfg_path} -> {pico_cfg}")
                cfg_path = pico_cfg
            args.half = False
            args.device = "cuda"
            log("[realtime-1660] FP32 + pico + roi_head=false selected "
                "(GTX 16-series has no Tensor Cores)")
        elif not info.get("cuda"):
            log("[realtime-1660] no CUDA GPU detected; this is only a CPU "
                "functional check, NOT a 1660S latency measurement")
            cfg_path = pico_cfg
            args.half = False
        else:
            log(f"[realtime-1660] {gpu_name} does not look like GTX 16-series; "
                "keeping the requested config")
    if args.ckpt and cfg_path != args.cfg:
        log("[!] --ckpt may have been trained with a different config; "
            "loading into pico uses strict=False. Retrain with "
            "configs/jingwei_p.yaml for reliable accuracy.")

    model, cfg = load_model(cfg_path, args.ckpt, args.device)
    if args.emit_unknown:
        cfg.fusion.emit_unknown_boxes = True
    if args.no_unknown:
        cfg.fusion.emit_unknown_boxes = False
    if args.half and args.device.startswith("cuda"):
        model.half()
    elif args.half:
        log("[!] --half ignored on CPU")
        args.half = False
    model.eval()
    if args.realtime_1660 and args.device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
    size = int(cfg.input_size[0])
    names = list(cfg.data.names) + ["未知异常"]
    proto = load_proto(ROOT / args.prototypes, cfg)
    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    if args.bench or args.realtime_1660:
        warmup = 5 if info.get("cuda") else 1
        iters = max(2, args.bench_iters) if info.get("cuda") else 2
        prec = "fp16" if args.half and info.get("cuda") else "fp32"
        try:
            r = measure(model, cfg, torch.device(args.device), prec,
                        warmup, iters)
            if info.get("cuda"):
                verdict = "PASS" if r["total_ms_p99"] <= TARGET_MS else "FAIL"
            else:
                verdict = "CPU-only (not a 1660S measurement)"
            log(f"[bench] {cfg.name} {prec}  net={r['net_ms_p50']} ms  "
                f"post={r['post_ms_p50']:.2f} ms  "
                f"P50={r['total_ms_p50']:.2f} ms  "
                f"P99={r['total_ms_p99']:.2f} ms  "
                f"{r['fps_p50']:.1f} FPS  P99<={TARGET_MS:.0f}ms: {verdict}")
            if (args.realtime_1660 and info.get("cuda") and
                    r["total_ms_p99"] > TARGET_MS):
                log("[!] PyTorch graph P99 > 25 ms. Build a TensorRT engine with "
                    "tools/export.py + trtexec and re-measure; TRT is usually "
                    "1.5-3x faster. Do not assume FP16 is faster on this GPU.")
        except RuntimeError as e:
            log(f"[bench] failed: {e}")

    src = Path(args.source)
    all_results = {}
    for name, bgr in iter_sources(src, args.dump_frames, out):
        if bgr is None:
            print(f"[warn] cannot read image: {console_safe(name)}")
            continue
        h0, w0 = bgr.shape[:2]
        # The network always sees a squashed size x size frame; predictions are
        # therefore in NETWORK pixels and must be mapped back before they mean
        # anything to the operator.  The old code drew them straight onto the
        # original frame, so every box on a non-1280 image landed in the wrong
        # place (usually completely outside the canvas).
        rgb = cv2.cvtColor(cv2.resize(bgr, (size, size)), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb.transpose(2, 0, 1)).float()[None] / 255.0
        if args.half:
            x = x.half()
        k = proto.assign(fabric_descriptor(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))) \
            if proto is not None else 0
        with torch.no_grad():
            det = model(x.to(args.device), proto_id=int(k))[0]
        boxes = det["boxes"].cpu().numpy()
        scores = det["scores"].cpu().numpy()
        labels = det["labels"].cpu().numpy()
        boxes_px = boxes_to_image_pixels(boxes, size, w0, h0)
        items = [dict(box=[round(float(v), 1) for v in bb],
                      score=round(float(s), 4),
                      label=names[int(l)] if int(l) < len(names) else str(int(l)))
                 for bb, s, l in zip(boxes_px, scores, labels)]
        n_unknown = int((labels == int(cfg.data.nc)).sum())
        all_results[name] = dict(
            image_size=[int(w0), int(h0)],
            network_size=[int(size), int(size)],
            fabric_prototype=int(k), detections=items,
            unknown_boxes=n_unknown,
            realtime_1660=bool(args.realtime_1660),
            emit_unknown=bool(cfg.fusion.get("emit_unknown_boxes", True)))

        if not args.no_draw:
            for bb, s, l in zip(boxes_px, scores, labels):
                c = COLORS[int(l) % len(COLORS)]
                x0, y0, x1, y1 = [int(v) for v in bb]
                cv2.rectangle(bgr, (x0, y0), (x1, y1), c, 2)
                # cv2.putText's Hershey fonts are ASCII-only: Chinese class
                # names are kept in results.json, but drawn as cls<id> here.
                label = ascii_safe(names[int(l)], f"cls{int(l)}") \
                    if int(l) < len(names) else f"cls{int(l)}"
                txt = f"{label} {s:.2f}"
                cv2.putText(bgr, txt, (x0, max(16, y0 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
            imwrite_unicode(out / name, bgr)
        print(f"{console_safe(name)}: {len(items)} detections "
              f"({n_unknown} unknown, prototype {k})")

    with (out / "results.json").open("w", encoding="utf-8") as fh:
        json.dump(all_results, fh, ensure_ascii=False, indent=2)
    print(f"results -> {console_safe(out)}")


if __name__ == "__main__":
    main()
