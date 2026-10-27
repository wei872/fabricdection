"""Single-image / folder / video inference demo.

    python tools/predict.py --cfg configs/jingwei_n.yaml --ckpt runs/bank/memory_bank.pt \
        --source path/to/image.jpg --out runs/predict

Chinese / Unicode image, folder and video names are supported on Windows and
Linux; see :mod:`jingwei.image_io`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from _common import ROOT, data_paths, load_model
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/jingwei_n.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--prototypes", default="runs/bank/fabric_prototypes.npz")
    ap.add_argument("--source", required=True, help="image, folder, or video file")
    ap.add_argument("--out", default="runs/predict")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-draw", action="store_true")
    args = ap.parse_args()

    model, cfg = load_model(args.cfg, args.ckpt, args.device)
    model.eval()
    size = int(cfg.input_size[0])
    names = list(cfg.data.names) + ["未知异常"]
    proto = load_proto(ROOT / args.prototypes, cfg)
    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    src = Path(args.source)
    if src.is_dir():
        files = list_images(src)
    elif src.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
        files = []
        with VideoCaptureUnicode(src) as cap:
            if not cap.isOpened():
                raise FileNotFoundError(f"cannot open video: {console_safe(src)}")
            i = 0
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                f = out / f"_frames/{i:06d}.jpg"
                f.parent.mkdir(parents=True, exist_ok=True)
                imwrite_unicode(f, fr)
                files.append(str(f))
                i += 1
    else:
        files = [str(src)]

    all_results = {}
    for f in files:
        bgr = imread_unicode(f, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[warn] cannot read image: {console_safe(f)}")
            continue
        rgb = cv2.cvtColor(cv2.resize(bgr, (size, size)), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb.transpose(2, 0, 1)).float()[None] / 255.0
        k = proto.assign(fabric_descriptor(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))) \
            if proto is not None else 0
        with torch.no_grad():
            det = model(x.to(args.device), proto_id=int(k))[0]
        boxes = det["boxes"].cpu().numpy()
        scores = det["scores"].cpu().numpy()
        labels = det["labels"].cpu().numpy()
        items = [dict(box=[round(float(v), 1) for v in bb],
                      score=round(float(s), 4),
                      label=names[int(l)] if int(l) < len(names) else str(int(l)))
                 for bb, s, l in zip(boxes, scores, labels)]
        all_results[Path(f).name] = dict(fabric_prototype=int(k), detections=items)

        if not args.no_draw:
            for bb, s, l in zip(boxes, scores, labels):
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
            imwrite_unicode(out / Path(f).name, bgr)
        print(f"{console_safe(Path(f).name)}: {len(items)} detections (prototype {k})")

    with (out / "results.json").open("w", encoding="utf-8") as fh:
        json.dump(all_results, fh, ensure_ascii=False, indent=2)
    print(f"results -> {console_safe(out)}")


if __name__ == "__main__":
    main()
