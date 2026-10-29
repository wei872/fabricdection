"""DINOv2 detector inference on images, folders or videos (Unicode paths supported)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import torch

from _common import load_model, resolve
from jingwei.data import list_images
from jingwei.image_io import (VideoCaptureUnicode, ascii_safe, console_safe,
                              imread_unicode, imwrite_unicode)
from jingwei.utils import scale_boxes

COLORS = [(int(255 * (.11 * i % 1)), int(255 * (.37 * i % 1)),
           int(255 * (.71 * i % 1))) for i in range(64)]


@torch.no_grad()
def predict_image(model, bgr, size, device, half=False):
    """Return boxes in ORIGINAL image coordinates, not network coordinates."""
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(cv2.resize(bgr, (size, size)), cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float()[None] / 255.
    x = x.to(device)
    det = model(x.half() if half else x)[0]
    return {**det, "boxes": scale_boxes(det["boxes"], (size, size), (w, h))}


def source_images(source):
    """Stream video frames rather than extracting the whole video to disk."""
    if not source.exists():
        raise FileNotFoundError(f"source does not exist: {source}")
    if source.is_dir():
        files = list_images(source)
        if not files:
            raise FileNotFoundError(f"no images under {source}")
        for path in files:
            yield Path(path).relative_to(source).as_posix(), imread_unicode(path, cv2.IMREAD_COLOR)
    elif source.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
        with VideoCaptureUnicode(source) as cap:
            if not cap.isOpened():
                raise FileNotFoundError(f"cannot open video: {source}")
            i = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield f"frames/{i:06d}.jpg", frame
                i += 1
    else:
        yield source.name, imread_unicode(source, cv2.IMREAD_COLOR)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cfg", default="configs/jingwei_dinov2.yaml")
    ap.add_argument("--ckpt", required=True, help="trained DINOv2 detector checkpoint")
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default="runs/predict")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--half", action="store_true", help="FP16 on CUDA only")
    ap.add_argument("--no-draw", action="store_true")
    args = ap.parse_args()
    if args.half and not args.device.startswith("cuda"):
        ap.error("--half is supported on CUDA only")
    model, cfg = load_model(args.cfg, args.ckpt, args.device)
    model.eval()
    if args.half:
        model.half()
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = cfg.data.names
    results = {}
    for name, bgr in source_images(Path(args.source)):
        if bgr is None:
            raise ValueError(f"cannot read image: {console_safe(name)}")
        det = predict_image(model, bgr, int(cfg.input_size[0]), args.device, args.half)
        boxes, scores, labels = (det[k].cpu().tolist() for k in ("boxes", "scores", "labels"))
        results[name] = {"detections": [
            dict(box=[round(v, 1) for v in box], score=round(score, 4),
                 class_id=label, label=names[label])
            for box, score, label in zip(boxes, scores, labels)]}
        if not args.no_draw:
            for box, score, label in zip(boxes, scores, labels):
                x0, y0, x1, y1 = map(round, box)
                color = COLORS[label % len(COLORS)]
                cv2.rectangle(bgr, (x0, y0), (x1, y1), color, 2)
                text = f"{ascii_safe(names[label], f'cls{label}')} {score:.2f}"
                cv2.putText(bgr, text, (x0, max(16, y0 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
            path = out / name
            path.parent.mkdir(parents=True, exist_ok=True)
            imwrite_unicode(path, bgr)
        print(console_safe(f"{name}: {len(boxes)} detections"))
    with (out / "results.json").open("w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(console_safe(f"results -> {out}"))


if __name__ == "__main__":
    main()
