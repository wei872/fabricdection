#!/usr/bin/env python3
"""相机原始条带（data/*.bmp）→ DATASET_SPEC.md 规范格式（v1）。

流程：ROI 检测 → 平场校正（可选）→ 1280² 重叠切块 → 正常/疑似笔迹分流
→ train/val 划分 → 可选合成缺陷注入（S 分支冷启动池）→ ultralytics 数据 yaml。

用法：
    python scripts/prepare_dataset.py --src data --dst dataset_fabric \
        --tile 1280 --overlap 0.15 --inject 2 --flatfield --seed 0

注意：Q1 未澄清前，脚本用"深色细长线"启发式检测疑似笔迹，命中 tile
仅记入 ignore_masks 并仍按正常样本使用（该像素区替换），请人工复核清单。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from PIL import Image

from fadnet.augment.synthdefects import IMPLEMENTED_KINDS, inject_defect
from fadnet.pipeline.flatfield import FlatFieldCorrector
from fadnet.pipeline.tiling import detect_fabric_roi, slice_tiles


def fabric_id_of(name: str) -> str:
    """KBA26050009_20261014_... → KBA2605（取首段下划线前缀，截 7 位）。"""
    return name.split("_")[0][:7]


def suspicious_pen_mask(gray: np.ndarray) -> np.ndarray:
    """疑似笔迹启发式：局部背景差 >40 且大面积细长暗结构（Q1 占位实现）。

    返回 bool 掩膜；真实部署前应改为人工标注或笔迹分类器。
    """
    g = gray.astype(np.float32)
    k = 25
    pad = k // 2
    cs = np.cumsum(np.cumsum(np.pad(g, pad + 1, mode="reflect"), 0), 1, dtype=np.float64)
    bg = (cs[k + 1:, k + 1:] - cs[:-k - 1, k + 1:] - cs[k + 1:, :-k - 1] + cs[:-k - 1, :-k - 1])
    bg = bg[: g.shape[0], : g.shape[1]] / ((k + 1) * (k + 1))
    return ((bg - g) > 40) & (g < 160)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data")
    ap.add_argument("--dst", default="dataset_fabric")
    ap.add_argument("--tile", type=int, default=1280)
    ap.add_argument("--overlap", type=float, default=0.15)
    ap.add_argument("--inject", type=int, default=0, help="每个正常 tile 注入 N 个合成缺陷")
    ap.add_argument("--flatfield", action="store_true")
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    src, dst = Path(args.src), Path(args.dst)
    bmps = sorted(src.glob("*.bmp"))
    assert bmps, f"no BMP under {src}"

    ff = FlatFieldCorrector() if args.flatfield else None
    records: list[dict] = []   # {tile_path, fabric_id, kind, label_path}
    synth_records: list[dict] = []

    for bmp in bmps:
        fabric_id = fabric_id_of(bmp.name)
        gray = np.asarray(Image.open(bmp).convert("L")).astype(np.float32)
        if ff:
            ff.observe(gray)
            gray = ff.correct(gray)
        pen = suspicious_pen_mask(gray)
        tiles = slice_tiles(gray, args.tile, args.overlap, roi=detect_fabric_roi(gray),
                            fabric_id=fabric_id)
        for ti, t in enumerate(tiles):
            name = f"{bmp.stem}_t{ti:02d}_{t.x0}_{t.y0}.png"
            out_dir = dst / "normal" / fabric_id / "images"
            out_dir.mkdir(parents=True, exist_ok=True)
            img = np.dstack([t.image] * 3).astype(np.uint8)
            Image.fromarray(img).save(out_dir / name)
            rec = {"tile": out_dir / name, "fabric_id": fabric_id}
            # 疑似笔迹 → ignore mask（Q1：仍为正常样本，训练时忽略区被替换）
            pm = pen[t.y0:t.y0 + args.tile, t.x0:t.x0 + args.tile]
            if pm.shape[0] < args.tile or pm.shape[1] < args.tile:   # 与 tile 同步补齐
                pm = np.pad(pm, ((0, args.tile - pm.shape[0]), (0, args.tile - pm.shape[1])))
            if pm.sum() > 200:
                md = dst / "ignore_masks" / fabric_id
                md.mkdir(parents=True, exist_ok=True)
                Image.fromarray((pm * 255).astype(np.uint8)).save(md / name)
                rec["ignore"] = True
            records.append(rec)
            # 合成缺陷池（S 分支冷启动）
            for si in range(args.inject):
                kind = IMPLEMENTED_KINDS[rng.integers(0, len(IMPLEMENTED_KINDS))]
                r = inject_defect(t.image.copy(), kind, rng)
                if not r.boxes:
                    continue
                sdir = dst / "synth" / "det"
                (sdir / "images").mkdir(parents=True, exist_ok=True)
                (sdir / "labels").mkdir(parents=True, exist_ok=True)
                sname = f"{bmp.stem}_t{ti:02d}_s{si}_{kind}.png"
                Image.fromarray(np.dstack([r.image] * 3).astype(np.uint8)).save(sdir / "images" / sname)
                (sdir / "labels" / sname.replace(".png", ".txt")).write_text(
                    "\n".join(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
                              for c, xc, yc, w, h in r.boxes))
                synth_records.append({"tile": sdir / "images" / sname})

    # ---- splits ----
    splits = dst / "splits"
    splits.mkdir(exist_ok=True)
    rng.shuffle(records)
    n_val = max(1, int(len(records) * args.val_ratio))
    val, train = records[:n_val], records[n_val:]
    (splits / "train_normal.txt").write_text(
        "\n".join(str(r["tile"]) for r in train))
    (splits / "val_normal.txt").write_text("\n".join(str(r["tile"]) for r in val))
    (splits / "val_det.txt").write_text("")       # 真实异常标注加入后由 Q2 流程生成
    (splits / "train_det.txt").write_text(
        "\n".join(str(r["tile"]) for r in synth_records))
    (splits / "calib_large.txt").write_text("")
    print(f"[splits] normal train={len(train)} val={len(val)} | synth_det={len(synth_records)}")

    # ---- 布种登记 + ultralytics 数据 yaml ----
    fabrics = sorted({r["fabric_id"] for r in records})
    meta = {f: {"fabric_code": f, "gloss": "unknown"} for f in fabrics}
    (dst / "fabric_meta.yaml").write_text(yaml.safe_dump(meta, allow_unicode=True))
    names = yaml.safe_load((Path("configs/data_fabric.yaml")).read_text())["names"]
    ultra = {"path": str(dst.resolve()), "train": ["synth/det/images"],
             "val": ["synth/det/images"], "nc": len(names),
             "names": {int(k): v for k, v in names.items()}}
    (dst / "data_ultralytics.yaml").write_text(yaml.safe_dump(ultra, allow_unicode=True))
    print(f"[done] dataset at {dst} | fabrics={fabrics}")
    print("[warn] Q1: 疑似笔迹掩膜见 ignore_masks/，请人工复核后再用于训练")


if __name__ == "__main__":
    main()
