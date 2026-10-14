#!/usr/bin/env python3
"""数据集准备 v2：以**用户已有 YOLO 格式标注数据集**为主入口（data/ 条带为备选）。

模式 A（默认，--yolo）：接入已标注 YOLO 数据集
  支持布局：root/images+root/labels；root/{train,val}/images+labels；
            root/data.yaml 或 root/classes.txt 提供类别名。
  流程：类别名映射(→CLASSES) → 大图重叠切块+跨块框转换 → 正常 tile 派生
        → 按源图分层划分 → 质量报告 → 缺量类合成补足(可选) → ultralytics 入口。
模式 B（--src，兼容 v1）：相机 BMP 条带（无标注）→ 正常 tiles + 合成缺陷池。

框转换规则（design/DATASET_SPEC §3）：r = 框∩tile / 框面积
  r ≥ keep_ratio(0.6)      → 保留（裁剪后框）
  0.15 ≤ r < keep_ratio    → 保留裁剪框（partial，训练可见截断形态）
  0.02 ≤ r < 0.15          → 进 ignore 掩膜（不学不报），tile 仍可为正常
  r < 0.02                 → 丢弃
正常 tile：无保留框 且 无 ignore 碎片 且 与任何原始框距离 > margin(32px)。

用法：
  python scripts/prepare_dataset.py --yolo /path/to/labeled_ds \
      --class-map configs/class_map.yaml --tile 1280 --overlap 0.15 \
      --min-per-class 50 --inject-topup
  python scripts/prepare_dataset.py --src data --dst dataset_fabric --inject 1 --flatfield
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from PIL import Image

from fadnet.augment.synthdefects import CLASSES, IMPLEMENTED_KINDS, inject_defect
from fadnet.pipeline.flatfield import FlatFieldCorrector
from fadnet.pipeline.tiling import detect_fabric_roi, slice_tiles

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLS_ID = {name: i for i, name in enumerate(CLASSES)}


# ---------------------------------------------------------------- 通用工具
def fabric_id_of(name: str) -> str:
    return name.split("_")[0][:7]


def suspicious_pen_mask(gray: np.ndarray) -> np.ndarray:
    """疑似笔迹启发式（仅 --pen 启用；Q1 占位实现，生产前人工复核）。"""
    g = gray.astype(np.float32)
    k = 25
    pad = k // 2
    cs = np.cumsum(np.cumsum(np.pad(g, pad + 1, mode="reflect"), 0), 1, dtype=np.float64)
    bg = (cs[k + 1:, k + 1:] - cs[:-k - 1, k + 1:] - cs[k + 1:, :-k - 1] + cs[:-k - 1, :-k - 1])
    bg = bg[: g.shape[0], : g.shape[1]] / ((k + 1) * (k + 1))
    return ((bg - g) > 40) & (g < 160)


def load_class_map(path: str | None, names: list[str]) -> dict[str, int]:
    """别名 → 本项目 CLASSES id。未映射类别名会告警并计入报告。"""
    alias: dict[str, int] = {n: CLS_ID[n] for n in names if n in CLS_ID}
    if path and Path(path).exists():
        raw = yaml.safe_load(Path(path).read_text()) or {}
        for ours, aliases in raw.items():
            if ours not in CLS_ID:
                print(f"[warn] class_map 中未知目标类: {ours}")
                continue
            for a in (aliases or []):
                alias[str(a).strip().lower()] = CLS_ID[ours]
    return alias


def discover_yolo(root: Path) -> tuple[list[dict], list[str]]:
    """返回 (样本列表[{img, label, split}], 源类别名列表)。

    兼容布局：root/images/{train,val}+root/labels/{train,val}（用户 source5 布局）；
    root/images+root/labels；root/{train,val}/images+labels；有 data.yaml/classes.txt 更佳。
    路径中含 train/val 目录名 → 记录 split hint（**尊重用户已有划分，不重划**）。
    无对应标签 txt 的图 = 正常样本。
    """
    names: list[str] = []
    if (root / "data.yaml").exists():
        d = yaml.safe_load((root / "data.yaml").read_text())
        nc = d.get("names")
        names = list(nc.values()) if isinstance(nc, dict) else list(nc or [])
    elif (root / "classes.txt").exists():
        names = (root / "classes.txt").read_text().split()
    imgs: list[dict] = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in IMG_EXT or not p.is_file():
            continue
        if any(part in ("labels",) for part in p.parts):
            continue
        lab, hint = None, None
        for cand in (p.parent.parent / "labels" / p.parent.name / (p.stem + ".txt"),
                     p.parent.parent.parent / "labels" / p.parent.name / (p.stem + ".txt"),
                     p.parent.parent / "labels" / (p.stem + ".txt"),
                     p.with_suffix(".txt")):
            if cand.exists():
                lab = cand
                break
        low = {q.lower() for q in p.parts}
        for s in ("train", "val", "test"):
            if s in low:
                hint = "val" if s == "test" else s
                break
        imgs.append({"img": p, "label": lab, "split": hint})
    if not names:
        ids = set()
        for s in imgs:
            if s["label"]:
                ids |= {int(l.split()[0]) for l in s["label"].read_text().splitlines() if l.strip()}
        names = [CLASSES[i] if i < len(CLASSES) else f"class_{i}" for i in sorted(ids)]
        print(f"[warn] 未找到 classes.txt/data.yaml：按 id 直接映射 CLASSES 猜测 {names}")
    return imgs, names


def read_boxes(label: Path | None) -> list[tuple[int, float, float, float, float]]:
    if label is None:
        return []
    out = []
    for ln in label.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        c, xc, yc, w, h = map(float, ln.split()[:5])
        out.append((int(c), xc, yc, w, h))
    return out


def tile_positions(w: int, h: int, tile: int, overlap: float) -> list[tuple[int, int]]:
    stride = int(tile * (1 - overlap))
    xs = list(range(0, max(w - tile, 0) + 1, stride)) or [0]
    ys = list(range(0, max(h - tile, 0) + 1, stride)) or [0]
    if xs[-1] != max(w - tile, 0):
        xs.append(max(w - tile, 0))
    if ys[-1] != max(h - tile, 0):
        ys.append(max(h - tile, 0))
    return [(x, y) for y in dict.fromkeys(ys) for x in dict.fromkeys(xs)]


def convert_boxes(boxes_px, tx0, ty0, tile, keep_ratio, frag_lo):
    """跨块框转换 → (kept[(cls, x0,y0,x1,y1)], ignores[(x0,y0,x1,y1)], touched_margin)"""
    kept, ignores = [], []
    margin_hit = False
    for cls, x0, y0, x1, y1 in boxes_px:
        ix0, iy0 = max(x0, tx0), max(y0, ty0)
        ix1, iy1 = min(x1, tx0 + tile), min(y1, ty0 + tile)
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        area = max(1.0, (x1 - x0) * (y1 - y0))
        r = inter / area
        # margin 判定（扩张 32px 是否触碰本 tile）
        if x1 + 32 > tx0 and x0 - 32 < tx0 + tile and y1 + 32 > ty0 and y0 - 32 < ty0 + tile:
            margin_hit = True
        if r >= keep_ratio or (0.15 <= r < keep_ratio and inter >= 32 * 32):
            kept.append((cls, ix0 - tx0, iy0 - ty0, ix1 - tx0, iy1 - ty0))
        elif r >= frag_lo:
            ignores.append((ix0 - tx0, iy0 - ty0, ix1 - tx0, iy1 - ty0))
    return kept, ignores, margin_hit


# ---------------------------------------------------------------- 写出
def save_tile(dst: Path, fabric: str, split: str, kind: str, name: str,
              img: np.ndarray, boxes: list, ignore: np.ndarray | None,
              lists: dict, stats: dict) -> None:
    """kind: normal|abnormal|synth；写图片/标签/掩膜并登记 split 清单。"""
    idir = dst / kind / fabric / "images"
    idir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(idir / name)
    lists[f"{split}_{kind}"].append(str((idir / name).resolve()))   # 绝对路径：ultralytics txt 清单要求
    if kind != "normal":
        ldir = dst / kind / fabric / "labels"
        ldir.mkdir(parents=True, exist_ok=True)
        h, w = img.shape[:2]
        (ldir / name.replace(".png", ".txt")).write_text(
            "\n".join(f"{c} {(x0+x1)/2/w:.6f} {(y0+y1)/2/h:.6f} {(x1-x0)/w:.6f} {(y1-y0)/h:.6f}"
                      for c, x0, y0, x1, y1 in boxes))
    if ignore is not None and ignore.sum() > 0:
        mdir = dst / "ignore_masks" / fabric
        mdir.mkdir(parents=True, exist_ok=True)
        Image.fromarray((ignore * 255).astype(np.uint8)).save(mdir / name)
    stats["tiles_" + kind] += 1


# ---------------------------------------------------------------- 模式 A：YOLO 接入
def run_yolo_mode(args, dst: Path) -> None:
    root = Path(args.yolo)
    samples, src_names = discover_yolo(root)
    alias = load_class_map(args.class_map, src_names)
    print(f"[yolo] images={len(samples)} src_classes={src_names}")

    lists: dict[str, list] = defaultdict(list)
    stats: dict = defaultdict(int)
    per_class: dict[int, dict] = {i: {"boxes": 0, "areas": [], "large": 0, "imgs": set()}
                                  for i in range(len(CLASSES))}
    unmapped: dict[str, int] = defaultdict(int)
    img_records: list[dict] = []     # 源图级：{i, split, mapped}

    # ---- 正常样本附加目录（只给图片、无标签；可选） ----
    if args.normal_dir:
        nd = Path(args.normal_dir)
        n0 = len(samples)
        for p in sorted(nd.rglob("*")):
            if p.suffix.lower() in IMG_EXT and p.is_file():
                low = {q.lower() for q in p.parts}
                hint = next(("val" if s == "test" else s
                             for s in ("train", "val", "test") if s in low), None)
                samples.append({"img": p, "label": None, "split": hint})
        print(f"[yolo] +normal_dir {nd} → +{len(samples)-n0} 正常图")

    # ---- 类别映射 + 总量统计 ----
    rng = np.random.default_rng(args.seed)
    mapped_of: list[list] = []
    tot_boxes: dict[int, int] = defaultdict(int)
    for i in range(len(samples)):
        mapped = []
        for c, xc, yc, w, h in read_boxes(samples[i]["label"]):
            name = src_names[c] if c < len(src_names) else f"class_{c}"
            cid = alias.get(str(name).strip().lower())
            if cid is None:
                unmapped[name] += 1
            else:
                mapped.append((cid, xc, yc, w, h))
                tot_boxes[cid] += 1
        mapped_of.append(mapped)

    # ---- 划分：优先尊重源 train/val；无 hint 的按分层贪心 / 正常图按比例 ----
    order = list(range(len(samples)))
    rng.shuffle(order)
    val_boxes: dict[int, int] = defaultdict(int)
    for i in order:                                   # 已有 hint=val 的框先计入
        if samples[i]["split"] == "val":
            for cid, *_ in mapped_of[i]:
                val_boxes[cid] += 1
    n_normal_nohint = sum(1 for i in order
                          if samples[i]["split"] is None and not mapped_of[i])
    normal_val_target = max(2, int(args.val_ratio * n_normal_nohint))
    for i in order:
        hint, mapped = samples[i]["split"], mapped_of[i]
        if hint:
            split = hint                              # ← 用户已有划分，原样保留
        elif mapped:
            split = ("val" if any(val_boxes[c] < max(1, int(args.val_ratio * tot_boxes[c]))
                                  for c, *_ in mapped) else "train")
            if split == "val":
                for c, *_ in mapped:
                    val_boxes[c] += 1
        else:
            split = "val" if stats["val_normal_imgs"] < normal_val_target else "train"
            if split == "val":
                stats["val_normal_imgs"] += 1
        stats[split + "_imgs"] += 1
        img_records.append({"i": i, "split": split, "mapped": mapped})

    # ---- 切块 + 框转换 + 写出 ----
    for rec in img_records:
        s = samples[rec["i"]]
        im = Image.open(s["img"])
        arr = np.asarray(im.convert("RGB"))
        h, w = arr.shape[:2]
        gray = np.asarray(im.convert("L")).astype(np.float32)
        pen = suspicious_pen_mask(gray) if args.pen else np.zeros_like(gray, bool)
        boxes_px = [(c, (xc - w_ / 2) * w, (yc - h_ / 2) * h, (xc + w_ / 2) * w, (yc + h_ / 2) * h)
                    for c, xc, yc, w_, h_ in rec["mapped"]]
        fabric = "global"
        for ti, (tx, ty) in enumerate(tile_positions(w, h, args.tile, args.overlap)):
            patch = arr[ty:ty + args.tile, tx:tx + args.tile]
            pm = pen[ty:ty + args.tile, tx:tx + args.tile]
            ph, pw = patch.shape[:2]
            if ph < args.tile or pw < args.tile:
                patch = np.pad(patch, ((0, args.tile - ph), (0, args.tile - pw), (0, 0)), mode="reflect")
                pm = np.pad(pm, ((0, args.tile - ph), (0, args.tile - pw)))
            kept, ignores, margin = convert_boxes(boxes_px, tx, ty, args.tile,
                                                  args.keep_ratio, args.frag_lo)
            name = f"{s['img'].stem}_t{ti:02d}_{tx}_{ty}.png"
            ign = pm.copy()
            for x0, y0, x1, y1 in ignores:
                ign[int(y0):int(y1), int(x0):int(x1)] = True
            if kept:
                for c, x0, y0, x1, y1 in kept:
                    per_class[c]["boxes"] += 1
                    per_class[c]["areas"].append((x1 - x0) * (y1 - y0))
                    per_class[c]["imgs"].add(rec["i"])
                    if (x1 - x0) * (y1 - y0) >= args.large_area:
                        per_class[c]["large"] += 1
                save_tile(dst, fabric, rec["split"], "abnormal", name, patch, kept,
                          ign if ign.any() else None, lists, stats)
                if any((x1 - x0) * (y1 - y0) >= args.large_area for _, x0, y0, x1, y1 in kept):
                    lists["calib_large"].append(str((dst / "abnormal" / fabric / "images" / name).resolve()))
            elif margin:
                stats["skipped_margin"] += 1        # 缺陷近邻块：不入正常集（防污染）
            else:
                save_tile(dst, fabric, rec["split"], "normal", name, patch, [],
                          ign if ign.any() else None, lists, stats)
                if ign.any():
                    stats["normal_with_ignore"] += 1
            stats["tiles_total"] += 1

    # ---- 缺量类合成补足 ----
    if args.inject_topup:
        normal_files = lists["train_normal"]
        for c in range(len(CLASSES)):
            kind = CLASSES[c]
            if kind not in IMPLEMENTED_KINDS:
                continue
            deficit = args.min_per_class - per_class[c]["boxes"]
            if deficit <= 0 or not normal_files:
                continue
            n_synth = min(deficit, 2 * max(per_class[c]["boxes"], 1))
            for k in range(n_synth):
                src = Path(normal_files[k % len(normal_files)])
                tile = np.asarray(Image.open(src).convert("L")).astype(np.float32)
                r = inject_defect(tile, kind, rng)
                if not r.boxes:
                    continue
                name = f"synth_{kind}_{stats['synth_n']:05d}.png"
                stats["synth_n"] += 1
                save_tile(dst, "global", "train", "synth", name,
                          np.dstack([r.image.astype(np.uint8)] * 3),
                          [(CLS_ID[kind], (b[1]-b[3]/2)*1280, (b[2]-b[4]/2)*1280,
                            (b[1]+b[3]/2)*1280, (b[2]+b[4]/2)*1280) for b in r.boxes],
                          None, lists, stats)
                per_class[c]["boxes"] += 1

    # ---- splits / 元数据 / ultralytics 入口 / 报告 ----
    if not lists["val_normal"] and lists["train_normal"]:   # 源无正常 val：自动留置
        n = max(1, int(len(lists["train_normal"]) * args.val_ratio))
        lists["val_normal"] = lists["train_normal"][:n]
        lists["train_normal"] = lists["train_normal"][n:]
        print(f"[warn] 源数据正常样本无 val 划分：自动留置 {n} 张正常 tile 作 val_normal")
    splits = dst / "splits"; splits.mkdir(exist_ok=True)
    (splits / "train_normal.txt").write_text("\n".join(lists["train_normal"]))
    (splits / "val_normal.txt").write_text("\n".join(lists["val_normal"]))
    (splits / "train_det.txt").write_text("\n".join(lists["train_abnormal"] + lists["train_synth"]))
    (splits / "val_det.txt").write_text("\n".join(lists["val_abnormal"]))
    (splits / "calib_large.txt").write_text("\n".join(dict.fromkeys(lists["calib_large"])))
    (dst / "fabric_meta.yaml").write_text(yaml.safe_dump(
        {"global": {"note": "免登记；来源目录: " + str(root)}}, allow_unicode=True))
    (dst / "data_ultralytics.yaml").write_text(yaml.safe_dump({
        "path": str(dst.resolve()),
        "train": [str((splits / "train_det.txt").resolve())],
        "val": [str((splits / "val_det.txt").resolve())],
        "nc": len(CLASSES), "names": {i: n for i, n in enumerate(CLASSES)}},
        allow_unicode=True))
    report = {"source": str(root), "src_classes": src_names,
              "unmapped_classes": dict(unmapped),
              "images": {"train": stats["train_imgs"], "val": stats["val_imgs"]},
              "tiles": {k: v for k, v in stats.items() if k.startswith("tiles") or k == "normal_with_ignore"},
              "classes": {CLASSES[c]: {"boxes": per_class[c]["boxes"],
                                       "large": per_class[c]["large"],
                                       "area_p50": float(np.percentile(per_class[c]["areas"], 50))
                                       if per_class[c]["areas"] else None,
                                       "images": len(per_class[c]["imgs"])}
                          for c in range(len(CLASSES)) if per_class[c]["boxes"]}}
    (dst / "dataset_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"[yolo] tiles: {dict(stats)}")
    print(f"[yolo] 类别统计（boxes/large/面积p50）:")
    for c in range(len(CLASSES)):
        if per_class[c]["boxes"]:
            a = per_class[c]["areas"]
            print(f"   {CLASSES[c]:16s} {per_class[c]['boxes']:5d} {per_class[c]['large']:4d} "
                  f"{np.percentile(a,50) if a else 0:9.0f}")
    if unmapped:
        print(f"[warn] 未映射类别（需在 configs/class_map.yaml 补别名或扩充 CLASSES）: {dict(unmapped)}")
    low = [CLASSES[c] for c in range(len(CLASSES))
           if 0 < per_class[c]["boxes"] < args.min_per_class]
    if low:
        print(f"[warn] 低于最低量({args.min_per_class})的类: {low} → 建议 --inject-topup 或补采")
    print(f"[done] → {dst}（报告 dataset_report.json）")


# ---------------------------------------------------------------- 模式 B：BMP 条带
def run_strip_mode(args, dst: Path) -> None:
    rng = np.random.default_rng(args.seed)
    src = Path(args.src)
    bmps = sorted(src.glob("*.bmp"))
    assert bmps, f"no BMP under {src}"
    ff = FlatFieldCorrector() if args.flatfield else None
    records, synth_records = [], []
    for bmp in bmps:
        fabric_id = fabric_id_of(bmp.name)
        gray = np.asarray(Image.open(bmp).convert("L")).astype(np.float32)
        if ff:
            ff.observe(gray)
            gray = ff.correct(gray)
        pen = suspicious_pen_mask(gray)
        for ti, t in enumerate(slice_tiles(gray, args.tile, args.overlap,
                                           roi=detect_fabric_roi(gray), fabric_id=fabric_id)):
            name = f"{bmp.stem}_t{ti:02d}_{t.x0}_{t.y0}.png"
            out_dir = dst / "normal" / fabric_id / "images"
            out_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.dstack([np.clip(t.image, 0, 255).astype(np.uint8)] * 3)).save(out_dir / name)
            rec = {"tile": out_dir / name}
            pm = pen[t.y0:t.y0 + args.tile, t.x0:t.x0 + args.tile]
            if pm.shape[0] < args.tile or pm.shape[1] < args.tile:
                pm = np.pad(pm, ((0, args.tile - pm.shape[0]), (0, args.tile - pm.shape[1])))
            if pm.sum() > 200:
                md = dst / "ignore_masks" / fabric_id
                md.mkdir(parents=True, exist_ok=True)
                Image.fromarray((pm * 255).astype(np.uint8)).save(md / name)
            records.append(rec)
            for si in range(args.inject):
                kind = IMPLEMENTED_KINDS[rng.integers(0, len(IMPLEMENTED_KINDS))]
                r = inject_defect(t.image.copy(), kind, rng)
                if not r.boxes:
                    continue
                sdir = dst / "synth" / "det"
                (sdir / "images").mkdir(parents=True, exist_ok=True)
                (sdir / "labels").mkdir(parents=True, exist_ok=True)
                sname = f"{bmp.stem}_t{ti:02d}_s{si}_{kind}.png"
                Image.fromarray(np.dstack([r.image.astype(np.uint8)] * 3)).save(sdir / "images" / sname)
                (sdir / "labels" / sname.replace(".png", ".txt")).write_text(
                    "\n".join(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for c, xc, yc, w, h in r.boxes))
                synth_records.append({"tile": sdir / "images" / sname})
    splits = dst / "splits"; splits.mkdir(exist_ok=True)
    rng.shuffle(records)
    n_val = max(1, int(len(records) * args.val_ratio))
    (splits / "train_normal.txt").write_text("\n".join(str(r["tile"].resolve()) for r in records[n_val:]))
    (splits / "val_normal.txt").write_text("\n".join(str(r["tile"].resolve()) for r in records[:n_val]))
    (splits / "train_det.txt").write_text("\n".join(str(r["tile"].resolve()) for r in synth_records))
    (splits / "val_det.txt").write_text("")
    (splits / "calib_large.txt").write_text("")
    names = CLASSES
    (dst / "data_ultralytics.yaml").write_text(yaml.safe_dump(
        {"path": str(dst.resolve()), "train": [str((splits / "train_det.txt").resolve())],
         "val": [str((splits / "train_det.txt").resolve())], "nc": len(names),
         "names": {i: n for i, n in enumerate(names)}}, allow_unicode=True))
    print(f"[splits] normal train={len(records)-n_val} val={n_val} | synth_det={len(synth_records)}")
    print(f"[done] dataset at {dst} | fabrics={sorted({fabric_id_of(b.name) for b in bmps})}")
    print("[warn] Q1: 疑似笔迹掩膜见 ignore_masks/，请人工复核后再用于训练")


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--cfg", default="configs/dataset.yaml")
    preargs, _ = pre.parse_known_args()
    ap = argparse.ArgumentParser()
    defs: dict = {}
    if Path(preargs.cfg).exists():                    # 数据集配置作默认值（CLI 可覆盖）
        c = yaml.safe_load(Path(preargs.cfg).read_text()) or {}
        src, pr = c.get("source", {}), c.get("prepare", {})
        defs = {k: v for k, v in {
            "yolo": src.get("yolo_root"), "normal_dir": src.get("normal_dir"),
            "class_map": pr.get("class_map", "configs/class_map.yaml"),
            "dst": pr.get("dst", "dataset_fabric"), "tile": pr.get("tile", 1280),
            "overlap": pr.get("overlap", 0.15), "keep_ratio": pr.get("keep_ratio", 0.6),
            "frag_lo": pr.get("frag_lo", 0.02), "large_area": pr.get("large_area", 2500.0),
            "min_per_class": pr.get("min_per_class", 50),
            "inject_topup": bool(pr.get("inject_topup", False)),
            "pen": bool(pr.get("pen", False)),
            "val_ratio": pr.get("val_ratio", 0.15)}.items() if v is not None}
        print(f"[cfg] 使用数据集配置 {preargs.cfg}")
    ap.add_argument("--cfg", default=preargs.cfg, help="数据集配置文件")
    ap.add_argument("--yolo", default=None, help="已标注 YOLO 数据集根目录（模式 A）")
    ap.add_argument("--normal-dir", default=None,
                    help="正常样本目录（只给图片无标签；混在 images/ 无标签亦可，二者可叠加）")
    ap.add_argument("--src", default=None, help="BMP 条带目录（模式 B）")
    ap.add_argument("--dst", default="dataset_fabric")
    ap.add_argument("--class-map", default="configs/class_map.yaml")
    ap.add_argument("--tile", type=int, default=1280)
    ap.add_argument("--overlap", type=float, default=0.15)
    ap.add_argument("--keep-ratio", type=float, default=0.6)
    ap.add_argument("--frag-lo", type=float, default=0.02)
    ap.add_argument("--large-area", type=float, default=2500.0)
    ap.add_argument("--min-per-class", type=int, default=50)
    ap.add_argument("--inject-topup", action="store_true", help="缺量类合成补足(≤2:1)")
    ap.add_argument("--inject", type=int, default=0, help="模式 B：每 tile 注入合成缺陷数")
    ap.add_argument("--flatfield", action="store_true")
    ap.add_argument("--pen", action="store_true", help="启用疑似笔迹 ignore 检测")
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.set_defaults(**defs)                           # 配置默认值最后注入（CLI 仍可覆盖）
    args = ap.parse_args()
    dst = Path(args.dst)
    assert args.yolo or args.src, "需要 --yolo 或 --src 之一（或在 configs/dataset.yaml 配置 source.yolo_root）"
    if args.yolo:
        run_yolo_mode(args, dst)
    else:
        run_strip_mode(args, dst)


if __name__ == "__main__":
    main()
