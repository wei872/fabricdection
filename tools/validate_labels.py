"""Validate YOLO label files (and their image caches) BEFORE CUDA training.

A single ``cls >= nc`` label, a 1-based dataset (``1..nc``), a non-finite box,
or a stale ``*.json`` cache can produce an opaque CUDA device-side assert deep
in ``jingwei/losses.py``.  This tool checks all of that on the host side, can
rewrite 1-based labels to the YOLO standard ``0..nc-1``, and can force a cache
rebuild so the next training run sees clean, 0-based labels.

Typical Windows usage::

    # 1) report only: does NOT modify anything
    python tools/validate_labels.py --cfg configs/jingwei_p.yaml ^
        --data D:/tyhn/yolo/dataset/source5_augmented/dataset.yaml ^
        --cache disk --cache-dir D:/tyhn/yolo/dataset/runs/cache

    # 2) if labels are 1..19, rewrite them to 0..18 and rebuild the cache
    python tools/validate_labels.py --cfg configs/jingwei_p.yaml ^
        --data D:/tyhn/yolo/dataset/source5_augmented/dataset.yaml ^
        --fix-one-based --cache disk ^
        --cache-dir D:/tyhn/yolo/dataset/runs/cache --rebuild-cache

Alternatively, leave the original .txt files alone and train with
``--label-offset 1`` (or ``--label-offset auto``).  In that case the runtime
sanitizer / cache builder shifts the ids for you.  Never feed raw class id
``19`` with ``nc=19`` into the detector head.

Paths, including Chinese / Unicode ones, are printed with ``console_safe`` so a
Windows console with a non-UTF-8 code page does not crash the checker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jingwei.cache import (CACHE_VERSION, build_dataset_cache, cache_paths,  # noqa: E402
                           cache_prefix, dataset_signature,
                           infer_label_offset_from_label_files)
from jingwei.data import infer_labels_dir, list_images                      # noqa: E402
from jingwei.image_io import console_safe                                   # noqa: E402
from jingwei.losses import resolve_label_offset                              # noqa: E402

WIN_ABS = re.compile(r"^[A-Za-z]:[\\/]")
ROW_KINDS = ("short_line", "parse_error", "non_integer_class")


def out(msg: str = "") -> None:
    print(console_safe(msg), flush=True)


def absolute_path(value, base: Path = ROOT) -> Path:
    """Windows drive paths are absolute even when this script runs on POSIX."""
    s = str(value).replace("\\", "/")
    if WIN_ABS.match(s):
        return Path(s)
    p = Path(s)
    return p if p.is_absolute() else Path(base) / p


def resolve_data_dir(value, data_yaml: Path, data: dict) -> Path:
    """Match ``tools/_common.py``: relative split paths resolve against ROOT.

    Also supports a standard YOLO ``path:`` root when the split path does not
    already include that root prefix.
    """
    s = str(value).replace("\\", "/")
    if WIN_ABS.match(s):
        return Path(s)
    p = Path(s)
    if p.is_absolute():
        return p
    candidates = [ROOT / p, data_yaml.parent / p]
    root_value = data.get("path")
    if root_value:
        r = str(root_value).replace("\\", "/")
        root = Path(r) if WIN_ABS.match(r) or Path(r).is_absolute() else ROOT / Path(r)
        if p.parts and root.name not in ("", ".", "..") and p.parts[0] == root.name:
            candidates.insert(0, ROOT / p)       # split already contains `path`
        else:
            candidates.insert(0, root / p)
    for c in candidates:
        if c.is_dir():
            return c
    return candidates[0]


def resolve_labels_dir(override, split: str, images_dir: Path) -> Path:
    if override:
        s = str(override).replace("\\", "/")
        s = (s.replace("{split}", split)
              .replace("{}", split)
              .replace("{images_dir}", str(images_dir)))
        return absolute_path(s, ROOT)
    return Path(infer_labels_dir(images_dir))


def parse_label_file(path: Path) -> list[dict]:
    """Parse a YOLO .txt into line records; keeps raw text for safe rewriting."""
    items: list[dict] = []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        lines = fh.readlines()
    for line in lines:
        eol = ""
        for e in ("\r\n", "\n", "\r"):
            if line.endswith(e):
                eol, raw = e, line[:-len(e)]
                break
        else:
            raw = line
        parts = raw.split()
        item = {"raw": raw, "eol": eol, "parts": parts, "kind": "blank"}
        if not parts:
            items.append(item)
            continue
        if len(parts) < 5:
            item["kind"] = "short_line"
            items.append(item)
            continue
        try:
            vals = [float(v) for v in parts[:5]]
        except ValueError:
            item["kind"] = "parse_error"
            items.append(item)
            continue
        c, cx, cy, bw, bh = vals
        if not all(math.isfinite(v) for v in vals):
            item["kind"] = "parse_error"
            items.append(item)
            continue
        if not float(c).is_integer():
            item["kind"] = "non_integer_class"
            items.append(item)
            continue
        item.update(kind="row", c=int(c), coords=(cx, cy, bw, bh))
        items.append(item)
    return items


def check_row(item: dict, offset: int, nc: int):
    """Return (error_kind, detail) or (None, None)."""
    c = int(item["c"]) - int(offset)
    if c < 0 or c >= nc:
        return "bad_class", (f"cls={item['c']} -> {c}; "
                             f"expected 0..{nc - 1} (or 1..{nc} with offset)")
    cx, cy, bw, bh = item["coords"]
    if not all(math.isfinite(v) for v in (cx, cy, bw, bh)):
        return "non_finite_box", "non-finite cx/cy/w/h"
    if bw <= 0 or bh <= 0:
        return "non_positive_box", f"w={bw:g} h={bh:g}"
    x0, y0 = cx - bw / 2, cy - bh / 2
    x1, y1 = cx + bw / 2, cy + bh / 2
    eps = 1e-6
    if x0 < -eps or y0 < -eps or x1 > 1 + eps or y1 > 1 + eps:
        return "out_of_bounds_box", (f"x0={x0:.4g} y0={y0:.4g} "
                                     f"x1={x1:.4g} y1={y1:.4g}")
    return None, None


def scan_sources(splits, data: dict, data_yaml: Path, labels_override):
    """First pass: parse every label file and collect raw class ids."""
    infos, all_ids = {}, []
    for split in splits:
        if split not in data:
            out(f"[skip] {split}: not in {data_yaml.name}")
            continue
        values = data[split] if isinstance(data[split], (list, tuple)) else [data[split]]
        images_dirs = [resolve_data_dir(v, data_yaml, data) for v in values]
        label_files = {}
        all_files, all_label_files = [], []
        n_images = n_missing = 0
        for images_dir in images_dirs:
            labels_dir = resolve_labels_dir(labels_override, split, images_dir)
            for img in list_images(images_dir):
                n_images += 1
                lab = labels_dir / (Path(img).stem + ".txt")
                all_files.append(img)
                all_label_files.append(lab)
                if not lab.is_file():
                    n_missing += 1
                    continue
                key = str(lab)
                if key in label_files:
                    continue
                parsed = parse_label_file(lab)
                label_files[key] = (lab, parsed, img)
                all_ids.extend(int(it["c"]) for it in parsed if it["kind"] == "row")
        infos[split] = {
            "images_dirs": images_dirs,
            "label_files": label_files,
            "all_files": all_files,
            "all_label_files": all_label_files,
            "n_images": n_images,
            "n_missing": n_missing,
        }
    return infos, all_ids


def validate_sources(infos, offset: int, nc: int, max_examples: int):
    stats = Counter()
    examples = []

    def add_example(split, path, line_no, kind, detail, raw):
        stats[kind] += 1
        if len(examples) < max_examples:
            examples.append(f"{split}:{console_safe(path)}:{line_no} "
                            f"{kind}: {detail} | {raw[:100]}")

    for split, info in infos.items():
        stats["images"] += info["n_images"]
        stats["missing_label_files"] += info["n_missing"]
        for lab, parsed, _img in info["label_files"].values():
            has_row = False
            for line_no, item in enumerate(parsed, 1):
                if item["kind"] == "row":
                    has_row = True
                    stats["rows"] += 1
                    kind, detail = check_row(item, offset, nc)
                    if kind:
                        add_example(split, lab, line_no, kind, detail, item["raw"])
                elif item["kind"] in ROW_KINDS:
                    add_example(split, lab, line_no, "malformed_line",
                                item["kind"], item["raw"])
            if not has_row:
                stats["empty_label_files"] += 1
    return stats, examples


def unique_backup(path: Path) -> Path:
    q = path.with_name(path.name + ".bak")
    i = 1
    while q.exists():
        q = path.with_name(path.name + f".bak{i}")
        i += 1
    return q


def write_fixed_file(path: Path, parsed, offset: int, backup: bool) -> int:
    lines, n_rows = [], 0
    for item in parsed:
        raw = item["raw"]
        if item["kind"] == "row":
            parts = list(item["parts"])
            parts[0] = str(int(item["c"]) - int(offset))
            raw = " ".join(parts)
            n_rows += 1
        lines.append(raw + item["eol"])
    if backup:
        shutil.copy2(path, unique_backup(path))
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("".join(lines))
    return n_rows


def cache_candidates(cache_dir: Path, split: str, img_size: int,
                     mode: str, images_dir: Path):
    tag = hashlib.sha1(str(images_dir).encode("utf-8", "replace")).hexdigest()[:8]
    prefix = cache_prefix(cache_dir, tag, split, img_size, mode)
    meta_path, data_path = cache_paths(prefix, mode)
    if meta_path.is_file():
        return [(meta_path, data_path)]
    found = []
    for mp in sorted(cache_dir.glob(f"{split}_{img_size}_{mode}_*.json")):
        ext = ".npy" if mode in ("disk", "ram") else ".bin"
        found.append((mp, mp.with_suffix(ext)))
    return found


def check_cache(infos, cache_dir: Path, mode: str, img_size: int,
                label_offset, nc: int, max_examples: int,
                delete_bad: bool, rebuild: bool, labels_override=None):
    stats, examples = Counter(), []
    if mode == "none":
        return stats, examples

    def expected_offset(info):
        """The offset the cache builder would resolve for THIS split.

        Comparing a 'auto' request against a cache that stores the resolved
        value (e.g. 1) must NOT look stale.
        """
        if str(label_offset).strip().lower() != "auto":
            return label_offset
        files = [str(lab) for lab, _p, _img in info["label_files"].values()]
        if not files:
            return label_offset
        return infer_label_offset_from_label_files(files, int(nc))

    def add_example(split, mp, kind, detail):
        stats[kind] += 1
        if len(examples) < max_examples:
            examples.append(f"cache {split}:{console_safe(mp.name)} "
                            f"{kind}: {detail}")

    def data_path_for(mp: Path) -> Path:
        ext = ".npy" if mode in ("disk", "ram") else ".bin"
        return mp.with_suffix(ext)

    def rebuild_one(split, info, mp: Path, off=None):
        images_dir = info["images_dirs"][0]
        labels_dir = resolve_labels_dir(labels_override, split, images_dir)
        tag = hashlib.sha1(str(images_dir).encode("utf-8", "replace")).hexdigest()[:8]
        prefix = cache_prefix(cache_dir, tag, split, img_size, mode)
        out(f"[{split}] rebuilding cache -> {console_safe(cache_paths(prefix, mode)[0].name)}")
        meta = build_dataset_cache(
            images_dir, labels_dir, img_size, prefix, mode, True,
            rebuild=True, log_fn=out,
            label_offset=label_offset if off is None else off, nc=nc)
        out(f"[{split}] rebuilt: {meta['n']} images, "
            f"dropped_bad_rows={meta.get('n_bad_labels', 0)}")

    for split, info in infos.items():
        split_off = expected_offset(info)
        candidates = []
        for images_dir in info["images_dirs"]:
            candidates.extend(cache_candidates(cache_dir, split, img_size,
                                               mode, images_dir))
        # de-duplicate paths
        seen, unique = set(), []
        for mp, dp in candidates:
            if str(mp) not in seen:
                seen.add(str(mp))
                unique.append((mp, dp))
        if not unique:
            out(f"[cache] {split}: no {mode} cache found under "
                f"{console_safe(cache_dir)}; training will build one")
            if rebuild:
                rebuild_one(split, info, cache_dir / f"{split}_{img_size}_{mode}.json",
                            split_off)
            continue

        for mp, dp in unique:
            try:
                meta = json.loads(mp.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError) as e:
                add_example(split, mp, "unreadable_cache", str(e))
                if rebuild:
                    rebuild_one(split, info, mp, split_off)
                continue
            if not meta.get("has_labels"):
                continue
            labels = meta.get("labels", [])
            stats["cache_records"] += int(meta.get("n", 0))
            if len(labels) != int(meta.get("n", -1)):
                add_example(split, mp, "cache_length_mismatch",
                            f"n={meta.get('n')} labels={len(labels)}")
            cache_ids, cache_errors = [], 0
            for i, rows in enumerate(labels):
                for row in rows or []:
                    if len(row) < 5:
                        cache_errors += 1
                        add_example(split, mp, "cache_bad_row", f"image {i}")
                        continue
                    try:
                        c = float(row[0])
                        coords = [float(v) for v in row[1:5]]
                    except (TypeError, ValueError):
                        cache_errors += 1
                        add_example(split, mp, "cache_bad_row", f"image {i}")
                        continue
                    if not math.isfinite(c) or not float(c).is_integer():
                        cache_errors += 1
                        add_example(split, mp, "cache_bad_row", f"image {i}")
                        continue
                    cache_ids.append(int(c))
            # Cached labels are ALWAYS stored 0-based: the builder already
            # applied the shift.  Re-applying it here would flag every id 0
            # as "-1" and make a perfectly good cache look invalid.
            off = 0
            for i, rows in enumerate(labels):
                for row in rows or []:
                    if len(row) < 5:
                        continue
                    try:
                        c = int(float(row[0]))
                        cx, cy, bw, bh = map(float, row[1:5])
                    except (TypeError, ValueError):
                        continue
                    item = {"c": c, "coords": (cx, cy, bw, bh)}
                    kind, detail = check_row(item, off, nc)
                    if kind:
                        cache_errors += 1
                        add_example(split, mp, f"cache_{kind}",
                                    f"image {i} cls={c} -> {c - off}")
            stale = []
            if meta.get("version") != CACHE_VERSION:
                stale.append(f"version={meta.get('version')} (need {CACHE_VERSION})")
            if meta.get("label_offset", "0") != str(split_off):
                stale.append(f"label_offset={meta.get('label_offset')} "
                             f"(need {split_off})")
            if meta.get("nc", nc) != nc:
                stale.append(f"nc={meta.get('nc')} (need {nc})")
            try:
                sig = dataset_signature(info["all_files"], info["all_label_files"])
                if meta.get("signature") != sig:
                    stale.append("signature changed")
            except OSError:
                stale.append("could not stat dataset files")
            if stale:
                add_example(split, mp, "stale_cache", "; ".join(stale))
            if cache_errors or stale:
                if rebuild:
                    rebuild_one(split, info, mp, split_off)
                elif delete_bad and (cache_errors or stale):
                    for p in (mp, dp, data_path_for(mp)):
                        try:
                            p.unlink(missing_ok=True)
                        except OSError:
                            pass
                    out(f"[cache] {split}: deleted invalid cache "
                        f"{console_safe(mp.name)}; next training run rebuilds it")
            else:
                out(f"[cache] {split}: OK {console_safe(mp.name)} "
                    f"({meta.get('n')} images, offset={meta.get('label_offset', '0')})")
    return stats, examples


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Check YOLO labels / caches before CUDA training.")
    ap.add_argument("--cfg", default=None,
                    help="model config; supplies nc / img_size / dataset_yaml")
    ap.add_argument("--data", default=None,
                    help="dataset yaml (default: cfg.data.dataset_yaml or configs/fabric19.yaml)")
    ap.add_argument("--nc", type=int, default=None)
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--splits", default="train,val",
                    help="comma-separated YOLO splits to check (default: train,val)")
    ap.add_argument("--labels-dir", default=None,
                    help="override labels dir; supports {split}, {} and {images_dir}")
    ap.add_argument("--label-offset", choices=["0", "1", "auto"], default=None,
                    help="class-id interpretation; default auto (detect 1-based)")
    ap.add_argument("--fix-one-based", action="store_true",
                    help="rewrite detected 1..nc labels to 0..nc-1 (with .bak)")
    ap.add_argument("--no-backup", action="store_true",
                    help="with --fix-one-based, do not create .txt.bak files")
    ap.add_argument("--cache", choices=["none", "disk", "ram", "bytes"],
                    default="none", help="also check the matching image cache")
    ap.add_argument("--cache-dir", default="runs/cache")
    ap.add_argument("--rebuild-cache", action="store_true",
                    help="rebuild checked caches after fixing / normalising labels")
    ap.add_argument("--delete-bad-cache", action="store_true",
                    help="delete invalid cache files so training rebuilds them")
    ap.add_argument("--max-examples", type=int, default=12)
    args = ap.parse_args()

    if args.rebuild_cache and args.cache == "none":
        ap.error("--rebuild-cache needs --cache disk / ram / bytes")
    if args.delete_bad_cache and args.cache == "none":
        ap.error("--delete-bad-cache needs --cache disk / ram / bytes")

    cfg = {}
    if args.cfg:
        cfg = yaml.safe_load(absolute_path(args.cfg).read_text(encoding="utf-8")) or {}
    data_yaml = absolute_path(
        args.data or cfg.get("data", {}).get("dataset_yaml")
        or "configs/fabric19.yaml")
    if not data_yaml.is_file():
        ap.error(f"dataset yaml not found: {data_yaml}")
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    nc = int(args.nc or cfg.get("data", {}).get("nc") or data.get("nc") or 19)
    img_size = int(args.img_size or (cfg.get("input_size") or [1280])[0])
    mode = args.label_offset if args.label_offset is not None else "auto"
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    cache_dir = absolute_path(args.cache_dir)

    out("=" * 78)
    out("JingWei YOLO label / cache validator")
    out(f"data={console_safe(data_yaml)}  cfg={args.cfg or '-'}  "
        f"nc={nc}  img_size={img_size}  splits={','.join(splits)}")
    out(f"label_offset_mode={mode}  cache={args.cache}  "
        f"cache_dir={console_safe(cache_dir)}")
    out("=" * 78)

    infos, all_ids = scan_sources(splits, data, data_yaml, args.labels_dir)
    if not infos:
        out("ERROR: no requested splits found in dataset yaml")
        return 2
    # Absurd ids (e.g. 99 or -1) are reported as bad_class, but must not stop
    # ``auto`` from recognising an otherwise clean 1..nc dataset.
    ids_for_offset = [i for i in all_ids if 0 <= i <= nc]
    ids_t = torch.tensor(ids_for_offset, dtype=torch.long)
    inferred_offset = resolve_label_offset(ids_t, mode, nc)
    raw_lo = min(all_ids) if all_ids else -1
    raw_hi = max(all_ids) if all_ids else -1
    out(f"raw class ids: count={len(all_ids)}  min={raw_lo}  max={raw_hi}  "
        f"offset_candidates={len(ids_for_offset)}  "
        f"inferred_shift={inferred_offset}")

    stats, examples = validate_sources(infos, inferred_offset, nc,
                                       args.max_examples)
    fixed = False
    if args.fix_one_based:
        if inferred_offset == 1:
            n_files = n_rows = 0
            for info in infos.values():
                for lab, parsed, _img in info["label_files"].values():
                    if any(it["kind"] == "row" for it in parsed):
                        n_rows += write_fixed_file(lab, parsed, 1,
                                                   not args.no_backup)
                        n_files += 1
            fixed = True
            out(f"FIXED 1-based labels: {n_files} files, {n_rows} rows "
                f"(backup={'no' if args.no_backup else '*.txt.bak'})")
            infos, all_ids = scan_sources(splits, data, data_yaml,
                                          args.labels_dir)
            stats, examples = validate_sources(infos, 0, nc,
                                               args.max_examples)
        else:
            out("FIX skipped: no global 1-based evidence. If you are SURE the "
                "labels are 1..nc, rerun with --label-offset 1 --fix-one-based.")

    out("-" * 78)
    out(f"source labels: images={stats['images']}  "
        f"label_files={sum(len(v['label_files']) for v in infos.values())}  "
        f"rows={stats['rows']}  missing_txt={stats['missing_label_files']}  "
        f"empty_txt={stats['empty_label_files']}")
    out(f"errors: bad_class={stats['bad_class']}  "
        f"bad_box={stats['non_finite_box'] + stats['non_positive_box'] + stats['out_of_bounds_box']}  "
        f"malformed_line={stats['malformed_line']}")
    for e in examples:
        out(f"  - {e}")

    # After fixing source .txt files, cached JSON must be 0-based too.
    cache_offset = 0 if fixed else mode
    cache_stats, cache_examples = check_cache(
        infos, cache_dir, args.cache, img_size, cache_offset, nc,
        args.max_examples, args.delete_bad_cache, args.rebuild_cache,
        args.labels_dir)
    if args.cache != "none":
        out("-" * 78)
        n_issues = sum(v for k, v in cache_stats.items()
                       if k != "cache_records")
        repaired = args.rebuild_cache or args.delete_bad_cache
        out(f"cache: records={cache_stats['cache_records']}  "
            f"issues_found={n_issues}"
            + ("  (repaired below)" if repaired and n_issues else ""))
        for e in cache_examples:
            out(f"  - {e}"
                + ("  [now rebuilt]" if repaired else ""))

    source_errors = (stats["bad_class"] + stats["non_finite_box"] +
                     stats["non_positive_box"] + stats["out_of_bounds_box"] +
                     stats["malformed_line"])
    cache_errors = sum(v for k, v in cache_stats.items()
                       if k != "cache_records" and k != "stale_cache")
    # --rebuild-cache / --delete-bad-cache already resolved any reported cache
    # problems; don't make a successful repair look like a failed validation.
    if args.rebuild_cache or args.delete_bad_cache:
        cache_errors = 0
        stale_left = False
    else:
        stale_left = bool(cache_stats["stale_cache"])

    out("=" * 78)
    if inferred_offset == 1 and not fixed:
        out("[info] labels look 1-based. No action is needed if you always train")
        out("       with --label-offset 1|auto (the cache build / runtime sanitizer")
        out("       shift them). To make the .txt files standard 0..nc-1 instead:")
        out(f"  python tools/validate_labels.py --data {console_safe(data_yaml)} "
            f"--nc {nc} --fix-one-based --label-offset {mode} "
            f"--cache {args.cache} --cache-dir {console_safe(cache_dir)} "
            f"--rebuild-cache")
        out("  ...or train with --label-offset 1 (or auto) WITHOUT editing .txt.")
    if source_errors or cache_errors or stale_left:
        out("RESULT: INVALID. Fix the reported labels / cache before CUDA training.")
        out("If you changed .txt files, also run training once with "
            "--rebuild-cache (or delete the old cache JSON/Npy/Bin).")
        return 2
    out("RESULT: OK. No invalid class ids / boxes / cached labels found.")
    out("Training still sanitizes targets at runtime, but this dataset is clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
