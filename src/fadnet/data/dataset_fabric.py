"""数据集读取（docs/DATASET_SPEC.md）

- NormalTileDataset：U 分支训练/标定（仅正常样本，支持 ignore 掩膜剔除）
- DetEvalDataset    ：评估/测试脚本用（识别 YOLO 框标注与 GT 掩膜）
注：S 分支监督训练直接由 ultralytics 吃 YOLO 目录，无需自定义 Dataset。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


IMG_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _load_grayscale3(path: str | Path) -> np.ndarray:
    from PIL import Image

    im = Image.open(path).convert("L")
    a = np.asarray(im, dtype=np.float32) / 255.0
    return np.repeat(a[..., None], 3, axis=-1)      # 灰度 → 3 通道


class NormalTileDataset(Dataset):
    """正常 tiles（U 分支训练 / 记忆库构建）。

    两种布局自动识别：
    A) 标准 YOLO 布局（最终数据集格式，用户 source5 同构）：
       root/images/{train,val}/*.png + root/labels/{train,val}/*.txt
       **无对应标签 txt 或空 txt 的图 = 正常样本**；ignore 掩膜可选
       root/ignore_masks/{split}/<同名>.png。
    B) 旧内部布局：root/normal/<fabric_id>/images/*.png，ignore 掩膜
       ignore_root/<fabric_id>/<同名>.png。
    """

    def __init__(self, root: str | Path, size: int = 896, fabrics: list[str] | None = None,
                 ignore_root: str | Path | None = None, max_per_fabric: int | None = None):
        self.root = Path(root)
        self.size = size
        self.ignore_root = Path(ignore_root) if ignore_root else None
        self.items: list[tuple[Path, str, Path | None]] = []
        if (self.root / "images").is_dir():                       # 布局 A
            for split in ("train", "val"):
                idir = self.root / "images" / split
                if not idir.is_dir():
                    continue
                for p in sorted(x for x in idir.iterdir()
                                if x.suffix.lower() in IMG_SUFFIXES):
                    lab = self.root / "labels" / split / (p.stem + ".txt")
                    if lab.exists() and lab.read_text().strip():
                        continue                                   # 有标注 = 异常图，跳过
                    mask = self.root / "ignore_masks" / split / p.name
                    self.items.append((p, split, mask if mask.exists() else None))
        else:                                                     # 布局 B
            for d in sorted(self.root.joinpath("normal").iterdir()):
                if not d.is_dir() or (fabrics and d.name not in fabrics):
                    continue
                imgs = sorted((d / "images").glob("*.png"))
                if max_per_fabric:
                    imgs = imgs[:max_per_fabric]
                for p in imgs:
                    mask = None
                    if self.ignore_root:
                        m = self.ignore_root / d.name / p.name
                        mask = m if m.exists() else None
                    self.items.append((p, d.name, mask))
        assert self.items, f"no normal images under {self.root}"
        if max_per_fabric and (self.root / "images").is_dir():
            self.items = self.items[:max_per_fabric]

    def __len__(self) -> int:
        return len(self.items)

    def _resize(self, x: np.ndarray) -> np.ndarray:
        from PIL import Image

        im = Image.fromarray((x[..., 0] * 255).astype(np.uint8))
        if im.size != (self.size, self.size):
            im = im.resize((self.size, self.size), Image.BILINEAR)
        a = np.asarray(im, dtype=np.float32) / 255.0
        return np.repeat(a[..., None], 3, axis=-1)

    def __getitem__(self, i: int) -> dict:
        img_path, key, mask_path = self.items[i]
        x = _load_grayscale3(img_path)
        x = self._resize(x)
        if mask_path is not None:                            # 忽略区替换为偏移正常区
            from PIL import Image
            mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32) > 0.5
            if mask.shape[:2] != (self.size, self.size):
                mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(
                    (self.size, self.size), Image.NEAREST), np.float32) > 127
            rolled = np.roll(x, (mask.shape[0] // 3, mask.shape[1] // 3), (0, 1))
            x[mask] = rolled[mask]
        t = torch.from_numpy(x).permute(2, 0, 1).float()
        t = (t - 0.5) / 0.5                                  # [-1,1]（ViT 侧再按需标准化）
        return {"image": t, "fabric_id": key, "path": str(img_path)}


class DetEvalDataset(Dataset):
    """评估用：异常/正常 tiles + YOLO 框 + 可选掩膜。"""

    def __init__(self, root: str | Path, which: str = "abnormal",
                 fabrics: list[str] | None = None, size: int = 1280):
        self.root = Path(root)
        self.size = size
        self.items: list[tuple[Path, Path | None, str]] = []
        if (self.root / "images").is_dir():                  # 标准 YOLO 布局：有标签者=异常
            for split in ("train", "val"):
                idir = self.root / "images" / split
                if not idir.is_dir():
                    continue
                for p in sorted(x for x in idir.iterdir()
                                if x.suffix.lower() in IMG_SUFFIXES):
                    lab = self.root / "labels" / split / (p.stem + ".txt")
                    if lab.exists() and lab.read_text().strip():
                        self.items.append((p, lab, split))
        else:                                                # 旧内部布局
            base = self.root / which
            for d in sorted(base.iterdir()):
                if not d.is_dir() or (fabrics and d.name not in fabrics):
                    continue
                for p in sorted((d / "images").glob("*.png")):
                    lab = d / "labels" / (p.stem + ".txt")
                    self.items.append((p, lab if lab.exists() else None, d.name))

    def __len__(self) -> int:
        return len(self.items)

    @staticmethod
    def read_yolo_label(path: Path, wh: int) -> np.ndarray:
        """返回 [K,5]：cls, x1, y1, x2, y2（像素）。"""
        rows = []
        for ln in path.read_text().strip().splitlines():
            c, xc, yc, w, h = map(float, ln.split())
            rows.append([c, (xc - w / 2) * wh, (yc - h / 2) * wh,
                         (xc + w / 2) * wh, (yc + h / 2) * wh])
        return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 5), np.float32)

    def __getitem__(self, i: int) -> dict:
        img_path, lab_path, fabric_id = self.items[i]
        x = _load_grayscale3(img_path)
        h, w = x.shape[:2]
        gt = self.read_yolo_label(lab_path, max(h, w)) if lab_path else np.zeros((0, 5), np.float32)
        return {"image": x, "gt": gt, "fabric_id": fabric_id, "path": str(img_path)}
