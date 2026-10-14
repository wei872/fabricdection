#!/usr/bin/env python3
"""U 分支训练：Dinomaly 重构通道 + PatchCore 记忆库 + 阈值标定（design §5.2 / §7.2）

仅使用正常样本。三阶段一次跑完：
  1) 冻结骨干 + NoiseyBottleneck + 线性注意力解码器 → Loose 余弦重构训练
  2) 同批正常样本特征 → k-center Coreset 记忆库（无梯度）
  3) val_normal 上校准 1%/99% 分位与 τ_u，写 configs/fabric_thresholds.yaml

用法：
  # 正式（DINOv3 权重本地路径；数据集见 docs/DATASET_SPEC.md）
  python scripts/train_ustream.py --root dataset_fabric --ckpt weights/dinov3_vitsp16.pth \
      --size 896 --iters 20000 --bs 16 --device cuda
  # 冒烟（随机初始化骨干，仅验证管线）
  python scripts/train_ustream.py --root dataset_fabric --iters 30 --bs 2 --smoke --max-per-fabric 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fadnet.backbone.vit import build_backbone
from fadnet.data.dataset_fabric import NormalTileDataset
from fadnet.ustream.dinomaly_branch import DinomalyReconBranch
from fadnet.ustream.heatmap import self_normalize
from fadnet.ustream.patchcore_memory import (MemoryBank, knn_anomaly_scores,
                                             neighborhood_aware, scores_to_heatmap,
                                             style_center)


def feature_grid(branch: DinomalyReconBranch, x: torch.Tensor, layer: int = 5) -> torch.Tensor:
    """取冻结骨干第 layer 层 token → [B,D,gh,gw]（记忆库/打分共用提取）。"""
    feats: dict[int, torch.Tensor] = {}
    h = branch.backbone.blocks[layer].register_forward_hook(
        lambda _m, _i, o: feats.__setitem__(0, o))
    with torch.no_grad():
        branch.backbone(x)
    h.remove()
    t = feats[0][:, -int((x.shape[-1] // 16) * (x.shape[-2] // 16)):, :]
    gh = gw = int(t.shape[1] ** 0.5)
    return t.transpose(1, 2).reshape(x.shape[0], -1, gh, gw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset_fabric")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--backbone", default="dinov3_vitsp16")
    ap.add_argument("--size", type=int, default=896)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--noise-p", type=float, default=0.2)
    ap.add_argument("--decoder-layers", type=int, default=2)
    ap.add_argument("--coreset", type=int, default=8192)
    ap.add_argument("--memory-layer", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="runs/ustream")
    ap.add_argument("--thresholds-out", default="configs/thresholds.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-per-fabric", type=int, default=None)
    args = ap.parse_args()

    dev = torch.device(args.device)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ds = NormalTileDataset(args.root, size=args.size,
                           ignore_root=Path(args.root) / "ignore_masks",
                           max_per_fabric=args.max_per_fabric)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True,
                    num_workers=0 if args.smoke else 4, drop_last=True)

    backbone = build_backbone(args.backbone, args.ckpt, img_size=args.size, smoke=args.smoke).to(dev)
    dim = backbone.blocks[0].norm1.normalized_shape[0]
    branch = DinomalyReconBranch(backbone, feat_dim=dim, decoder_layers=args.decoder_layers,
                                 noise_p=args.noise_p).to(dev)
    opt = torch.optim.AdamW([p for p in branch.parameters() if p.requires_grad], lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iters)

    # ---------- 1) Dinomaly 重构训练（正常样本 only） ----------
    branch.train()
    t0, step = time.time(), 0
    while step < args.iters:
        for batch in dl:
            x = batch["image"].to(dev)
            encs, outs = branch(x)
            loss = branch.loss(encs, outs)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
            step += 1
            if step % 50 == 0 or step == 1:
                print(f"[train] {step}/{args.iters} loss={loss.item():.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e} {(time.time()-t0)/step:.2f}s/it")
            if step >= args.iters:
                break
    torch.save({"decoder": {k: v for k, v in branch.state_dict().items()
                            if not k.startswith("backbone.")},
                "config": vars(args)}, out / "ustream_decoder.pt")
    print(f"[train] done, decoder saved to {out/'ustream_decoder.pt'}")
    branch.eval()

    # ---------- 2) PatchCore 全局记忆库（style_center 后布种无关，免登记） ----------
    bank = MemoryBank(dim=dim)
    feats_all: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in dl:
            x = batch["image"].to(dev)
            fmap = neighborhood_aware(style_center(feature_grid(branch, x, args.memory_layer)))
            feats = fmap.flatten(2).permute(0, 2, 1).reshape(-1, dim)
            feats_all.append(feats[:: max(1, feats.shape[0] // 4096)])
    bank.build(torch.cat(feats_all), n_centers=args.coreset)
    print(f"[patchcore] global coreset={tuple(bank.centers.shape)}")
    bank.save(str(out / "memory_bank.pt"))

    # ---------- 3) 全局阈值标定（自归一化 z 空间，val_normal p99） ----------
    zs: list[np.ndarray] = []
    with torch.no_grad():
        for batch in dl_limited(args, ds, dev):
            x = batch["image"].to(dev)
            a_rec = branch.anomaly_map(x)
            fmap = neighborhood_aware(style_center(feature_grid(branch, x, args.memory_layer)))
            q = fmap.flatten(2).permute(0, 2, 1).reshape(-1, dim)
            g = fmap.shape[-1]
            a_knn = scores_to_heatmap(knn_anomaly_scores(q, bank.centers), g, g,
                                      (x.shape[-2], x.shape[-1]))
            z = torch.maximum(self_normalize(a_rec), self_normalize(a_knn))
            zs.append(z.flatten().cpu().numpy())
    z = np.concatenate(zs)
    tau_u = float(np.percentile(z, 99.0))
    thr = {"*": {"tau_u": round(tau_u, 3), "tau_large": None,
                 "bg_quantile": 90, "provisional": True}}
    Path(args.thresholds_out).write_text(yaml.safe_dump(thr, allow_unicode=True))
    print(f"[calib] global tau_u={tau_u:.3f}（val normal p99，z 空间全局通用）"
          f" → {args.thresholds_out}")
    print("[note] tau_large 需在含大缺陷的 calib 集上按 design §7.2 填定（当前 provisional）")


def dl_limited(args, ds, dev, split: str = "val_normal.txt"):
    split_path = Path(args.root) / "splits" / split
    if not split_path.exists() or not split_path.read_text().strip():
        split_path = Path(args.root) / "splits" / "train_normal.txt"
    files = {Path(p).resolve() for p in split_path.read_text().strip().splitlines() if p.strip()}
    idx = [i for i, it in enumerate(ds.items) if it[0].resolve() in files]
    sub = torch.utils.data.Subset(ds, idx)
    return DataLoader(sub, batch_size=max(1, args.bs // 2), shuffle=False,
                      num_workers=0)


if __name__ == "__main__":
    main()
