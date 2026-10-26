"""Verification suite for the JingWei reference implementation.

Run from the repo root::

    python -m unittest discover -s tests -v

No pytest / no GPU / no real data needed. It checks shapes at the required
1280x1280 input, trainability, the deterministic safety net, fabric-type
invariance, memory-bank algebra, and the dataset readers.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jingwei import build_model, load_config                       # noqa: E402
from jingwei.fabric import (fabric_descriptor, fit_fabric_prototypes,  # noqa: E402
                            greedy_coreset, spherical_kmeans)
from jingwei.fusion import safety_net_numpy                       # noqa: E402
from jingwei.layers import LocalContrastNorm                     # noqa: E402
from jingwei.losses import hungarian_match                       # noqa: E402
from jingwei.utils import count_parameters, profile_macs          # noqa: E402

CFG_N = str(ROOT / "configs/jingwei_n.yaml")


def weave(period, bright=128, refl=0.0, size=512):
    yy, xx = np.mgrid[0:size, 0:size]
    g = (np.sin(2 * np.pi * xx / period) * np.sin(2 * np.pi * yy / (period * 1.3))) * 35
    field = refl * 60 * np.sin(2 * np.pi * xx / size)
    return np.clip(bright + g + field, 0, 255).astype(np.uint8)


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.cfg = load_config(CFG_N)
        cls.model = build_model(cls.cfg)
        cls.model.eval()
        cls.x = torch.randn(1, 3, 1280, 1280)

    # ---------------------------------------------------------------- build / shapes
    def test_01_build_and_channels(self):
        self.assertEqual(self.model.backbone.out_channels, (48, 96, 192, 384))
        self.assertEqual(self.model.backbone.strides, (4, 8, 16, 32))
        p = count_parameters(self.model)
        self.assertLess(p / 1e6, 8.0, "nano variant should stay under 8 M params")
        print(f"\n  params = {p / 1e6:.2f} M")

    def test_02_export_shapes_at_1280(self):
        with torch.no_grad():
            outs = self.model.export_forward(self.x)
        expected = [(1, 19, 160, 160), (1, 4, 160, 160), (1, 1, 160, 160),
                    (1, 19, 40, 40), (1, 4, 40, 40), (1, 1, 40, 40),
                    (1, 1, 160, 160), (1, 3, 160, 160)]  # + context
        self.assertEqual([tuple(t.shape) for t in outs], expected)
        for t in outs:
            self.assertTrue(torch.isfinite(t).all())

    def test_03_train_forward_backward(self):
        m = self.model
        m.train()
        m.set_stage("full")
        targets = [dict(boxes=torch.tensor([[100., 100., 240., 170.],
                                            [620., 500., 940., 545.]]),
                        labels=torch.tensor([1, 17]))]
        out = m(self.x, targets=targets, proto_id=0, it=500)
        losses = out["losses"]
        self.assertIn("rec", losses)
        for k, v in losses.items():
            self.assertTrue(torch.isfinite(v), f"{k} is not finite: {v}")
        losses["total"].backward()
        grads = [p.grad for p in m.parameters() if p.grad is not None]
        self.assertGreater(len(grads), 20)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        m.eval()

    def test_04_eval_detections(self):
        with torch.no_grad():
            det = self.model(self.x, proto_id=0)
        self.assertIsInstance(det, list)
        d = det[0]
        for key in ("boxes", "scores", "labels"):
            self.assertIn(key, d)
            self.assertEqual(d[key].shape[0], d["boxes"].shape[0])
        self.assertIn("n_safety_boxes", d)

    def test_05_compute_budget(self):
        macs = profile_macs(self.model, [torch.zeros(1, 3, 1280, 1280)])
        self.assertLess(macs["total"] / 1e9, 20.0,
                        "nano must stay under 20 G MACs at 1280x1280")
        print(f"  MACs @1280 = {macs['total'] / 1e9:.2f} G")

    def test_06_strip_train_only(self):
        m = build_model(self.cfg).eval()
        m.strip_train_only()
        with torch.no_grad():
            outs = m.export_forward(torch.randn(1, 3, 256, 256))
        self.assertEqual(len(outs), 8)   # 7 base + context

    # ---------------------------------------------------------------- safety net
    def test_07_safety_net_emits_large_defect(self):
        a = np.zeros((160, 160), np.float32)
        a[30:90, 40:130] = 0.9                      # ~0.1% of image area
        out = safety_net_numpy(a, np.zeros((0, 4), np.float32),
                               np.zeros(0, np.float32), np.zeros(0, np.int64),
                               (1280, 1280), self.cfg, unknown_label=19)
        self.assertEqual(out["n_safety_boxes"], 1)
        self.assertEqual(len(out["boxes"]), 1)
        self.assertEqual(int(out["labels"][0]), 19)
        b = out["boxes"][0]
        self.assertGreater(b[2] - b[0], 300)       # it's a LARGE defect
        self.assertGreater(b[3] - b[1], 200)

    def test_08_safety_net_no_false_box_on_normal(self):
        a = np.full((160, 160), 0.05, np.float32)
        out = safety_net_numpy(a, np.zeros((0, 4), np.float32),
                               np.zeros(0, np.float32), np.zeros(0, np.int64),
                               (1280, 1280), self.cfg, 19)
        self.assertEqual(len(out["boxes"]), 0)

    def test_09_safety_net_boosts_existing_box(self):
        a = np.zeros((160, 160), np.float32)
        a[30:90, 40:130] = 0.9
        boxes = np.array([[300., 250., 1000., 700.]], np.float32)
        scores = np.array([0.4], np.float32)
        labels = np.array([10], np.int64)
        out = safety_net_numpy(a, boxes, scores, labels, (1280, 1280), self.cfg, 19)
        self.assertEqual(out["n_safety_boxes"], 0, "no duplicate box should be emitted")
        self.assertGreater(float(out["scores"][0]), 0.4, "confidence should be boosted")

    # ---------------------------------------------------------------- fabric robustness
    def test_10_descriptor_is_reflectance_invariant(self):
        for period in (5, 9, 13, 19):
            d1 = fabric_descriptor(weave(period, 100, 0.0))
            d2 = fabric_descriptor(weave(period, 170, 0.9))
            self.assertAlmostEqual(d1[0], d2[0], places=4,
                                   msg="warp period must not depend on 反光")
            self.assertAlmostEqual(d1[0], float(period), delta=0.6,
                                   msg="warp period should be recovered")
            self.assertAlmostEqual(d1[1], d2[1], places=4)

    def test_11_local_contrast_norm_removes_illumination(self):
        lcn = LocalContrastNorm(k=15, learnable=False).eval()
        base = torch.from_numpy(weave(9, 128, 0.0, 128).astype(np.float32) / 255.0)
        base = base[None, None].expand(1, 3, -1, -1).contiguous()
        yy, xx = np.mgrid[0:128, 0:128]
        field = (0.25 * np.sin(2 * np.pi * xx / 128)).astype(np.float32)
        lit = (base + torch.from_numpy(field)[None, None]).clamp(0, 1)
        with torch.no_grad():
            a, b = lcn(base), lcn(lit)
        rel = (a - b).abs().mean() / (a.abs().mean() + 1e-6)
        self.assertLess(rel, 0.6, "output should be much less sensitive than raw pixels")

    def test_12_prototypes_and_memory_bank_algebra(self):
        D = np.stack([fabric_descriptor(weave(p, b, r)) for p in (5, 9, 13, 19)
                      for b in (100, 160) for r in (0.0, 0.8)])
        proto, labels = fit_fabric_prototypes(D, k=4)
        self.assertEqual(proto.centers.shape, (4, 8))
        self.assertEqual(labels.shape, (16,))
        # the 4 images of each period (2 brightness x 2 reflection) must land in
        # exactly ONE fabric prototype: the descriptor is photometrically invariant
        for g in range(4):
            group = labels[g * 4:(g + 1) * 4]
            self.assertEqual(len(set(group)), 1, f"period group {g} split: {group}")
            self.assertLess(np.abs(D[g * 4:(g + 1) * 4] - D[g * 4]).max(), 0.01)

        X = np.random.RandomState(0).randn(2000, 128).astype(np.float32)
        X /= np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8
        c = spherical_kmeans(X, 128, iters=8)
        self.assertEqual(c.shape, (128, 128))
        self.assertTrue(np.allclose(np.linalg.norm(c, axis=-1), 1.0, atol=1e-4))
        self.assertEqual(greedy_coreset(X, 64).shape, (64, 128))

        bank = self.model.normality.bank
        bank.set_prototype(0, c)
        stored = bank.centroids[0]
        q = torch.from_numpy(X[:1].reshape(1, 128, 1, 1))
        # exact equivalence: 1 - max_k <q, c_k> (conv) == explicit cosine max
        explicit = 1.0 - (q.reshape(1, 128) @ stored.T).max()
        got = bank.raw_distance(q, 0).reshape(-1)[0]
        self.assertAlmostEqual(float(got), float(explicit), places=4)
        s = bank.score(torch.from_numpy(X[:1].reshape(1, 128, 1, 1)), 0)
        self.assertEqual(s.shape, (1, 1, 1, 1))
        self.assertTrue(((s >= 0) & (s <= 1)).all())

    # ---------------------------------------------------------------- losses / data
    def test_13_hungarian_assignment(self):
        m = build_model(self.cfg).eval()
        with torch.no_grad():
            feats = m.neck(m.backbone(torch.randn(1, 3, 256, 256)))
            preds = m.head(feats, training=True)
        targets = [dict(boxes=torch.tensor([[30., 30., 90., 70.],
                                            [140., 120., 220., 190.]]),
                        labels=torch.tensor([1, 17]))]
        matches = hungarian_match(preds, targets, (8, 32), 19, 1.0, 800)
        self.assertEqual(sum(len(x) for x in matches), 2)

    def test_14_dataset_readers(self):
        import cv2
        from jingwei.data import NormalImageDataset, YOLODefectDataset, collate
        with tempfile.TemporaryDirectory() as tmp:
            imgs = Path(tmp) / "images" / "train"
            labs = Path(tmp) / "labels" / "train"
            normal = Path(tmp) / "normal" / "images" / "train"
            for d in (imgs, labs, normal):
                d.mkdir(parents=True, exist_ok=True)
            for i in range(3):
                cv2.imwrite(str(imgs / f"{i}.jpg"), weave(9 + i, 128, 0.2, 128))
                (labs / f"{i}.txt").write_text("1 0.25 0.30 0.10 0.08\n17 0.7 0.6 0.2 0.1\n")
                cv2.imwrite(str(normal / f"n{i}.jpg"), weave(7, 140, 0.4, 128))
            ds = YOLODefectDataset(imgs, labs, img_size=128, augment=False)
            item = ds[0]
            self.assertEqual(tuple(item["image"].shape), (3, 128, 128))
            self.assertEqual(tuple(item["boxes"].shape), (2, 4))
            self.assertEqual(tuple(item["labels"].shape), (2,))
            nd = NormalImageDataset(normal, img_size=128, augment=False,
                                    with_descriptor=True)
            self.assertEqual(tuple(nd[0]["descriptor"].shape), (8,))
            batch = collate([ds[0], ds[1]])
            self.assertEqual(tuple(batch["image"].shape), (2, 3, 128, 128))
            self.assertIsInstance(batch["boxes"], list)

    def test_15_tools_import_cleanly(self):
        import py_compile
        for f in sorted((ROOT / "tools").glob("*.py")):
            py_compile.compile(str(f), doraise=True)
        for f in sorted((ROOT / "jingwei").glob("*.py")):
            py_compile.compile(str(f), doraise=True)

    # ---------------------------------------------- 1660 SUPER (no Tensor Core) variant
    def test_16_pico_variant_budget(self):
        cfg = load_config(str(ROOT / "configs/jingwei_p.yaml"))
        m = build_model(cfg).eval()
        macs = profile_macs(m, [torch.zeros(1, 3, 1280, 1280)])
        self.assertLess(macs["total"] / 1e9, 9.0, "pico must fit a GTX 1660 SUPER")
        self.assertFalse(cfg.fusion.roi_head, "pico disables ROI head for TRT export")
        with torch.no_grad():
            outs = m.export_forward(torch.randn(1, 3, 1280, 1280))
        self.assertEqual(len(outs), 8)               # 7 base + context
        self.assertEqual(tuple(outs[6].shape), (1, 1, 128, 128))   # anomaly
        self.assertEqual(tuple(outs[7].shape), (1, 3, 128, 128))   # context
        print(f"  pico params = {count_parameters(m)/1e6:.2f} M, "
              f"MACs = {macs['total']/1e9:.2f} G")

    def test_17_export_postprocess_equivalence(self):
        """export_forward + graph-external postprocess == direct inference."""
        for cfgp in ("configs/jingwei_n.yaml", "configs/jingwei_p.yaml"):
            m = build_model(str(ROOT / cfgp)).eval()
            rf = bool(m.cfg.fusion.roi_head)
            x = torch.randn(1, 3, 256, 256)
            with torch.no_grad():
                outs = m.export_forward(x, 0, return_feats=rf)
                a = m.postprocess_from_tensors(outs, (256, 256))[0]
                b = m(x, proto_id=0)[0]
            self.assertEqual(len(outs), 10 if rf else 8)  # + context
            self.assertEqual(set(a.keys()), set(b.keys()))
            for k in ("boxes", "scores", "labels"):
                self.assertEqual(a[k].shape, b[k].shape, f"{cfgp}:{k}")

    def test_18_dwconv_switch(self):
        """model.dwconv=false must still build & run (A/B on 1660 SUPER)."""
        cfg = load_config(str(ROOT / "configs/jingwei_p.yaml"))
        cfg.model.dwconv = False
        m = build_model(cfg).eval()
        with torch.no_grad():
            outs = m.export_forward(torch.randn(1, 3, 256, 256))
        self.assertEqual(len(outs), 8)

    def test_19_pico_onnx_export(self):
        """jingwei_p's 160->64 normality grid must still export to static ONNX."""
        try:
            import onnx
        except ImportError:
            self.skipTest("onnx not installed")

        cfg = load_config(str(ROOT / "configs/jingwei_p.yaml"))
        m = build_model(cfg).eval().strip_train_only()
        if m.normality is not None:
            m.normality.bank.set_active(0)

        class W(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                return self.model.export_forward(x, 0, return_feats=False)

        names = ["cls3", "reg3", "obj3", "cls5", "reg5", "obj5", "anomaly", "context"]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "jingwei_p.onnx"
            torch.onnx.export(
                W(m), torch.zeros(1, 3, 1280, 1280), str(path),
                opset_version=17, input_names=["images"], output_names=names,
                dynamic_axes=None, do_constant_folding=True)
            graph = onnx.load(str(path))
            onnx.checker.check_model(graph)
            outs = [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim])
                    for o in graph.graph.output]
        self.assertEqual([n for n, _ in outs], names)
        self.assertEqual(outs[6][1], [1, 1, 128, 128])    # anomaly
        self.assertEqual(outs[7][1], [1, 3, 128, 128])    # context

    # ---------------------------------------------------------- stage-0 DINO teacher
    def test_20_dino_distill_module(self):
        """synthetic DINO teacher -> student adaptor -> cosine loss backward."""
        from jingwei.dino import (DINODistillLoss, DINOStudentAdaptor,
                                  load_teacher, make_views)

        torch.manual_seed(0)
        m = build_model(str(ROOT / "configs/jingwei_p.yaml"))
        teacher = load_teacher("synthetic", device="cpu", synthetic_dim=64)
        adaptor = DINOStudentAdaptor(m, teacher.spec.embed_dim)

        # stage 0: only backbone / neck / patch / target_proj are trainable
        for p in m.parameters():
            p.requires_grad_(False)
        for mod in (m.backbone, m.neck, m.normality.patch,
                    m.normality.recon.target_proj):
            for p in mod.parameters():
                p.requires_grad_(True)
        self.assertFalse(any(p.requires_grad for p in m.head.parameters()))

        x = torch.rand(2, 3, 448, 448)
        tv, sv = make_views(x, 224, strong=True)
        tmap = teacher.patch_map(tv)
        self.assertEqual(tuple(tmap.shape), (2, 64, 16, 16))  # 224 / 14

        bb = m.backbone(sv)
        feats = m.neck(bb)
        q = m.normality.patch(feats)
        rec_target = m.normality.recon.target_proj(feats["n5"])
        smaps = adaptor(feats, q, rec_target)
        loss = DINODistillLoss()(smaps, tmap)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all()
                           for p in m.normality.recon.target_proj.parameters()))
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all()
                           for p in adaptor.parameters()))

    def test_21_dino_stage0_cli_smoke(self):
        """tools/pretrain_dino.py runs with no data / no download and saves a ckpt."""
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "dino"
            proc = subprocess.run(
                [sys.executable, "tools/pretrain_dino.py", "--smoke", "2",
                 "--iters", "1", "--batch", "1", "--cfg", "configs/jingwei_p.yaml",
                 "--out", str(out)],
                cwd=ROOT, capture_output=True, text=True, timeout=180)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertTrue((out / "dino_init.pt").is_file())
            state = torch.load(out / "dino_init.pt", map_location="cpu")
            self.assertEqual(state["teacher"], "synthetic")
            self.assertIn("model", state)

    # ------------------------------------------------ foreground / edge fold-in
    def test_22_foreground_context_estimation(self):
        """Per-frame cloth mask follows 布幅; edge band marks normal fold-in."""
        from jingwei.foreground import estimate_fabric_context

        G, H, W = 160, 1280, 1280
        yy, xx = np.mgrid[0:H, 0:W]
        weave = (np.sin(2 * np.pi * xx / 7) * np.sin(2 * np.pi * yy / 11)) * 0.25 + 0.5
        img = np.ones((H, W, 3), np.float32) * 0.5
        fabric = np.zeros((H, W), np.float32)
        fabric[:, 200:1080] = 1.0
        img = img * (1 - fabric[..., None]) + (0.45 + 0.2 * weave[..., None]) * fabric[..., None]
        img[:, 205:250] *= 0.35                       # normal fold-in shadow
        x = torch.from_numpy(img.transpose(2, 0, 1)).float()[None]
        fm, eb = estimate_fabric_context(x, G, band_cells=8.0)  # 64px @1280/160
        fm, eb = fm[0, 0].numpy(), eb[0, 0].numpy()
        self.assertGreater(fm[:, 100:140].mean(), 0.5, "cloth centre is fabric")
        self.assertLess(fm[:, :20].mean(), 0.2, "outside web is background")
        self.assertGreater(eb[:, 25:40].mean(), 0.15, "fold-in band detected")
        self.assertLess(eb[:, 70:90].mean(), 0.05, "cloth centre is not edge")

    def test_23_safety_net_rejects_fold_keeps_real_defect(self):
        """Fold-like edge component is suppressed; centre/foreground defects survive."""
        G = 160
        cfg = load_config(CFG_N)
        ctx = np.zeros((3, G, G), np.float32)
        ctx[1, :, 20:140] = 1.0            # fabric_mask
        ctx[2, :, 20:30] = 0.8            # left edge_band
        ctx[2, :, 130:140] = 0.8          # right edge_band

        empty_b = np.zeros((0, 4), np.float32)
        empty_s = np.zeros(0, np.float32)
        empty_l = np.zeros(0, np.int64)

        # long thin strip at the left selvedge, low defect_fg, moderate peak
        a_fold = np.zeros((G, G), np.float32)
        a_fold[20:140, 22:28] = 0.70
        out_fold = safety_net_numpy(a_fold, empty_b, empty_s, empty_l,
                                    (1280, 1280), cfg, 19, context=ctx)
        self.assertEqual(out_fold["n_safety_boxes"], 0,
                         "normal fold-in must NOT become 未知异常")

        # large central blob: not edge, not fold-shaped
        a_def = np.zeros((G, G), np.float32)
        a_def[50:100, 60:110] = 0.90
        out_def = safety_net_numpy(a_def, empty_b, empty_s, empty_l,
                                   (1280, 1280), cfg, 19, context=ctx)
        self.assertEqual(out_def["n_safety_boxes"], 1,
                         "large centre defect must still be emitted")

        # real edge defect: learned foreground overrides fold suppression
        ctx2 = ctx.copy()
        ctx2[0, 20:140, 22:28] = 0.9
        out_edge = safety_net_numpy(a_fold, empty_b, empty_s, empty_l,
                                    (1280, 1280), cfg, 19, context=ctx2)
        self.assertEqual(out_edge["n_safety_boxes"], 1,
                         "high defect_fg in edge band must NOT be suppressed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
