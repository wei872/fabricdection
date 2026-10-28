"""Verification suite for the JingWei reference implementation.

Run from the repo root::

    python -m unittest discover -s tests -v

No pytest / no GPU / no real data needed. It checks shapes at the required
1280x1280 input, trainability, the deterministic safety net, fabric-type
invariance, memory-bank algebra, the dataset readers, Chinese / Unicode
file paths, non-square fabric augmentation, training progress logs,
full checkpoint resume, numpy RNG state checkpointing, dynamic
local+global attention foreground context, validation metrics, YOLO-style
image cache, unknown-defect boxes, and CUDA label / index guards (36 tests).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))   # import tools/_common.py helpers

from jingwei import build_model, load_config                       # noqa: E402
from jingwei.data import collate, to_model_input                    # noqa: E402
from jingwei.fabric import (fabric_descriptor, fit_fabric_prototypes,  # noqa: E402
                            greedy_coreset, spherical_kmeans)
from jingwei.fusion import safety_net_numpy                       # noqa: E402
from jingwei.image_io import (VideoCaptureUnicode, ascii_safe,     # noqa: E402
                              console_safe, imread_unicode,
                              imwrite_unicode)
from jingwei.layers import LocalContrastNorm                     # noqa: E402
from jingwei.losses import (build_dense_targets, hungarian_match,  # noqa: E402
                            resolve_label_offset,
                            sanitize_detection_targets)
from jingwei.utils import (count_parameters, profile_macs,         # noqa: E402
                           torch_load)

CFG_N = str(ROOT / "configs/jingwei_n.yaml")
CFG_P = str(ROOT / "configs/jingwei_p.yaml")


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
        from jingwei.data import NormalImageDataset, YOLODefectDataset, collate
        with tempfile.TemporaryDirectory() as tmp:
            imgs = Path(tmp) / "images" / "train"
            labs = Path(tmp) / "labels" / "train"
            normal = Path(tmp) / "normal" / "images" / "train"
            for d in (imgs, labs, normal):
                d.mkdir(parents=True, exist_ok=True)
            # ASCII, uppercase extension, and Chinese filenames / label stems.
            names = ["0.jpg", "1.JPG", "中文_布边折入.png"]
            for i, name in enumerate(names):
                imwrite_unicode(imgs / name, weave(9 + i, 128, 0.2, 128))
                (labs / f"{Path(name).stem}.txt").write_text(
                    "1 0.25 0.30 0.10 0.08\n17 0.7 0.6 0.2 0.1\n", encoding="utf-8")
                imwrite_unicode(normal / f"n{i}.jpg", weave(7, 140, 0.4, 128))
            imwrite_unicode(normal / "正常图.WEBP", weave(7, 140, 0.4, 128))
            ds = YOLODefectDataset(imgs, labs, img_size=128, augment=False)
            self.assertEqual(len(ds), 3)
            item = ds[0]
            # datasets hand uint8 HWC to the loader; the float CHW cast happens
            # once per batch on the GPU (jingwei.data.to_model_input)
            self.assertEqual(item["image"].dtype, torch.uint8)
            self.assertEqual(tuple(item["image"].shape), (128, 128, 3))
            self.assertEqual(tuple(to_model_input(item["image"]).shape),
                             (1, 3, 128, 128))
            self.assertEqual(tuple(item["boxes"].shape), (2, 4))
            self.assertEqual(tuple(item["labels"].shape), (2,))
            nd = NormalImageDataset(normal, img_size=128, augment=False,
                                    with_descriptor=True)
            self.assertEqual(len(nd), 4)
            self.assertEqual(tuple(nd[0]["descriptor"].shape), (8,))
            batch = collate([ds[0], ds[1]])
            self.assertEqual(tuple(batch["image"].shape), (2, 128, 128, 3))
            self.assertEqual(batch["image"].dtype, torch.uint8)
            x = to_model_input(batch["image"])
            self.assertEqual(tuple(x.shape), (2, 3, 128, 128))
            self.assertEqual(x.dtype, torch.float32)
            self.assertLessEqual(float(x.max()), 1.0)
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
            state = torch_load(out / "dino_init.pt", map_location="cpu")
            self.assertEqual(state["teacher"], "synthetic")
            self.assertIn("model", state)

    # ------------------------------------------------ foreground / edge fold-in
    def test_22_dynamic_foreground_context_targets(self):
        """Dynamic fabric/fold targets follow moving 布幅, not fixed image edges."""
        from jingwei.foreground import build_fabric_context_targets

        G, size = 128, 320

        def make(shift, fold=False):
            yy, xx = np.mgrid[0:size, 0:size]
            weave = ((np.sin(2 * np.pi * xx / 7) *
                      np.sin(2 * np.pi * yy / 11)) * 0.25 + 0.5)
            bg = np.ones((size, size, 3), np.float32) * 0.5
            cloth_img = (0.35 + 0.3 * weave[..., None]).astype(np.float32)
            fabric = np.zeros((size, size), np.float32)
            fabric[:, shift:shift + 180] = 1.0
            img = (bg * (1 - fabric[..., None]) +
                   cloth_img * fabric[..., None])
            if fold:
                img[:, shift:shift + 20] *= 0.4     # normal fold-in shadow
            img = np.clip(img, 0.0, 1.0)
            return torch.from_numpy(img.transpose(2, 0, 1)).float()[None]

        fm1, fb1, g1 = build_fabric_context_targets(make(30), G)
        fm2, fb2, g2 = build_fabric_context_targets(make(120, fold=True), G)
        fm1, fb1, fm2, fb2 = (t[0, 0].numpy()
                              for t in (fm1, fb1, fm2, fb2))
        self.assertEqual(fm1.shape, (G, G))
        self.assertEqual(fb1.shape, (G, G))
        self.assertEqual(fm2.shape, (G, G))
        self.assertEqual(fb2.shape, (G, G))

        c1a, c1b = 30 * G // size, (30 + 180) * G // size
        c2a, c2b = 120 * G // size, (120 + 180) * G // size
        self.assertGreater(fm1[:, c1a + 10:c1b - 10].mean(),
                           fm1[:, :max(1, c1a - 5)].mean(),
                           "cloth region must be fabric in shifted image 1")
        self.assertGreater(fm2[:, c2a + 20:c2b - 10].mean(),
                           fm2[:, :max(1, c2a - 5)].mean(),
                           "cloth region must be fabric in shifted image 2")
        self.assertFalse(np.allclose(fm1, fm2, atol=0.1),
                         "fabric target must move with 布幅, not be static")

        # fold_band peak must be near a learned cloth boundary (c2a), and
        # must NOT be pinned to the left image border.
        peak2 = int(fb2.mean(axis=0).argmax())
        self.assertLess(abs(peak2 - c2a), 10,
                        "dynamic fold band should follow shifted cloth edge")
        self.assertGreater(fb2[:, max(0, peak2 - 2):peak2 + 3].mean(),
                           fb2[:, :5].mean(),
                           "fold band at cloth boundary > fixed image border")
        peak1 = int(fb1.mean(axis=0).argmax())
        self.assertGreater(fb1[:, max(0, peak1 - 2):peak1 + 3].mean(),
                           fb1[:, :5].mean(),
                           "fold band follows cloth edge in image 1 too")

        self.assertNotAlmostEqual(float(g1), float(g2), places=3,
                                  msg="global cloth prior changes with 布幅")
        self.assertGreaterEqual(float(g1), 0.0)
        self.assertLessEqual(float(g2), 1.0)

    def test_23_safety_net_rejects_fold_keeps_real_defect(self):
        """Fold-like edge component is suppressed; centre/foreground defects survive."""
        G = 160
        cfg = load_config(CFG_N)
        ctx = np.zeros((3, G, G), np.float32)
        ctx[1, :, 20:140] = 1.0            # fabric_mask
        ctx[2, :, 20:30] = 0.8            # learned dynamic left fold_band
        ctx[2, :, 130:140] = 0.8          # learned dynamic right fold_band

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

    # ------------------------------------------------------- Chinese / Unicode paths
    def test_24_unicode_paths(self):
        """Chinese image / folder / video / prototype names must not error."""
        from jingwei.data import list_images, read_image

        # ASCII-only helpers -------------------------------------------------
        self.assertEqual(ascii_safe("油点线", "cls3"), "cls3")
        self.assertEqual(ascii_safe("ABC 123", "cls3"), "ABC 123")
        escaped = console_safe("中文_路径")
        self.assertNotIn("中", escaped)
        self.assertIn("\\u", escaped)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "布匹数据_中文目录"
            root.mkdir(parents=True, exist_ok=True)
            img_cn = root / "中文_测试图.jpg"
            img_up = root / "大写扩展.PNG"
            self.assertTrue(imwrite_unicode(img_cn, weave(9, 128, 0.2, 128)))
            self.assertTrue(imwrite_unicode(img_up, weave(11, 128, 0.2, 128)))
            self.assertIsNone(imread_unicode(root / "不存在.jpg"))

            gray = imread_unicode(img_cn, cv2.IMREAD_GRAYSCALE)
            self.assertIsNotNone(gray)
            self.assertEqual(gray.ndim, 2)
            self.assertEqual(fabric_descriptor(gray).shape, (8,))
            rgb = read_image(img_cn)
            self.assertEqual(rgb.shape[:2], (128, 128))
            self.assertEqual(rgb.shape[2], 3)

            found = list_images(root)
            self.assertEqual(len(found), 2)
            self.assertTrue(any("中文_测试图.jpg" in f for f in found))
            self.assertTrue(any("大写扩展.PNG" in f for f in found))

            # tools/_common.descriptors_for_folder reads Unicode paths -------
            tools_dir = str(ROOT / "tools")
            sys.path.insert(0, tools_dir)
            try:
                import _common
                files, descs, proto, labels = _common.descriptors_for_folder(root, 2)
            finally:
                sys.path.remove(tools_dir)
            self.assertEqual(len(files), 2)
            self.assertEqual(descs.shape, (2, 8))
            self.assertEqual(labels.shape, (2,))

            # Chinese-named .npz prototype file -----------------------------
            proto_path = root / "布种原型.npz"
            np.savez(proto_path,
                     centers=np.zeros((2, 8), np.float32),
                     mean=np.zeros(8, np.float32),
                     std=np.ones(8, np.float32))
            self.assertTrue(proto_path.exists())

            # Chinese-named video + VideoCaptureUnicode ----------------------
            ascii_video = root / "_tmp_ascii.avi"
            writer = cv2.VideoWriter(str(ascii_video),
                                     cv2.VideoWriter_fourcc(*"MJPG"), 5, (64, 64))
            self.assertTrue(writer.isOpened())
            writer.write(np.full((64, 64, 3), 110, np.uint8))
            writer.release()
            video_cn = root / "布匹视频_测试.avi"
            shutil.copyfile(ascii_video, video_cn)
            with VideoCaptureUnicode(video_cn) as cap:
                self.assertTrue(cap.isOpened())
                ok, frame = cap.read()
                self.assertTrue(ok)
                self.assertEqual(frame.shape[:2], (64, 64))
                tmp_copy = cap.temp_path
            self.assertIsNotNone(tmp_copy)
            self.assertFalse(os.path.exists(tmp_copy))

            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "ascii:backslashreplace"

            # predict.py on a Chinese image, writing a Chinese output name ---
            out_img = root / "中文输出目录"
            cmd_img = [sys.executable, "tools/predict.py",
                       "--cfg", CFG_P, "--source", str(img_cn),
                       "--out", str(out_img), "--device", "cpu",
                       "--prototypes", str(proto_path)]
            proc = subprocess.run(cmd_img, cwd=str(ROOT), env=env,
                                  capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=180)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertTrue((out_img / img_cn.name).exists())
            res = json.loads((out_img / "results.json").read_text(encoding="utf-8"))
            self.assertIn(img_cn.name, res)
            self.assertIn("\\u", proc.stdout)       # console_safe escaped it
            self.assertNotIn("中文", proc.stdout)

            # predict.py on a Chinese video path -----------------------------
            out_vid = root / "视频输出"
            cmd_vid = [sys.executable, "tools/predict.py",
                       "--cfg", CFG_P, "--source", str(video_cn),
                       "--out", str(out_vid), "--device", "cpu",
                       "--prototypes", str(proto_path), "--no-draw"]
            proc_v = subprocess.run(cmd_vid, cwd=str(ROOT), env=env,
                                    capture_output=True, text=True,
                                    encoding="utf-8", errors="replace", timeout=180)
            self.assertEqual(proc_v.returncode, 0, proc_v.stdout + proc_v.stderr)
            self.assertTrue((out_vid / "_frames" / "000000.jpg").exists())
            res_v = json.loads((out_vid / "results.json").read_text(encoding="utf-8"))
            self.assertIn("000000.jpg", res_v)
            self.assertNotIn("布匹视频", proc_v.stdout)

    # --------------------------------------- non-square fabric augmentation
    def test_25_non_square_photometric_augment(self):
        """Reflectance field must broadcast on long non-square RGB frames."""
        from jingwei.data import (NormalImageDataset, YOLODefectDataset,
                                  fabric_photometric, geometric_augment)

        class ZeroRng:
            """Deterministically enable every augmentation branch."""
            def uniform(self, a, b):
                return (a + b) / 2.0
            def random(self):
                return 0.0
            def normal(self, loc, scale, size):
                return np.zeros(size, np.float32)
            def integers(self, a, b):
                return a

        # 1040x2448-like aspect ratio, scaled down for CI speed.
        img = np.full((104, 244, 3), 128, np.uint8)
        out = fabric_photometric(img, ZeroRng())
        self.assertEqual(out.shape, img.shape)
        self.assertEqual(out.dtype, np.uint8)

        boxes = np.array([[10., 10., 50., 40.],
                          [100., 50., 200., 90.]], np.float32)
        out2, boxes2 = geometric_augment(out, boxes.copy(), ZeroRng(), 128)
        self.assertEqual(out2.shape, (128, 128, 3))
        self.assertEqual(boxes2.shape, boxes.shape)
        self.assertTrue(np.isfinite(boxes2).all())
        self.assertTrue((boxes2 >= 0).all() and (boxes2 <= 128).all())

        # Full dataset path with augment=True and a seed that enables the
        # reflectance branch on a non-square image.
        with tempfile.TemporaryDirectory() as tmp:
            imgs = Path(tmp) / "images" / "train"
            labs = Path(tmp) / "labels" / "train"
            normal = Path(tmp) / "normal" / "images" / "train"
            for d in (imgs, labs, normal):
                d.mkdir(parents=True, exist_ok=True)
            gray = weave(9, 128, 0.2, 128)[:, :96]     # 128 x 96, non-square
            imwrite_unicode(imgs / "长条布幅.jpg", gray)
            (labs / "长条布幅.txt").write_text(
                "1 0.25 0.30 0.10 0.08\n", encoding="utf-8")
            imwrite_unicode(normal / "正常长条.jpg", gray)
            ds = YOLODefectDataset(imgs, labs, img_size=64,
                                   augment=True, seed=2)
            item = ds[0]
            self.assertEqual(tuple(item["image"].shape), (64, 64, 3))
            nd = NormalImageDataset(normal, img_size=64, augment=True, seed=2,
                                    with_descriptor=True)
            item2 = nd[0]
            self.assertEqual(tuple(item2["image"].shape), (64, 64, 3))
            self.assertEqual(tuple(item2["descriptor"].shape), (8,))

    # ---------------------------------------------------------- training logs
    def test_26_training_progress_logs(self):
        """Stage 0 / A / B must print startup + per-iteration progress."""
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "ascii:backslashreplace"

        def run(cmd):
            proc = subprocess.run(cmd, cwd=str(ROOT), env=env,
                                  capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=180)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            return proc.stdout

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)

            det = run([sys.executable, "tools/train_detector.py",
                       "--cfg", CFG_P, "--smoke", "1", "--epochs", "1",
                       "--batch", "1", "--device", "cpu", "--num-workers", "0",
                       "--log-interval", "1", "--out", str(out / "det")])
            for token in ("JingWei stage A", "batches/epoch", "log_interval",
                          "epoch   1/1  iter", "eta=", "checkpoint ->", "saved ->"):
                self.assertIn(token, det)

            norm = run([sys.executable, "tools/train_normality.py",
                        "--cfg", CFG_P, "--smoke", "1", "--iters", "1",
                        "--batch", "1", "--device", "cpu",
                        "--log-interval", "1", "--out", str(out / "norm")])
            for token in ("JingWei stage B", "iter      1/1", "eta=", "saved ->"):
                self.assertIn(token, norm)

            dino = run([sys.executable, "tools/pretrain_dino.py",
                        "--cfg", CFG_P, "--smoke", "1", "--iters", "1",
                        "--batch", "1", "--log-interval", "1",
                        "--out", str(out / "dino")])
            for token in ("JingWei stage 0", "iter      1/1", "eta=", "next:"):
                self.assertIn(token, dino)

    # ------------------------------------------------------------- resume training
    def test_27_resume_training(self):
        """Full checkpoints restore optimizer / scheduler / RNG in stages 0/A/B."""
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "ascii:backslashreplace"

        def run(cmd):
            proc = subprocess.run(cmd, cwd=str(ROOT), env=env,
                                  capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=180)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            return proc.stdout

        def common(*extra):
            return [sys.executable, *extra, "--cfg", CFG_P, "--device", "cpu",
                    "--seed", "7", "--log-interval", "0"]

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)

            # ---- stage A: 1 epoch -> full checkpoint -> resume to 2 epochs ----
            run(common("tools/train_detector.py", "--smoke", "1", "--epochs", "1",
                       "--batch", "1", "--num-workers", "0", "--save-interval", "1",
                       "--out", str(out / "det_a")))
            det_ckpt = out / "det_a" / "last.pt"
            self.assertTrue(det_ckpt.is_file())
            self.assertTrue((out / "det_a" / "epoch_0001.pt").is_file())
            s = torch_load(det_ckpt, map_location="cpu")
            self.assertIn("optimizer", s)
            self.assertIn("scheduler", s)
            self.assertIn("rng", s)
            self.assertEqual(s["train"]["stage"], "detector")
            self.assertEqual(s["train"]["epoch"], 1)

            det2 = run(common("tools/train_detector.py", "--smoke", "1",
                              "--epochs", "2", "--batch", "1", "--num-workers", "0",
                              "--resume", str(det_ckpt),
                              "--out", str(out / "det_b")))
            self.assertIn("mode=full", det2)
            self.assertIn("resumed full checkpoint", det2)
            self.assertIn("start_epoch=2/2", det2)
            self.assertIn("epoch   2/2", det2)
            self.assertNotIn("epoch   1/2", det2)
            s2 = torch_load(out / "det_b" / "last.pt", map_location="cpu")
            self.assertEqual(s2["train"]["epoch"], 2)

            # ---- stage B: 1 iter -> full checkpoint -> resume to 2 iters ----
            run(common("tools/train_normality.py", "--smoke", "1", "--iters", "1",
                       "--batch", "1", "--save-interval", "1",
                       "--out", str(out / "norm_a")))
            norm_ckpt = out / "norm_a" / "last.pt"
            self.assertTrue(norm_ckpt.is_file())
            self.assertTrue((out / "norm_a" / "normality.pt").is_file())
            s = torch_load(norm_ckpt, map_location="cpu")
            self.assertIn("optimizer", s)
            self.assertEqual(s["train"]["stage"], "normality")
            self.assertEqual(s["train"]["iter"], 1)

            norm2 = run(common("tools/train_normality.py", "--smoke", "1",
                               "--iters", "2", "--batch", "1",
                               "--resume", str(norm_ckpt),
                               "--out", str(out / "norm_b")))
            self.assertIn("mode=full", norm2)
            self.assertIn("start_iter=1/2", norm2)
            self.assertIn("resumed full checkpoint", norm2)
            s2 = torch_load(out / "norm_b" / "last.pt", map_location="cpu")
            self.assertEqual(s2["train"]["iter"], 2)

            # ---- stage 0: 1 iter -> full checkpoint -> resume to 2 iters ----
            run(common("tools/pretrain_dino.py", "--smoke", "1", "--iters", "1",
                       "--batch", "1", "--save-interval", "1",
                       "--out", str(out / "dino_a")))
            dino_ckpt = out / "dino_a" / "last.pt"
            self.assertTrue(dino_ckpt.is_file())
            self.assertTrue((out / "dino_a" / "dino_init.pt").is_file())
            s = torch_load(dino_ckpt, map_location="cpu")
            self.assertIn("optimizer", s)
            self.assertIn("dino_adaptor", s)
            self.assertIn("teacher_model", s)
            self.assertEqual(s["train"]["stage"], "dino")
            self.assertEqual(s["train"]["iter"], 1)

            dino2 = run(common("tools/pretrain_dino.py", "--smoke", "1",
                               "--iters", "2", "--batch", "1",
                               "--resume", str(dino_ckpt),
                               "--out", str(out / "dino_b")))
            self.assertIn("mode=full", dino2)
            self.assertIn("start_iter=1/2", dino2)
            self.assertIn("resumed full checkpoint", dino2)
            s2 = torch_load(out / "dino_b" / "last.pt", map_location="cpu")
            self.assertEqual(s2["train"]["iter"], 2)
            self.assertTrue((out / "dino_b" / "dino_init.pt").is_file())

            # Stored iters are reused when --iters is omitted; already done.
            done = run(common("tools/pretrain_dino.py", "--smoke", "1",
                              "--batch", "1", "--resume",
                              str(out / "dino_b" / "last.pt"),
                              "--out", str(out / "dino_done")))
            self.assertIn("nothing to do", done)
            self.assertTrue((out / "dino_done" / "dino_init.pt").is_file())

            # Weight-only warm start (dino_init.pt) must still work.
            warm = run(common("tools/train_detector.py", "--smoke", "1",
                              "--epochs", "1", "--batch", "1",
                              "--num-workers", "0", "--resume",
                              str(out / "dino_b" / "dino_init.pt"),
                              "--out", str(out / "warm")))
            self.assertIn("mode=weights-only", warm)
            self.assertIn("starting from epoch 1", warm)

    # ------------------------------------------------------- numpy RNG state
    def test_28_numpy_rng_state_checkpoint(self):
        """BitGenerator state must survive weights_only=True on Windows / numpy 2.x.

        Regression for:
          AttributeError: 'dict' object has no attribute 'tolist'
        PCG64/PCG64DXSM expose ``state`` as a dict; Philox/SFC64/MT19937 use
        ndarray fields.  The checkpoint must contain no raw numpy arrays.
        """
        from _common import (_generator_state, _restore_generator_state,
                             collect_rng_state, restore_rng_state,
                             save_checkpoint)
        from jingwei.data import NormalImageDataset

        for name in ("PCG64", "PCG64DXSM", "Philox", "SFC64", "MT19937"):
            cls = getattr(np.random, name, None)
            if cls is None:
                continue
            rng = np.random.Generator(cls())
            for _ in range(5):
                rng.random()
            st = _generator_state(rng)
            self.assertEqual(st.get("__jingwei_rng_format__"), 2)
            expected = [rng.random() for _ in range(4)]
            with tempfile.TemporaryDirectory() as d:
                p = Path(d) / "rng.pt"
                save_checkpoint({"rng": st}, p)
                loaded = torch_load(p, map_location="cpu")["rng"]
            _restore_generator_state(rng, loaded)
            self.assertEqual([rng.random() for _ in range(4)], expected,
                             f"state mismatch for {name}")

        # Real dataset object, not the smoke dataset (which has no .rng).
        with tempfile.TemporaryDirectory() as d:
            img_dir = Path(d) / "normaltrain" / "images"
            img_dir.mkdir(parents=True)
            for i in range(3):
                imwrite_unicode(str(img_dir / f"img_{i}.jpg"),
                                np.full((64, 96, 3), 40 + i * 30,
                                        dtype=np.uint8))
            ds = NormalImageDataset(img_dir, img_size=64, augment=True,
                                    seed=11, with_descriptor=False)
            self.assertTrue(hasattr(ds, "rng"))
            for _ in range(3):
                ds.rng.random()
            state = collect_rng_state(ds)
            self.assertIn("numpy", state)
            with tempfile.TemporaryDirectory() as d2:
                p = Path(d2) / "ckpt.pt"
                save_checkpoint({"rng": state}, p)
                loaded = torch_load(p, map_location="cpu")["rng"]
            expected = [ds.rng.random() for _ in range(5)]
            restore_rng_state(loaded, ds)
            self.assertEqual([ds.rng.random() for _ in range(5)], expected)

    # ------------------------------------------- local + global attention fg context
    def test_29_local_global_attention_foreground(self):
        """Learned fg context has local+global branches, dynamic FiLM and loss."""
        from jingwei.foreground import FabricContext
        from jingwei.normality import LinearAttention2

        torch.manual_seed(0)
        cfg = load_config(CFG_P)
        self.assertEqual(str(cfg.foreground.mode), "local_global_attention")
        ctx = FabricContext(cfg, channels=(16, 32, 64, 128))
        G = int(cfg.normality.out_grid)
        feats = {"n3": torch.randn(1, 32, 160, 160),
                 "n5": torch.randn(1, 128, 40, 40)}

        # global branch really is LinearAttention2 (Dinomaly ELU+1).
        self.assertIsInstance(ctx.attention.blocks[0].attn, LinearAttention2)
        self.assertEqual(len(ctx.attention.blocks),
                         int(cfg.foreground.attention_blocks))
        self.assertEqual(tuple(ctx.attention.cls_token.shape),
                         (1, 1, int(cfg.foreground.attention_dim)))

        parts = ctx.forward_parts(feats)
        self.assertEqual(tuple(parts["context"].shape), (1, 3, G, G))
        self.assertEqual(tuple(parts["defect_fg"].shape), (1, 1, G, G))
        self.assertEqual(tuple(parts["fabric_mask"].shape), (1, 1, G, G))
        self.assertEqual(tuple(parts["fold_band"].shape), (1, 1, G, G))
        self.assertEqual(tuple(parts["global_logit"].shape), (1, 1))
        self.assertTrue(torch.equal(parts["context"][:, 2:3],
                                    parts["fold_band"]))
        self.assertTrue(torch.isfinite(parts["context"]).all())

        # Inference context depends on FEATURES, not on raw RGB / fixed borders:
        # same feats + different image -> identical context.
        x1 = torch.rand(1, 3, 1280, 1280)
        x2 = torch.rand(1, 3, 1280, 1280)
        anomaly = torch.rand(1, 1, G, G)
        eff1, c1, _ = ctx.forward(x1, feats, anomaly)
        _, c2, _ = ctx.forward(x2, feats, anomaly)
        self.assertTrue(torch.allclose(c1, c2))
        self.assertEqual(tuple(eff1.shape), (1, 1, G, G))
        self.assertLessEqual(float(eff1.max()), float(anomaly.max()) + 1e-6)
        feats2 = {"n3": torch.randn(1, 32, 160, 160) * 0.5 + 1.0,
                  "n5": torch.randn(1, 128, 40, 40) * 0.5 - 1.0}
        c3 = ctx.forward_parts(feats2)["context"]
        self.assertFalse(torch.allclose(c1, c3, atol=1e-3),
                         "context must respond to local/global feature changes")

        # Training loss is finite and reaches BOTH local and global parameters.
        tgt_fg = torch.zeros(1, 1, G, G)
        tgt_fg[..., 40:80, 40:80] = 1.0
        tgt_fabric, tgt_fold, tgt_global = ctx.build_targets(x1, tgt_fg)
        loss, loss_parts = ctx.loss(parts, tgt_fg, tgt_fabric,
                                    tgt_fold, tgt_global)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(loss_parts),
                         {"defect", "fabric", "fold", "global"})
        loss.backward()
        named = dict(ctx.named_parameters())
        for key in ("attention.blocks.0.attn.qkv.weight",
                    "attention.cls_token",
                    "attention.local.0.weight",
                    "attention.fg_head.weight"):
            self.assertIsNotNone(named[key].grad, key)
            self.assertTrue(torch.isfinite(named[key].grad).all(), key)

    # ------------------------------------------------- validation metrics in training
    def test_30_validation_metrics_and_best_checkpoint(self):
        """Stage A must print val precision/recall/mAP/large recall and save best.pt."""
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "ascii:backslashreplace"
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "det"
            cmd = [sys.executable, "tools/train_detector.py",
                   "--cfg", CFG_P, "--smoke", "2", "--epochs", "1",
                   "--batch", "1", "--num-workers", "0", "--device", "cpu",
                   "--log-interval", "0", "--val-interval", "1",
                   "--val-max", "2", "--normal-max", "2", "--cache", "none",
                   "--out", str(out)]
            proc = subprocess.run(cmd, cwd=str(ROOT), env=env,
                                  capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=240)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            text = proc.stdout
            for token in ("val   epoch", "mAP50=", "precision=", "recall=",
                          "f1=", "large_recall=", "normal_fpr=",
                          "unknown_boxes=", "best   ->"):
                self.assertIn(token, text)
            self.assertTrue((out / "last.pt").is_file())
            self.assertTrue((out / "best.pt").is_file())

    # ------------------------------------------------------- YOLO-style image cache
    def test_31_yolo_style_image_cache(self):
        """disk/ram/bytes caches reproduce the original dataset + Unicode paths."""
        from jingwei.cache import (build_dataset_cache, cache_paths,
                                   cache_prefix, load_cached_dataset)
        from jingwei.data import YOLODefectDataset, infer_labels_dir

        size = 64
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img_dir = root / "布幅图片" / "images"
            lab_dir = root / "布幅图片" / "labels"
            img_dir.mkdir(parents=True)
            lab_dir.mkdir(parents=True)
            for i in range(3):
                img = np.full((80, 120, 3), 40 + i * 40, np.uint8)
                cv2.putText(img, str(i), (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (200, 200, 200), 2)
                self.assertTrue(imwrite_unicode(img_dir / f"布匹_{i}.jpg", img))
                (lab_dir / f"布匹_{i}.txt").write_text(
                    f"{i % 2} 0.5 0.5 0.4 0.3\n", encoding="utf-8")

            inferred = infer_labels_dir(img_dir)
            self.assertTrue(inferred.endswith("labels"), inferred)
            orig = YOLODefectDataset(img_dir, inferred, img_size=size,
                                     augment=False, seed=0)
            self.assertEqual(len(orig[0]["labels"]), 1)

            for mode in ("disk", "ram", "bytes"):
                prefix = cache_prefix(root / "cache", "tag", "train",
                                      size, mode)
                meta = build_dataset_cache(
                    img_dir, inferred, size, prefix, mode, True,
                    log_fn=lambda m: None)
                self.assertEqual(meta["n"], 3)
                ds = load_cached_dataset(prefix, mode, False, 0)
                o, c = orig[1], ds[1]
                self.assertEqual(tuple(c["image"].shape),
                                 tuple(o["image"].shape))
                self.assertEqual(c["image"].dtype, torch.uint8)
                self.assertLess(float((c["image"].float() -
                                       o["image"].float()).abs().max()),
                                1e-6, mode)
                self.assertTrue(torch.allclose(c["boxes"], o["boxes"]), mode)
                self.assertTrue(torch.equal(c["labels"], o["labels"]), mode)
                self.assertIn("布匹_1", c["path"])
                meta2 = build_dataset_cache(
                    img_dir, inferred, size, prefix, mode, True,
                    log_fn=lambda m: None)
                self.assertEqual(meta["signature"], meta2["signature"])
                mp, dp = cache_paths(prefix, mode)
                self.assertTrue(mp.is_file() and dp.is_file())

            # CLI pre-build like YOLO's train.cache, with a Unicode dataset path
            import yaml
            data_yaml = root / "data.yaml"
            data_yaml.write_text(yaml.safe_dump({
                "train": str(root / "train" / "images"),
                "val": str(root / "val" / "images"),
                "normaltrain": str(root / "normaltrain" / "images"),
                "normalval": str(root / "normalval" / "images"),
            }, allow_unicode=True), encoding="utf-8")
            for split in ("train", "val", "normaltrain", "normalval"):
                d = root / split / "images"
                d.mkdir(parents=True)
                (root / split / "labels").mkdir(parents=True, exist_ok=True)
                imwrite_unicode(d / "图_0.jpg",
                                np.full((48, 64, 3), 90, np.uint8))
                if split in ("train", "val"):
                    (root / split / "labels" / "图_0.txt").write_text(
                        "1 0.5 0.5 0.3 0.3\n", encoding="utf-8")
            cfg_yaml = root / "tiny.yaml"
            cfg_yaml.write_text("name: tiny\ninput_size: [64, 64]\n"
                                "data:\n  dataset_yaml: " +
                                str(data_yaml).replace("\\", "/") +
                                "\n  nc: 19\n  names: [a, b]\n",
                                encoding="utf-8")
            cache_dir = root / "cli_cache"
            cmd = [sys.executable, "tools/cache_dataset.py",
                   "--cfg", str(cfg_yaml), "--data", str(data_yaml),
                   "--splits", "train,val,normaltrain,normalval",
                   "--mode", "bytes", "--cache-dir", str(cache_dir),
                   "--img-size", "64"]
            proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True,
                                  text=True, encoding="utf-8",
                                  errors="replace", timeout=120)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertIn("cache ready", proc.stdout)
            self.assertTrue(any(cache_dir.glob("train_64_bytes_*.bin")))
            self.assertTrue(any(cache_dir.glob("val_64_bytes_*.json")))

    # --------------------------------------------------- unknown defect type boxes
    def test_32_unknown_defect_boxes(self):
        """Small anomalous components also emit 未知异常 boxes, large ones stay."""
        G = 160
        cfg = load_config(CFG_N)
        self.assertTrue(bool(cfg.fusion.emit_unknown_boxes))
        ctx = np.zeros((3, G, G), np.float32)
        ctx[1] = 1.0                       # all fabric, no fold band
        empty_b = np.zeros((0, 4), np.float32)
        empty_s = np.zeros(0, np.float32)
        empty_l = np.zeros(0, np.int64)

        small = np.zeros((G, G), np.float32)
        small[80:83, 80:82] = 0.90        # 6 cells < large_min, >= unknown_min
        out = safety_net_numpy(small, empty_b, empty_s, empty_l,
                               (1280, 1280), cfg, 19, context=ctx)
        self.assertEqual(out["n_safety_boxes"], 1)
        self.assertEqual(out["n_unknown_boxes"], 1)
        self.assertIn(19, out["labels"].tolist())

        cfg.fusion.emit_unknown_boxes = False
        off = safety_net_numpy(small, empty_b, empty_s, empty_l,
                               (1280, 1280), cfg, 19, context=ctx)
        self.assertEqual(off["n_safety_boxes"], 0)

        cfg.fusion.emit_unknown_boxes = True
        large = np.zeros((G, G), np.float32)
        large[50:100, 60:110] = 0.95
        big = safety_net_numpy(large, empty_b, empty_s, empty_l,
                               (1280, 1280), cfg, 19, context=ctx)
        self.assertEqual(big["n_safety_boxes"], 1)
        self.assertEqual(big["n_unknown_boxes"], 0)
        self.assertIn(19, big["labels"].tolist())

    # --------------------------------------------------- CUDA device-side assert guards
    def test_33_sanitize_detection_targets_and_label_offset(self):
        """Invalid ids / boxes are dropped on CPU; 1..nc ids shift exactly once."""
        nc, size = 3, (16, 16)
        raw = [dict(
            boxes=torch.tensor([[1., 1., 5., 5.],
                                [6., 6., 10., 10.],
                                [20., 20., 30., 30.],     # out of bounds
                                [1., 1., -2., 5.]]),      # x1 <= x0
            labels=torch.tensor([1, 3, 2, 2]))]           # 1-based, one is cls=3
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            clean = sanitize_detection_targets(raw, nc, size, "cpu", "auto")
        self.assertEqual(clean[0]["labels"].tolist(), [0, 2])
        self.assertEqual(clean[0]["boxes"].shape, (2, 4))
        self.assertTrue(bool((clean[0]["boxes"] >= 0).all()))

        # already-0-based data is not shifted by an explicit 1-based request
        zero = [dict(boxes=torch.tensor([[1., 1., 5., 5.]]),
                     labels=torch.tensor([0]))]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            again = sanitize_detection_targets(zero, nc, size, "cpu", "1")
        self.assertEqual(again[0]["labels"].tolist(), [0])

        # mode='auto' only shifts when an out-of-range id proves 1-based labels
        self.assertEqual(resolve_label_offset(torch.tensor([1, 2, 3]), "auto", nc), 1)
        self.assertEqual(resolve_label_offset(torch.tensor([0, 2]), "auto", nc), 0)
        self.assertEqual(resolve_label_offset(torch.tensor([0, 2]), "1", nc), 0)
        # a stray 99 must not stop auto from shifting the valid 1-based rows
        mixed = [dict(boxes=torch.tensor([[1., 1., 5., 5.],
                                          [6., 6., 10., 10.],
                                          [11., 11., 13., 13.]]),
                      labels=torch.tensor([1, 3, 99]))]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            shifted = sanitize_detection_targets(mixed, nc, size, "cpu", "auto")
        self.assertEqual(shifted[0]["labels"].tolist(), [0, 2])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            bad = sanitize_detection_targets(
                [dict(boxes=torch.tensor([[1., 1., 5., 5.],
                                          [6., 6., 10., 10.]]),
                      labels=torch.tensor([0, 3]))],
                nc, size, "cpu", "0")
        self.assertEqual(bad[0]["labels"].tolist(), [0])

    def test_34_dense_target_fallback_never_indexes_oob(self):
        """Tiny edge boxes with no centre-sampling cell used idx into arange(G)."""
        B, nc, H, W, stride = 1, 3, 2, 2, 8
        pred = {"cls": torch.zeros(B, nc, H, W),
                "reg": torch.zeros(B, 4, H, W)}
        targets = [dict(
            boxes=torch.tensor([[0.0, 0.0, 0.2, 0.2],
                                [15.8, 15.8, 16.0, 16.0]]),
            labels=torch.tensor([0, 2]))]
        # Before the fix: nearest grid id 3 indexed arange(G=2) -> IndexKernel.cu
        # device-side assert on CUDA / IndexError on CPU.
        t_cls, t_reg, t_qual, t_fg, pos, pts = build_dense_targets(
            pred, targets, stride, nc, 1.0, radius=0.1)
        self.assertEqual(tuple(t_cls.shape), (B, nc, H * W))
        self.assertEqual(int(pos.sum()), 2)        # both boxes got a fallback cell
        self.assertEqual(float(t_cls[0, 0, 0]), 1.0)
        self.assertEqual(float(t_cls[0, 2, 3]), 1.0)
        for t in (t_reg, t_qual, t_fg, pts):
            self.assertTrue(torch.isfinite(t).all())

    def test_35_validate_labels_cli_fix_and_rebuild_cache(self):
        """1-based .txt + stale cache -> clear validator report, then clean fix."""
        import hashlib
        import yaml
        from jingwei.cache import (build_dataset_cache, cache_paths,
                                   cache_prefix)
        size, nc = 32, 3
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "布幅数据"
            splits = {}
            for split, cls in (("train", 3), ("val", 1)):
                img_dir = data_root / "images" / split
                lab_dir = data_root / "labels" / split
                img_dir.mkdir(parents=True)
                lab_dir.mkdir(parents=True)
                imwrite_unicode(img_dir / f"布匹_{split}.jpg",
                                np.full((24, 40, 3), 120, np.uint8))
                (lab_dir / f"布匹_{split}.txt").write_text(
                    f"{cls} 0.5 0.5 0.4 0.3\n", encoding="utf-8")
                splits[split] = (img_dir, lab_dir)

            data_yaml = root / "data.yaml"
            data_yaml.write_text(yaml.safe_dump({
                "train": str(splits["train"][0]),
                "val": str(splits["val"][0]),
                "nc": nc,
                "names": ["a", "b", "c"],
            }, allow_unicode=True), encoding="utf-8")
            cfg_yaml = root / "tiny.yaml"
            cfg_yaml.write_text(
                "name: tiny\ninput_size: [32, 32]\n"
                "data:\n  nc: 3\n  dataset_yaml: " +
                str(data_yaml).replace("\\", "/") +
                "\n  names: [a, b, c]\n", encoding="utf-8")

            # Build a STALE cache with raw 1-based labels and no nc metadata.
            cache_dir = root / "cache"
            cache_dir.mkdir(parents=True)
            for split, (img_dir, lab_dir) in splits.items():
                tag = hashlib.sha1(str(img_dir).encode("utf-8", "replace")).hexdigest()[:8]
                prefix = cache_prefix(cache_dir, tag, split, size, "bytes")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    build_dataset_cache(img_dir, lab_dir, size, prefix,
                                        "bytes", True, log_fn=lambda m: None)
            self.assertTrue(any(cache_dir.glob("train_32_bytes_*.json")))

            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "ascii:backslashreplace"

            def run(*extra):
                cmd = [sys.executable, "tools/validate_labels.py",
                       "--cfg", str(cfg_yaml), "--data", str(data_yaml),
                       "--nc", str(nc), "--img-size", str(size),
                       "--cache", "bytes", "--cache-dir", str(cache_dir), *extra]
                return subprocess.run(cmd, cwd=str(ROOT), env=env,
                                      capture_output=True, text=True,
                                      encoding="utf-8", errors="replace",
                                      timeout=120)

            bad = run("--label-offset", "0", "--max-examples", "3")
            self.assertEqual(bad.returncode, 2, bad.stdout + bad.stderr)
            self.assertIn("bad_class=", bad.stdout)
            self.assertIn("cache_bad_class", bad.stdout)
            self.assertIn("RESULT: INVALID", bad.stdout)

            fixed = run("--fix-one-based", "--label-offset", "auto",
                        "--rebuild-cache", "--no-backup")
            self.assertEqual(fixed.returncode, 0, fixed.stdout + fixed.stderr)
            self.assertIn("FIXED 1-based labels", fixed.stdout)
            self.assertIn("RESULT: OK", fixed.stdout)

            # .txt files are now 0-based; rebuilt cache JSON contains 0-based ids.
            for split, (_img, lab_dir) in splits.items():
                txt = (lab_dir / f"布匹_{split}.txt").read_text(encoding="utf-8")
                self.assertLess(int(txt.split()[0]), nc)
            with (next(cache_dir.glob("train_32_bytes_*.json"))).open(
                    "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            ids = [int(r[0]) for rows in meta["labels"] for r in rows]
            self.assertTrue(ids)
            self.assertTrue(all(0 <= i < nc for i in ids), ids)
            self.assertEqual(meta["label_offset"], "0")
            self.assertEqual(meta["nc"], nc)

    def test_36_cache_builder_normalises_one_based_labels(self):
        """Cache build with --label-offset auto stores only 0..nc-1 ids."""
        from jingwei.cache import build_dataset_cache, cache_paths, cache_prefix
        size, nc = 24, 3
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img_dir = root / "images" / "train"
            lab_dir = root / "labels" / "train"
            img_dir.mkdir(parents=True)
            lab_dir.mkdir(parents=True)
            imwrite_unicode(img_dir / "布_0.jpg",
                            np.full((16, 24, 3), 100, np.uint8))
            (lab_dir / "布_0.txt").write_text(
                "1 0.5 0.5 0.3 0.3\n"
                "3 0.4 0.4 0.2 0.2\n"
                "99 0.5 0.5 0.1 0.1\n"       # invalid: must be dropped
                "2 0.5 0.5 0.0 0.2\n",      # zero width: must be dropped
                encoding="utf-8")
            prefix = cache_prefix(root / "cache", "tag", "train", size, "bytes")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                meta = build_dataset_cache(
                    img_dir, lab_dir, size, prefix, "bytes", True,
                    log_fn=lambda m: None, label_offset="auto", nc=nc)
            self.assertEqual(meta["n_bad_labels"], 2)
            ids = [int(r[0]) for rows in meta["labels"] for r in rows]
            self.assertEqual(ids, [0, 2])
            self.assertEqual(meta["label_offset"], "1")   # resolved, not 'auto'
            self.assertEqual(meta["label_offset_requested"], "auto")
            mp, dp = cache_paths(prefix, "bytes")
            self.assertTrue(mp.is_file() and dp.is_file())

            # 'auto' must be resolved per SPLIT, not per image: a file whose
            # ids never reach nc (here only 1..2) must still be shifted when
            # another file proves the dataset is 1-based.
            from jingwei.cache import infer_label_offset_from_label_files
            second = lab_dir / "布_1.txt"
            second.write_text("1 0.5 0.5 0.3 0.3\n2 0.4 0.4 0.2 0.2\n",
                              encoding="utf-8")
            imwrite_unicode(img_dir / "布_1.jpg",
                            np.full((16, 24, 3), 140, np.uint8))
            files = sorted(str(p) for p in lab_dir.glob("*.txt"))
            self.assertEqual(infer_label_offset_from_label_files(files, nc), 1)
            prefix2 = cache_prefix(root / "cache2", "tag", "train", size, "bytes")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                meta2 = build_dataset_cache(
                    img_dir, lab_dir, size, prefix2, "bytes", True,
                    log_fn=lambda m: None, label_offset="auto", nc=nc)
            per_image = [[int(r[0]) for r in rows] for rows in meta2["labels"]]
            for rows in per_image:
                self.assertTrue(rows, per_image)
                self.assertTrue(all(0 <= i < nc for i in rows), per_image)
            self.assertIn([0, 1], per_image, per_image)


    # ---------------------------------------------------------- dataloader speed
    def test_37_augmentation_speed_budget(self):
        """Augmentation must stay in the single-digit-millisecond range.

        Regression guard: the old reflectance branch blurred a 1280x1280
        *float32* field with sigmaX = w / 12 (~106 px), which costs ~0.5 s per
        image on OpenCV, so ``--num-workers 0`` training crawled at <1 it/s and
        a 29k-image dataset could never finish.  Budget here is deliberately
        generous (60 ms) so slow CI machines still pass.
        """
        from jingwei.data import fabric_photometric, geometric_augment
        img = (np.random.default_rng(0).random((1280, 1280, 3)) * 255).astype(np.uint8)
        boxes = np.array([[100., 120., 600., 700.]], np.float32)
        rng = np.random.default_rng(7)

        t0 = time.perf_counter()
        n = 5
        for _ in range(n):
            fabric_photometric(img, rng)
        photo_ms = (time.perf_counter() - t0) / n * 1000
        t0 = time.perf_counter()
        for _ in range(n):
            geometric_augment(img.copy(), boxes.copy(), rng, 1280)
        geo_ms = (time.perf_counter() - t0) / n * 1000
        self.assertLess(photo_ms, 60.0, f"fabric_photometric {photo_ms:.1f} ms")
        self.assertLess(geo_ms, 60.0, f"geometric_augment {geo_ms:.1f} ms")
        print(f"\n  aug speed: photometric {photo_ms:.1f} ms  "
              f"geometric {geo_ms:.1f} ms per 1280x1280 image")

    def test_38_cached_dataset_pickles_without_pixels(self):
        """DataLoader workers must not be sent the memmap contents.

        Pickling ``np.memmap`` serialises the whole array: 29533 x 1280 x 1280
        x 3 uint8 is ~145 GB, which made ``--num-workers 4`` die with
        ``MemoryError`` at worker start-up on Windows.  The dataset now ships the
        cache path only and re-opens the memmap inside each worker.
        """
        import pickle
        from jingwei.cache import (build_dataset_cache, cache_prefix,
                                   load_cached_dataset)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img_dir, lab_dir = root / "images", root / "labels"
            img_dir.mkdir()
            lab_dir.mkdir()
            for i in range(4):
                imwrite_unicode(img_dir / f"im{i}.jpg",
                                np.full((32, 40, 3), 90 + i * 20, np.uint8))
                (lab_dir / f"im{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n",
                                                    encoding="utf-8")
            prefix = cache_prefix(root / "cache", "tag", "train", 32, "disk")
            build_dataset_cache(img_dir, lab_dir, 32, prefix, "disk", True,
                                log_fn=lambda m: None, label_offset=0, nc=19)
            ds = load_cached_dataset(prefix, "disk", True, 0)
            blob = pickle.dumps(ds)              # what DataLoader workers get
            self.assertLess(len(blob), 200_000, len(blob))
            clone = pickle.loads(blob)
            self.assertEqual(len(clone), 4)
            a, b = ds[2]["image"], clone[2]["image"]
            self.assertTrue(torch.equal(a, b))
            self.assertEqual(tuple(clone[0]["boxes"].shape), (1, 4))

    def test_39_geometric_augment_box_alignment(self):
        """Boxes must follow flips / rot90 / resize exactly (pixel check)."""
        from jingwei.data import geometric_augment
        src = np.zeros((256, 256, 3), np.uint8)
        src[40:100, 120:200] = 255                     # y0,y1 / x0,x1
        box = np.array([[120., 40., 200., 100.]], np.float32)
        for seed in range(60):
            rng = np.random.default_rng(seed)
            img, boxes = geometric_augment(src.copy(), box.copy(), rng, 128)
            if not len(boxes):
                continue
            x0, y0, x1, y1 = [float(v) for v in boxes[0]]
            bright = img[:, :, 0]
            ys = np.where(bright.mean(1) > 200)[0]
            xs = np.where(bright.mean(0) > 200)[0]
            if len(ys) == 0 or len(xs) == 0:
                continue
            self.assertLess(abs(xs[0] - x0), 3, (seed, boxes[0], xs[0]))
            self.assertLess(abs(xs[-1] + 1 - x1), 3, (seed, boxes[0], xs[-1]))
            self.assertLess(abs(ys[0] - y0), 3, (seed, boxes[0], ys[0]))
            self.assertLess(abs(ys[-1] + 1 - y1), 3, (seed, boxes[0], ys[-1]))

    def test_40_uint8_batch_pipeline_matches_float_reference(self):
        """collate -> to_model_input must equal the old per-sample float cast."""
        from jingwei.data import (collate, image_hw, to_chw_float,
                                  to_model_input)
        a = (np.random.default_rng(1).random((16, 24, 3)) * 255).astype(np.uint8)
        b = (np.random.default_rng(2).random((16, 24, 3)) * 255).astype(np.uint8)
        batch = collate([dict(image=torch.from_numpy(a)),
                         dict(image=torch.from_numpy(b))])
        self.assertEqual(batch["image"].dtype, torch.uint8)
        x = to_model_input(batch["image"])
        ref = torch.stack([to_chw_float(a), to_chw_float(b)], 0)
        self.assertTrue(torch.allclose(x, ref, atol=1e-6))
        self.assertEqual(image_hw(batch["image"]), (16, 24))
        self.assertEqual(image_hw(ref), (16, 24))

    def test_41_amp_and_loader_flags(self):
        """--amp resolves sanely and bf16 autocast keeps the loss finite."""
        from train_detector import pick_amp_dtype
        self.assertIsNone(pick_amp_dtype("off", "cpu"))
        self.assertIsNone(pick_amp_dtype("auto", "cpu"))
        self.assertIs(pick_amp_dtype("bf16", "cuda"), torch.bfloat16)
        self.assertIs(pick_amp_dtype("fp16", "cuda"), torch.float16)

        # bf16 autocast forward+backward on CPU exercises the same kernels the
        # GPU mixed-precision run uses; the detection loss must stay finite.
        model = build_model(CFG_P)
        model.set_stage("detector")
        model.train()
        x = torch.rand(1, 3, 128, 128)
        targets = [dict(boxes=torch.tensor([[10., 10., 60., 60.]]),
                        labels=torch.tensor([2]))]
        tg = sanitize_detection_targets(targets, 19, (128, 128), "cpu", 0)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            out = model(x, targets=tg)
        loss = out["losses"]["total"]
        self.assertTrue(torch.isfinite(loss), loss)
        loss.float().backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))


if __name__ == "__main__":
    unittest.main(verbosity=2)
