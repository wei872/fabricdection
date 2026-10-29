"""Verification suite for the JingWei reference implementation.

Run from the repo root::

    python -m unittest discover -s tests -v

No pytest / no GPU / no real data needed. It checks shapes at the required
1280x1280 input, trainability, the deterministic safety net, fabric-type
invariance, memory-bank algebra, the dataset readers, Chinese / Unicode
file paths, non-square fabric augmentation, training progress logs,
full checkpoint resume, numpy RNG state checkpointing, dynamic
local+global attention foreground context, validation metrics, and
unknown-defect boxes / objectness gate cold start / warm start / negative mixing (64 tests).
"""
from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))   # import tools/_common.py helpers

from jingwei import build_model, load_config                       # noqa: E402
from jingwei.fabric import (fabric_descriptor, fit_fabric_prototypes,  # noqa: E402
                            greedy_coreset, spherical_kmeans)
from jingwei.fusion import safety_net_numpy                       # noqa: E402
from jingwei.image_io import (VideoCaptureUnicode, ascii_safe,     # noqa: E402
                              console_safe, imread_unicode,
                              imwrite_unicode)
from jingwei.layers import LocalContrastNorm                     # noqa: E402
from jingwei.losses import hungarian_match, o2o_negative_count   # noqa: E402
from jingwei.utils import (count_parameters, profile_macs,         # noqa: E402
                           torch_load)
from _common import subset_indices                                 # noqa: E402

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
            self.assertEqual(tuple(item["image"].shape), (3, 128, 128))
            self.assertEqual(tuple(item["boxes"].shape), (2, 4))
            self.assertEqual(tuple(item["labels"].shape), (2,))
            nd = NormalImageDataset(normal, img_size=128, augment=False,
                                    with_descriptor=True)
            self.assertEqual(len(nd), 4)
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
                       "--prototypes", str(proto_path)]
            proc_v = subprocess.run(cmd_vid, cwd=str(ROOT), env=env,
                                    capture_output=True, text=True,
                                    encoding="utf-8", errors="replace", timeout=180)
            self.assertEqual(proc_v.returncode, 0, proc_v.stdout + proc_v.stderr)
            # Streaming decode: the annotated frame itself is the output,
            # no intermediate out/_frames dump (see tools/predict.py).
            self.assertTrue((out_vid / "000000.jpg").exists())
            self.assertFalse((out_vid / "_frames").exists())
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
            self.assertEqual(tuple(item["image"].shape), (3, 64, 64))
            nd = NormalImageDataset(normal, img_size=64, augment=True, seed=2,
                                    with_descriptor=True)
            item2 = nd[0]
            self.assertEqual(tuple(item2["image"].shape), (3, 64, 64))
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
                   "--val-max", "2", "--normal-max", "2",
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

    # --------------------------------------------------- detector-only validation
    def test_33_subset_indices_deterministic(self):
        a = subset_indices(1000, 50, 7)
        b = subset_indices(1000, 50, 7)
        c = subset_indices(1000, 50, 8)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(len(a), 50)
        self.assertEqual(len(set(a)), 50)
        self.assertTrue(all(0 <= i < 1000 for i in a))
        self.assertIsNone(subset_indices(10, 10, 0))
        self.assertIsNone(subset_indices(10, 0, 0))

    def test_34_detector_stage_eval_skips_normality(self):
        m = build_model(self.cfg).eval()
        m.set_stage("detector")
        called = {"n": 0}

        def boom(*args, **kwargs):
            called["n"] += 1
            raise AssertionError("normality ran during detector-stage eval")

        m.normality.forward = boom
        with torch.no_grad():
            d = m(torch.randn(1, 3, 256, 256), proto_id=0)[0]
        self.assertEqual(called["n"], 0)
        # the detector-only path now returns the SAME keys (zeros) so
        # downstream code never has to branch on the training stage.
        self.assertEqual(d["n_safety_boxes"], 0)
        self.assertEqual(d["n_global_components"], 0)
        self.assertEqual(d["boxes"].shape[0], d["scores"].shape[0])

    def test_35_o2o_negative_floor(self):
        self.assertEqual(o2o_negative_count(25600, 0, 0.02, 2048), 2048)
        self.assertEqual(o2o_negative_count(1600, 8, 0.02, 2048), 1592)
        self.assertEqual(o2o_negative_count(100, 99, 0.02, 2048), 1)
        self.assertEqual(o2o_negative_count(100, 100, 0.02, 2048), 0)

    def test_36_pr_diagnostics(self):
        from evaluate import _best_f1, _pr_at_thresholds
        records = [[(0.9, True), (0.8, False)], [(0.7, True)]]
        best_f1, best_thr = _best_f1(records, 2)
        self.assertAlmostEqual(best_f1, 0.8, places=6)
        self.assertAlmostEqual(best_thr, 0.7, places=6)
        pr = _pr_at_thresholds(records, 2, (0.75, 0.7))
        self.assertAlmostEqual(pr[0.75]["precision"], 0.5, places=6)
        self.assertAlmostEqual(pr[0.7]["recall"], 1.0, places=6)
        self.assertEqual(pr[0.7]["tp"], 2)


    # ------------------------------------------------- performance / calibration
    def test_37_local_contrast_norm_separable_box(self):
        """The separable box filter must equal the old k x k window exactly."""
        lcn = LocalContrastNorm(k=15, learnable=False).eval()
        rng = np.random.default_rng(0)
        x = torch.from_numpy(rng.random((1, 3, 96, 111)).astype(np.float32))
        ref = torch.nn.functional.avg_pool2d(x, 15, 1, 7,
                                             count_include_pad=False)
        got = lcn._box(x)
        self.assertEqual(got.shape, ref.shape)
        self.assertLess(float((got - ref).abs().max()), 1e-5,
                        "separable 1-D box passes must be numerically exact")
        # corners are where a naive separable implementation usually breaks
        for sl in ((slice(0, 4), slice(0, 4)), (slice(-4, None), slice(-4, None))):
            self.assertLess(float((got[:, :, sl[0], sl[1]] -
                                   ref[:, :, sl[0], sl[1]]).abs().max()), 1e-5)
        with torch.no_grad():
            out = lcn(x)
        self.assertTrue(torch.isfinite(out).all())

    def test_38_anomaly_fusion_preserves_calibration(self):
        """`gate` must not shift the calibrated operating point (0.5)."""
        from jingwei.normality import AnomalyFusion

        for mode in ("gate", "fixed"):
            fuse = AnomalyFusion(mode, 0.6, 0.4).eval()
            for v in (0.01, 0.05, 0.25, 0.5, 0.75, 0.99):
                a = torch.full((1, 1, 3, 3), v)
                got = float(fuse(a, a)[0, 0, 0, 0])
                self.assertAlmostEqual(got, v, places=3,
                                       msg=f"{mode} mode shifted {v} -> {got}")
            # monotone in each input
            lo = float(fuse(torch.full((1, 1, 2, 2), 0.1),
                            torch.full((1, 1, 2, 2), 0.1))[0, 0, 0, 0])
            hi = float(fuse(torch.full((1, 1, 2, 2), 0.9),
                            torch.full((1, 1, 2, 2), 0.9))[0, 0, 0, 0])
            self.assertGreater(hi, lo)
            self.assertGreater(lo, 0.0)
            self.assertLess(hi, 1.0)

    def test_39_safety_net_ignores_global_anomaly(self):
        """A whole-frame anomaly (light flicker / mis-calibration) is not a defect."""
        G = 160
        cfg = load_config(CFG_N)
        ctx = np.zeros((3, G, G), np.float32)
        ctx[1] = 0.95                                   # all cloth
        flooded = np.full((G, G), 0.55, np.float32)
        empty_b = np.zeros((0, 4), np.float32)
        empty_s = np.zeros(0, np.float32)
        empty_l = np.zeros(0, np.int64)
        out = safety_net_numpy(flooded, empty_b, empty_s, empty_l,
                               (1280, 1280), cfg, 19, context=ctx)
        self.assertEqual(out["n_safety_boxes"], 0,
                         "full-frame component must not become a box")
        self.assertEqual(out["n_global_components"], 1)
        # a genuinely large but LOCAL defect is still emitted
        local = np.zeros((G, G), np.float32)
        local[20:140, 20:140] = 0.9                     # 12100 cells = 47%
        out2 = safety_net_numpy(local, empty_b, empty_s, empty_l,
                               (1280, 1280), cfg, 19, context=ctx)
        self.assertEqual(out2["n_safety_boxes"], 1)

    def test_40_predict_boxes_back_to_image_pixels(self):
        """Detections must be reported/drawn in ORIGINAL image coordinates."""
        from predict import boxes_to_image_pixels
        net = 1280
        boxes = np.array([[128., 256., 640., 512.]], np.float32)
        got = boxes_to_image_pixels(boxes, net, 640, 360)
        np.testing.assert_allclose(
            got, np.array([[64., 72., 320., 144.]], np.float32), rtol=1e-5)
        # clipping: a box outside the network frame stays inside the image
        clipped = boxes_to_image_pixels(
            np.array([[-50., -50., 5000., 5000.]], np.float32), net, 100, 50)
        np.testing.assert_allclose(clipped,
                                   np.array([[0., 0., 100., 50.]], np.float32))
        self.assertEqual(boxes_to_image_pixels(np.zeros((0, 4), np.float32),
                                              net, 10, 10).shape, (0, 4))

    def test_41_list_images_single_pass(self):
        """One directory walk must return exactly what the 14-pattern walk did."""
        from jingwei.data import IMG_EXTS, list_images
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "数据"
            (root / "sub" / "deeper").mkdir(parents=True)
            names = ["a.jpg", "b.JPG", "c.PNG", "d.WebP", "e.tif", "f.txt",
                     "sub/g.jpeg", "sub/deeper/h.bmp"]
            for n in names:
                if Path(n).suffix.lower() == ".txt":
                    (root / n).parent.mkdir(parents=True, exist_ok=True)
                    (root / n).write_text("not an image", encoding="utf-8")
                else:
                    imwrite_unicode(root / n, weave(9, 128, 0.1, 32))
            (root / "looks_like_image.jpg").mkdir()   # directory, not a file
            legacy = set()
            for ext in IMG_EXTS:
                legacy.update(root.rglob(ext))
            legacy = sorted(str(f) for f in legacy if f.is_file())
            found = list_images(root)
            # every image the old 14-pattern walk found is still found ...
            self.assertTrue(set(legacy) <= set(found))
            # ... plus mixed-case extensions ("d.WebP"), which the case
            # sensitive rglob("*.webp")/rglob("*.WEBP") patterns missed even
            # though imread_unicode can decode them.
            self.assertEqual(len(found), 7)
            self.assertTrue(any(f.endswith("d.WebP") for f in found))
            self.assertFalse(any("looks_like_image" in f for f in found))
            self.assertFalse(any(f.endswith("f.txt") for f in found))
            self.assertEqual(list_images(root / "does_not_exist"), [])

    def test_42_checkpoint_key_mismatch_is_reported(self):
        """Partially-matching checkpoints must not load silently."""
        from _common import load_state_dict_report
        m = build_model(self.cfg)
        sd = m.state_dict()
        dropped = next(k for k in sd if k.startswith("backbone."))
        partial = {k: v for k, v in sd.items() if k != dropped}
        partial["bogus_layer.weight"] = torch.zeros(1)
        fresh = build_model(self.cfg)
        msgs = load_state_dict_report(fresh, partial, tag="unit-test")
        self.assertEqual(len(msgs), 2)
        self.assertIn("missing", msgs[0])
        self.assertIn("unexpected", msgs[1])
        # every key that WAS present must have been loaded
        before = fresh.state_dict()[dropped].clone()
        self.assertTrue(torch.equal(fresh.state_dict()[dropped], before))
        for k, v in partial.items():
            if k in fresh.state_dict():
                self.assertTrue(torch.equal(fresh.state_dict()[k], v), k)
        clean = load_state_dict_report(build_model(self.cfg), sd, tag="clean")
        self.assertEqual(clean, [])


    def test_43_worker_seeding_covers_python_random(self):
        """Stage-0 views use ``random``; every worker must get its own stream."""
        from jingwei.data import _seed_worker

        def draw(worker_id):
            random.seed(0)
            _seed_worker(worker_id, 7)
            return [random.random() for _ in range(4)]

        self.assertEqual(draw(3), draw(3), "same worker id -> reproducible")
        self.assertNotEqual(draw(3), draw(4),
                            "different workers must not share one stream")

    # ------------------------------------------------------- precision / decode
    @staticmethod
    def _blob_preds(nc=19, h=160, w=160, blob=(60, 72, 60, 72), obj_logit=3.0):
        """One confident defect blob on the P3 map, nothing on P5."""
        obj = torch.full((1, 1, h, w), -8.0)
        y0, y1, x0, x1 = blob
        obj[:, :, y0:y1, x0:x1] = obj_logit
        cls = torch.full((1, nc, h, w), -6.0)
        cls[:, 3, :, :] = 4.0
        reg = torch.full((1, 4, h, w), 2.0)
        p3 = dict(cls=cls, reg=reg, obj_o2o=obj)
        p5 = dict(cls=torch.full((1, nc, 40, 40), -6.0),
                  reg=torch.full((1, 4, 40, 40), 2.0),
                  obj_o2o=torch.full((1, 1, 40, 40), -8.0))
        return [p3, p5]

    def test_44_peak_filter_prevents_duplicate_boxes(self):
        """One defect must decode to ONE box, not one box per plateau cell."""
        m = build_model(self.cfg).eval()
        preds = self._blob_preds()
        on = m.head.decode(preds, (1280, 1280))[0]
        self.assertEqual(len(on["boxes"]), 1,
                         "3x3 peak filter must collapse one defect to one box")

        # A/B: disabling the filter reproduces the old (precision-killing)
        # behaviour of one box per high-objectness cell.
        m.head.peak_filter = False
        off = m.head.decode(preds, (1280, 1280))[0]
        self.assertGreater(len(off["boxes"]), 100,
                           "without the filter every plateau cell becomes a box")
        # sanity: the surviving box is the one in the blob
        b = on["boxes"][0]
        self.assertLess(abs(float(b[0] + b[2]) / 2 - 1280 * 66 / 160), 60)
        m.head.peak_filter = True

    def test_45_peak_filter_keeps_separate_defects(self):
        """Two well-separated defects -> two peaks, not one."""
        m = build_model(self.cfg).eval()
        obj = torch.full((1, 1, 160, 160), -8.0)
        obj[:, :, 20:26, 20:26] = 3.0
        obj[:, :, 120:126, 120:126] = 2.0
        cls = torch.full((1, 19, 160, 160), -6.0)
        cls[:, 5, :, :] = 4.0
        reg = torch.full((1, 4, 160, 160), 2.0)
        preds = [dict(cls=cls, reg=reg, obj_o2o=obj),
                 dict(cls=torch.full((1, 19, 40, 40), -6.0),
                      reg=torch.full((1, 4, 40, 40), 2.0),
                      obj_o2o=torch.full((1, 1, 40, 40), -8.0))]
        d = m.head.decode(preds, (1280, 1280))[0]
        self.assertEqual(len(d["boxes"]), 2)
        self.assertEqual(sorted(d["labels"].tolist()), [5, 5])

    def test_46_selection_metric_is_precision_aware(self):
        """`best.pt` must not be chosen by a metric that ignores precision."""
        import re
        src = (ROOT / "tools" / "train_detector.py").read_text(encoding="utf-8")
        # the old objective only used mAP50 + large_recall (both recall-ish)
        self.assertNotIn("score = 0.5 * val_metrics[\"mAP50\"]", src)
        self.assertIn("best_f1", src)
        self.assertRegex(src, r"score = float\(val_metrics\[\"best_f1\"\]\)")
        # and the per-epoch log must expose the FP breakdown
        self.assertIn("wrong_class=", src)
        self.assertIn("boxes/img=", src)

    def test_47_fusion_calibration_ranks_by_f1(self):
        """Stage D must measure (and rank by) precision, not recall alone."""
        src = (ROOT / "tools" / "calibrate_fusion.py").read_text(encoding="utf-8")
        self.assertIn("detection_precision_from_cache", src)
        self.assertIn("F1=", src)
        self.assertIn("--min-precision", src)

    def test_48_val_loader_reports_boxes_per_image(self):
        """evaluate_map must expose dets/image so precision is diagnosable."""
        from evaluate import evaluate_map

        class Stub(torch.nn.Module):
            def __init__(self, n_box):
                super().__init__()
                self.n_box = n_box

            def forward(self, x):
                out = []
                for _ in range(x.shape[0]):
                    b = torch.tensor([[10., 10., 100., 100.]] * self.n_box)
                    out.append(dict(boxes=b, scores=torch.tensor([0.9] * self.n_box),
                                    labels=torch.zeros(self.n_box, dtype=torch.long)))
                return out

        batch = dict(image=torch.zeros(1, 3, 128, 128),
                     boxes=[torch.tensor([[10., 10., 100., 100.]])],  # IoU 1
                     labels=[torch.tensor([0])])
        _, _, pr = evaluate_map(Stub(6), [batch], "cpu", 2, conf_thresh=0.5)
        self.assertAlmostEqual(pr["dets_per_image"], 6.0)
        # 1 TP + 5 duplicates -> precision 1/6 (class-aware TP/FP)
        self.assertAlmostEqual(pr["precision"], 1 / 6, places=4)
        self.assertEqual(pr["fp_breakdown"]["duplicate_same_gt"], 5)


    # ------------------------------------------------ precision round 2 (v0.2.0)
    def test_49_cross_scale_dedup(self):
        """The same defect found by P3 AND P5 (nested boxes) -> one box."""
        m = build_model(self.cfg).eval()
        cls3 = torch.full((1, 19, 160, 160), -6.0)
        cls3[:, 3] = 4.0
        cls5 = torch.full((1, 19, 40, 40), -6.0)
        cls5[:, 3] = 4.0
        o3 = torch.full((1, 1, 160, 160), -8.0)
        o3[:, :, 60:72, 60:72] = 3.0
        o5 = torch.full((1, 1, 40, 40), -8.0)
        o5[:, :, 16:20, 16:20] = 3.0
        preds = [dict(cls=cls3, reg=torch.full((1, 4, 160, 160), 1.0), obj_o2o=o3),
                 dict(cls=cls5, reg=torch.full((1, 4, 40, 40), 4.0), obj_o2o=o5)]
        on = m.head.decode(preds, (1280, 1280))[0]
        self.assertEqual(len(on["boxes"]), 1,
                         "nested P3/P5 duplicate must be suppressed")
        m.head.dedup = False
        off = m.head.decode(preds, (1280, 1280))[0]
        self.assertEqual(len(off["boxes"]), 2)
        m.head.dedup = True
        # two defects far apart on different scales must both survive
        o3b = torch.full((1, 1, 160, 160), -8.0)
        o3b[:, :, 20:26, 20:26] = 3.0
        o5b = torch.full((1, 1, 40, 40), -8.0)
        o5b[:, :, 25:27, 25:27] = 3.0
        two = m.head.decode([dict(cls=cls3, reg=torch.full((1, 4, 160, 160), 1.0),
                                  obj_o2o=o3b),
                             dict(cls=cls5, reg=torch.full((1, 4, 40, 40), 4.0),
                                  obj_o2o=o5b)], (1280, 1280))[0]
        self.assertEqual(len(two["boxes"]), 2)

    def test_50_objectness_supervises_every_cell(self):
        """ohem leaves ~92% of the frame without any background gradient."""
        from jingwei.losses import DetectionLoss

        tgt = [dict(boxes=torch.tensor([[480., 480., 500., 500.]]),
                    labels=torch.tensor([3]))]

        def run(mode, base_logit=-4.0):
            cfg = load_config(CFG_N)
            cfg.train.obj_o2o_mode = mode
            obj = torch.full((1, 1, 160, 160), float(base_logit))
            obj[:, :, 60, 60] = 4.0        # the matched positive
            obj3 = obj.clone().requires_grad_(True)
            preds = [dict(cls=torch.full((1, 19, 160, 160), -3.0),
                          reg=torch.full((1, 4, 160, 160), 1.0),
                          obj=obj.clone(), obj_o2o=obj3, fg=None),
                     dict(cls=torch.full((1, 19, 40, 40), -3.0),
                          reg=torch.full((1, 4, 40, 40), 1.0),
                          obj=obj.new_full((1, 1, 40, 40), -4.0),
                          obj_o2o=obj.new_full((1, 1, 40, 40), -4.0).requires_grad_(True),
                          fg=None)]
            out = DetectionLoss(cfg)(preds, tgt, (1280, 1280))
            out["obj_o2o"].backward()
            g = obj3.grad
            return out, int((g.abs() > 0).sum()), float(g[0, 0, 5, 150])

        out_ohem, n_ohem, far_ohem = run("ohem")
        out_focal, n_focal, far_focal = run("focal")
        for o in (out_ohem, out_focal):
            self.assertTrue(torch.isfinite(o["total"]))
            self.assertGreater(float(o["obj_o2o"]), 0.0)
        self.assertEqual(n_focal, 160 * 160, "focal must touch every cell")
        self.assertLess(n_ohem, 160 * 160 // 2,
                        "ohem only touches the sampled hard negatives")
        self.assertEqual(far_ohem, 0.0,
                         "far background cells get NO gradient under ohem")
        self.assertNotEqual(far_focal, 0.0,
                            "focal must supervise far background cells too")
        # NOTE (v0.2.1): the old assertion here was
        # ``focal_loss > ohem_loss``, i.e. a comparison of raw magnitudes.
        # With the balanced normalisation that is not meaningful any more --
        # focal legitimately collapses to ~3e-6/cell once the background is
        # already right (that is the whole point of the (1-p_t)^gamma factor),
        # while ohem's top-k mean always pays for its hardest cells.
        # What must hold is the property that matters: *wrong* background is
        # still punished hard, and it is punished because of the background
        # term (the balanced normalisation keeps that term alive).
        out_wrong, _, _ = run("focal", base_logit=0.0)     # p = 0.5 everywhere
        self.assertGreater(float(out_wrong["obj_o2o"]), 0.05,
                           "a background predicted at p=0.5 must still cost a lot")
        self.assertGreater(float(out_wrong["obj_o2o"]),
                           float(out_focal["obj_o2o"]))

    def test_51_precision_diagnosis_helpers(self):
        """The diagnosis tool must classify FPs the same way evaluate.py does."""
        from diagnose_precision import _match, report_at

        gt_b = torch.tensor([[10., 10., 100., 100.], [200., 200., 300., 300.]])
        gt_l = torch.tensor([3, 7])
        boxes = torch.tensor([
            [10., 10., 100., 100.],      # exact TP (class 3)
            [11., 11., 101., 101.],      # duplicate of the same GT
            [200., 200., 300., 300.],    # right box, wrong class (3 instead of 7)
            [400., 400., 500., 500.],    # background
        ])
        labels = torch.tensor([3, 3, 3, 5])
        scores = torch.tensor([0.9, 0.8, 0.7, 0.6])
        kinds = [k for k, _ in _match(boxes, labels, scores, gt_b, gt_l, 19, 0.5)]
        self.assertEqual(kinds, ["tp", "duplicate", "wrong_class", "background"])

        frames = [dict(boxes=boxes, scores=scores, labels=labels,
                       scale=torch.tensor([0, 0, 1, 1]),
                       gt_boxes=gt_b, gt_labels=gt_l)]
        r = report_at(frames, 0.0, 19, 0.5)
        self.assertEqual((r["tp"], r["fp"], r["fn"]), (1, 3, 1))
        self.assertEqual(r["cats"]["duplicate"], 1)
        self.assertEqual(r["cats"]["wrong_class"], 1)
        self.assertEqual(r["cats"]["background"], 1)
        self.assertAlmostEqual(r["precision"], 0.25)
        self.assertAlmostEqual(r["recall"], 0.5)
        self.assertEqual(r["per_scale"][(0, "tp")], 1)
        self.assertEqual(r["per_scale"][(1, "fp")], 2)
        self.assertEqual(r["confusions"][(7, 3)], 1)
        # a high threshold must remove everything
        self.assertEqual(report_at(frames, 0.95, 19, 0.5)["n_det"], 0)

    def test_52_dedup_survives_fusion(self):
        """scale_idx must travel through the safety net so per-scale FP stats work."""
        from jingwei.fusion import fuse_detections

        cfg = load_config(CFG_N)
        det = dict(boxes=torch.tensor([[100., 100., 300., 300.],
                                       [400., 400., 600., 600.]]),
                   scores=torch.tensor([0.9, 0.8]),
                   labels=torch.tensor([3, 4]),
                   obj=torch.tensor([0.9, 0.8]),
                   points=torch.zeros(2, 2),
                   scale_idx=torch.tensor([0, 1]))
        anomaly = torch.zeros(1, 1, 160, 160)      # no extra safety boxes
        out = fuse_detections(det, anomaly, (1280, 1280), cfg, 19)
        self.assertIn("scale_idx", out)
        self.assertEqual(out["scale_idx"].tolist(), [0, 1])

    # ------------------------------------------------ precision round 3 (v0.2.1)
    def test_53_quality_target_never_collapses_to_zero(self):
        """Cold start: predicted IoU is ~0, so the quality target must not be 0.

        The dense target used to be the *predicted* IoU of the matched cell.
        On a fresh model every predicted box is a degenerate dot, that IoU is
        0, and the branch was therefore trained towards ``target = 0`` on the
        positive cells themselves (``obj_o2m`` ~0.025 for a whole run) -- the
        objectness/classification heads could never wake up.
        """
        from jingwei.losses import build_dense_targets

        cfg = load_config(CFG_N)
        pred = dict(cls=torch.zeros(1, 19, 160, 160),
                    reg=torch.zeros(1, 4, 160, 160),        # cold start
                    obj=torch.zeros(1, 1, 160, 160))
        tgt = [dict(boxes=torch.tensor([[480., 480., 560., 560.]]),
                    labels=torch.tensor([3]))]
        for floor in (0.5, 0.0):
            _, _, t_qual, _, pos, _ = build_dense_targets(
                pred, tgt, 8, 19, float(cfg.model.reg_scale), quality_floor=floor)
            self.assertGreater(int(pos.sum()), 0)
            vals = t_qual[pos.expand(-1, 1, -1)]
            self.assertGreater(float(vals.min()), 0.0,
                               "a positive cell must never be taught objectness=0")
            self.assertLessEqual(float(vals.max()), 1.0)
            if floor > 0:
                self.assertGreaterEqual(float(vals.min()), floor - 1e-6)

    def test_54_classification_is_trained_up_on_a_defect(self):
        """The deployed cls head must be pushed UP on a positive cell.

        With the quality-scaled target the gradient at a defect cell was
        *positive* (i.e. push the class probability down towards the ~0.02
        predicted IoU), which is the direct cause of the dead score gate.
        """
        from jingwei.losses import DetectionLoss

        cfg = load_config(CFG_N)
        B, HW = 1, 160 * 160
        cls = torch.full((B, 19, 160, 160), -2.0, requires_grad=True)
        preds = [dict(cls=cls,
                      reg=torch.zeros(B, 4, 160, 160),
                      obj=torch.zeros(B, 1, 160, 160),
                      obj_o2o=torch.zeros(B, 1, 160, 160), fg=None),
                 dict(cls=torch.full((B, 19, 40, 40), -2.0),
                      reg=torch.zeros(B, 4, 40, 40),
                      obj=torch.zeros(B, 1, 40, 40),
                      obj_o2o=torch.zeros(B, 1, 40, 40), fg=None)]
        tgt = [dict(boxes=torch.tensor([[480., 480., 560., 560.]]),
                    labels=torch.tensor([3]))]
        out = DetectionLoss(cfg)(preds, tgt, (1280, 1280))
        out["cls_o2m"].backward()
        from jingwei.losses import build_dense_targets
        _, _, _, _, pos, _ = build_dense_targets(preds[0], tgt, 8, 19, 1.0)
        flat_pos = pos.reshape(-1)
        g = cls.grad.reshape(B, 19, HW)[0, 3]
        self.assertLess(float(g[flat_pos].mean()), -0.05,
                        "gradient wrt cls logit at a positive must be negative"
                        " (increase the class probability)")
        self.assertGreater(torch.isfinite(g).all().item(), 0)

    def test_55_objectness_positives_own_their_loss_term(self):
        """The positive cells must own a real share of `obj_o2o` (not 0.7%).

        Measured by masking the matched cell: with a balanced normalisation the
        loss drops to ~0 when the positive is already predicted, while the old
        ``sum / n_pos`` form was dominated by the ~25k background cells and
        barely noticed the positive at all.
        """
        from jingwei.losses import DetectionLoss, hungarian_match

        cfg = load_config(CFG_N)
        tgt = [dict(boxes=torch.tensor([[480., 480., 560., 560.]]),
                    labels=torch.tensor([3]))]
        loss_fn = DetectionLoss(cfg)

        shapes = [(1, 1, 160, 160), (1, 1, 40, 40)]

        def make(pos_logit=None, hit=None):
            preds = []
            for si, shp in enumerate(shapes):
                obj = torch.full(shp, -2.0).reshape(-1).clone()
                if hit is not None and hit[0] == si:
                    obj[hit[1]] = pos_logit
                obj = obj.reshape(shp)
                preds.append(dict(cls=torch.full((1, 19, shp[2], shp[3]), -2.0),
                                  reg=torch.full((1, 4, shp[2], shp[3]), 1.0),
                                  obj=obj.clone(), obj_o2o=obj, fg=None))
            return preds

        mm = hungarian_match(make(), tgt, loss_fn.strides, loss_fn.nc,
                             loss_fn.reg_scale, loss_fn.o2o_candidates)
        hit = [(si, flat) for si, m in enumerate(mm) for _, flat, _ in m]
        self.assertTrue(hit, "the fixture must produce an o2o match")
        hit = hit[0]

        def obj_loss(pos_logit):
            out = loss_fn(make(pos_logit, hit), tgt, (1280, 1280))
            return float(out["obj_o2o"])

        low = obj_loss(-2.0)      # positive predicted ~0.12 -> big positive loss
        high = obj_loss(12.0)     # positive predicted ~1.0 -> positive term ~0
        self.assertGreater(low, 0.05)
        self.assertLess(low, 20.0, "balanced normalisation keeps the term O(1)")
        self.assertGreater(low - high, 0.3 * low,
                           "the positive cell must own >=30% of obj_o2o;"
                           " a sum/n_pos form is owned by background cells")

    def test_56_head_opens_the_score_gate(self):
        """Regression for `boxes/img = 0.0`: after N Adam steps the gate must open.

        Head-only optimisation on a frozen feature map -- this is the exact
        300-step budget of a 55-iter epoch set, so the logged run is reproduced
        without a backbone, a dataset or a GPU.
        """
        from jingwei.head import NMSFreeDetect
        from jingwei.losses import DetectionLoss

        cfg = load_config(CFG_P)
        torch.manual_seed(0)
        head = NMSFreeDetect(cfg, [0, 96, 0, 384])
        for hd in head.heads:
            torch.nn.init.constant_(
                hd.obj_o2o.bias, float(cfg.model.get("obj_bias_init", -2.0)))
        head.train()
        loss_fn = DetectionLoss(cfg)
        g = torch.Generator().manual_seed(0)
        feats = {"n3": torch.randn(1, 96, 40, 40, generator=g),
                 "n5": torch.randn(1, 384, 10, 10, generator=g)}
        tgt = [dict(boxes=torch.tensor([[110., 110., 210., 210.]]),
                    labels=torch.tensor([3]))]
        opt = torch.optim.AdamW([p for p in head.parameters() if p.requires_grad],
                                lr=float(cfg.train.detector_lr))
        thresh = float(cfg.model.score_thresh)
        opened_at = None
        for it in range(300):
            opt.zero_grad(set_to_none=True)
            preds = head(feats, training=True)
            loss_fn(preds, tgt, (320, 320))["total"].backward()
            opt.step()
            if it % 25 == 24:
                with torch.no_grad():
                    p = head(feats, training=False)
                    # ALL scales: P3 alone can stay at the init level while P5
                    # has already learned the object (this masked a dead-gate
                    # report in an earlier version of the probe).
                    s = torch.cat([
                        (q["obj_o2o"].sigmoid() *
                         q["cls"].sigmoid().max(1, keepdim=True).values).reshape(1, -1)
                        for q in p], 1)
                    if float(s.max()) > thresh:
                        opened_at = it + 1
                        break
        self.assertIsNotNone(
            opened_at,
            f"300 Adam steps at lr={float(cfg.train.detector_lr)} did not push "
            f"the deployed score above {thresh} -- the objectness gate is dead")
        dets = head.decode(head(feats, training=False), (320, 320))[0]
        self.assertGreater(len(dets["boxes"]), 0)
        self.assertGreater(head.last_stats["score_max"], 0.0)

    def test_57_validation_reports_gate_liveness(self):
        """`gate_max < conf` (nothing emitted) must be distinguishable from bad boxes."""
        from evaluate import evaluate_map

        class Stub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = torch.nn.Module()
                self.head.last_stats = dict(score_max=0.004, obj_max=0.02,
                                            cls_max=0.20, n_above=0, n_out=0)

            def forward(self, x):
                return [dict(boxes=torch.zeros(0, 4), scores=torch.zeros(0),
                             labels=torch.zeros(0, dtype=torch.long))
                        for _ in range(x.shape[0])]

        batch = dict(image=torch.zeros(1, 3, 128, 128),
                     boxes=[torch.tensor([[10., 10., 100., 100.]])],
                     labels=[torch.tensor([0])])
        map50, _, pr = evaluate_map(Stub(), [batch], "cpu", 2, conf_thresh=0.15)
        self.assertEqual(map50, 0.0)
        self.assertAlmostEqual(pr["gate_max"], 0.004, places=6)
        self.assertAlmostEqual(pr["obj_max"], 0.02, places=6)
        self.assertLess(pr["gate_max"], 0.15,
                        "gate_max below conf == nothing emitted, not bad boxes")
        self.assertEqual(pr["n_above"], 0)

    # ------------------------------------------------ precision round 4 (v0.2.2)
    def test_58_background_weight_pushes_the_field_down(self):
        """`neg_weight` must actually move the background side of a term.

        The v0.2.1 run ended at ``obj_o2o = 0.086`` with ``obj ~ 0.5`` *on
        background*: a 50/50 balanced term has no "switch off" pressure, so the
        deployed score ``obj*cls`` cleared the 0.15 gate on ordinary fabric
        (55 boxes/img, precision 0.3%, normal_fpr 1.0).
        """
        from jingwei.losses import _balanced_bce

        bce = torch.ones(1, 4)
        is_pos = torch.tensor([[True, False, False, False]])
        base = float(_balanced_bce(bce, is_pos))                 # 1 (pos) + 1 (neg)
        heavier = float(_balanced_bce(bce, is_pos, neg_weight=2.5))
        self.assertAlmostEqual(base, 2.0, places=6)
        self.assertAlmostEqual(heavier, 1.0 + 2.5, places=6)
        self.assertGreater(heavier, base)
        # and it must not touch the positive side
        self.assertAlmostEqual(
            float(_balanced_bce(bce, is_pos, pos_weight=2.0)), 3.0, places=6)

    def test_59_objectness_reads_the_regression_stem(self):
        """The deployed objectness head may not be a bare 1x1 conv on the neck.

        A 1x1 conv cannot turn a smooth neck map into a one-cell peak, and the
        measured field was flat (obj ~ 0.5 on background).  ``obj_head_from``
        now defaults to ``reg``: two more depthwise convs give it a 9-cell
        receptive field (~72 px at stride 8) with no extra backbone work.
        """
        from jingwei.head import NMSFreeDetect

        cfg = load_config(CFG_P)
        cfg = cfg.copy()
        cfg.model.obj_head_from = "reg"
        head = NMSFreeDetect(cfg, [0, 96, 0, 288])
        for hd in head.heads:
            self.assertEqual(hd.obj_from, "reg")
            self.assertIsInstance(hd.obj_stem, torch.nn.Sequential)
            h = hd.obj_stem[0].pw.out_channels
            self.assertEqual(hd.obj_o2o.weight.shape[1], h,
                             "objectness must read the stem, not the neck")
            self.assertEqual(hd.obj_pred.weight.shape[1], h)
        # A/B path stays available.
        cfg.model.obj_head_from = "neck"
        head_n = NMSFreeDetect(cfg, [0, 96, 0, 288])
        self.assertIsInstance(head_n.heads[0].obj_stem, torch.nn.Identity)
        self.assertEqual(head_n.heads[0].obj_o2o.weight.shape[1], 96)

    def test_60_decode_reports_the_background_level(self):
        """`score_p99` separates "found a defect" from "the field is flat"."""
        from jingwei.head import NMSFreeDetect

        cfg = load_config(CFG_P)
        head = NMSFreeDetect(cfg, [0, 96, 0, 288]).eval()
        flat = 0.25                                     # sigmoid(0) * sigmoid(0)
        preds = [dict(cls=torch.zeros(1, cfg.data.nc, 40, 40),
                      reg=torch.ones(1, 4, 40, 40),
                      obj_o2o=torch.zeros(1, 1, 40, 40), obj=None),
                 dict(cls=torch.zeros(1, cfg.data.nc, 10, 10),
                      reg=torch.ones(1, 4, 10, 10),
                      obj_o2o=torch.zeros(1, 1, 10, 10), obj=None)]
        dets = head.decode(preds, (320, 320))
        st = head.last_stats
        self.assertAlmostEqual(st["score_max"], flat, places=3)
        self.assertAlmostEqual(st["score_p99"], flat, places=3)
        self.assertAlmostEqual(st["obj_mean"], 0.5, places=3)
        self.assertGreater(len(dets[0]["boxes"]), 0,
                           "score 0.25 > gate 0.15 -> boxes must be emitted")
        self.assertGreater(st["n_above"], 0)

    def test_61_the_deployed_gate_opens_on_the_shipped_config(self):
        """End-to-end cold start: >=1 box above the gate within 120 Adam steps.

        This is the regression for the real run that logged ``boxes/img = 0.0``
        for two epochs.  Measured with the shipped knobs: the gate is open by
        step ~25; with the v0.2.1 knobs (``obj_bias_init=-4``, neck 1x1
        objectness) it is *still shut after 300 steps* (score 0.03).
        """
        from jingwei.model import build_model
        from jingwei.losses import DetectionLoss

        cfg = load_config(CFG_P)
        torch.manual_seed(0)
        model = build_model(cfg).eval()
        imgs = torch.rand(2, 3, 320, 320)
        with torch.no_grad():
            feats = model.neck(model.backbone(imgs))
        head = model.head
        head.train()
        loss_fn = DetectionLoss(cfg)
        tgt = [dict(boxes=torch.tensor([[100., 100., 160., 160.]]),
                    labels=torch.tensor([3])),
               dict(boxes=torch.zeros(0, 4), labels=torch.zeros(0, dtype=torch.long))]
        opt = torch.optim.AdamW(head.parameters(), lr=float(cfg.train.detector_lr))
        thresh = float(cfg.fusion.final_score_thresh)
        self.assertAlmostEqual(thresh, float(cfg.model.score_thresh), places=9,
                               msg="one absolute threshold (CODE_REVIEW §2.13)")
        opened_at = None
        for it in range(120):
            opt.zero_grad(set_to_none=True)
            out = loss_fn(head(feats, training=True), tgt, (320, 320))
            out["total"].backward()
            opt.step()
            if it % 10 == 9:
                with torch.no_grad():
                    p = head(feats, training=False)
                    sc = torch.cat([(q["obj_o2o"].sigmoid() *
                                     q["cls"].sigmoid().max(1, keepdim=True).values).reshape(-1)
                                    for q in p])
                    if float(sc.max()) > thresh:
                        opened_at = it + 1
                        break
        self.assertIsNotNone(opened_at, "the deployed gate never opened in 120 steps")
        with torch.no_grad():
            dets = head.decode(head(feats, training=False), (320, 320))
        st = head.last_stats
        self.assertGreater(float(st["score_p99"]), 0.0)
        self.assertLessEqual(float(st["score_p99"]), float(st["score_max"]) + 1e-6)
        self.assertGreater(sum(len(d["boxes"]) for d in dets), 0,
                           "an open gate must produce boxes")

    def test_62_best_f1_reports_its_operating_point_precision(self):
        """precision must be reported at the F1-optimal conf, not only at the
        shipped one (the real run printed P=0.0026 at conf 0.15 while the model's
        own optimum was ``0.0867@0.74`` -- 0.3% vs ~5.5%)."""
        import numpy as np
        from evaluate import _best_f1, normal_fpr

        records = [[(0.9, True), (0.8, True), (0.7, False), (0.6, False)]]
        best_f1, thr, p, r = _best_f1(records, 2, with_pr=True)
        self.assertAlmostEqual(p, 1.0, places=6)
        self.assertAlmostEqual(r, 1.0, places=6)
        self.assertAlmostEqual(thr, 0.8, places=6)
        self.assertAlmostEqual(best_f1, 1.0, places=6)
        self.assertEqual(len(_best_f1(records, 2)), 2)

        class Stub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = torch.nn.Module()
                self.head.last_stats = dict(score_max=0.31, score_p99=0.28)

            def forward(self, x):
                return [dict(boxes=torch.ones(3, 4), scores=torch.ones(3)),
                        dict(boxes=torch.zeros(0, 4), scores=torch.zeros(0))]

        fpr, nfp, n, bpi, smax = normal_fpr(
            Stub(), [dict(image=torch.zeros(2, 3, 64, 64))], "cpu", return_extra=True)
        self.assertAlmostEqual(fpr, 0.5, places=6)
        self.assertAlmostEqual(bpi, 1.5, places=6)
        self.assertAlmostEqual(smax, 0.31, places=6)
        self.assertEqual((nfp, n), (1, 2))
        self.assertTrue(np.isfinite(bpi))

    def test_63_warm_start_survives_a_changed_head_shape(self):
        """`--resume runs/dino/dino_init.pt` must not crash after v0.2.2.

        v0.2.2 moved the objectness heads from the neck (``c`` channels) to the
        regression stem (``h = c*head_width``).  ``load_state_dict(strict=False)``
        only tolerates missing / unexpected *keys*, so it still raised
        ``RuntimeError: size mismatch for head.heads.0.obj_pred.weight ...``
        before a single batch ran.  Only those tensors are incompatible: the
        backbone/neck/other head weights are perfectly reusable, so the loader
        now skips them, says so, and downgrades a full checkpoint to a
        weight-only warm start (the optimizer state is indexed by parameter
        position and cannot be carried over either).
        """
        from _common import (load_state_dict_report, optimizer_state_compatible,
                             resume_checkpoint, split_shape_mismatch)
        from jingwei.model import build_model

        cfg = load_config(CFG_P)
        old_cfg = cfg.copy()
        old_cfg.model.obj_head_from = "neck"          # the pre-v0.2.2 layout
        old_cfg.model.obj_bias_init = -4.0
        torch.manual_seed(0)
        old = build_model(old_cfg)
        torch.manual_seed(0)
        new = build_model(cfg)

        old_sd = old.state_dict()
        self.assertEqual(tuple(old_sd["head.heads.0.obj_o2o.weight"].shape), (1, 72, 1, 1))
        self.assertEqual(tuple(old_sd["head.heads.1.obj_o2o.weight"].shape), (1, 288, 1, 1))

        keep, reshaped = split_shape_mismatch(new, old_sd)
        self.assertEqual(len(reshaped), 4, "2 scales x (obj_pred, obj_o2o)")
        self.assertEqual(len(keep), len(old_sd) - 4)
        # the biases (shape (1,)) are compatible and must survive
        self.assertIn("head.heads.0.obj_pred.bias", keep)

        msgs = load_state_dict_report(new, old_sd, tag="dino_init")
        self.assertTrue(any("shape changed" in m for m in msgs),
                        "a shape mismatch must be reported, not swallowed")
        # ... and the rest of the weights really were loaded into ``new``.
        loaded = new.state_dict()
        for key in ("head.heads.0.cls_pred.weight", "head.heads.1.reg_pred.weight",
                    "head.heads.0.obj_o2o.bias"):
            self.assertTrue(torch.equal(old_sd[key], loaded[key]), key)
        self.assertIn("head.heads.0.obj_stem.0.dw.weight", loaded,
                      "the new objectness stem exists and keeps its own init")

        # A *full* checkpoint of the old layout: weights load, optimizer state
        # must be dropped instead of raising or silently mis-assigning momentum.
        opt_old = torch.optim.AdamW(old.parameters(), lr=1e-3)
        state = dict(model=old_sd, optimizer=opt_old.state_dict(), train=dict(stage="detector"))
        opt_new = torch.optim.AdamW(new.parameters(), lr=1e-3)
        self.assertFalse(optimizer_state_compatible(opt_new, opt_old.state_dict()))
        info = resume_checkpoint(state, "detector", new, opt_new, None)
        self.assertFalse(info["full"])
        self.assertEqual(info["epoch"], 0)

        # Same architecture: the optimizer state is still restored (full resume).
        opt_same = torch.optim.AdamW(old.parameters(), lr=1e-3)
        opt_same.zero_grad()
        info2 = resume_checkpoint(dict(model=old_sd, optimizer=opt_same.state_dict(),
                                       train=dict(stage="detector")),
                                  "detector", old, torch.optim.AdamW(old.parameters(), lr=1e-3), None)
        self.assertTrue(info2["full"])

    # ------------------------------------------------ precision round 5 (v0.2.3)
    def test_64_defect_free_frames_are_mixed_in_as_pure_negatives(self):
        """A whole frame of normal fabric must be a training example.

        Stage A used to see defect images only, so every "background" cell it
        learned from was fabric inside a *defective* frame -- the model learned
        "something is always here" (normal_fpr = 1.0, 55 boxes/img).  Mixing in
        ``normaltrain`` frames with EMPTY targets fixes that prior directly.
        """
        from torch.utils.data import Dataset

        from jingwei.data import collate
        from _common import NegativeMixDataset

        class Fake(Dataset):
            def __init__(self, n, labelled):
                self.n, self.labelled = n, labelled

            def __len__(self):
                return self.n

            def __getitem__(self, i):
                d = dict(image=torch.zeros(3, 8, 8) + i * 0.01)
                if self.labelled:
                    d["boxes"] = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
                    d["labels"] = torch.tensor([7])
                return d

        pos, neg = Fake(16, True), Fake(5, False)

        only_neg = NegativeMixDataset(pos, neg, 1.0, seed=0)
        self.assertEqual(len(only_neg), len(pos), "epoch length follows the positives")
        for i in range(len(only_neg)):
            d = only_neg[i]
            self.assertEqual(tuple(d["boxes"].shape), (0, 4))
            self.assertEqual(int(d["labels"].numel()), 0)

        only_pos = NegativeMixDataset(pos, neg, 0.0, seed=0)
        for i in range(len(only_pos)):
            self.assertEqual(tuple(only_pos[i]["boxes"].shape), (1, 4))

        mixed = NegativeMixDataset(pos, neg, 0.25, seed=0)
        n_neg = sum(int(mixed[i]["boxes"].numel() == 0) for i in range(400))
        self.assertTrue(0.15 <= n_neg / 400.0 <= 0.35, f"got {n_neg}/400 negatives")
        # every sample carries both keys, or `collate` (which inspects batch[0])
        # raises KeyError on the others
        batch = collate([mixed[i] for i in range(4)])
        self.assertEqual(len(batch["boxes"]), 4)
        self.assertIn("labels", batch)
        self.assertEqual(batch["image"].shape[0], 4)

    def test_65_shipped_weights_fund_the_deployed_branches(self):
        """The o2m terms are training-only; obj_o2o decides whether a box exists.

        Measured on a real run (epoch 9, raw per-term losses).  With the weights
        shipped before v0.2.3 the split was box_o2m 38% / obj_o2m 24% /
        box_o2o 9% / cls_o2m 7% / cls_o2o 5% / **obj_o2o 2%** -- 62% of the
        gradient went to machinery that does not run at inference, which is why
        the background objectness never came down.
        """
        RAW = dict(cls_o2m=0.458, box_o2m=1.311, obj_o2m=1.663,
                   cls_o2o=0.302, box_o2o=0.450, obj_o2o=0.101)
        for name in ("jingwei_n", "jingwei_p", "jingwei_s"):
            cfg = load_config(str(ROOT / f"configs/{name}.yaml"))
            w = {k: float(v) for k, v in dict(cfg.train.loss_weights).items()}
            contrib = {k: w[k] * v for k, v in RAW.items()}
            tot = sum(contrib.values())
            dep = sum(contrib[k] for k in ("obj_o2o", "cls_o2o", "box_o2o"))
            self.assertGreaterEqual(contrib["obj_o2o"] / tot, 0.15,
                                    f"{name}: obj_o2o is starved")
            self.assertGreaterEqual(dep / tot, 0.45,
                                    f"{name}: deployed branches own {dep / tot:.0%}")
            self.assertGreaterEqual(float(cfg.train.obj_o2o_neg_weight), 1.5,
                                    "background pressure must stay > 1")
            self.assertGreater(float(cfg.train.obj_o2m_neg_weight), 1.0)
            self.assertGreater(float(cfg.train.get("normal_image_frac", 0.0)), 0.0,
                               "defect-free frames must be mixed in")


if __name__ == "__main__":
    unittest.main(verbosity=2)
