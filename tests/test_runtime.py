"""Current pipeline regression tests: data, metrics, checkpoints and public CLIs."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools"), str(ROOT / "tests")]
from helpers import FakeEncoder, local_config
from jingwei import build_model, load_config
from jingwei.data import YOLODefectDataset, collate, fabric_photometric
from jingwei.image_io import imread_unicode, imwrite_unicode
from _common import (collect_rng_state, load_model, restore_rng_state,
                     resume_checkpoint, save_checkpoint, subset_indices)
from jingwei.utils import torch_load
from evaluate import _pr_at_thresholds, validate_detector
from predict import predict_image, source_images
from export import ExportWrapper


class TestRuntime(unittest.TestCase):
    def cfg(self):
        cfg = load_config(ROOT / "configs/jingwei_dinov2.yaml")
        cfg.unknown.enabled = False
        cfg.input_size = [128, 128]
        cfg.model.dino_input_size = 56
        return cfg

    def test_tensor_export_and_native_inference_match(self):
        with patch("torch.hub.load", return_value=FakeEncoder()):
            model = build_model(self.cfg()).eval()
        x = torch.rand(2, 3, 128, 128)
        tensors = ExportWrapper(model)(x)
        self.assertEqual(len(tensors), 6)
        native = model(x)
        restored = model.postprocess_from_tensors(tensors, (128, 128))
        for a, b in zip(native, restored):
            self.assertEqual(a.keys(), b.keys())
            for k in a:
                torch.testing.assert_close(a[k], b[k])
        with self.assertRaises(ValueError):
            model.postprocess_from_tensors(tensors + (x,), (128, 128))

    def test_predict_boxes_use_original_coordinates(self):
        class Static(torch.nn.Module):
            def forward(self, x):
                return [dict(boxes=x.new_tensor([[32., 16., 96., 80.]]),
                             scores=x.new_tensor([.8]), labels=torch.tensor([1]))]
        image = np.zeros((256, 512, 3), np.uint8)
        result = predict_image(Static(), image, 128, "cpu")
        torch.testing.assert_close(result["boxes"], torch.tensor([[128., 32., 384., 160.]]))

    def test_unicode_dataset_and_non_square_resize(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, labels = root / "images/train", root / "labels/train"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            name = "布面 缺陷"
            image = np.full((40, 100, 3), 127, np.uint8)
            self.assertTrue(imwrite_unicode(images / f"{name}.png", image))
            (labels / f"{name}.txt").write_text("1 0.5 0.5 0.2 0.5\n", encoding="utf-8")
            ds = YOLODefectDataset(images, img_size=128, augment=False)
            item = ds[0]
            torch.testing.assert_close(item["boxes"], torch.tensor([[51.2, 32., 76.8, 96.]]))
            self.assertEqual(item["labels"].tolist(), [1])
            self.assertEqual(collate([item, item])["image"].shape, (2, 3, 128, 128))
            self.assertEqual(imread_unicode(images / f"{name}.png").shape, image.shape)

    def test_photometric_augmentation_accepts_long_frames(self):
        image = np.full((104, 244, 3), 127, np.uint8)
        for seed in range(8):
            result = fabric_photometric(image.copy(), np.random.default_rng(seed))
            self.assertEqual(result.shape, image.shape)
            self.assertEqual(result.dtype, np.uint8)

    def test_numpy_rng_checkpoint_roundtrip(self):
        for kind in (np.random.PCG64, np.random.Philox, np.random.SFC64, np.random.MT19937):
            ds = type("Dataset", (), {})()
            ds.rng = np.random.Generator(kind(12))
            state = collect_rng_state(ds)
            expected = ds.rng.random(8)
            restore_rng_state(state, ds)
            np.testing.assert_array_equal(ds.rng.random(8), expected)

    def test_full_checkpoint_restores_optimizer_scheduler_and_rng(self):
        import copy
        import random
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        scheduler.step()
        state = copy.deepcopy(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                   scheduler=scheduler.state_dict(), rng=collect_rng_state(),
                                   train=dict(backbone="dinov2", epoch=3, iter=6)))
        expected_random = (random.random(), np.random.rand(), torch.rand(4))
        restored = torch.nn.Linear(2, 1)
        opt = torch.optim.AdamW(restored.parameters(), lr=.8)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
        info = resume_checkpoint(state, restored, opt, sched)
        self.assertTrue(info["full"])
        self.assertEqual((info["epoch"], info["iter"]), (3, 6))
        self.assertEqual(scheduler.state_dict(), sched.state_dict())
        self.assertEqual(opt.param_groups[0]["lr"], optimizer.param_groups[0]["lr"])
        for source, target in zip(model.parameters(), restored.parameters()):
            torch.testing.assert_close(source, target)
            torch.testing.assert_close(optimizer.state[source]["exp_avg"], opt.state[target]["exp_avg"])
        self.assertEqual(random.random(), expected_random[0])
        self.assertEqual(np.random.rand(), expected_random[1])
        torch.testing.assert_close(torch.rand(4), expected_random[2])

    def test_resume_rejects_weight_only_but_explicit_warmstart_accepts_it(self):
        model = torch.nn.Linear(2, 1)
        opt = torch.optim.AdamW(model.parameters())
        state = dict(model=model.state_dict())
        with self.assertRaisesRegex(ValueError, "--pretrained"):
            resume_checkpoint(state, model, opt)
        self.assertFalse(resume_checkpoint(state, model)["full"])

    def test_failed_atomic_save_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "last.pt"
            save_checkpoint(dict(marker=17), path)
            def fail_save(state, temporary):
                Path(temporary).write_bytes(b"incomplete checkpoint")
                raise OSError("disk full")
            with patch("_common.torch.save", side_effect=fail_save):
                with self.assertRaisesRegex(OSError, "disk full"):
                    save_checkpoint(dict(marker=99), path)
            self.assertEqual(torch_load(path)["marker"], 17)
            self.assertFalse(path.with_name(path.name + ".tmp").exists())

    def test_video_source_streams_unicode_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "布面视频.avi"
            # OpenCV's writer may not support Unicode on Windows; create an
            # ASCII temporary video and rename it before testing the reader.
            ascii_path = Path(tmp) / "input.avi"
            writer = cv2.VideoWriter(str(ascii_path), cv2.VideoWriter_fourcc(*"MJPG"), 5., (64, 32))
            self.assertTrue(writer.isOpened())
            try:
                for value in (0, 64, 128):
                    writer.write(np.full((32, 64, 3), value, np.uint8))
            finally:
                writer.release()
            ascii_path.rename(path)
            frames = list(source_images(path))
            self.assertEqual([name for name, _ in frames], [f"frames/{i:06d}.jpg" for i in range(3)])
            self.assertTrue(all(frame.shape == (32, 64, 3) for _, frame in frames))

    def test_directory_source_preserves_same_named_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("first", "second"):
                (root / name).mkdir()
                imwrite_unicode(root / name / "布匹.png", np.zeros((32, 32, 3), np.uint8))
            self.assertEqual([name for name, _ in source_images(root)],
                             ["first/布匹.png", "second/布匹.png"])

    def test_subset_indices_reproducible(self):
        self.assertEqual(subset_indices(100, 10, 7), subset_indices(100, 10, 7))
        self.assertNotEqual(subset_indices(100, 10, 7), subset_indices(100, 10, 8))
        self.assertIsNone(subset_indices(10, 0, 7))

    def test_metrics_match_same_class_before_selecting_iou(self):
        class Static(torch.nn.Module):
            def forward(self, x):
                # Exact overlap with the OTHER class must not hide a valid
                # IoU=.9 same-class match. Both GTs occupy overlapping regions.
                return [dict(boxes=torch.tensor([[0., 0., 10., 10.]]),
                             scores=torch.tensor([.9]), labels=torch.tensor([1]))]
        batch = dict(image=torch.rand(1, 3, 128, 128),
                     boxes=[torch.tensor([[0., 0., 10., 10.], [0., 0., 9., 10.]])],
                     labels=[torch.tensor([0, 1])])
        m = validate_detector(Static(), [batch], None, "cpu", self.cfg())
        self.assertEqual(m["tp"], 1)
        self.assertEqual(m["fp"], 0)
        self.assertEqual(m["fn"], 1)
        self.assertEqual(m["precision"], 1.)
        self.assertTrue(math.isnan(m["normal_fpr"]))
        self.assertEqual(m["normal_n"], 0)

    def test_no_ground_truth_still_reports_false_positives_at_thresholds(self):
        result = _pr_at_thresholds([[(.9, False)]], 0, [.5])[.5]
        self.assertEqual(result["fp"], 1)
        self.assertEqual(result["tp"], 0)
        self.assertEqual(result["recall"], 0.)

    def test_rejects_legacy_configurations(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = yaml.safe_load((ROOT / "configs/jingwei_dinov2.yaml").read_text(encoding="utf-8"))
            path = Path(tmp) / "old.yaml"
            raw["model"]["backbone"] = "weavecsp"
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "legacy configuration"):
                load_config(path)

    def test_rejects_invalid_input_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = yaml.safe_load((ROOT / "configs/jingwei_dinov2.yaml").read_text(encoding="utf-8"))
            path = Path(tmp) / "bad.yaml"
            raw["input_size"] = [127, 128]
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "input_size"):
                load_config(path)

    def test_all_public_help_commands_work_without_model_download(self):
        for script in ("train_detector", "evaluate", "predict", "bench_device", "export", "calibrate_anomaly"):
            result = subprocess.run([sys.executable, f"tools/{script}.py", "--help"],
                                    cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("usage:", result.stdout)
            self.assertNotIn("--realtime-1660", result.stdout)
            self.assertNotIn("--emit-unknown", result.stdout)

    def test_prediction_evaluation_and_benchmark_clis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = local_config(root)
            model, cfg = load_model(str(config))
            checkpoint = root / "detector.pt"
            save_checkpoint({"model": model.state_dict()}, checkpoint)
            images, labels = root / "images/val", root / "labels/val"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            imwrite_unicode(images / "布匹.png", np.full((64, 100, 3), 127, np.uint8))
            (labels / "布匹.txt").write_text("1 0.5 0.5 0.25 0.25\n", encoding="utf-8")
            data = root / "data.yaml"
            data.write_text(yaml.safe_dump({"train": str(images), "val": str(images)}), encoding="utf-8")
            env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                   "PYTHONIOENCODING": "utf-8"}
            def run(script, *extra):
                result = subprocess.run([sys.executable, f"tools/{script}.py", "--cfg", str(config),
                                         "--ckpt", str(checkpoint), "--device", "cpu", *extra],
                                        cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return result.stdout
            prediction = root / "predictions"
            run("predict", "--source", str(images), "--out", str(prediction))
            results = json.loads((prediction / "results.json").read_text(encoding="utf-8"))
            self.assertIn("布匹.png", results)
            self.assertTrue((prediction / "布匹.png").exists())
            text = run("evaluate", "--data", str(data), "--workers", "0", "--diag")
            self.assertIn("normal FPR: N/A", text)
            self.assertIn("GT=1", text)
            data.write_text(yaml.safe_dump({"val": str(images), "normalval": str(images)}), encoding="utf-8")
            text = run("evaluate", "--data", str(data), "--workers", "0")
            self.assertNotIn("normal FPR: N/A", text)
            self.assertRegex(text, r"normal FPR: [0-9.]+ \([01]/1\)")
            bench = root / "bench.json"
            run("bench_device", "--warmup", "0", "--iters", "1", "--out", str(bench))
            self.assertGreater(json.loads(bench.read_text())["p50_ms"], 0.)
            # Strict loading prevents a partial/old backbone checkpoint from
            # silently becoming a random frozen feature extractor.
            save_checkpoint({"model": {"unrelated": torch.ones(1)}}, checkpoint)
            with self.assertRaises(RuntimeError):
                load_model(str(config), str(checkpoint))


if __name__ == "__main__":
    unittest.main()
