"""Open-set plumbing, NOT real DINOv2 accuracy or GTX1660 speed validation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools"), str(ROOT / "tests")]
from helpers import local_config
from jingwei.data import MixedFabricDataset, YOLODefectDataset, geometric_augment
from jingwei.image_io import imwrite_unicode
from jingwei.novelty import NormalPrototypes
from jingwei.utils import torch_load
from calibrate_anomaly import conformal_threshold
from _common import configure_inference, load_model


class TestOpenSet(unittest.TestCase):
    def test_only_explicit_normals_update_prototypes_and_updates_invalidate_calibration(self):
        normal = torch.zeros(1, 3, 4, 4, requires_grad=True)
        with torch.no_grad():
            normal[:, 0] = 1
        module = NormalPrototypes(3, k=4)
        module.update(normal)
        self.assertIsNone(normal.grad)
        self.assertFalse(module.centers.requires_grad)
        torch.testing.assert_close(module.score(normal), torch.zeros(1, 4, 4))
        module.calibrated.fill_(True)
        count = module.updates.item()
        module.update(normal[:0])
        self.assertEqual(module.updates.item(), count)
        self.assertTrue(module.calibrated)
        module.update(normal)
        self.assertFalse(module.calibrated)

    def test_novel_patch_region_coordinates_and_normal_no_alert(self):
        normal = torch.zeros(1, 3, 4, 4)
        normal[:, 0] = 1
        module = NormalPrototypes(3, k=2)
        module.update(normal)
        module.threshold.fill_(.5)
        self.assertEqual(len(module.regions(normal, (64, 128))[0]["unknown_boxes"]), 0)
        novel = normal.clone()
        novel[:, :, :2, :2] = 0
        novel[:, 1, :2, :2] = 1
        regions = module.regions(novel, (64, 128))[0]
        torch.testing.assert_close(regions["unknown_boxes"], torch.tensor([[0., 0., 64., 32.]]))
        self.assertGreater(float(regions["unknown_scores"][0]), .9)

    def test_threshold_image_max_order_statistic_requires_enough_samples(self):
        self.assertEqual(conformal_threshold([.1, .2, .3], .25), .3)
        with self.assertRaisesRegex(ValueError, "Too few"):
            conformal_threshold([.1, .2, .3], .01)
        with self.assertRaises(ValueError):
            conformal_threshold([], .01)

    def test_fp32_statistics_survive_half_conversion(self):
        module = NormalPrototypes(3, k=2)
        module.centers.fill_(.123456789)
        module.threshold.fill_(.123456789)
        centers, threshold = module.centers.clone(), module.threshold.clone()
        module.half()
        self.assertEqual(module.centers.dtype, torch.float32)
        torch.testing.assert_close(module.centers, centers, rtol=0, atol=0)
        torch.testing.assert_close(module.threshold, threshold, rtol=0, atol=0)

    def test_mixed_dataset_negatives_have_empty_boxes_and_share_rng(self):
        class Data:
            def __init__(self, normal=False):
                self.rng = np.random.default_rng(9)
                self.normal = normal
            def __len__(self): return 4
            def __getitem__(self, i):
                x = dict(image=torch.zeros(3, 32, 32), path=str(i))
                if not self.normal:
                    x.update(boxes=torch.tensor([[1., 1., 4., 4.]]), labels=torch.tensor([1]))
                return x
        mixed = MixedFabricDataset(Data(), Data(True), .5)
        self.assertEqual(len(mixed), 8)
        self.assertFalse(mixed[0]["is_normal"])
        self.assertTrue(mixed[7]["is_normal"])
        self.assertEqual(mixed[7]["boxes"].shape, (0, 4))
        rng = np.random.default_rng(10)
        mixed.rng = rng
        self.assertIs(mixed.defects.rng, rng)
        self.assertIs(mixed.normals.rng, rng)

    def test_density_scaling_is_not_cancelled_by_final_resize(self):
        class RNG:
            calls = 0
            def random(self):
                self.calls += 1
                return 1. if self.calls < 4 else 0.
            def uniform(self, low, high, size): return np.array([1.2, .8])
        boxes = np.array([[20., 20., 60., 60.]], np.float32)
        image = np.full((100, 100, 3), 120, np.uint8)
        result, boxes = geometric_augment(image, boxes, RNG(), 100)
        np.testing.assert_allclose(boxes, [[14., 26., 62., 58.]], atol=1e-5)
        self.assertEqual(result.shape, image.shape)

    def test_one_encoder_pass_and_normal_only_mask_with_detector_gradients(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = local_config(tmp)
            cfg = yaml.safe_load(cfg_path.read_text())
            cfg["unknown"].update(enabled=True, prototypes=4)
            cfg_path.write_text(yaml.safe_dump(cfg))
            model, _ = load_model(str(cfg_path))
            model.train()
            captured = {}
            original = model.backbone.encoder.get_intermediate_layers
            def encode(*args, **kwargs):
                maps = original(*args, **kwargs)
                captured["normal"] = maps[1][:1].detach().clone()
                return maps
            targets = [dict(boxes=torch.empty(0, 4), labels=torch.empty(0, dtype=torch.long)),
                       dict(boxes=torch.tensor([[20., 20., 60., 60.]]), labels=torch.tensor([1]))]
            with patch.object(model.backbone.encoder, "get_intermediate_layers", side_effect=encode) as call:
                result = model(torch.rand(2, 3, 128, 128), targets, normal_mask=torch.tensor([True, False]))
                result["losses"]["total"].backward()
                self.assertEqual(call.call_count, 1)
            expected = NormalPrototypes(12, k=4)
            expected.update(captured["normal"])
            torch.testing.assert_close(model.novelty.centers, expected.centers)
            self.assertTrue(all(p.grad is None for p in model.backbone.encoder.parameters()))
            self.assertIsNotNone(model.backbone.adapters[1].conv.weight.grad)
            model.novelty.threshold.fill_(0.)
            model.novelty.calibrated.fill_(True)
            model.eval()
            with patch.object(model.backbone.encoder, "get_intermediate_layers", wraps=original) as call:
                with torch.no_grad():
                    detections = model(torch.rand(1, 3, 128, 128))
                self.assertEqual(call.call_count, 1)
                self.assertIn("unknown_boxes", detections[0])

    def test_nested_labels_keep_relative_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder, label in (("cloth_a", 1), ("cloth_b", 2)):
                images = root / "images/train" / folder
                labels = root / "labels/train" / folder
                images.mkdir(parents=True)
                labels.mkdir(parents=True)
                imwrite_unicode(images / "same.png", np.zeros((32, 32, 3), np.uint8))
                (labels / "same.txt").write_text(f"{label} 0.5 0.5 0.5 0.5\n")
            dataset = YOLODefectDataset(root / "images/train", img_size=128, augment=False)
            self.assertEqual([dataset[i]["labels"].tolist() for i in range(2)], [[1], [2]])

    def test_training_calibration_prediction_evaluation_and_acceptance_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg_path = local_config(root)
            cfg = yaml.safe_load(cfg_path.read_text())
            cfg["unknown"]["enabled"] = True
            cfg["unknown"]["prototypes"] = 4
            cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            data = {}
            for split in ("train", "val", "unknownval", "normaltrain", "normalcal", "normalval"):
                images = root / "images" / split
                labels = root / "labels" / split
                images.mkdir(parents=True)
                labels.mkdir(parents=True)
                data[split] = str(images)
                for i in range(3):
                    imwrite_unicode(images / f"布面{i}.png", np.full((64, 96, 3), 80 + i * 20, np.uint8))
                    if not split.startswith("normal"):
                        (labels / f"布面{i}.txt").write_text("1 0.5 0.5 0.25 0.25\n")
            data_path = root / "data.yaml"
            data_path.write_text(yaml.safe_dump(data), encoding="utf-8")
            env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONIOENCODING": "utf-8"}
            def run(tool, *extra, code=0):
                result = subprocess.run([sys.executable, f"tools/{tool}.py", "--cfg", str(cfg_path),
                                         "--device", "cpu", *extra], cwd=ROOT, env=env,
                                        capture_output=True, text=True, timeout=90)
                self.assertEqual(result.returncode, code, result.stdout + result.stderr)
                return result.stdout + result.stderr
            out = root / "training"
            run("train_detector", "--data", str(data_path), "--epochs", "1", "--batch", "2",
                "--no-augment", "--num-workers", "0", "--val-workers", "0", "--out", str(out))
            checkpoint = out / "last.pt"
            self.assertGreater(int(torch_load(checkpoint)["model"]["novelty.updates"]), 0)
            model, _ = load_model(str(cfg_path), str(checkpoint))
            with self.assertRaisesRegex(ValueError, "not calibrated"):
                configure_inference(model)
            configure_inference(model, known_only=True)
            calibrated = root / "calibrated.pt"
            run("calibrate_anomaly", "--data", str(data_path), "--ckpt", str(checkpoint),
                "--out", str(calibrated), "--target-fpr", ".25")
            state = torch_load(calibrated)
            self.assertTrue(state["model"]["novelty.calibrated"])
            prediction = root / "predictions"
            run("predict", "--ckpt", str(calibrated), "--source", data["val"], "--out", str(prediction))
            result = json.loads((prediction / "results.json").read_text(encoding="utf-8"))
            self.assertIn("unknown_regions", result["布面0.png"])
            metrics = run("evaluate", "--ckpt", str(calibrated), "--data", str(data_path), "--workers", "0")
            self.assertIn("unknown active: True", metrics)
            self.assertIn("box_recall", metrics)
            report = root / "bench.json"
            run("bench_device", "--ckpt", str(calibrated), "--source", data["val"],
                "--iters", "2", "--warmup", "0", "--require-target", "--out", str(report), code=2)
            report = json.loads(report.read_text())
            self.assertFalse(report["gtx1660_full_acceptance"])
            self.assertTrue(report["unknown_enabled"])
            self.assertEqual(report["scope"], "real_frame_preprocess_h2d_model_postprocess_d2h")
            model, _ = load_model(str(cfg_path), str(calibrated))
            model.backbone.encoder_size += 14
            with self.assertRaisesRegex(ValueError, "sizes differ"):
                configure_inference(model)
            model.backbone.encoder_size -= 14
            model.novelty.precision_bits.fill_(16)
            with self.assertRaisesRegex(ValueError, "precision differs"):
                configure_inference(model)


if __name__ == "__main__":
    unittest.main()
