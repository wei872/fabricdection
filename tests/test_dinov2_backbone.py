"""Direct-backbone contracts tested without network / official weight downloads."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools"), str(ROOT / "tests")]
from jingwei import build_model, load_config
from jingwei.dinov2_backbone import DINOv2Backbone
from jingwei.head import DecoupledHead
from jingwei.utils import classwise_nms, torch_load
from _common import load_model
from helpers import FakeEncoder


class TestDirectDINO(unittest.TestCase):
    def cfg(self):
        cfg = load_config(ROOT / "configs/jingwei_dinov2.yaml")
        cfg.input_size = [128, 128]
        cfg.model.dino_input_size = 56
        return cfg

    def test_pretrained_request_shapes_freeze_and_adapter_gradients(self):
        cfg = self.cfg()
        with patch("torch.hub.load", return_value=FakeEncoder()) as hub:
            backbone = DINOv2Backbone(cfg)
        hub.assert_called_once_with("facebookresearch/dinov2", "dinov2_vitb14",
                                    source="github", pretrained=True)
        backbone.train()
        self.assertFalse(backbone.encoder.training)
        self.assertTrue(backbone.adapters.training)
        feats = backbone(torch.rand(2, 3, 128, 160))
        for name, c, stride in zip(("d", "p3", "p4", "p5"),
                                  backbone.out_channels, (4, 8, 16, 32)):
            self.assertEqual(feats[name].shape, (2, c, 128 // stride, 160 // stride))
        sum(f.square().mean() for f in feats.values()).backward()
        self.assertTrue(all(p.grad is None for p in backbone.encoder.parameters()))
        self.assertTrue(all(p.grad is not None for p in backbone.adapters.parameters()))

    def test_optional_encoder_finetuning(self):
        cfg = self.cfg()
        cfg.model.dino_freeze = False
        with patch("torch.hub.load", return_value=FakeEncoder()):
            model = DINOv2Backbone(cfg).train()
        model(torch.rand(2, 3, 128, 128))["p3"].square().mean().backward()
        self.assertTrue(model.encoder.training)
        self.assertIsNotNone(model.encoder.proj.weight.grad)

    def test_detector_loss_backward_and_eval(self):
        with patch("torch.hub.load", return_value=FakeEncoder()):
            model = build_model(self.cfg()).train()
        targets = [dict(boxes=torch.tensor([[24., 24., 80., 72.]]),
                        labels=torch.tensor([1])) for _ in range(2)]
        result = model(torch.rand(2, 3, 128, 128), targets)
        self.assertTrue(torch.isfinite(result["losses"]["total"]))
        result["losses"]["total"].backward()
        self.assertIsNotNone(model.head.heads[0].cls_pred.weight.grad)
        self.assertIsNotNone(model.backbone.adapters[1].conv.weight.grad)
        model.eval()
        self.assertEqual(len(model(torch.rand(2, 3, 128, 128))), 2)

    def test_loading_failure_is_not_silent_random_fallback(self):
        with patch("torch.hub.load", side_effect=OSError("weights unavailable")):
            with self.assertRaisesRegex(RuntimeError, "No random-backbone fallback"):
                DINOv2Backbone(self.cfg())

    def test_checkpoint_restores_encoder_without_pretrained_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.cfg()
            path = Path(tmp) / "cfg.yaml"
            path.write_text(yaml.safe_dump(json.loads(json.dumps(cfg))), encoding="utf-8")
            with patch("torch.hub.load", return_value=FakeEncoder()):
                model = build_model(copy.deepcopy(cfg))
            ckpt = Path(tmp) / "det.pt"
            torch.save({"model": model.state_dict()}, ckpt)
            with patch("torch.hub.load", return_value=FakeEncoder()) as hub:
                restored, loaded_cfg = load_model(str(path), str(ckpt))
            self.assertFalse(hub.call_args.kwargs["pretrained"])
            self.assertTrue(loaded_cfg.model.dino_pretrained)
            for k, v in model.state_dict().items():
                torch.testing.assert_close(v, restored.state_dict()[k])
            incomplete = model.state_dict()
            del incomplete["backbone.encoder.proj.weight"]
            torch.save({"model": incomplete}, ckpt)
            with patch("torch.hub.load", return_value=FakeEncoder()):
                with self.assertRaises(RuntimeError):
                    load_model(str(path), str(ckpt))

    def test_local_official_weight_loading_is_strict(self):
        cfg = self.cfg()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "encoder.pth"
            original = FakeEncoder()
            torch.save(original.state_dict(), path)
            cfg.model.dino_weights = str(path)
            with patch("torch.hub.load", return_value=FakeEncoder()) as hub:
                backbone = DINOv2Backbone(cfg)
            self.assertFalse(hub.call_args.kwargs["pretrained"])
            torch.testing.assert_close(backbone.encoder.proj.weight, original.proj.weight)
            torch.save({"wrong": torch.ones(1)}, path)
            with patch("torch.hub.load", return_value=FakeEncoder()):
                with self.assertRaises(RuntimeError):
                    DINOv2Backbone(cfg)

    def test_softplus_regression_has_gradient_for_negative_logits(self):
        head = DecoupledHead(16, 19, 1., False, "softplus").eval()
        nn.init.zeros_(head.reg_pred.weight)
        nn.init.constant_(head.reg_pred.bias, -5.)
        out = head(torch.rand(2, 16, 8, 8))["reg"]
        self.assertTrue((out > 0).all())
        out.sum().backward()
        self.assertTrue((head.reg_pred.bias.grad > 0).all())

    def test_classwise_nms_keeps_different_classes(self):
        boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.],
                              [0., 0., 10., 10.], [30., 30., 40., 40.]])
        scores = torch.tensor([.9, .8, .7, .6])
        labels = torch.tensor([1, 1, 2, 1])
        self.assertEqual(classwise_nms(boxes, scores, labels, .5).tolist(), [0, 2, 3])
        self.assertEqual(classwise_nms(boxes[:0], scores[:0], labels[:0]).numel(), 0)

    def test_decode_nms_is_opt_in_and_suppresses_cross_scale_duplicates(self):
        from jingwei.utils import boxes_to_distances
        from jingwei.losses import _grid_points
        with patch("torch.hub.load", return_value=FakeEncoder()):
            model = build_model(self.cfg())
        predictions = []
        for stride in (8, 32):
            pts = _grid_points(1, 2, stride, "cpu", torch.float32)
            cls = torch.full((1, 19, 1, 2), -10.)
            cls[:, 1] = 10.
            box = torch.tensor([[-32., -32., 96., 96.]]).expand(2, 4)
            distances = boxes_to_distances(pts, box).T.reshape(1, 4, 1, 2) / stride
            predictions.append({"cls": cls, "reg": distances,
                                "obj_o2o": torch.full((1, 1, 1, 2), 10.)})
        self.assertEqual(len(model.head.decode(predictions, (128, 128))[0]["boxes"]), 1)
        model.head.nms_iou = 0.
        self.assertEqual(len(model.head.decode(predictions, (128, 128))[0]["boxes"]), 4)

    def test_direct_cli_fresh_resume_and_pretrained(self):
        # A tiny local hub fixture validates the REAL CLI, optimizer and scheduler.
        # It is not a real pretrained model; never use it for accuracy conclusions.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hub = root / "hub"
            hub.mkdir()
            (hub / "hubconf.py").write_text('''
import torch
from torch import nn
class Fake(nn.Module):
    embed_dim = 12
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(nn.Linear(12, 12) for _ in range(12))
        self.proj = nn.Conv2d(3, 12, 14, 14)
    def get_intermediate_layers(self, x, n, reshape):
        f = self.proj(x)
        return tuple(f for _ in n)
def dinov2_vitb14(pretrained=True):
    print("FIXTURE pretrained=" + str(pretrained))
    return Fake()
''', encoding="utf-8")
            cfg = self.cfg()
            cfg.model.dino_repo = str(hub)
            config = root / "cfg.yaml"
            config.write_text(yaml.safe_dump(json.loads(json.dumps(cfg))), encoding="utf-8")
            env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
            def run(out, *extra):
                cmd = [sys.executable, "tools/train_detector.py", "--cfg", str(config),
                       "--smoke", "2", "--epochs", "2", "--batch", "2", "--device", "cpu",
                       "--amp", "off", "--val-interval", "1", "--out", str(out), *extra]
                result = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True,
                                        text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return result.stdout
            out = root / "first"
            text = run(out, "--epochs", "1", "--no-augment")
            self.assertIn("DIRECT pretrained DINOv2 backbone", text)
            self.assertIn("FIXTURE pretrained=True", text)
            self.assertIn("best_metric=best_f1", text)
            self.assertIn("TP=", text)
            state = torch_load(out / "last.pt", map_location="cpu")
            self.assertEqual(state["train"]["backbone"], "dinov2")
            self.assertLess(state["train"]["iter"], 2)
            text = run(root / "resume", "--resume", str(out / "last.pt"))
            self.assertIn("FIXTURE pretrained=False", text)
            self.assertIn("mode=full", text)
            self.assertIn("start_epoch=2/2", text)
            text = run(root / "warm", "--pretrained", str(out / "last.pt"), "--lr", "0.0001")
            self.assertIn("mode=weights-only", text)
            self.assertIn("global_iter=0", text)
            fresh = torch_load(root / "warm" / "last.pt", map_location="cpu")
            self.assertEqual(fresh["scheduler"]["base_lrs"], [.0001])


if __name__ == "__main__":
    unittest.main()
