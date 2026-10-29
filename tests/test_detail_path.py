"""Native-pixel information routes, not real-data quality/FPS guarantees."""
import copy
import importlib.util
import tempfile
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools'), str(ROOT / 'tests')]
from helpers import FakeEncoder
from jingwei import build_model, load_config
from jingwei.detail import LocalDetail
from jingwei.novelty import NormalPrototypes, texture_descriptor
from _common import restore_detector_weights


class ConstantEncoder(FakeEncoder):
    def get_intermediate_layers(self, x, n, reshape):
        f = x.new_ones(len(x), self.embed_dim, x.shape[-2] // 14, x.shape[-1] // 14)
        return (f, f, f, f)


class TestNativeDetail(unittest.TestCase):
    def config(self, detail=True):
        cfg = load_config(ROOT / 'configs/jingwei_dinov2.yaml')
        cfg.input_size = [128, 128]
        cfg.model.dino_input_size = 56
        cfg.model.local_detail = detail
        cfg.unknown.texture_detail = detail
        return cfg

    def model(self, cfg, encoder=None):
        with patch('torch.hub.load', return_value=encoder or FakeEncoder()):
            return build_model(cfg)

    def test_space_to_depth_exposes_every_input_pixel(self):
        x = torch.randn(1, 3, 128, 160)
        detail = LocalDetail(48).eval()
        torch.testing.assert_close(F.pixel_shuffle(detail.pack(x), 2), x, rtol=0, atol=0)
        self.assertEqual(detail(x).shape, (1, 48, 16, 20))
        self.assertLess(sum(p.numel() for p in detail.parameters()), 5000)

    def test_known_features_change_even_when_all_vit_maps_are_identical(self):
        torch.manual_seed(11)
        x = torch.full((1, 3, 128, 128), .5)
        y = x.clone()
        y[:, :, 40:48:2, 40:48] = 1.
        y[:, :, 41:48:2, 40:48] = 0.
        for enabled in (False, True):
            model = self.model(self.config(enabled), ConstantEncoder()).eval()
            a, b = model.backbone(x), model.backbone(y)
            torch.testing.assert_close(a['p4'], b['p4'], rtol=0, atol=0)
            if enabled:
                self.assertGreater(float((a['p3'] - b['p3']).abs().max()), 0.)
            else:
                torch.testing.assert_close(a['p3'], b['p3'], rtol=0, atol=0)

    def test_detail_learns_while_vit_remains_frozen(self):
        model = self.model(self.config()).train()
        model.backbone(torch.rand(2, 3, 128, 128))['p3'].square().mean().backward()
        branch = model.backbone.local_detail
        self.assertGreater(float(branch.layers[0].conv.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(branch.gate.grad.abs()), 0.)
        self.assertTrue(all(p.grad is None for p in model.backbone.encoder.parameters()))

    def test_unknown_detects_native_texture_change_with_constant_semantics(self):
        image = torch.full((1, 3, 32, 32), .5)
        raw = torch.ones(1, 12, 1, 1)
        novelty = NormalPrototypes(12, k=2, texture_detail=True)
        novelty.update(raw, texture_descriptor(image))
        torch.testing.assert_close(novelty.score(raw, texture_descriptor(image)), torch.zeros(1, 4, 4))
        changed = image.clone()
        changed[:, :, 8:16:2, 8:16] = 0.
        changed[:, :, 9:16:2, 8:16] = 1.
        novelty.threshold.fill_(.2)
        result = novelty.regions(raw, (32, 32), texture_descriptor(changed))[0]
        torch.testing.assert_close(result['unknown_boxes'], torch.tensor([[8., 8., 16., 16.]]))
        self.assertGreater(float(result['unknown_scores'][0]), .9)
        self.assertFalse(texture_descriptor(changed.requires_grad_()).requires_grad)

    def test_texture_brightness_offset_invariance_and_half_statistics(self):
        x = torch.rand(1, 3, 32, 32) * .5
        torch.testing.assert_close(texture_descriptor(x), texture_descriptor(x + .2), atol=3e-6, rtol=1e-5)
        torch.testing.assert_close(texture_descriptor(x), texture_descriptor(1.5 * x + .1), atol=3e-6, rtol=1e-5)
        novelty = NormalPrototypes(12, k=2, texture_detail=True)
        raw = torch.ones(1, 12, 1, 1)
        novelty.update(raw, texture_descriptor(x))
        before = novelty.texture.centers.clone()
        novelty.calibrated.fill_(True)
        novelty.half()
        self.assertEqual(novelty.texture.centers.dtype, torch.float32)
        torch.testing.assert_close(novelty.texture.centers, before, rtol=0, atol=0)
        novelty.update(raw, texture_descriptor(x))
        self.assertFalse(novelty.calibrated)

    @unittest.skipUnless(importlib.util.find_spec('onnx') and importlib.util.find_spec('onnxruntime'),
                         'optional ONNX tooling not installed')
    def test_known_onnx_six_tensors_match_native_with_detail(self):
        import onnx
        import onnxruntime as ort
        from export import ExportWrapper
        model = self.model(self.config()).eval()
        image = torch.rand(1, 3, 128, 128)
        wrapper = ExportWrapper(model)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'known.onnx'
            with torch.no_grad():
                expected = wrapper(image)
                torch.onnx.export(wrapper, image, str(output), opset_version=17,
                                  input_names=['images'])
            onnx.checker.check_model(onnx.load(str(output)))
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            session = ort.InferenceSession(str(output), options, providers=['CPUExecutionProvider'])
            actual = session.run(None, {'images': image.numpy()})
            self.assertEqual(len(actual), 6)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(torch.from_numpy(a), b, rtol=1e-4, atol=1e-5)

    def test_missing_native_features_are_not_silently_ignored(self):
        novelty = NormalPrototypes(12, k=2, texture_detail=True)
        raw = torch.ones(1, 12, 1, 1)
        with self.assertRaisesRegex(ValueError, 'Native texture'):
            novelty.update(raw)
        novelty.update(raw, texture_descriptor(torch.rand(1, 3, 32, 32)))
        with self.assertRaisesRegex(ValueError, 'Native texture'):
            novelty.score(raw)

    def test_legacy_upgrade_restores_old_weights_and_initializes_only_new_branches(self):
        old = self.model(self.config(False))
        old.novelty.calibrated.fill_(True)
        weights = copy.deepcopy(old.state_dict())
        new = self.model(self.config(True))
        branch = copy.deepcopy(new.backbone.local_detail.state_dict())
        restore_detector_weights(new, weights, allow_detail_upgrade=True)
        for k, value in weights.items():
            if k != 'novelty.calibrated':
                torch.testing.assert_close(new.state_dict()[k], value, rtol=0, atol=0)
        for k, value in branch.items():
            torch.testing.assert_close(new.backbone.local_detail.state_dict()[k], value, rtol=0, atol=0)
        self.assertFalse(new.novelty.calibrated)
        self.assertEqual(int(new.novelty.texture.updates), 0)
        with self.assertRaises(RuntimeError):
            restore_detector_weights(new, weights)  # inference/full resume strict

    def test_upgrade_cannot_hide_missing_head_or_partially_missing_new_branch(self):
        old = self.model(self.config(False))
        new = self.model(self.config(True))
        broken = dict(old.state_dict())
        del broken['head.heads.0.cls_pred.weight']
        with self.assertRaises(RuntimeError):
            restore_detector_weights(new, broken, allow_detail_upgrade=True)
        partial = dict(new.state_dict())
        del partial['backbone.local_detail.gate']
        with self.assertRaises(RuntimeError):
            restore_detector_weights(new, partial, allow_detail_upgrade=True)


if __name__ == '__main__':
    unittest.main()
