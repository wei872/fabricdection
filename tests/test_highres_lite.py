"""Capped semantic pyramid + native fine grid. No real accuracy/FPS claims."""
import copy
import json
import os
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools'), str(ROOT / 'tests')]
from helpers import FakeEncoder, local_config
from jingwei import build_model, load_config
from jingwei.utils import grid_center_coverage
from jingwei.novelty import texture_descriptor
from _common import configure_inference, load_model, resume_checkpoint
from evaluate import RawScoreStats, validate_detector


class TestHighResolutionLite(unittest.TestCase):
    def config(self, native=128, pyramid=64, unknown=False):
        cfg = load_config(ROOT / 'configs/jingwei_dinov2.yaml')
        cfg.input_size = [native, native]
        cfg.model.pyramid_size = pyramid
        cfg.model.dino_input_size = 56
        cfg.unknown.enabled = unknown
        return cfg

    def model(self, cfg):
        with patch('torch.hub.load', return_value=FakeEncoder()):
            return build_model(cfg)

    def test_native_fine_grid_but_pan_and_coarse_head_stay_small(self):
        model = self.model(self.config()).train()
        target = [dict(boxes=torch.tensor([[20., 20., 50., 50.]]), labels=torch.tensor([1]))] * 2
        observed = []
        handle = model.neck.register_forward_pre_hook(lambda module, inputs: observed.append(inputs[0]['p3'].shape))
        result = model(torch.rand(2, 3, 128, 128), target)
        handle.remove()
        self.assertEqual(observed[0][-2:], (8, 8))  # PAN uses 64/8, NOT 128/8
        self.assertEqual(result['preds'][0]['cls'].shape[-2:], (16, 16))
        self.assertEqual(result['preds'][1]['cls'].shape[-2:], (2, 2))
        self.assertEqual(model.head.strides, (8, 64))
        self.assertEqual(model.det_loss.strides, model.head.strides)
        result['losses']['total'].backward()
        self.assertGreater(float(model.backbone.local_detail.layers[0].conv.weight.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for p in model.backbone.encoder.parameters()))

    def test_capped_profile_rejects_silently_resized_tensor_inputs(self):
        model = self.model(self.config()).eval()
        with self.assertRaisesRegex(ValueError, 'native input dimensions'):
            model(torch.rand(1, 3, 64, 64))

    def test_residual_is_native_detail_not_another_coarse_interpolation(self):
        model = self.model(self.config()).eval()
        x = torch.rand(1, 3, 128, 128)
        with torch.no_grad():
            features = model.backbone(x, include_detail=False)
            native = model.backbone.local_detail((x - model.backbone.mean) / model.backbone.std)
            coarse = torch.nn.functional.avg_pool2d(native, 2, 2)
            residual = native - torch.nn.functional.interpolate(coarse, size=(16, 16), mode='bilinear', align_corners=False)
        torch.testing.assert_close(features['native_residual'], residual)
        self.assertGreater(float(residual.abs().max()), 0.)

    def test_640_style_path_is_unchanged_when_pyramid_not_capped(self):
        model = self.model(self.config(native=128, pyramid=None)).eval()
        x = torch.rand(1, 3, 128, 128)
        with torch.no_grad():
            backbone = model.backbone(x, include_detail=False)
            reference = model.neck(backbone)
            actual = model._features(backbone)
        self.assertNotIn('native_residual', backbone)
        self.assertEqual(model.head.strides, (8, 32))
        for key in reference:
            torch.testing.assert_close(actual[key], reference[key], rtol=0, atol=0)

    def test_coarse_decode_uses_true_native_stride(self):
        model = self.model(self.config()).eval()
        predictions = [dict(cls=torch.full((1, 19, h, h), -10.),
                            reg=torch.full((1, 4, h, h), .5),
                            obj_o2o=torch.full((1, 1, h, h), -10.)) for h in (16, 2)]
        predictions[1]['cls'][0, 0, 0, 0] = 10
        predictions[1]['obj_o2o'][0, 0, 0, 0] = 10
        result = model.head.decode(predictions, (128, 128))[0]
        torch.testing.assert_close(result['boxes'], torch.tensor([[0., 0., 64., 64.]]))
        torch.testing.assert_close(result['points'], torch.tensor([[32., 32.]]))
        self.assertEqual(result['scale_idx'].tolist(), [1])

    def test_score_diagnostic_coverage_uses_profile_strides(self):
        model = self.model(self.config()).eval()
        box = torch.tensor([[31., 31., 33., 33.]])
        self.assertFalse(grid_center_coverage(box, (128, 128), (8, 32))[0])
        predictions = [dict(cls=torch.full((1, 19, h, h), -10.),
                            reg=torch.full((1, 4, h, h), .5),
                            obj_o2o=torch.full((1, 1, h, h), -10.)) for h in (16, 2)]
        predictions[1]['cls'][0, 0, 0, 0] = 10
        predictions[1]['obj_o2o'][0, 0, 0, 0] = 10
        predictions[1]['reg'][0, :, 0, 0] = 1 / 64
        batch = dict(image=torch.rand(1, 3, 128, 128), boxes=[box], labels=[torch.tensor([0])])
        with patch.object(model.head, 'forward', return_value=predictions):
            with RawScoreStats(model.head) as stats:
                metrics = validate_detector(model, [batch], None, 'cpu', model.cfg, score_stats=stats)
        self.assertEqual(metrics['tp'], 1)
        self.assertEqual(metrics['gt_grid_coverage'], dict(covered=1, total=1, fraction=1.))

    def test_folding_matches_six_native_tensors_and_prevents_training(self):
        model = self.model(self.config())
        with self.assertRaises(RuntimeError):
            model.fuse_for_inference()
        model.eval()
        x = torch.rand(1, 3, 128, 128)
        with torch.no_grad():
            before = model.export_forward(x)
            model.fuse_for_inference()
            after = model.export_forward(x)
        self.assertGreater(model.fused_bn_count, 0)
        self.assertFalse(any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules()))
        for a, b in zip(before, after):
            torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)
        count = model.fused_bn_count
        self.assertIs(model.fuse_for_inference(), model)
        self.assertEqual(count, model.fused_bn_count)
        with self.assertRaisesRegex(RuntimeError, 'inference-only'):
            model.train()

    def test_fusion_can_be_disabled_for_baseline_measurements(self):
        model = self.model(self.config())
        configure_inference(model, known_only=True, fuse=False)
        self.assertFalse(getattr(model, '_inference_fused', False))
        configure_inference(model, known_only=True)
        self.assertTrue(model._inference_fused)

    def test_profile_changes_no_weight_shapes_and_resets_geometry_dependent_banks(self):
        old = self.model(self.config(native=64, pyramid=None, unknown=True))
        new = self.model(self.config(unknown=True))
        self.assertEqual({k: v.shape for k, v in old.state_dict().items()},
                         {k: v.shape for k, v in new.state_dict().items()})
        old.novelty.updates.fill_(7)
        old.novelty.texture.updates.fill_(7)
        old.novelty.centers.fill_(1)
        old.novelty.texture.centers.fill_(1)
        old.novelty.calibrated.fill_(True)
        checkpoint = dict(model=copy.deepcopy(old.state_dict()), train={'architecture': old.architecture_signature})
        result = resume_checkpoint(checkpoint, new)
        self.assertFalse(result['full'])
        self.assertEqual(int(new.novelty.updates), 0)
        self.assertEqual(int(new.novelty.texture.updates), 0)
        self.assertFalse(new.novelty.calibrated)
        torch.testing.assert_close(new.head.heads[0].cls_pred.weight, old.head.heads[0].cls_pred.weight)
        self.assertFalse(new.architecture_compatible(old.architecture_signature))

    def test_full_resume_rejects_different_geometry_before_loading_weights(self):
        old = self.model(self.config(native=64, pyramid=None))
        new = self.model(self.config())
        optimizer = torch.optim.AdamW(new.parameters())
        state = dict(model=old.state_dict(), optimizer=optimizer.state_dict(), rng={},
                     train=dict(epoch=1, iter=1, backbone='dinov2', architecture=old.architecture_signature))
        with self.assertRaisesRegex(ValueError, '--pretrained'):
            resume_checkpoint(state, new, optimizer)

    def test_fusion_leaves_calibrated_unknown_scores_unchanged(self):
        model = self.model(self.config(unknown=True)).eval()
        normal = torch.rand(1, 3, 128, 128)
        test = torch.rand(1, 3, 128, 128)
        with torch.no_grad():
            raw = model.backbone(normal, return_raw=True)['raw']
            model.novelty.update(raw, texture_descriptor(normal))
            model.novelty.threshold.fill_(.1)
            model.novelty.calibrated.fill_(True)
            before = model(test)[0]
            model.fuse_for_inference()
            after = model(test)[0]
        torch.testing.assert_close(before['unknown_boxes'], after['unknown_boxes'], rtol=0, atol=0)
        torch.testing.assert_close(before['unknown_scores'], after['unknown_scores'], rtol=0, atol=0)

    def test_legacy_640_resume_signature_only_accepted_for_actual_640_geometry(self):
        legacy = dict(local_detail=True, texture_detail=True)
        model = self.model(self.config(native=640, pyramid=None, unknown=True))
        self.assertTrue(model.architecture_compatible(legacy))
        self.assertFalse(self.model(self.config(unknown=True)).architecture_compatible(legacy))

    def test_wrong_inference_config_rejected_before_weight_restore(self):
        model = self.model(self.config())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / 'model.pt'
            torch.save(dict(model=model.state_dict(), train={'architecture': model.architecture_signature}), checkpoint)
            cfg = self.config()
            path = root / 'cfg.yaml'
            path.write_text(yaml.safe_dump(json.loads(json.dumps(cfg))), encoding='utf-8')
            torch.save(dict(model=model.state_dict()), checkpoint)
            with patch('torch.hub.load', return_value=FakeEncoder()):
                with self.assertRaisesRegex(ValueError, 'recorded checkpoint geometry'):
                    load_model(str(path), str(checkpoint))
            torch.save(dict(model=model.state_dict(), train={'architecture': model.architecture_signature}), checkpoint)
            cfg.model.pyramid_size = None
            path.write_text(yaml.safe_dump(json.loads(json.dumps(cfg))), encoding='utf-8')
            with patch('torch.hub.load', return_value=FakeEncoder()):
                with self.assertRaisesRegex(ValueError, 'geometry'):
                    load_model(str(path), str(checkpoint))

    def test_export_metadata_contains_correct_nonlegacy_decoder_strides(self):
        from export import export_metadata
        model = self.model(self.config()).eval()
        metadata = export_metadata(model, model.cfg)
        self.assertEqual(metadata['prediction_strides'], [8, 64])
        self.assertEqual(metadata['input_shape'], [1, 3, 128, 128])
        self.assertEqual(len(metadata['outputs']), 6)
        json.dumps(metadata)

    def test_capped_profile_training_and_full_resume_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = local_config(root)
            raw = yaml.safe_load(config.read_text())
            raw['model']['pyramid_size'] = 64
            config.write_text(yaml.safe_dump(raw), encoding='utf-8')
            out = root / 'train'
            def run(*extra):
                command = [sys.executable, 'tools/train_detector.py', '--cfg', str(config),
                           '--smoke', '2', '--epochs', '1', '--batch', '2', '--device', 'cpu',
                           '--val-interval', '0', '--out', str(out), *extra]
                result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                        env={**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}, timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return result.stdout
            first = run()
            self.assertIn('native_input=128 semantic_pyramid=64 prediction_strides=(8, 64)', first)
            resumed = run('--resume', str(out / 'last.pt'), '--epochs', '2')
            self.assertIn('mode=full', resumed)
            self.assertIn('start_epoch=2/2', resumed)

    def test_invalid_caps_rejected_and_real_1280_profile_is_valid(self):
        cfg = load_config(ROOT / 'configs/jingwei_dinov2_1280lite.yaml')
        self.assertEqual(cfg.input_size, [1280, 1280])
        self.assertEqual(cfg.model.pyramid_size, 640)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'invalid.yaml'
            for value in (0, 63, 96, 256, '64', True):
                cfg = self.config(pyramid=value)
                path.write_text(yaml.safe_dump(json.loads(json.dumps(cfg))), encoding='utf-8')
                with self.assertRaises(ValueError):
                    load_config(path)


if __name__ == '__main__':
    unittest.main()
