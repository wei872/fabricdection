"""Convergence controls and explicit file initialization, not accuracy claims."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools'), str(ROOT / 'tests')]
from _training_control import cosine_factor, schedule_spec, EarlyStopper
from _common import load_model
from helpers import FakeEncoder, local_config
from jingwei.utils import torch_load


class TestTrainingControl(unittest.TestCase):
    def test_500_epoch_budget_does_not_stretch_explicit_40_epoch_decay(self):
        spec = schedule_spec(500, 10, 1, cfg_decay=50, explicit_decay=40)
        self.assertEqual(spec['decay_steps'], 400)
        self.assertAlmostEqual(cosine_factor(400, 10, 400), .05)
        self.assertAlmostEqual(cosine_factor(5000, 10, 400), .05)
        self.assertAlmostEqual(cosine_factor(0, 10, 400), .1)
        self.assertAlmostEqual(cosine_factor(10, 10, 400), 1.)

    def test_floor_is_monotonic_without_warmup_and_never_rebounds(self):
        values = [cosine_factor(step, 0, 40) for step in range(501)]
        self.assertEqual(values[0], 1.)
        self.assertTrue(all(a >= b for a, b in zip(values, values[1:])))
        self.assertTrue(all(abs(value - .05) < 1e-12 for value in values[40:]))

    def test_config_horizon_capped_by_short_training_budget(self):
        self.assertEqual(schedule_spec(30, 10, 1, cfg_decay=50)['decay_epochs'], 30)
        self.assertEqual(schedule_spec(500, 10, 1, cfg_decay=50)['decay_epochs'], 50)
        for bad in (0, -1, True, '50'):
            with self.assertRaises(ValueError):
                schedule_spec(500, 10, 1, cfg_decay=bad)

    def test_resume_preserves_step_horizon_despite_new_budget_or_batch_count(self):
        original = schedule_spec(50, 10, 1, cfg_decay=40)
        resumed = schedule_spec(500, 20, 2, cfg_decay=50, saved={'lr_schedule': original})
        self.assertEqual(resumed, original)
        with self.assertRaisesRegex(ValueError, '--pretrained'):
            schedule_spec(500, 20, 2, explicit_decay=50, saved={'lr_schedule': original})

    def test_legacy_resume_reconstructs_old_500_epoch_horizon_not_new_default(self):
        old = {'epochs': 500, 'batches_per_epoch': 440}
        spec = schedule_spec(600, 220, 1, cfg_decay=50, saved=old)
        self.assertEqual(spec['decay_steps'], 220000)
        self.assertEqual(spec['warmup_steps'], 440)
        self.assertEqual(schedule_spec(600, 220, 0, saved=old)['kind'], 'legacy_cosine_annealing')

    def test_early_stop_counts_validation_events_after_min_epochs(self):
        stop = EarlyStopper(patience=2, min_delta=.01, min_epochs=5)
        self.assertFalse(stop.step(.5, 1))
        self.assertFalse(stop.step(.505, 4))
        self.assertEqual(stop.bad_checks, 0)
        self.assertFalse(stop.step(.505, 5))
        self.assertTrue(stop.step(.505, 10))  # two validation events, not five epochs

    def test_meaningful_improvement_resets_patience_and_roundtrips(self):
        stop = EarlyStopper(2, .01, 0)
        stop.step(.5, 1)
        stop.step(.505, 2)
        self.assertFalse(stop.step(.52, 3))
        self.assertEqual(stop.bad_checks, 0)
        restored = EarlyStopper(2, .01, 0)
        restored.load_state_dict(stop.state_dict())
        self.assertEqual(restored.state_dict(), stop.state_dict())
        self.assertFalse(restored.step(.52, 4))
        self.assertTrue(restored.step(.52, 5))
        disabled = EarlyStopper(0, .01, 0)
        self.assertFalse(any(disabled.step(.5, i) for i in range(100)))

    def test_invalid_controls_rejected(self):
        for args in ((-1, .01, 0), (1, -.1, 0), (1, float('nan'), 0), (1, .1, -1)):
            with self.assertRaises(ValueError):
                EarlyStopper(*args)
        with self.assertRaises(ValueError):
            EarlyStopper().step(float('nan'), 1)

    def test_local_file_load_is_strict_and_never_requests_hub_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = local_config(root)
            original = FakeEncoder()
            weights = root / 'official_fixture.pth'
            torch.save(original.state_dict(), weights)
            with patch('torch.hub.load', return_value=FakeEncoder()) as hub:
                model, cfg = load_model(str(config), dino_weights=str(weights), dino_repo=str(root / 'hub'))
            self.assertFalse(hub.call_args.kwargs['pretrained'])
            self.assertEqual(hub.call_args.kwargs['source'], 'local')
            self.assertIn('official_file=', model.backbone.initialization_source)
            self.assertTrue(all(not p.requires_grad for p in model.backbone.encoder.parameters()))
            with patch.dict(os.environ, {'HOME': str(root)}), patch('torch.hub.load', return_value=FakeEncoder()):
                load_model(str(config), dino_weights='~/official_fixture.pth', dino_repo='~/hub')
            for k, v in original.state_dict().items():
                torch.testing.assert_close(model.backbone.encoder.state_dict()[k], v, rtol=0, atol=0)
            with patch('torch.hub.load') as hub:
                with self.assertRaises(FileNotFoundError):
                    load_model(str(config), dino_weights=str(root / 'missing.pth'))
                hub.assert_not_called()
            broken = dict(original.state_dict())
            broken.pop('proj.weight')
            torch.save(broken, weights)
            with patch('torch.hub.load', return_value=FakeEncoder()):
                with self.assertRaisesRegex(RuntimeError, 'No random-backbone fallback'):
                    load_model(str(config), dino_weights=str(weights))

    def test_loss_diagnostics_are_detached_and_reconstruct_existing_box_losses(self):
        from jingwei import load_config
        from jingwei.losses import DetectionLoss
        cfg = load_config(ROOT / 'configs/jingwei_dinov2.yaml')
        cfg.data.nc = 1
        criterion = DetectionLoss(cfg)
        for positive in (True, False):
            predictions = [dict(cls=torch.full((1, 1, h, h), -2., requires_grad=True),
                                reg=torch.full((1, 4, h, h), .1, requires_grad=True),
                                obj=torch.zeros(1, 1, h, h, requires_grad=True),
                                obj_o2o=torch.zeros(1, 1, h, h, requires_grad=True), fg=None)
                           for h in (4, 1)]
            targets = [dict(boxes=torch.tensor([[0., 0., 16., 16.]]) if positive else torch.empty(0, 4),
                            labels=torch.zeros(int(positive), dtype=torch.long))]
            losses = criterion(predictions, targets, (32, 32))
            diag = criterion.last_diagnostics
            for branch in ('o2m', 'o2o'):
                torch.testing.assert_close(losses['box_' + branch],
                                           diag[branch + '_ciou'] + diag[branch + '_pixel_l1_x0.1'])
            self.assertTrue(all(not v.requires_grad for v in diag.values()))
            losses['total'].backward()
            self.assertTrue(torch.isfinite(predictions[0]['cls'].grad).all())

    def test_official_file_cli_early_stops_saves_state_and_preserves_backbone(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = local_config(root)
            backbone = FakeEncoder()
            weights = root / 'official_fixture.pth'
            torch.save(backbone.state_dict(), weights)
            out = root / 'train'
            command = [sys.executable, 'tools/train_detector.py', '--cfg', str(config),
                       '--smoke', '2', '--epochs', '10', '--batch', '2', '--device', 'cpu',
                       '--dino-weights', str(weights), '--dino-repo', str(root / 'hub'),
                       '--lr-decay-epochs', '2', '--early-stop-patience', '1',
                       '--early-stop-min-delta', '2', '--early-stop-min-epochs', '0', '--out', str(out)]
            # min_delta=2 deliberately prevents any improvement after first
            # validation (F1 in [0,1]); this is a deterministic control test.
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=90,
                                    env={**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('backbone_source=official_file=', result.stdout)
            self.assertIn('Early stopping at epoch 2', result.stdout)
            self.assertIn('boxdiag', result.stdout)
            self.assertIn('grad preclip', result.stdout)
            state = torch_load(out / 'last.pt')
            self.assertEqual(state['train']['epoch'], 2)
            self.assertEqual(state['train']['lr_schedule']['decay_epochs'], 2)
            self.assertEqual(state['train']['early_stopping']['bad_checks'], 1)
            self.assertTrue((out / 'best.pt').is_file())
            for k, v in backbone.state_dict().items():
                torch.testing.assert_close(state['model']['backbone.encoder.' + k], v, rtol=0, atol=0)
            resume = [sys.executable, 'tools/train_detector.py', '--cfg', str(config),
                      '--smoke', '2', '--epochs', '12', '--batch', '2', '--device', 'cpu',
                      '--resume', str(out / 'last.pt'), '--out', str(out)]
            result = subprocess.run(resume, cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('already early-stopped', result.stdout)
            self.assertEqual(torch_load(out / 'last.pt')['train']['epoch'], 2)
            mismatch = subprocess.run(resume + ['--lr-decay-epochs', '3'], cwd=ROOT,
                                      capture_output=True, text=True, timeout=60)
            self.assertNotEqual(mismatch.returncode, 0)
            self.assertIn('Cannot change LR horizon', mismatch.stderr)
            continued = subprocess.run(resume + ['--epochs', '3', '--early-stop-patience', '0'],
                                       cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(continued.returncode, 0, continued.stdout + continued.stderr)
            state = torch_load(out / 'last.pt')
            self.assertEqual(state['train']['epoch'], 3)
            self.assertEqual(state['train']['lr_schedule']['decay_epochs'], 2)
            self.assertEqual(state['train']['early_stopping']['patience'], 0)
            conflict = subprocess.run(command + ['--pretrained', str(out / 'best.pt')], cwd=ROOT,
                                      capture_output=True, text=True, timeout=30)
            self.assertEqual(conflict.returncode, 2)


if __name__ == '__main__':
    unittest.main()
