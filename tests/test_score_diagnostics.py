"""Diagnostic scores cannot change predictions, checkpoints or P/R cutoffs."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools"), str(ROOT / "tests")]
from helpers import FakeEncoder
from jingwei import build_model, load_config
from evaluate import RawScoreStats, average_precision, set_candidate_conf, validate_detector


class TestScoreDiagnostics(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(ROOT / "configs/jingwei_dinov2.yaml")
        self.cfg.input_size = [128, 128]
        self.cfg.model.dino_input_size = 56
        self.cfg.unknown.enabled = False
        with patch("torch.hub.load", return_value=FakeEncoder()):
            self.model = build_model(self.cfg).eval()
        self.predictions = []
        for scale in range(2):
            cls = torch.full((1, 19, 1, 1), -10.)
            if scale == 0:
                cls[:, 0] = -2.  # sigmoid(-2)*sigmoid(1.5) = .09746, below .15
            self.predictions.append(dict(cls=cls, reg=torch.full((1, 4, 1, 1), .5),
                                         obj_o2o=torch.full((1, 1, 1, 1), 1.5)))
        self.batch = dict(image=torch.rand(1, 3, 128, 128),
                          boxes=[torch.tensor([[0., 0., 8., 8.]])], labels=[torch.tensor([0])])

    def test_stats_expose_score_filtered_tp_and_keep_reporting_cutoff(self):
        state_before = {k: v.clone() for k, v in self.model.state_dict().items()}
        with patch.object(self.model.head, "forward", return_value=self.predictions):
            baseline = validate_detector(self.model, [self.batch], None, "cpu", self.cfg)
            self.assertEqual(baseline["n_det"], 0)
            self.assertEqual(baseline["recall"], 0.)
            with RawScoreStats(self.model.head) as stats:
                observed = validate_detector(self.model, [self.batch], [dict(image=self.batch["image"])],
                                             "cpu", self.cfg, score_stats=stats)
            self.assertEqual(observed["n_det"], baseline["n_det"])
            self.assertEqual(observed["gt_grid_coverage"], dict(covered=1, total=1, fraction=1.))
            raw = observed["raw_score_stats"]
            self.assertEqual(raw["val"]["images"], 1)
            self.assertEqual(raw["normalval"]["images"], 1)
            self.assertAlmostEqual(raw["val"]["max_joint_score"]["max"], .0974568, places=5)
            self.assertEqual(raw["val"]["raw_cells_before_topk"]["0.15"]["images_with_candidates"], 0)
            self.assertEqual(raw["val"]["raw_cells_before_topk"]["0.05"]["images_with_candidates"], 1)
            set_candidate_conf(self.model, .001)
            with RawScoreStats(self.model.head) as stats:
                diagnostic = validate_detector(self.model, [self.batch], [dict(image=self.batch["image"])], "cpu", self.cfg,
                                                conf_thresh=.15, score_stats=stats)
            self.assertEqual(diagnostic["n_det"], 1)
            # Primary P/R stays at .15, so it still reports 0 TP. The .05
            # diagnostic reveals an otherwise perfect low-confidence detection.
            self.assertEqual(diagnostic["conf_thresh"], .15)
            self.assertEqual(diagnostic["tp"], 0)
            self.assertEqual(observed["normal_known_fpr"], 0.)
            self.assertEqual(diagnostic["normal_known_fpr"], 1.)  # candidate .001, not P/R .15
            self.assertEqual(diagnostic["pr_at"][.05]["tp"], 1)
            self.assertGreater(diagnostic["mAP50"], .9)
        for k, v in state_before.items():
            torch.testing.assert_close(self.model.state_dict()[k], v, rtol=0, atol=0)
        self.assertEqual(len(self.model.head._forward_hooks), 0)

    def test_numpy2_trapezoid_path_avoids_deprecated_trapz(self):
        with patch("evaluate.np.trapezoid", create=True, return_value=.375) as integrate:
            value = average_precision(np.array([0., 1.]), np.array([1., 1.]))
        self.assertEqual(value, .375)
        integrate.assert_called_once()

    def test_hook_removed_after_exception(self):
        with self.assertRaisesRegex(RuntimeError, "diagnostic failure"):
            with RawScoreStats(self.model.head):
                self.assertEqual(len(self.model.head._forward_hooks), 1)
                raise RuntimeError("diagnostic failure")
        self.assertEqual(len(self.model.head._forward_hooks), 0)

    def test_invalid_override_cannot_mutate_filters(self):
        for value in (-.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                set_candidate_conf(self.model, value)
        self.assertEqual(self.model.head.score_thresh, .05)
        self.assertEqual(self.model.cfg.inference.final_score_thresh, .15)


if __name__ == "__main__":
    unittest.main()
