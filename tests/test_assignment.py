"""Geometry-feasible supervision for a nonnegative l/t/r/b head.

These counterexamples validate assignment, not real-data accuracy/FPS gains.
"""
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jingwei import load_config
from jingwei.losses import DetectionLoss, _grid_points, build_dense_targets, hungarian_match
from jingwei.utils import boxes_from_distances, boxes_to_distances, grid_center_coverage


class TestGeometryAssignment(unittest.TestCase):
    strides = (8, 32)

    def predictions(self, batch=1):
        predictions = []
        for h, w in ((8, 8), (2, 2)):
            predictions.append(dict(cls=torch.full((batch, 1, h, w), -10.),
                                    obj_o2o=torch.full((batch, 1, h, w), -10.),
                                    obj=torch.zeros(batch, 1, h, w),
                                    reg=torch.full((batch, 4, h, w), .1), fg=None))
        # Deliberately overconfident WRONG cell, spatially far from GT.
        predictions[0]["cls"][:, :, 0, 0] = 10.
        predictions[0]["obj_o2o"][:, :, 0, 0] = 10.
        return predictions

    def target(self, boxes):
        return dict(boxes=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                    labels=torch.zeros(len(boxes), dtype=torch.long))

    def assert_feasible_unique(self, predictions, targets, matches):
        cells, objects = set(), set()
        for si, group in enumerate(matches):
            pts = _grid_points(*predictions[si]["cls"].shape[-2:], self.strides[si], "cpu", torch.float32)
            for batch, flat, gt in group:
                self.assertNotIn((si, batch, flat), cells)
                self.assertNotIn((batch, gt), objects)
                cells.add((si, batch, flat))
                objects.add((batch, gt))
                distances = boxes_to_distances(pts[flat:flat + 1], targets[batch]["boxes"][gt:gt + 1])
                self.assertTrue(bool((distances >= 0).all()), (pts[flat], targets[batch]["boxes"][gt]))
                torch.testing.assert_close(boxes_from_distances(pts[flat:flat + 1], distances),
                                           targets[batch]["boxes"][gt:gt + 1])
        return objects

    def test_confident_outside_cell_never_becomes_positive(self):
        predictions = self.predictions()
        targets = [self.target([[40., 40., 56., 56.]])]
        for budget in (2, 128):
            matches = hungarian_match(predictions, targets, self.strides, 1, 1., budget)
            self.assertEqual(self.assert_feasible_unique(predictions, targets, matches), {(0, 0)})
            self.assertNotIn((0, 0, 0), matches[0])

    def test_low_confidence_gt_cells_survive_tiny_score_budget(self):
        predictions = self.predictions()
        targets = [self.target([[40., 40., 56., 56.], [8., 40., 24., 56.]])]
        matches = hungarian_match(predictions, targets, self.strides, 1, 1., max_candidates=2)
        self.assertEqual(self.assert_feasible_unique(predictions, targets, matches), {(0, 0), (0, 1)})

    def test_unrepresentable_gt_not_forced_and_dense_targets_stay_nonnegative(self):
        predictions = self.predictions()
        targets = [self.target([[5., 5., 6., 6.]])]
        matches = hungarian_match(predictions, targets, self.strides, 1, 1., 2)
        self.assertEqual(matches, [[], []])
        for p, stride in zip(predictions, self.strides):
            _, distances, _, _, positive, _ = build_dense_targets(p, targets, stride, 1, 1.)
            self.assertFalse(positive.any())
            self.assertTrue((distances >= 0).all())

    def test_infeasible_row_does_not_steal_other_gt_match(self):
        predictions = self.predictions()
        targets = [self.target([[5., 5., 6., 6.], [40., 40., 56., 56.]])]
        matches = hungarian_match(predictions, targets, self.strides, 1, 1., 2)
        self.assertEqual(self.assert_feasible_unique(predictions, targets, matches), {(0, 1)})

    def test_crowded_same_cell_has_no_duplicate_or_outside_positive(self):
        predictions = self.predictions()
        targets = [self.target([[3., 3., 5., 5.]] * 3)]
        matches = hungarian_match(predictions, targets, self.strides, 1, 1., 2)
        self.assertEqual(len(self.assert_feasible_unique(predictions, targets, matches)), 1)

    def test_feasible_cardinality_precedes_confidence_preference(self):
        predictions = self.predictions()
        targets = [self.target([[0., 0., 64., 64.], [3., 3., 5., 5.]])]
        matches = hungarian_match(predictions, targets, self.strides, 1, 1., 2)
        self.assertEqual(self.assert_feasible_unique(predictions, targets, matches), {(0, 0), (0, 1)})
        self.assertIn((0, 0, 1), matches[0])  # the tiny GT gets its only feasible cell

    def test_batch_and_empty_normal_image_isolation(self):
        predictions = self.predictions(batch=3)
        targets = [self.target([]), self.target([[40., 40., 56., 56.]]), self.target([[8., 40., 24., 56.]])]
        matches = hungarian_match(predictions, targets, self.strides, 1, 1., 2)
        self.assertEqual(self.assert_feasible_unique(predictions, targets, matches), {(1, 0), (2, 0)})

    def test_criterion_reports_unmatched_and_supports_backward(self):
        cfg = load_config(ROOT / "configs/jingwei_dinov2.yaml")
        cfg.data.nc = 1
        criterion = DetectionLoss(cfg)
        predictions = self.predictions()
        for p in predictions:
            for key in ("cls", "obj_o2o", "obj", "reg"):
                p[key].requires_grad_()
        targets = [self.target([[5., 5., 6., 6.], [40., 40., 56., 56.]])]
        losses = criterion(predictions, targets, (64, 64))
        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertEqual(criterion.last_assignment, dict(matched_gt=1, total_gt=2, unmatched_gt=1))
        losses["total"].backward()
        self.assertTrue(any(p["reg"].grad is not None for p in predictions))

    def test_analytic_coverage_matches_visible_grid_including_boundaries(self):
        generator = torch.Generator().manual_seed(17)
        starts = torch.rand(200, 2, generator=generator) * 80 - 8
        boxes = torch.cat((starts, starts + torch.rand(200, 2, generator=generator) * 24 + .1), 1)
        boxes = torch.cat((boxes, torch.tensor([[4., 4., 12., 12.], [5., 5., 6., 6.],
                                               [60., 60., 70., 70.], [80., 80., 90., 90.]])))
        points = torch.cat([_grid_points(64 // s, 64 // s, s, "cpu", torch.float32) for s in self.strides])
        expected = ((points[None] >= boxes[:, None, :2]) & (points[None] <= boxes[:, None, 2:])).all(-1).any(-1)
        torch.testing.assert_close(grid_center_coverage(boxes, (64, 64)), expected)
        self.assertEqual(grid_center_coverage(torch.empty(0, 4), (64, 64)).shape, (0,))
        self.assertEqual(grid_center_coverage(boxes[-4:], (64, 64)).tolist(), [True, False, True, False])


if __name__ == "__main__":
    unittest.main()
