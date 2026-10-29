"""Regression checks for stage-A box supervision (no data / GPU / teacher).

Run: python -m unittest discover -s tests -p test_detection_loss.py -v
"""
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jingwei.config import load_config
from jingwei.losses import DetectionLoss, _grid_points, build_dense_targets
from jingwei.utils import box_ciou, boxes_from_distances, boxes_to_distances


class TestDenseBoxLoss(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(ROOT / "configs/jingwei_n.yaml")
        self.criterion = DetectionLoss(self.cfg)
        self.targets = [
            {"boxes": torch.tensor([[-9., -13., 39., 37.]]),
             "labels": torch.tensor([1])},
            {"boxes": torch.tensor([[-5., -7., 43., 41.]]),
             "labels": torch.tensor([2])},
        ]

    def predictions(self, perturb=0.0):
        preds = []
        for stride, h, w in ((8, 4, 4), (32, 1, 1)):
            points = _grid_points(h, w, stride, "cpu", torch.float32)
            regs = []
            for target in self.targets:
                box = target["boxes"].expand(len(points), 4)
                dist = boxes_to_distances(points, box) + perturb
                regs.append(dist.T.reshape(4, h, w) /
                            (self.criterion.reg_scale * stride))
            preds.append({
                "cls": torch.full((2, 19, h, w), -2., requires_grad=True),
                "reg": torch.stack(regs).requires_grad_(),
                "obj": torch.zeros(2, 1, h, w, requires_grad=True),
                "obj_o2o": torch.zeros(2, 1, h, w, requires_grad=True),
                "fg": None,
            })
        return preds

    def test_perfect_boxes_have_zero_dense_box_loss(self):
        preds = self.predictions()
        _, _, _, _, pos, _ = build_dense_targets(
            preds[0], self.targets, 8, 19, self.criterion.reg_scale)
        self.assertTrue(bool((pos.sum((1, 2)) > 1).all()))
        losses = self.criterion(preds, self.targets, (128, 128))
        self.assertLess(abs(float(losses["box_o2m"])), 1e-5)
        self.assertTrue(bool(torch.isfinite(losses["total"])))

    def test_loss_and_gradients_match_per_cell_reference(self):
        preds = self.predictions(perturb=2.0)
        actual = self.criterion(preds, self.targets, (128, 128))["box_o2m"]
        expected = torch.zeros(())
        # Deliberately index one cell at a time rather than reusing the loss's
        # vectorised selection: this catches channel/cell layout regressions.
        for p, stride in zip(preds, (8, 32)):
            _, _, _, _, pos, pts = build_dense_targets(
                p, self.targets, stride, 19, self.criterion.reg_scale)
            distances, gt_distances, points = [], [], []
            h, w = p["reg"].shape[-2:]
            for b, _, flat in pos.nonzero().tolist():
                y, x = divmod(flat, w)
                distances.append(p["reg"][b, :, y, x] *
                                 self.criterion.reg_scale * stride)
                points.append(pts[flat])
                gt_distances.append(boxes_to_distances(
                    pts[flat:flat + 1], self.targets[b]["boxes"])[0])
            pd, td, points = map(torch.stack, (distances, gt_distances, points))
            expected = expected + (1 - box_ciou(
                boxes_from_distances(points, pd),
                boxes_from_distances(points, td))).mean() + 0.1 * F.l1_loss(pd, td)
        torch.testing.assert_close(actual, expected)
        tensors = [p["reg"] for p in preds]
        actual_grads = torch.autograd.grad(actual, tensors, retain_graph=True)
        expected_grads = torch.autograd.grad(expected, tensors)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad)
            self.assertGreater(float(actual_grad.abs().sum()), 0.)

    def test_empty_targets_are_finite_and_backward_works(self):
        preds = self.predictions()
        targets = [{"boxes": torch.empty(0, 4),
                    "labels": torch.empty(0, dtype=torch.long)} for _ in range(2)]
        losses = self.criterion(preds, targets, (128, 128))
        self.assertEqual(float(losses["box_o2m"]), 0.)
        self.assertTrue(bool(torch.isfinite(losses["total"])))
        losses["total"].backward()
        self.assertIsNotNone(preds[0]["cls"].grad)


if __name__ == "__main__":
    unittest.main()
