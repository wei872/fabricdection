"""Deterministic supervision/gradient checks; not measured accuracy gains."""
from pathlib import Path
import sys
import unittest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jingwei import load_config
from jingwei.losses import (DetectionLoss, balanced_class_focal_loss,
                            build_dense_targets, build_o2o_quality_targets)


class TestQualitySupervision(unittest.TestCase):
    def config(self):
        cfg = load_config(ROOT / 'configs/jingwei_dinov2.yaml')
        cfg.data.nc = 1
        return cfg

    def predictions(self, distance=.125):
        return [dict(cls=torch.full((1, 1, h, h), -2., requires_grad=True),
                     reg=torch.full((1, 4, h, h), distance, requires_grad=True),
                     obj=torch.zeros(1, 1, h, h, requires_grad=True),
                     obj_o2o=torch.full((1, 1, h, h), 2., requires_grad=True), fg=None)
                for h in (4, 1)]

    def target(self, size=8.):
        return [dict(boxes=torch.tensor([[0., 0., size, size]]), labels=torch.tensor([0]))]

    def test_poor_boxes_do_not_reverse_correct_class_gradient(self):
        predictions = self.predictions(.01)
        target = self.target(32.)
        classes, _, quality, _, pos, _ = build_dense_targets(predictions[0], target, 8, 1, 1.)
        logits = predictions[0]['cls'].flatten(2)
        self.assertTrue(pos.any())
        self.assertLess(float(quality[pos].max()), .001)
        old_target = classes * quality
        old = (F.binary_cross_entropy_with_logits(logits, old_target, reduction='none') *
               (logits.sigmoid() - old_target).square()).sum() / (pos.sum() + 1.)
        new = balanced_class_focal_loss(logits, classes, torch.ones(1))
        old_grad = torch.autograd.grad(old, logits, retain_graph=True)[0]
        new_grad = torch.autograd.grad(new, logits)[0]
        self.assertTrue((old_grad[pos] > 0).all())  # old loss suppresses true class
        self.assertTrue((new_grad[pos] < 0).all())  # gradient descent now raises it

    def test_normal_negative_loss_does_not_scale_with_background_grid(self):
        a = torch.full((1, 19, 64), -2., requires_grad=True)
        b = torch.full((1, 19, 6400), -2., requires_grad=True)
        loss_a = balanced_class_focal_loss(a, torch.zeros_like(a), torch.ones(19))
        loss_b = balanced_class_focal_loss(b, torch.zeros_like(b), torch.ones(19))
        torch.testing.assert_close(loss_a, loss_b)
        loss_b.backward()
        self.assertGreater(float(b.grad.sum()), 0.)

    def test_per_image_balancing_matches_separate_images(self):
        torch.manual_seed(5)
        logits = torch.randn(2, 3, 100)
        target = torch.zeros_like(logits)
        target[0, 1, :7] = 1
        together = balanced_class_focal_loss(logits, target, torch.ones(3), min_neg=8)
        separate = sum(balanced_class_focal_loss(x[None], y[None], torch.ones(3), min_neg=8)
                       for x, y in zip(logits, target)) / 2
        torch.testing.assert_close(together, separate)

    def test_quality_targets_equal_iou_and_are_detached(self):
        pred = self.predictions()[0]
        quality, mask = build_o2o_quality_targets(pred, [(0, 0, 0)], self.target(), 8, 1.)
        self.assertEqual(float(quality[0, 0]), .0625)
        self.assertEqual(int(mask.sum()), 1)
        self.assertFalse(quality.requires_grad)
        perfect = self.predictions(.5)[0]
        quality, _ = build_o2o_quality_targets(perfect, [(0, 0, 0)], self.target(), 8, 1.)
        self.assertAlmostEqual(float(quality[0, 0]), 1.)

    def test_overconfident_bad_box_is_pushed_down_good_box_up(self):
        for distance, sign in ((.125, 1), (.5, -1)):
            preds = self.predictions(distance)
            criterion = DetectionLoss(self.config())
            criterion(preds, self.target(), (32, 32))['obj_o2o'].backward()
            self.assertGreater(sign * float(preds[0]['obj_o2o'].grad[0, 0, 0, 0]), 0.)
            self.assertIsNone(preds[0]['reg'].grad)  # quality target cannot cheat through regression

    def test_zero_quality_positive_is_not_mined_as_negative(self):
        cfg = self.config()
        cfg.train.o2o_quality_floor = 0.
        preds = self.predictions(0.)
        DetectionLoss(cfg)(preds, self.target(), (32, 32))['obj_o2o'].backward()
        self.assertAlmostEqual(float(preds[0]['obj_o2o'].grad[0, 0, 0, 0]),
                               float(torch.tensor(2.).sigmoid()), places=6)

    def test_deployed_match_outside_dense_center_is_not_a_dense_negative(self):
        p = dict(cls=torch.full((1, 2, 8, 8), -2.), reg=torch.ones(1, 4, 8, 8))
        target = [dict(boxes=torch.tensor([[0., 0., 64., 64.], [0., 0., 8., 8.]]),
                       labels=torch.tensor([0, 1]))]
        before = build_dense_targets(p, target, 8, 2, 1.)
        # Dense chooses the small class-1 GT at point (4,4). Deployed matching
        # can allocate that cell to GT 0; all shared targets must agree with it.
        self.assertEqual(float(before[0][0, 1, 0]), 1.)
        after = build_dense_targets(p, target, 8, 2, 1., matches=[(0, 0, 0)])
        self.assertEqual(after[0][0, :, 0].tolist(), [1., 0.])
        self.assertEqual(after[1][0, :, 0].tolist(), [4., 4., 60., 60.])
        self.assertTrue(after[4][0, 0, 0])

    def test_bad_quality_config_rejected(self):
        for key, value in (('o2o_quality_floor', 1.), ('o2o_quality_floor', float('nan')),
                           ('cls_min_neg', 0), ('cls_neg_ratio', -1.), ('cls_neg_ratio', float('inf'))):
            cfg = self.config()
            cfg.train[key] = value
            with self.assertRaises(ValueError):
                DetectionLoss(cfg)


if __name__ == '__main__':
    unittest.main()
