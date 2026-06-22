from __future__ import annotations

import unittest

import torch

from hpbg_rl.belief_losses import BeliefDistillationLoss


class BeliefLossTests(unittest.TestCase):
    def test_masked_loss_ignores_invalid_nodes_and_warmup(self):
        loss_fn = BeliefDistillationLoss(weight=0.5, warmup_updates=2, ramp_updates=0)
        prediction = torch.zeros(1, 4, 3)
        target = torch.zeros(1, 4, 3)
        prediction[:, 2:] = 10.0
        target[:, 2:] = -10.0
        valid_mask = torch.tensor([[True, True, False, False]])

        warmup = loss_fn(prediction, target, valid_mask, update_step=0)
        active = loss_fn(prediction, target, valid_mask, update_step=2)

        self.assertLess(float(warmup["loss"]), 1e-5)
        self.assertEqual(float(warmup["weighted_loss"]), 0.0)
        self.assertEqual(warmup["lambda_eff"], 0.0)
        self.assertLess(float(active["loss"]), 1e-5)
        self.assertLess(float(active["weighted_loss"]), 1e-5)
        self.assertEqual(active["lambda_eff"], 0.5)
        self.assertIn("explored_loss", active)
        self.assertIn("oracle_loss", active)
        self.assertIn("potential_loss", active)

    def test_target_is_detached_and_prediction_receives_gradient(self):
        loss_fn = BeliefDistillationLoss(weight=1.0, warmup_updates=0, ramp_updates=0)
        prediction = torch.full((1, 2, 3), 0.5, requires_grad=True)
        target = torch.ones(1, 2, 3, requires_grad=True)
        valid_mask = torch.tensor([[True, True]])

        metrics = loss_fn(prediction, target, valid_mask, update_step=0)
        metrics["weighted_loss"].backward()

        self.assertIsNotNone(prediction.grad)
        self.assertIsNone(target.grad)

    def test_shape_mismatch_fails_fast(self):
        loss_fn = BeliefDistillationLoss(weight=0.5, warmup_updates=0, ramp_updates=0)
        with self.assertRaises(ValueError):
            loss_fn(torch.zeros(1, 2, 3), torch.zeros(1, 2, 2), torch.ones(1, 2, dtype=torch.bool), 0)


if __name__ == "__main__":
    unittest.main()