import unittest

import torch
import torch.optim as optim

from ariadne_wavelet_attnbias.driver import (
    compute_distill_lambda,
    compute_distill_loss,
    compute_distill_teacher_probs,
    get_policy_optimizer_update_count,
    set_policy_optimizer_update_count,
)


class PrivilegedDistillTests(unittest.TestCase):
    def test_teacher_distribution_respects_mask_and_loss_is_finite(self):
        q_values = torch.tensor([[[1.0], [4.0], [2.0]]], dtype=torch.float32)
        edge_padding_mask = torch.tensor([[[0, 1, 0]]], dtype=torch.bool)
        logp = torch.log_softmax(torch.tensor([[1.0, -1e8, 0.5]], dtype=torch.float32), dim=-1)

        teacher_probs = compute_distill_teacher_probs(q_values, edge_padding_mask, tau=1.0)
        distill_loss = compute_distill_loss(logp, teacher_probs)

        self.assertAlmostEqual(float(teacher_probs[0, 1].item()), 0.0, places=8)
        self.assertAlmostEqual(float(teacher_probs.sum(dim=-1).item()), 1.0, places=6)
        self.assertTrue(torch.isfinite(distill_loss))

    def test_warmup_lambda_progression(self):
        self.assertAlmostEqual(compute_distill_lambda(0, 0.2, 10), 0.0, places=8)
        self.assertAlmostEqual(compute_distill_lambda(5, 0.2, 10), 0.1, places=8)
        self.assertAlmostEqual(compute_distill_lambda(10, 0.2, 10), 0.2, places=8)
        self.assertAlmostEqual(compute_distill_lambda(15, 0.2, 10), 0.2, places=8)

    def test_optimizer_update_count_roundtrip(self):
        param = torch.nn.Parameter(torch.ones(1))
        optimizer = optim.Adam([param], lr=1e-3)
        set_policy_optimizer_update_count(optimizer, 7)

        state_dict = optimizer.state_dict()
        restored = optim.Adam([torch.nn.Parameter(torch.ones(1))], lr=1e-3)
        restored.load_state_dict(state_dict)

        self.assertEqual(get_policy_optimizer_update_count(restored), 7)


if __name__ == "__main__":
    unittest.main()
