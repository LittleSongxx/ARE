import unittest

import torch

from ARiADNE.driver import (
    _append_perf_metrics,
    _actor_update_due,
    _maybe_sync_target_networks,
    _resolve_entropy_target_tensor,
    _resolve_updates_to_run,
)
from ARiADNE.parameter import RuntimeConfig


class DriverScheduleTests(unittest.TestCase):
    def test_perf_metric_accumulator_preserves_dynamic_episode_metrics(self):
        perf_metrics = {}
        _append_perf_metrics(
            perf_metrics,
            {
                "travel_dist": 3.0,
                "reward_info": 1.5,
                "reward_terminal": 7.0,
            },
        )
        self.assertEqual(perf_metrics["travel_dist"], [3.0])
        self.assertEqual(perf_metrics["reward_info"], [1.5])
        self.assertEqual(perf_metrics["reward_terminal"], [7.0])

    def test_replay_ratio_budget_is_capped_by_train_updates(self):
        config = RuntimeConfig(train_updates_per_iter=4, replay_ratio=2.0)
        self.assertEqual(_resolve_updates_to_run(config, pending_update_budget=10.0), 4)
        self.assertEqual(_resolve_updates_to_run(config, pending_update_budget=2.9), 2)

        baseline = RuntimeConfig(train_updates_per_iter=3, replay_ratio=0.0)
        self.assertEqual(_resolve_updates_to_run(baseline, pending_update_budget=0.0), 3)

    def test_policy_delay_only_triggers_periodic_actor_updates(self):
        self.assertFalse(_actor_update_due(learner_update_step=0, policy_delay=2))
        self.assertTrue(_actor_update_due(learner_update_step=1, policy_delay=2))
        self.assertFalse(_actor_update_due(learner_update_step=2, policy_delay=2))
        self.assertTrue(_actor_update_due(learner_update_step=0, policy_delay=1))

    def test_soft_target_update_moves_target_without_hard_reset(self):
        target_q1 = torch.nn.Linear(1, 1, bias=False)
        target_q2 = torch.nn.Linear(1, 1, bias=False)
        source_q1 = torch.nn.Linear(1, 1, bias=False)
        source_q2 = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            target_q1.weight.fill_(0.0)
            target_q2.weight.fill_(0.0)
            source_q1.weight.fill_(1.0)
            source_q2.weight.fill_(1.0)

        counter = _maybe_sync_target_networks(
            RuntimeConfig(enable_soft_target_update=True, tau=0.25),
            17,
            target_q1,
            target_q2,
            source_q1,
            source_q2,
        )
        self.assertEqual(counter, 17)
        self.assertAlmostEqual(float(target_q1.weight.item()), 0.25, places=6)
        self.assertAlmostEqual(float(target_q2.weight.item()), 0.25, places=6)

    def test_hard_target_sync_resets_counter_and_copies_weights(self):
        target_q1 = torch.nn.Linear(1, 1, bias=False)
        target_q2 = torch.nn.Linear(1, 1, bias=False)
        source_q1 = torch.nn.Linear(1, 1, bias=False)
        source_q2 = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            target_q1.weight.fill_(0.0)
            target_q2.weight.fill_(0.0)
            source_q1.weight.fill_(2.0)
            source_q2.weight.fill_(3.0)

        counter = _maybe_sync_target_networks(
            RuntimeConfig(enable_soft_target_update=False, tau=0.0),
            64,
            target_q1,
            target_q2,
            source_q1,
            source_q2,
        )
        self.assertEqual(counter, 1)
        self.assertAlmostEqual(float(target_q1.weight.item()), 2.0, places=6)
        self.assertAlmostEqual(float(target_q2.weight.item()), 3.0, places=6)

    def test_adaptive_entropy_target_uses_valid_actions(self):
        edge_padding_mask = torch.tensor(
            [
                [[0, 0, 0, 1]],
                [[0, 1, 1, 1]],
            ],
            dtype=torch.int64,
        )
        config = RuntimeConfig(enable_adaptive_entropy_target=True, entropy_target_scale=0.05)
        entropy_target_tensor, n_valid_mean, entropy_target_mean = _resolve_entropy_target_tensor(
            edge_padding_mask,
            config,
            default_entropy_target=0.1,
        )

        expected = 0.05 * torch.log(torch.tensor([3.0, 1.0]))
        self.assertTrue(torch.allclose(entropy_target_tensor, expected))
        self.assertAlmostEqual(n_valid_mean, 2.0, places=6)
        self.assertAlmostEqual(entropy_target_mean, float(entropy_target_tensor.mean().item()), places=6)


if __name__ == "__main__":
    unittest.main()
