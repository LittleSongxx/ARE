from __future__ import annotations

import unittest

import numpy as np
import torch

from ARiADNE.replay_buffer import ReplayBuffer, TRANSITION_FIELDS


def _transition(marker: int, with_bias: bool = False) -> dict[str, torch.Tensor | None]:
    transition: dict[str, torch.Tensor | None] = {}
    for field_name in TRANSITION_FIELDS:
        if "attn_bias" in field_name:
            transition[field_name] = torch.full((1, 1, 2, 2), float(marker), dtype=torch.float32) if with_bias else None
        elif field_name in {"done", "n_step_actual"}:
            transition[field_name] = torch.tensor([[int(marker)]], dtype=torch.int64)
        elif field_name == "action":
            transition[field_name] = torch.tensor([[int(marker)]], dtype=torch.int64)
        else:
            transition[field_name] = torch.tensor([[float(marker)]], dtype=torch.float32)
    return transition


class ReplayBufferTests(unittest.TestCase):
    def test_uniform_sampling_returns_unit_is_weights(self):
        np.random.seed(0)
        replay_buffer = ReplayBuffer(capacity=4, prioritized=False)
        for marker in range(4):
            replay_buffer.push(_transition(marker))

        sample = replay_buffer.sample(batch_size=3)
        self.assertEqual(replay_buffer.size, 4)
        self.assertEqual(tuple(sample.batch["reward"].shape), (3, 1, 1))
        self.assertTrue(torch.all(sample.is_weights == 1.0))

    def test_capacity_overwrite_keeps_latest_transitions(self):
        np.random.seed(1)
        replay_buffer = ReplayBuffer(capacity=3, prioritized=False)
        for marker in range(5):
            replay_buffer.push(_transition(marker))

        sample = replay_buffer.sample(batch_size=3)
        sampled_rewards = {float(value.item()) for value in sample.batch["reward"]}
        self.assertEqual(sampled_rewards, {2.0, 3.0, 4.0})

    def test_prioritized_sampling_prefers_higher_priority(self):
        np.random.seed(2)
        replay_buffer = ReplayBuffer(capacity=4, prioritized=True, alpha=1.0)
        for marker in range(4):
            replay_buffer.push(_transition(marker))
        replay_buffer.update_priorities([0, 1, 2, 3], [1.0, 1.0, 1.0, 10.0])

        counts = {marker: 0 for marker in range(4)}
        for _ in range(200):
            sample = replay_buffer.sample(batch_size=1, beta=0.4)
            reward_marker = int(sample.batch["reward"][0].item())
            counts[reward_marker] += 1
            self.assertLessEqual(float(sample.is_weights.max().item()), 1.0)
            self.assertGreater(float(sample.is_weights.min().item()), 0.0)

        self.assertGreater(counts[3], counts[0])
        self.assertGreater(counts[3], counts[1])
        self.assertGreater(counts[3], counts[2])

    def test_new_insertions_inherit_current_max_priority(self):
        replay_buffer = ReplayBuffer(capacity=4, prioritized=True, alpha=0.6)
        for marker in range(3):
            replay_buffer.push(_transition(marker))
        replay_buffer.update_priorities([0, 1, 2], [0.5, 3.0, 1.0])
        replay_buffer.push(_transition(9))

        self.assertAlmostEqual(float(replay_buffer._priorities[3]), 3.0)


if __name__ == "__main__":
    unittest.main()
