import unittest

import torch

from ARiADNE.replay_buffer import (
    CRITIC_NEXT_OBSERVATION_SLOTS,
    GAMMA_POW_SLOT,
    NEXT_OBSERVATION_SLOTS,
    N_STEP_ACTUAL_SLOT,
    empty_episode_buffer,
)
from ARiADNE.worker import Worker


def _marker(value, dtype=torch.float32):
    return torch.tensor([[value]], dtype=dtype)


def _make_episode_buffer(rewards, dones):
    episode_buffer = empty_episode_buffer()
    num_steps = len(rewards)
    for slot_index in range(len(episode_buffer)):
        for step_index in range(num_steps):
            if slot_index == Worker.REWARD_SLOT:
                episode_buffer[slot_index].append(_marker(rewards[step_index], dtype=torch.float32))
            elif slot_index == Worker.DONE_SLOT:
                episode_buffer[slot_index].append(_marker(dones[step_index], dtype=torch.int64))
            elif slot_index == GAMMA_POW_SLOT:
                episode_buffer[slot_index].append(_marker(1.0, dtype=torch.float32))
            elif slot_index == N_STEP_ACTUAL_SLOT:
                episode_buffer[slot_index].append(_marker(1, dtype=torch.int64))
            else:
                episode_buffer[slot_index].append(_marker(slot_index * 100 + step_index, dtype=torch.float32))
    return episode_buffer


class NStepReturnTests(unittest.TestCase):
    def test_n_step_postprocess_keeps_buffer_shape_and_updates_targets(self):
        episode_buffer = _make_episode_buffer([1.0, 2.0, 3.0, 4.0], [0, 0, 1, 0])

        processed = Worker.apply_n_step_returns(episode_buffer, n_step=3, gamma=0.9)

        expected_rewards = [1.0 + 0.9 * 2.0 + 0.9 * 0.9 * 3.0, 2.0 + 0.9 * 3.0, 3.0, 4.0]
        expected_dones = [1, 1, 1, 0]
        expected_gamma_pow = [0.9 ** 3, 0.9 ** 2, 0.9, 0.9]
        expected_next_indices = [2, 2, 2, 3]

        self.assertEqual(len(processed), len(episode_buffer))
        for slot_index in range(len(processed)):
            self.assertEqual(len(processed[slot_index]), 4)

        for index, expected_reward in enumerate(expected_rewards):
            self.assertAlmostEqual(float(processed[Worker.REWARD_SLOT][index].item()), expected_reward, places=6)
            self.assertEqual(int(processed[Worker.DONE_SLOT][index].item()), expected_dones[index])
            self.assertAlmostEqual(float(processed[GAMMA_POW_SLOT][index].item()), expected_gamma_pow[index], places=6)
            self.assertEqual(int(processed[N_STEP_ACTUAL_SLOT][index].item()), expected_next_indices[index] - index + 1)

        for slot_index in NEXT_OBSERVATION_SLOTS + CRITIC_NEXT_OBSERVATION_SLOTS:
            for index, next_index in enumerate(expected_next_indices):
                self.assertEqual(
                    float(processed[slot_index][index].item()),
                    float(episode_buffer[slot_index][next_index].item()),
                )

    def test_tail_without_terminal_keeps_done_zero_and_uses_available_steps(self):
        episode_buffer = _make_episode_buffer([1.0, 2.0], [0, 0])

        processed = Worker.apply_n_step_returns(episode_buffer, n_step=3, gamma=0.5)

        self.assertAlmostEqual(float(processed[Worker.REWARD_SLOT][0].item()), 1.0 + 0.5 * 2.0, places=6)
        self.assertEqual(int(processed[Worker.DONE_SLOT][0].item()), 0)
        self.assertAlmostEqual(float(processed[GAMMA_POW_SLOT][0].item()), 0.5 ** 2, places=6)
        self.assertEqual(int(processed[N_STEP_ACTUAL_SLOT][0].item()), 2)

    def test_n_step_disabled_returns_original_buffer(self):
        episode_buffer = _make_episode_buffer([1.0, 2.0], [0, 0])

        processed = Worker.apply_n_step_returns(episode_buffer, n_step=1, gamma=0.9)

        self.assertIs(processed, episode_buffer)


if __name__ == "__main__":
    unittest.main()
