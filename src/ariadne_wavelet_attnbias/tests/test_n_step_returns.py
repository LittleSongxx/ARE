import unittest

import torch

from ariadne_wavelet_attnbias.worker import Worker


def _marker(value, dtype=torch.float32):
    return torch.tensor([[[value]]], dtype=dtype)


class NStepReturnTests(unittest.TestCase):
    def _make_episode_buffer(self):
        episode_buffer = [[] for _ in range(27)]
        rewards = [1.0, 2.0, 3.0, 4.0]
        dones = [0, 0, 1, 0]
        for index in range(len(rewards)):
            episode_buffer[7].append(_marker(rewards[index], dtype=torch.float32))
            episode_buffer[8].append(_marker(dones[index], dtype=torch.int64))
            for slot in Worker.NEXT_OBSERVATION_SLOTS + Worker.CRITIC_NEXT_OBSERVATION_SLOTS:
                episode_buffer[slot].append(_marker(slot * 100 + index, dtype=torch.float32))
        for slot in range(27):
            if episode_buffer[slot]:
                continue
            for index in range(len(rewards)):
                episode_buffer[slot].append(_marker(slot * 10 + index, dtype=torch.float32))
        return episode_buffer

    def test_n_step_postprocess_keeps_buffer_shape_and_updates_targets(self):
        episode_buffer = self._make_episode_buffer()

        processed = Worker.apply_n_step_returns(episode_buffer, n_step=3, gamma=0.9)

        expected_rewards = [1.0 + 0.9 * 2.0 + 0.9 * 0.9 * 3.0, 2.0 + 0.9 * 3.0, 3.0, 4.0]
        expected_dones = [1, 1, 1, 1]
        expected_next_indices = [2, 2, 2, 3]

        self.assertEqual(len(processed), 27)
        for slot in range(27):
            self.assertEqual(len(processed[slot]), 4)

        for index, expected_reward in enumerate(expected_rewards):
            self.assertAlmostEqual(float(processed[7][index].item()), expected_reward, places=6)
            self.assertEqual(int(processed[8][index].item()), expected_dones[index])

        for slot in Worker.NEXT_OBSERVATION_SLOTS + Worker.CRITIC_NEXT_OBSERVATION_SLOTS:
            for index, next_index in enumerate(expected_next_indices):
                self.assertEqual(
                    float(processed[slot][index].item()),
                    float(episode_buffer[slot][next_index].item()),
                )

    def test_n_step_disabled_returns_original_buffer(self):
        episode_buffer = self._make_episode_buffer()

        processed = Worker.apply_n_step_returns(episode_buffer, n_step=1, gamma=0.9)

        self.assertIs(processed, episode_buffer)


if __name__ == "__main__":
    unittest.main()
