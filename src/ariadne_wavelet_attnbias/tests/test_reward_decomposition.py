import unittest
from unittest.mock import patch

import numpy as np

from ariadne_wavelet_attnbias.env import Env
from ariadne_wavelet_attnbias.parameter import FRONTIER_CELL_SIZE, SENSOR_RANGE, UPDATING_MAP_SIZE, RLOptions


class RewardDecompositionTests(unittest.TestCase):
    def _make_env(self, rl_options: RLOptions):
        env = Env.__new__(Env)
        env.rl_options = rl_options
        env.global_frontiers = {(0, 0), (1, 1)}
        env.belief_info = object()
        env.old_belief = np.zeros((2, 2), dtype=np.int16)
        env.robot_belief = np.ones((2, 2), dtype=np.int16)
        env.last_reward_components = {}
        return env

    def test_legacy_reward_is_preserved_when_decomposition_disabled(self):
        env = self._make_env(RLOptions(use_reward_decomposition=False))
        dist = 5.0
        expected_dist = -dist / UPDATING_MAP_SIZE * 5
        expected_info = 1.0 / (SENSOR_RANGE * 3.14 // FRONTIER_CELL_SIZE)

        with patch("ariadne_wavelet_attnbias.env.get_frontier_in_map", return_value={(1, 1)}):
            reward = Env.calculate_reward(env, dist)

        self.assertAlmostEqual(reward, expected_dist + expected_info, places=6)
        self.assertEqual(env.last_reward_components, {})

    def test_reward_components_sum_to_total_with_terminal_bonus(self):
        env = self._make_env(
            RLOptions(
                use_reward_decomposition=True,
                r_info_w=2.0,
                r_dist_w=0.5,
                r_safe_w=3.0,
                r_terminal_bonus=7.0,
            )
        )
        dist = 5.0

        with patch("ariadne_wavelet_attnbias.env.get_frontier_in_map", return_value={(1, 1)}):
            reward = Env.calculate_reward(env, dist)
        reward = Env.apply_terminal_bonus(env, reward)

        components = env.last_reward_components
        self.assertAlmostEqual(
            components["r_info"] + components["r_dist"] + components["r_safe"] + components["r_terminal"],
            reward,
            places=6,
        )
        self.assertAlmostEqual(components["total"], reward, places=6)
        self.assertAlmostEqual(components["r_terminal"], 7.0, places=6)


if __name__ == "__main__":
    unittest.main()
