from __future__ import annotations

import unittest

import numpy as np

from hpbg_rl.expert_reward import apply_expert_reward, compute_oracle_node_targets, compute_state_exploration_potential
from hpbg_rl.parameter import FREE, OCCUPIED, UNKNOWN
from hpbg_rl.utils import MapInfo


class ExpertRewardTests(unittest.TestCase):
    def test_potential_based_shaping_formula(self):
        shaped = apply_expert_reward(
            reward=1.0,
            previous_potential=0.25,
            next_potential=0.5,
            gamma=0.9,
            shaping_weight=2.0,
            oracle_gain_weight=0.0,
        )
        self.assertAlmostEqual(shaped, 1.0 + 2.0 * (0.9 * 0.5 - 0.25))

    def test_state_potential_uses_ground_truth_free_area(self):
        truth = np.asarray([[FREE, FREE, OCCUPIED], [FREE, OCCUPIED, FREE]], dtype=int)
        belief = np.asarray([[FREE, UNKNOWN, OCCUPIED], [FREE, OCCUPIED, UNKNOWN]], dtype=int)
        self.assertAlmostEqual(compute_state_exploration_potential(belief, truth), 0.5)

    def test_oracle_node_targets_shape_range_and_finiteness(self):
        truth = np.full((15, 15), FREE, dtype=int)
        truth[0, :] = OCCUPIED
        truth[:, 0] = OCCUPIED
        belief = np.full_like(truth, UNKNOWN)
        belief[2:5, 2:5] = FREE
        ground_truth_info = MapInfo(truth, 0.0, 0.0, 0.4)
        belief_info = MapInfo(belief, 0.0, 0.0, 0.4)
        node_coords = np.asarray([[1.2, 1.2], [4.0, 4.0]], dtype=float)

        targets = compute_oracle_node_targets(node_coords, ground_truth_info, belief_info, sensor_range=1.2)

        self.assertEqual(targets.oracle_utility.shape, (2,))
        self.assertEqual(targets.expert_potential.shape, (2,))
        self.assertTrue(np.isfinite(targets.oracle_utility).all())
        self.assertTrue(np.isfinite(targets.expert_potential).all())
        self.assertTrue(np.all(targets.oracle_utility >= 0.0))
        self.assertTrue(np.all(targets.oracle_utility <= 1.0))
        self.assertTrue(np.all(targets.expert_potential >= 0.0))
        self.assertTrue(np.all(targets.expert_potential <= 1.0))


if __name__ == "__main__":
    unittest.main()