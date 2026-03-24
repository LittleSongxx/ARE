from __future__ import annotations

import unittest

import numpy as np

from wpg_rl.ground_truth_node_manager import build_actor_to_critic_index


class OverlapMappingTests(unittest.TestCase):
    def test_actor_to_critic_index_handles_subset_with_different_order(self):
        critic_node_coords = np.array(
            [
                [4.0, 0.0],
                [0.0, 0.0],
                [8.0, 0.0],
                [2.0, 0.0],
                [6.0, 0.0],
            ]
        )
        actor_node_coords = np.array(
            [
                [6.0001, 0.0],
                [0.0, 0.0],
                [4.0, 0.0001],
            ]
        )

        actor_to_critic_index = build_actor_to_critic_index(
            actor_node_coords,
            critic_node_coords,
            padded_size=5,
            digits=3,
        )

        self.assertEqual(actor_to_critic_index[:3].tolist(), [4, 1, 0])
        self.assertEqual(actor_to_critic_index[3:].tolist(), [-1, -1])


if __name__ == "__main__":
    unittest.main()
