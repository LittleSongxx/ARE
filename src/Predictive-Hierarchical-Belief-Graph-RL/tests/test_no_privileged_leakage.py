from __future__ import annotations

import unittest

import numpy as np
import torch

from hpbg_rl.agent import Agent
from hpbg_rl.parameter import CRITIC_NODE_INPUT_DIM, NODE_INPUT_DIM


class _DummyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, *args, **kwargs):  # pragma: no cover - not used here
        raise NotImplementedError


class NoPrivilegedLeakageTests(unittest.TestCase):
    def test_actor_observation_uses_online_width_only(self):
        agent = Agent(_DummyPolicy(), device="cpu")
        agent.node_coords = np.asarray([[0.0, 0.0], [4.0, 0.0], [8.0, 0.0]], dtype=float)
        agent.utility = np.asarray([2.0, 0.0, 1.0], dtype=float)
        agent.guidepost = np.asarray([1.0, 0.0, 0.0], dtype=float)
        agent.current_index = 0
        agent.adjacent_matrix = np.asarray(
            [
                [0, 0, 1],
                [0, 0, 0],
                [1, 0, 0],
            ],
            dtype=int,
        )
        agent.neighbor_indices = np.asarray([0, 1], dtype=int)

        observation = agent.get_observation()
        node_inputs = observation[0]

        self.assertEqual(node_inputs.shape[-1], NODE_INPUT_DIM)
        self.assertNotEqual(node_inputs.shape[-1], CRITIC_NODE_INPUT_DIM)
        self.assertEqual(CRITIC_NODE_INPUT_DIM - NODE_INPUT_DIM, 3)
        self.assertTrue(torch.isfinite(node_inputs).all())


if __name__ == "__main__":
    unittest.main()