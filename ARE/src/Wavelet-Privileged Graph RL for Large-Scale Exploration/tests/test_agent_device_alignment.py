from __future__ import annotations

import unittest

import numpy as np
import torch

from wpg_rl.agent import Agent


class _DummyPolicy(torch.nn.Module):
    def __init__(self, device: str | torch.device = "cpu"):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1, device=device))

    def forward(self, *args, **kwargs):  # pragma: no cover - unused in this test
        raise NotImplementedError


class AgentDeviceAlignmentTests(unittest.TestCase):
    def test_get_observation_places_all_tensors_on_agent_device(self):
        device = torch.device("meta")
        agent = Agent(_DummyPolicy(device=device), device=device)
        agent.node_coords = np.asarray([[0.0, 0.0], [4.0, 0.0]], dtype=float)
        agent.utility = np.asarray([1.0, 0.0], dtype=float)
        agent.guidepost = np.asarray([1.0, 0.0], dtype=float)
        agent.current_index = 0
        agent.adjacent_matrix = np.asarray([[0, 0], [0, 0]], dtype=int)
        agent.neighbor_indices = np.asarray([0, 1], dtype=int)

        observation = agent.get_observation()

        devices = [tensor.device for tensor in observation]
        self.assertTrue(all(device == target for target in devices))


if __name__ == "__main__":
    unittest.main()
