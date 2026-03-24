import unittest

import numpy as np

from ARiADNE_Wavelet.agent import Agent
from ARiADNE_Wavelet.model import PolicyNet
from ARiADNE_Wavelet.parameter import EMBEDDING_DIM, NODE_INPUT_DIM


class HistoryFeatureTests(unittest.TestCase):
    def test_transition_feedback_is_encoded_into_history_vector(self):
        model = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM, enable_wavelet_history=False)
        agent = Agent(model, device="cpu", plot=False)

        agent.utility = np.array([2.0, 1.0, 0.0], dtype=np.float32)
        agent.guidepost = np.array([0.0, 1.0, 1.0], dtype=np.float32)
        agent.frontier = {(0.0, 0.0), (1.0, 1.0)}
        agent._previous_frontier_count = 1

        agent.record_transition_feedback(
            reward_proxy=0.75,
            explored_delta=0.12,
            travel_delta=0.6,
            selected_node_index=3,
            selected_node_utility=1.8,
        )
        agent._update_history_buffer()

        self.assertEqual(len(agent.history_buffer), 1)
        history_vec = agent.history_buffer[-1]

        reward_idx = agent.history_feature_builder.feature_set.index("reward_proxy")
        explored_idx = agent.history_feature_builder.feature_set.index("explored_delta")
        self.assertAlmostEqual(float(history_vec[reward_idx]), 0.75, places=6)
        self.assertAlmostEqual(float(history_vec[explored_idx]), 0.12, places=6)


if __name__ == "__main__":
    unittest.main()
