import unittest

import numpy as np
import torch

from ARiADNE.env import Env
from ARiADNE.model import PolicyNet
from ARiADNE.node_manager_infer import NodeManager
from ARiADNE.parameter import EMBEDDING_DIM, RuntimeConfig, get_node_input_dim
from ARiADNE.utils import get_collision_counter, get_frontier_in_map, reset_collision_counter


class InferenceWaveletGateTests(unittest.TestCase):
    def test_wavelet_guided_sampling_and_skip_do_not_increase_collision_checks(self):
        torch.manual_seed(0)
        np.random.seed(0)
        env = Env(0, plot=False)
        baseline_cfg = RuntimeConfig(use_wavelet_feature=True, run_session="infer_base")
        optimized_cfg = baseline_cfg.with_overrides(
            wavelet_skip_utility_updates=True,
            wavelet_guided_node_sampling=True,
            wavelet_adaptive_dth=True,
        )

        baseline_manager = NodeManager(runtime_config=baseline_cfg, start=env.robot_location.copy())
        optimized_manager = NodeManager(runtime_config=optimized_cfg, start=env.robot_location.copy())
        frontiers = get_frontier_in_map(env.belief_info)

        reset_collision_counter()
        baseline_manager.update_graph(env.robot_location, frontiers, env.belief_info, env.belief_info)
        baseline_manager.update_graph(env.robot_location, frontiers, env.belief_info, env.belief_info)
        baseline_checks = get_collision_counter()

        reset_collision_counter()
        optimized_manager.update_graph(env.robot_location, frontiers, env.belief_info, env.belief_info)
        optimized_manager.update_graph(env.robot_location, frontiers, env.belief_info, env.belief_info)
        optimized_checks = get_collision_counter()

        self.assertLessEqual(optimized_checks, baseline_checks)
        self.assertGreater(len(optimized_manager.nodes_dict), 0)


if __name__ == "__main__":
    unittest.main()
