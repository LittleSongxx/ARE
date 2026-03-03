import unittest

import numpy as np
import torch

from ARiADNE.agent import Agent
from ARiADNE.env import Env
from ARiADNE.ground_truth_node_manager import GroundTruthNodeManager
from ARiADNE.model import PolicyNet
from ARiADNE.parameter import EMBEDDING_DIM, NODE_INPUT_DIM, RuntimeConfig, get_critic_node_input_dim
from ARiADNE.utils import multiscale_haar_energy_map
from ARiADNE.worker import Worker


class WaveletShapeTests(unittest.TestCase):
    def test_multiscale_haar_energy_map_shape_and_range(self):
        grid = np.array(
            [
                [255, 255, 127, 1],
                [255, 127, 127, 1],
                [255, 255, 255, 1],
                [1, 1, 127, 127],
            ],
            dtype=np.int16,
        )
        wavelet_map = multiscale_haar_energy_map(grid)
        self.assertEqual(wavelet_map.shape, grid.shape)
        self.assertGreaterEqual(float(wavelet_map.min()), 0.0)
        self.assertLessEqual(float(wavelet_map.max()), 1.0)

    def test_agent_observation_last_dim(self):
        env = Env(0, plot=False)
        model = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM)
        agent = Agent(model, device="cpu", plot=False)
        agent.update_planning_state(env.belief_info, env.robot_location)
        observation = agent.get_observation()
        self.assertEqual(observation[0].shape[-1], NODE_INPUT_DIM)
        self.assertEqual(len(observation), 7)

    def test_ground_truth_observation_last_dim(self):
        env = Env(0, plot=False)
        model = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM)
        agent = Agent(model, device="cpu", plot=False)
        agent.update_planning_state(env.belief_info, env.robot_location)
        manager = GroundTruthNodeManager(agent.node_manager, env.ground_truth_info, device="cpu", plot=False)
        observation = manager.get_ground_truth_observation(env.robot_location)
        self.assertEqual(observation[0].shape[-1], get_critic_node_input_dim(RuntimeConfig()))
        self.assertEqual(len(observation), 7)

    def test_worker_coordinate_assertions_still_pass(self):
        torch.manual_seed(7)
        np.random.seed(7)
        model = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM)
        runtime_config = RuntimeConfig(
            max_episodes=1,
            num_meta_agent=1,
            max_episode_step=1,
            minimum_buffer_size=1,
            batch_size=1,
            replay_size=1,
            save_img_gap=1,
            summary_window=1,
            train_updates_per_iter=1,
            run_session="test_wavelet_shapes",
        )
        worker = Worker(0, model, 0, runtime_config=runtime_config, device="cpu", save_image=False)
        worker.robot.update_planning_state(worker.env.belief_info, worker.env.robot_location)
        observation = worker.robot.get_observation()
        gt_observation = worker.ground_truth_node_manager.get_ground_truth_observation(worker.env.robot_location)
        worker.save_observation(observation, gt_observation)
        self.assertEqual(len(worker.episode_buffer[0]), 1)


if __name__ == "__main__":
    unittest.main()
