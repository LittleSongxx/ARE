from __future__ import annotations

import numpy as np
import torch

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    PACKAGE_ROOT = Path(__file__).resolve().parent
    if str(PACKAGE_ROOT.parent) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT.parent))

    from ARiADNE.agent import Agent
    from ARiADNE.env import Env
    from ARiADNE.ground_truth_node_manager import GroundTruthNodeManager
    from ARiADNE.parameter import EMBEDDING_DIM, RuntimeConfig, get_gifs_path, NODE_INPUT_DIM
    from ARiADNE.utils import build_artifact_stem, ensure_bucket_dir, finalize_episode_artifacts
else:
    from .agent import Agent
    from .env import Env
    from .ground_truth_node_manager import GroundTruthNodeManager
    from .parameter import EMBEDDING_DIM, RuntimeConfig, get_gifs_path, NODE_INPUT_DIM
    from .utils import build_artifact_stem, ensure_bucket_dir, finalize_episode_artifacts


class Worker:
    def __init__(
        self,
        meta_agent_id,
        policy_net,
        global_step,
        runtime_config: RuntimeConfig,
        device="cpu",
        save_image=False,
        forced_map_path=None,
    ):
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.device = device
        self.runtime_config = runtime_config
        self.episode_artifact_stem = build_artifact_stem(global_step)
        self.output_dir = ensure_bucket_dir(
            get_gifs_path(runtime_config),
            global_step,
            runtime_config.result_bucket_episodes,
        )
        self.episode_return = 0.0
        self.episode_steps = 0

        self.env = Env(
            global_step,
            plot=self.save_image,
            output_dir=self.output_dir,
            artifact_stem=self.episode_artifact_stem,
            forced_map_path=forced_map_path,
        )
        self.robot = Agent(policy_net, runtime_config=self.runtime_config, device=self.device, plot=self.save_image)
        self.ground_truth_node_manager = GroundTruthNodeManager(
            self.robot.node_manager,
            self.env.ground_truth_info,
            runtime_config=self.runtime_config,
            device=self.device,
            plot=self.save_image,
        )

        self.episode_buffer = [[] for _ in range(31)]
        self.perf_metrics = {}

    @staticmethod
    def _buffer_tensor(tensor: torch.Tensor | None):
        if tensor is None:
            return None
        return tensor.detach().to("cpu")

    def run_episode(self):
        done = False
        self.robot.update_planning_state(self.env.belief_info, self.env.robot_location)
        observation = self.robot.get_observation()
        ground_truth_observation = self.ground_truth_node_manager.get_ground_truth_observation(self.env.robot_location)

        if self.save_image:
            self.robot.plot_env()
            self.ground_truth_node_manager.plot_ground_truth_env(self.env.robot_location)
            self.env.plot_env(0)

        for i in range(self.runtime_config.max_episode_step):
            self.save_observation(observation, ground_truth_observation)

            next_location, action_index = self.robot.select_next_waypoint(observation)
            self.save_action(action_index)

            node = self.robot.node_manager.nodes_dict.find((self.robot.location[0], self.robot.location[1]))
            check = np.array(list(node.data.neighbor_set)).reshape(-1, 2)
            assert next_location[0] + next_location[1] * 1j in check[:, 0] + check[:, 1] * 1j
            assert next_location[0] != self.robot.location[0] or next_location[1] != self.robot.location[1]

            reward = self.env.step(next_location)
            self.robot.update_planning_state(self.env.belief_info, self.env.robot_location)
            if self.robot.utility.sum() == 0:
                done = True
                reward += 20
            self.episode_return += float(reward)
            self.episode_steps = i + 1
            self.save_reward_done(reward, done)

            observation = self.robot.get_observation()
            ground_truth_observation = self.ground_truth_node_manager.get_ground_truth_observation(self.env.robot_location)
            self.save_next_observations(observation, ground_truth_observation)

            if self.save_image:
                self.robot.plot_env()
                self.ground_truth_node_manager.plot_ground_truth_env(self.env.robot_location)
                self.env.plot_env(i + 1)

            if done:
                break

        self.perf_metrics["travel_dist"] = self.env.travel_dist
        self.perf_metrics["explored_rate"] = self.env.explored_rate
        self.perf_metrics["success_rate"] = done
        self.perf_metrics["episode_return"] = self.episode_return
        self.perf_metrics["episode_steps"] = self.episode_steps

        if self.save_image:
            finalize_episode_artifacts(self.output_dir, self.episode_artifact_stem, self.env.frame_files)

    def _append_optional_bias(self, slot_index, attn_bias):
        buffered = self._buffer_tensor(attn_bias)
        self.episode_buffer[slot_index].append(buffered)

    def save_observation(self, observation, ground_truth_observation):
        (
            node_inputs,
            node_padding_mask,
            edge_mask,
            current_index,
            current_edge,
            edge_padding_mask,
            attn_bias,
        ) = observation
        self.episode_buffer[0] += self._buffer_tensor(node_inputs)
        self.episode_buffer[1] += self._buffer_tensor(node_padding_mask.bool())
        self.episode_buffer[2] += self._buffer_tensor(edge_mask.bool())
        self.episode_buffer[3] += self._buffer_tensor(current_index)
        self.episode_buffer[4] += self._buffer_tensor(current_edge)
        self.episode_buffer[5] += self._buffer_tensor(edge_padding_mask.bool())
        self._append_optional_bias(27, attn_bias)

        (
            critic_node_inputs,
            critic_node_padding_mask,
            critic_edge_mask,
            critic_current_index,
            critic_current_edge,
            critic_edge_padding_mask,
            critic_attn_bias,
        ) = ground_truth_observation
        self.episode_buffer[15] += self._buffer_tensor(critic_node_inputs)
        self.episode_buffer[16] += self._buffer_tensor(critic_node_padding_mask.bool())
        self.episode_buffer[17] += self._buffer_tensor(critic_edge_mask.bool())
        self.episode_buffer[18] += self._buffer_tensor(critic_current_index)
        self.episode_buffer[19] += self._buffer_tensor(critic_current_edge)
        self.episode_buffer[20] += self._buffer_tensor(critic_edge_padding_mask.bool())
        self._append_optional_bias(29, critic_attn_bias)

        assert torch.all(current_edge == critic_current_edge)
        assert torch.all(
            node_inputs[0, current_index.item(), :2] == critic_node_inputs[0, critic_current_index.item(), :2]
        )
        assert torch.all(
            torch.gather(node_inputs, 1, current_edge.repeat(1, 1, 2))
            == torch.gather(critic_node_inputs, 1, critic_current_edge.repeat(1, 1, 2))
        )

    def save_action(self, action_index):
        self.episode_buffer[6] += self._buffer_tensor(action_index.reshape(1, 1, 1))

    def save_reward_done(self, reward, done):
        reward_tensor = torch.tensor([reward], dtype=torch.float32, device=self.device).reshape(1, 1, 1)
        done_tensor = torch.tensor([int(done)], dtype=torch.int64, device=self.device).reshape(1, 1, 1)
        self.episode_buffer[7] += self._buffer_tensor(reward_tensor)
        self.episode_buffer[8] += self._buffer_tensor(done_tensor)

    def save_next_observations(self, observation, ground_truth_observation):
        (
            node_inputs,
            node_padding_mask,
            edge_mask,
            current_index,
            current_edge,
            edge_padding_mask,
            attn_bias,
        ) = observation
        self.episode_buffer[9] += self._buffer_tensor(node_inputs)
        self.episode_buffer[10] += self._buffer_tensor(node_padding_mask.bool())
        self.episode_buffer[11] += self._buffer_tensor(edge_mask.bool())
        self.episode_buffer[12] += self._buffer_tensor(current_index)
        self.episode_buffer[13] += self._buffer_tensor(current_edge)
        self.episode_buffer[14] += self._buffer_tensor(edge_padding_mask.bool())
        self._append_optional_bias(28, attn_bias)

        (
            critic_node_inputs,
            critic_node_padding_mask,
            critic_edge_mask,
            critic_current_index,
            critic_current_edge,
            critic_edge_padding_mask,
            critic_attn_bias,
        ) = ground_truth_observation
        self.episode_buffer[21] += self._buffer_tensor(critic_node_inputs)
        self.episode_buffer[22] += self._buffer_tensor(critic_node_padding_mask.bool())
        self.episode_buffer[23] += self._buffer_tensor(critic_edge_mask.bool())
        self.episode_buffer[24] += self._buffer_tensor(critic_current_index)
        self.episode_buffer[25] += self._buffer_tensor(critic_current_edge)
        self.episode_buffer[26] += self._buffer_tensor(critic_edge_padding_mask.bool())
        self._append_optional_bias(30, critic_attn_bias)


if __name__ == "__main__":
    torch.manual_seed(4777)
    np.random.seed(4777)
    from .model import PolicyNet

    config = RuntimeConfig(run_session="worker_debug")
    model = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM)
    worker = Worker(0, model, 77, runtime_config=config, save_image=False)
    worker.run_episode()
