from __future__ import annotations

import numpy as np
import torch

from .agent import Agent
from .env import Env
from .ground_truth_node_manager import GroundTruthNodeManager
from .parameter import EMBEDDING_DIM, GAMMA, NODE_INPUT_DIM, RuntimeConfig, get_gifs_path
from .utils import build_artifact_stem, ensure_bucket_dir, finalize_episode_artifacts


class Worker:
    NEXT_OBSERVATION_SLOTS = (9, 10, 11, 12, 13, 14)
    CRITIC_NEXT_OBSERVATION_SLOTS = (21, 22, 23, 24, 25, 26)

    def __init__(
        self,
        meta_agent_id,
        policy_net,
        global_step,
        runtime_config: RuntimeConfig,
        device="cpu",
        save_image=False,
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
        self.reward_component_sums = {
            "reward_info": 0.0,
            "reward_dist": 0.0,
            "reward_safe": 0.0,
            "reward_terminal": 0.0,
        }

        self.env = Env(
            global_step,
            plot=self.save_image,
            output_dir=self.output_dir,
            artifact_stem=self.episode_artifact_stem,
            runtime_config=self.runtime_config,
        )
        self.robot = Agent(policy_net, self.device, self.save_image)
        self.ground_truth_node_manager = GroundTruthNodeManager(
            self.robot.node_manager,
            self.env.ground_truth_info,
            device=self.device,
            plot=self.save_image,
        )

        self.episode_buffer = [[] for _ in range(27)]
        self.perf_metrics = {}

    @staticmethod
    def _buffer_tensor(tensor: torch.Tensor) -> torch.Tensor:
        # Keep replay data on CPU before it goes through Ray object transport.
        return tensor.detach().to("cpu")

    @staticmethod
    def apply_n_step_returns(
        episode_buffer: list[list[torch.Tensor]],
        n_step: int,
        gamma: float,
    ) -> list[list[torch.Tensor]]:
        if n_step <= 1 or not episode_buffer or not episode_buffer[7]:
            return episode_buffer

        processed_buffer = [list(slot) for slot in episode_buffer]
        reward_buffer = episode_buffer[7]
        done_buffer = episode_buffer[8]
        buffer_size = len(reward_buffer)

        for index in range(buffer_size):
            reward_n = 0.0
            done_n = 0
            final_index = index
            available_steps = min(n_step, buffer_size - index)

            for offset in range(available_steps):
                reward_index = index + offset
                reward_n += (gamma ** offset) * float(reward_buffer[reward_index].item())
                final_index = reward_index
                if int(done_buffer[reward_index].item()) == 1:
                    done_n = 1
                    break
            else:
                if available_steps < n_step:
                    done_n = 1

            processed_buffer[7][index] = reward_buffer[index].clone().fill_(reward_n)
            processed_buffer[8][index] = done_buffer[index].clone().fill_(done_n)
            for slot_index in Worker.NEXT_OBSERVATION_SLOTS + Worker.CRITIC_NEXT_OBSERVATION_SLOTS:
                processed_buffer[slot_index][index] = episode_buffer[slot_index][final_index]

        return processed_buffer

    def _accumulate_reward_components(self) -> None:
        if not self.runtime_config.rl_options.use_reward_decomposition:
            return
        self.reward_component_sums["reward_info"] += float(self.env.last_reward_components.get("r_info", 0.0))
        self.reward_component_sums["reward_dist"] += float(self.env.last_reward_components.get("r_dist", 0.0))
        self.reward_component_sums["reward_safe"] += float(self.env.last_reward_components.get("r_safe", 0.0))
        self.reward_component_sums["reward_terminal"] += float(self.env.last_reward_components.get("r_terminal", 0.0))

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
                reward = self.env.apply_terminal_bonus(reward)
            self._accumulate_reward_components()
            self.episode_return += float(reward)
            self.episode_steps = i + 1
            self.save_reward_done(reward, done)

            observation = self.robot.get_observation()
            ground_truth_observation = self.ground_truth_node_manager.get_ground_truth_observation(
                self.env.robot_location
            )
            self.save_next_observations(observation, ground_truth_observation)

            if self.save_image:
                self.robot.plot_env()
                self.ground_truth_node_manager.plot_ground_truth_env(self.env.robot_location)
                self.env.plot_env(i + 1)

            if done:
                break

        if self.runtime_config.rl_options.use_n_step_return:
            self.episode_buffer = self.apply_n_step_returns(
                self.episode_buffer,
                self.runtime_config.rl_options.n_step,
                GAMMA,
            )

        self.perf_metrics["travel_dist"] = self.env.travel_dist
        self.perf_metrics["explored_rate"] = self.env.explored_rate
        self.perf_metrics["success_rate"] = done
        self.perf_metrics["episode_return"] = self.episode_return
        self.perf_metrics["episode_steps"] = self.episode_steps
        if self.runtime_config.rl_options.use_reward_decomposition:
            self.perf_metrics.update(self.reward_component_sums)
        if self.runtime_config.rl_options.use_curriculum and self.env.curriculum_level_index is not None:
            self.perf_metrics["curriculum_level_index"] = int(self.env.curriculum_level_index)

        if self.save_image:
            finalize_episode_artifacts(self.output_dir, self.episode_artifact_stem, self.env.frame_files)

    def save_observation(self, observation, ground_truth_observation):
        node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask = observation
        self.episode_buffer[0] += self._buffer_tensor(node_inputs)
        self.episode_buffer[1] += self._buffer_tensor(node_padding_mask.bool())
        self.episode_buffer[2] += self._buffer_tensor(edge_mask.bool())
        self.episode_buffer[3] += self._buffer_tensor(current_index)
        self.episode_buffer[4] += self._buffer_tensor(current_edge)
        self.episode_buffer[5] += self._buffer_tensor(edge_padding_mask.bool())

        (
            critic_node_inputs,
            critic_node_padding_mask,
            critic_edge_mask,
            critic_current_index,
            critic_current_edge,
            critic_edge_padding_mask,
        ) = ground_truth_observation
        self.episode_buffer[15] += self._buffer_tensor(critic_node_inputs)
        self.episode_buffer[16] += self._buffer_tensor(critic_node_padding_mask.bool())
        self.episode_buffer[17] += self._buffer_tensor(critic_edge_mask.bool())
        self.episode_buffer[18] += self._buffer_tensor(critic_current_index)
        self.episode_buffer[19] += self._buffer_tensor(critic_current_edge)
        self.episode_buffer[20] += self._buffer_tensor(critic_edge_padding_mask.bool())

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
        node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask = observation
        self.episode_buffer[9] += self._buffer_tensor(node_inputs)
        self.episode_buffer[10] += self._buffer_tensor(node_padding_mask.bool())
        self.episode_buffer[11] += self._buffer_tensor(edge_mask.bool())
        self.episode_buffer[12] += self._buffer_tensor(current_index)
        self.episode_buffer[13] += self._buffer_tensor(current_edge)
        self.episode_buffer[14] += self._buffer_tensor(edge_padding_mask.bool())

        (
            critic_node_inputs,
            critic_node_padding_mask,
            critic_edge_mask,
            critic_current_index,
            critic_current_edge,
            critic_edge_padding_mask,
        ) = ground_truth_observation
        self.episode_buffer[21] += self._buffer_tensor(critic_node_inputs)
        self.episode_buffer[22] += self._buffer_tensor(critic_node_padding_mask.bool())
        self.episode_buffer[23] += self._buffer_tensor(critic_edge_mask.bool())
        self.episode_buffer[24] += self._buffer_tensor(critic_current_index)
        self.episode_buffer[25] += self._buffer_tensor(critic_current_edge)
        self.episode_buffer[26] += self._buffer_tensor(critic_edge_padding_mask.bool())


if __name__ == "__main__":
    torch.manual_seed(4777)
    np.random.seed(4777)
    from .model import PolicyNet

    model = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM)
    worker = Worker(0, model, 77, RuntimeConfig(), save_image=False)
    worker.run_episode()
