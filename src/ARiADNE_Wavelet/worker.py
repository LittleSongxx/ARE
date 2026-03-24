from __future__ import annotations

import torch
import numpy as np

from .agent import Agent
from .env import Env
from .parameter import EMBEDDING_DIM, K_SIZE, NODE_INPUT_DIM, RuntimeConfig, get_gifs_path
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
        )
        self.robot = Agent(
            policy_net,
            self.device,
            self.save_image,
            history_len=runtime_config.history_len,
            history_input_dim=runtime_config.history_input_dim,
            history_feature_set=runtime_config.history_feature_set,
        )

        # Replay layout (18 items):
        # 0..5 obs(legacy 6), 6 action, 7 reward, 8 done,
        # 9..14 next_obs(legacy 6), 15 history_inputs, 16 next_history_inputs, 17 utility_target_hint
        self.episode_buffer = [[] for _ in range(runtime_config.replay_channels)]
        self._step_action_slots: list[int] = []
        self._step_rewards: list[float] = []
        self._step_dones: list[bool] = []
        self.perf_metrics = {}

    @staticmethod
    def _buffer_tensor(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().to("cpu")

    def run_episode(self):
        done = False
        self.robot.reset_history()
        self.robot.update_planning_state(self.env.belief_info, self.env.robot_location)
        observation = self.robot.get_observation()

        if self.save_image:
            self.robot.plot_env()
            self.env.plot_env(0)

        for i in range(self.runtime_config.max_episode_step):
            self.save_observation(observation)

            next_location, action_index, next_node_index = self.robot.select_next_waypoint(observation)
            self.save_action(action_index)

            node = self.robot.node_manager.nodes_dict.find((self.robot.location[0], self.robot.location[1]))
            check = np.array(list(node.data.neighbor_set)).reshape(-1, 2)
            assert next_location[0] + next_location[1] * 1j in check[:, 0] + check[:, 1] * 1j

            travel_before = float(self.env.travel_dist)
            explored_before = float(self.env.explored_rate)
            reward = self.env.step(next_location)
            travel_delta = float(self.env.travel_dist - travel_before)
            explored_delta = float(self.env.explored_rate - explored_before)
            selected_node_utility = float(self.robot.utility[next_node_index]) if self.robot.utility is not None else 0.0
            self.robot.record_transition_feedback(
                reward_proxy=float(reward),
                explored_delta=explored_delta,
                travel_delta=travel_delta,
                selected_node_index=next_node_index,
                selected_node_utility=selected_node_utility,
            )
            self.robot.update_planning_state(self.env.belief_info, self.env.robot_location)
            if self.robot.utility.sum() == 0:
                done = True
                reward += 20

            self.episode_return += float(reward)
            self.episode_steps = i + 1
            self.save_reward_done(reward, done)
            self._step_rewards.append(float(reward))
            self._step_dones.append(bool(done))

            observation = self.robot.get_observation()
            self.save_next_observations(observation)

            if self.save_image:
                self.robot.plot_env()
                self.env.plot_env(i + 1)

            if done:
                break

        self._finalize_utility_targets()

        self.perf_metrics["travel_dist"] = self.env.travel_dist
        self.perf_metrics["explored_rate"] = self.env.explored_rate
        self.perf_metrics["success_rate"] = done
        self.perf_metrics["episode_return"] = self.episode_return
        self.perf_metrics["episode_steps"] = self.episode_steps

        if self.save_image:
            finalize_episode_artifacts(
                self.output_dir,
                self.episode_artifact_stem,
                self.env.frame_files,
                frame_rate=self.runtime_config.gif_frame_rate,
            )

    def save_observation(self, observation):
        (
            node_inputs,
            node_padding_mask,
            edge_mask,
            current_index,
            current_edge,
            edge_padding_mask,
            history_inputs,
        ) = observation
        self.episode_buffer[0] += self._buffer_tensor(node_inputs)
        self.episode_buffer[1] += self._buffer_tensor(node_padding_mask.bool())
        self.episode_buffer[2] += self._buffer_tensor(edge_mask.bool())
        self.episode_buffer[3] += self._buffer_tensor(current_index)
        self.episode_buffer[4] += self._buffer_tensor(current_edge)
        self.episode_buffer[5] += self._buffer_tensor(edge_padding_mask.bool())
        self.episode_buffer[15] += self._buffer_tensor(history_inputs)
        placeholder = torch.full((1, K_SIZE, 1), float("nan"), dtype=torch.float32, device=self.device)
        self.episode_buffer[17] += self._buffer_tensor(placeholder)

    def save_action(self, action_index):
        action_slot = int(action_index.item())
        self._step_action_slots.append(action_slot)
        self.episode_buffer[6] += self._buffer_tensor(action_index.reshape(1, 1, 1))

    def save_reward_done(self, reward, done):
        reward_tensor = torch.tensor([reward], dtype=torch.float32, device=self.device).reshape(1, 1, 1)
        done_tensor = torch.tensor([int(done)], dtype=torch.int64, device=self.device).reshape(1, 1, 1)
        self.episode_buffer[7] += self._buffer_tensor(reward_tensor)
        self.episode_buffer[8] += self._buffer_tensor(done_tensor)

    def save_next_observations(self, observation):
        (
            node_inputs,
            node_padding_mask,
            edge_mask,
            current_index,
            current_edge,
            edge_padding_mask,
            history_inputs,
        ) = observation
        self.episode_buffer[9] += self._buffer_tensor(node_inputs)
        self.episode_buffer[10] += self._buffer_tensor(node_padding_mask.bool())
        self.episode_buffer[11] += self._buffer_tensor(edge_mask.bool())
        self.episode_buffer[12] += self._buffer_tensor(current_index)
        self.episode_buffer[13] += self._buffer_tensor(current_edge)
        self.episode_buffer[14] += self._buffer_tensor(edge_padding_mask.bool())
        self.episode_buffer[16] += self._buffer_tensor(history_inputs)

    def _finalize_utility_targets(self) -> None:
        if len(self._step_rewards) == 0 or len(self.episode_buffer[17]) == 0:
            return

        target_type = str(self.runtime_config.utility_target_type).strip().lower()
        if target_type != "n_step_return":
            return

        horizon = max(int(self.runtime_config.utility_target_horizon), 1)
        gamma = float(self.runtime_config.utility_target_gamma)
        episode_len = len(self._step_rewards)

        for t in range(episode_len):
            discounted_return = 0.0
            discount = 1.0
            for delta in range(horizon):
                idx = t + delta
                if idx >= episode_len:
                    break
                discounted_return += discount * float(self._step_rewards[idx])
                if self._step_dones[idx]:
                    break
                discount *= gamma

            if t >= len(self._step_action_slots):
                continue
            slot = int(self._step_action_slots[t])
            if not (0 <= slot < K_SIZE):
                continue
            self.episode_buffer[17][t][slot, 0] = float(discounted_return)


if __name__ == "__main__":
    torch.manual_seed(4777)
    np.random.seed(4777)
    from .model import PolicyNet

    model = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM)
    worker = Worker(0, model, 77, RuntimeConfig(run_session="worker_debug"), save_image=False)
    worker.run_episode()
