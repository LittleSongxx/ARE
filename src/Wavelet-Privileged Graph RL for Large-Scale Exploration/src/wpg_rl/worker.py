from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .agent import Agent
from .env import Env
from .ground_truth_node_manager import GroundTruthNodeManager, build_actor_to_critic_index
from .parameter import EMBEDDING_DIM, MAX_EPISODE_STEP, NODE_INPUT_DIM, NODE_PADDING_SIZE, RESULT_BUCKET_EPISODES, gifs_path
from .utils import ensure_episode_bucket_dir, make_gif


class Worker:
    def __init__(self, meta_agent_id, policy_net, global_step, device="cpu", save_image=False):
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.device = device
        self.gif_episode_dir = (
            ensure_episode_bucket_dir(gifs_path, global_step, RESULT_BUCKET_EPISODES) if self.save_image else None
        )

        self.env = Env(global_step, plot=self.save_image, gifs_dir=self.gif_episode_dir)
        self.robot = Agent(policy_net, self.device, self.save_image)
        self.ground_truth_node_manager = GroundTruthNodeManager(
            self.robot.node_manager,
            self.env.ground_truth_info,
            device=self.device,
            plot=self.save_image,
        )

        self.episode_buffer = [[] for _ in range(28)]
        self.perf_metrics = {}

    def run_episode(self):
        done = False
        self.robot.update_planning_state(self.env.belief_info, self.env.robot_location)
        observation = self.robot.get_observation()
        ground_truth_observation = self.ground_truth_node_manager.get_ground_truth_observation(self.env.robot_location)

        if self.save_image:
            Path(self.gif_episode_dir).mkdir(parents=True, exist_ok=True)
            self.robot.plot_env()
            self.ground_truth_node_manager.plot_ground_truth_env(self.env.robot_location)
            self.env.plot_env(0)

        for i in range(MAX_EPISODE_STEP):
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

        self.perf_metrics["travel_dist"] = self.env.travel_dist
        self.perf_metrics["explored_rate"] = self.env.explored_rate
        self.perf_metrics["success_rate"] = done
        self.perf_metrics["episode_steps"] = i + 1
        self.perf_metrics["episode_return"] = float(
            sum(item.squeeze().item() for item in self.episode_buffer[7]) if self.episode_buffer[7] else 0.0
        )

        if self.save_image:
            make_gif(self.gif_episode_dir, self.global_step, self.env.frame_files, self.env.explored_rate)

    def save_observation(self, observation, ground_truth_observation):
        node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask = observation
        self.episode_buffer[0] += node_inputs
        self.episode_buffer[1] += node_padding_mask.bool()
        self.episode_buffer[2] += edge_mask.bool()
        self.episode_buffer[3] += current_index
        self.episode_buffer[4] += current_edge
        self.episode_buffer[5] += edge_padding_mask.bool()

        (
            critic_node_inputs,
            critic_node_padding_mask,
            critic_edge_mask,
            raw_critic_current_index,
            _raw_critic_current_edge,
            _raw_critic_edge_padding_mask,
        ) = ground_truth_observation
        actor_to_critic_index = self.build_actor_to_critic_index_tensor()
        critic_current_index, critic_current_edge, critic_edge_padding_mask = self.align_actor_indices_to_critic(
            current_index,
            current_edge,
            edge_padding_mask,
            actor_to_critic_index,
            raw_critic_current_index,
        )
        self.episode_buffer[15] += critic_node_inputs
        self.episode_buffer[16] += critic_node_padding_mask.bool()
        self.episode_buffer[17] += critic_edge_mask.bool()
        self.episode_buffer[18] += critic_current_index
        self.episode_buffer[19] += critic_current_edge
        self.episode_buffer[20] += critic_edge_padding_mask.bool()
        self.episode_buffer[27] += actor_to_critic_index.unsqueeze(0)

    def save_action(self, action_index):
        self.episode_buffer[6] += action_index.reshape(1, 1, 1)

    def save_reward_done(self, reward, done):
        self.episode_buffer[7] += torch.FloatTensor([reward]).reshape(1, 1, 1).to(self.device)
        self.episode_buffer[8] += torch.tensor([int(done)]).reshape(1, 1, 1).to(self.device)

    def save_next_observations(self, observation, ground_truth_observation):
        node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask = observation
        self.episode_buffer[9] += node_inputs
        self.episode_buffer[10] += node_padding_mask.bool()
        self.episode_buffer[11] += edge_mask.bool()
        self.episode_buffer[12] += current_index
        self.episode_buffer[13] += current_edge
        self.episode_buffer[14] += edge_padding_mask.bool()

        (
            critic_node_inputs,
            critic_node_padding_mask,
            critic_edge_mask,
            raw_critic_current_index,
            _raw_critic_current_edge,
            _raw_critic_edge_padding_mask,
        ) = ground_truth_observation
        actor_to_critic_index = self.build_actor_to_critic_index_tensor()
        critic_current_index, critic_current_edge, critic_edge_padding_mask = self.align_actor_indices_to_critic(
            current_index,
            current_edge,
            edge_padding_mask,
            actor_to_critic_index,
            raw_critic_current_index,
        )
        self.episode_buffer[21] += critic_node_inputs
        self.episode_buffer[22] += critic_node_padding_mask.bool()
        self.episode_buffer[23] += critic_edge_mask.bool()
        self.episode_buffer[24] += critic_current_index
        self.episode_buffer[25] += critic_current_edge
        self.episode_buffer[26] += critic_edge_padding_mask.bool()

    def build_actor_to_critic_index_tensor(self) -> torch.Tensor:
        actor_to_critic_index = build_actor_to_critic_index(
            self.robot.node_coords,
            self.ground_truth_node_manager.ground_truth_node_coords,
            padded_size=NODE_PADDING_SIZE,
        )
        return torch.as_tensor(actor_to_critic_index, dtype=torch.long, device=self.device)

    def align_actor_indices_to_critic(
        self,
        actor_current_index,
        actor_current_edge,
        actor_edge_padding_mask,
        actor_to_critic_index,
        raw_critic_current_index,
    ):
        actor_to_critic_index = actor_to_critic_index.long()
        critic_current_index = torch.gather(
            actor_to_critic_index.unsqueeze(0),
            1,
            actor_current_index.view(1, 1).long(),
        ).view_as(actor_current_index)
        if torch.any(critic_current_index < 0):
            raise RuntimeError("Current actor node is missing from privileged critic graph overlap mapping")

        critic_current_edge = torch.gather(
            actor_to_critic_index.unsqueeze(0),
            1,
            actor_current_edge.squeeze(-1).long(),
        ).unsqueeze(-1)
        invalid_edge_mask = critic_current_edge.squeeze(-1).lt(0)
        critic_edge_padding_mask = actor_edge_padding_mask.bool() | invalid_edge_mask.unsqueeze(1)
        critic_current_edge = critic_current_edge.clamp_min(0)

        valid_actor_mask = ~actor_edge_padding_mask.squeeze(0).squeeze(0).bool()
        if torch.any(invalid_edge_mask.squeeze(0)[valid_actor_mask]):
            raise RuntimeError("Actor candidate edge is missing from privileged critic graph overlap mapping")
        assert torch.equal(critic_current_index, raw_critic_current_index), (
            critic_current_index,
            raw_critic_current_index,
        )
        return critic_current_index, critic_current_edge, critic_edge_padding_mask
