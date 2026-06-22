from __future__ import annotations

import time

import numpy as np
import torch

from agent import Agent
from benchmark_metrics import compute_benchmark_metrics
from env import Env
from ground_truth_node_manager import GroundTruthNodeManager
from parameter import (
    EMBEDDING_DIM,
    FREE,
    MAX_EPISODE_STEP,
    NODE_INPUT_DIM,
    RESULT_BUCKET_EPISODES,
    gifs_path,
)
from runtime_utils import derive_episode_seed, set_global_seeds
from utils import ensure_episode_bucket_dir, make_gif


class Worker:
    def __init__(
        self,
        meta_agent_id,
        policy_net,
        global_step,
        device="cpu",
        save_image=False,
        gifs_dir=None,
        forced_map_path=None,
        artifact_prefix=None,
        base_seed=None,
    ):
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.device = device
        self.base_seed = 0 if base_seed is None else int(base_seed)
        self.episode_seed = derive_episode_seed(
            self.base_seed, global_step, meta_agent_id=self.meta_agent_id
        )
        set_global_seeds(self.episode_seed)
        self.output_dir = None
        if self.save_image:
            if gifs_dir is not None:
                self.output_dir = gifs_dir
            else:
                self.output_dir = ensure_episode_bucket_dir(
                    gifs_path, global_step, RESULT_BUCKET_EPISODES
                )

        artifact_name = artifact_prefix or str(global_step)
        self.env = Env(
            global_step,
            plot=self.save_image,
            gifs_dir=self.output_dir,
            forced_map_path=forced_map_path,
            artifact_prefix=artifact_name,
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

    def _accumulate_planning_profile(
        self, totals: dict[str, float], fallback_sec: float
    ) -> None:
        profile = self.robot.last_planning_profile
        totals["graph_update_time_sec"] += float(
            profile.get("planning_total_sec", fallback_sec)
        )
        totals["node_manager_update_time_sec"] += float(
            profile.get("node_manager_update_sec", 0.0)
        )
        totals["dense_graph_build_time_sec"] += float(
            profile.get("dense_graph_build_sec", 0.0)
        )
        totals["graph_rarefaction_time_sec"] += float(
            profile.get("graph_rarefaction_sec", 0.0)
        )

    def run_episode(self):
        done = False
        episode_return = 0.0
        steps_taken = 0
        timing_totals = {
            "graph_update_time_sec": 0.0,
            "observation_time_sec": 0.0,
            "policy_inference_time_sec": 0.0,
            "env_step_time_sec": 0.0,
            "node_manager_update_time_sec": 0.0,
            "dense_graph_build_time_sec": 0.0,
            "graph_rarefaction_time_sec": 0.0,
        }

        planning_start = time.perf_counter()
        self.robot.update_planning_state(self.env.belief_info, self.env.robot_location)
        self._accumulate_planning_profile(
            timing_totals, time.perf_counter() - planning_start
        )
        observation_start = time.perf_counter()
        observation = self.robot.get_observation()
        timing_totals["observation_time_sec"] += time.perf_counter() - observation_start
        ground_truth_observation = (
            self.ground_truth_node_manager.get_ground_truth_observation(
                self.env.robot_location
            )
        )

        if self.save_image:
            self.robot.plot_env()
            self.ground_truth_node_manager.plot_ground_truth_env(
                self.env.robot_location
            )
            self.env.plot_env(0)

        if self.robot.utility.sum() == 0:
            done = True

        for step in range(MAX_EPISODE_STEP):
            if done:
                break

            self.save_observation(observation, ground_truth_observation)

            policy_start = time.perf_counter()
            next_location, action_index = self.robot.select_next_waypoint(observation)
            timing_totals["policy_inference_time_sec"] += (
                time.perf_counter() - policy_start
            )
            self.save_action(action_index)

            node = self.robot.node_manager.nodes_dict.find(
                (self.robot.location[0], self.robot.location[1])
            )
            check = np.array(list(node.data.neighbor_set)).reshape(-1, 2)
            assert (
                next_location[0] + next_location[1] * 1j
                in check[:, 0] + check[:, 1] * 1j
            ), (
                next_location,
                self.robot.location,
                node.data.neighbor_set,
            )
            assert (
                next_location[0] != self.robot.location[0]
                or next_location[1] != self.robot.location[1]
            )

            env_step_start = time.perf_counter()
            reward = self.env.step(next_location)
            timing_totals["env_step_time_sec"] += time.perf_counter() - env_step_start
            planning_start = time.perf_counter()
            self.robot.update_planning_state(
                self.env.belief_info, self.env.robot_location
            )
            self._accumulate_planning_profile(
                timing_totals, time.perf_counter() - planning_start
            )
            if self.robot.utility.sum() == 0:
                done = True
                reward += 20

            episode_return += float(reward)
            steps_taken = step + 1
            self.save_reward_done(reward, done)

            observation_start = time.perf_counter()
            observation = self.robot.get_observation()
            timing_totals["observation_time_sec"] += (
                time.perf_counter() - observation_start
            )
            ground_truth_observation = (
                self.ground_truth_node_manager.get_ground_truth_observation(
                    self.env.robot_location
                )
            )
            self.save_next_observations(observation, ground_truth_observation)

            if self.save_image:
                self.robot.plot_env()
                self.ground_truth_node_manager.plot_ground_truth_env(
                    self.env.robot_location
                )
                self.env.plot_env(steps_taken)

        self.perf_metrics["travel_dist"] = self.env.travel_dist
        self.perf_metrics["explored_rate"] = self.env.explored_rate
        self.perf_metrics["success_rate"] = float(bool(done))
        self.perf_metrics["episode_return"] = float(episode_return)
        self.perf_metrics["episode_steps"] = float(steps_taken)
        self.perf_metrics["episode_seed"] = int(self.episode_seed)
        self.perf_metrics.update(
            compute_benchmark_metrics(
                explored_rate=self.env.explored_rate,
                travel_dist=self.env.travel_dist,
                steps_taken=steps_taken,
                total_free_cells=int(np.sum(self.env.ground_truth == FREE)),
                cell_size=float(self.env.cell_size),
                episode_wall_time_sec=sum(
                    timing_totals[key]
                    for key in (
                        "graph_update_time_sec",
                        "observation_time_sec",
                        "policy_inference_time_sec",
                        "env_step_time_sec",
                    )
                ),
                graph_update_time_sec=timing_totals["graph_update_time_sec"],
                observation_time_sec=timing_totals["observation_time_sec"],
                policy_inference_time_sec=timing_totals["policy_inference_time_sec"],
                env_step_time_sec=timing_totals["env_step_time_sec"],
                node_manager_update_time_sec=timing_totals[
                    "node_manager_update_time_sec"
                ],
                dense_graph_build_time_sec=timing_totals["dense_graph_build_time_sec"],
                graph_rarefaction_time_sec=timing_totals["graph_rarefaction_time_sec"],
            )
        )

        if self.save_image and self.output_dir is not None:
            make_gif(
                self.output_dir,
                self.global_step,
                self.env.frame_files,
                self.env.explored_rate,
            )

    def save_observation(self, observation, ground_truth_observation):
        (
            node_inputs,
            node_padding_mask,
            edge_mask,
            current_index,
            current_edge,
            edge_padding_mask,
        ) = observation
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
            critic_current_index,
            critic_current_edge,
            critic_edge_padding_mask,
        ) = ground_truth_observation
        self.episode_buffer[15] += critic_node_inputs
        self.episode_buffer[16] += critic_node_padding_mask.bool()
        self.episode_buffer[17] += critic_edge_mask.bool()
        self.episode_buffer[18] += critic_current_index
        self.episode_buffer[19] += critic_current_edge
        self.episode_buffer[20] += critic_edge_padding_mask.bool()

        assert torch.all(current_edge == critic_current_edge), (
            current_edge,
            critic_current_edge,
            current_index,
            critic_current_index,
        )
        assert torch.all(
            node_inputs[0, current_index.item(), :2]
            == critic_node_inputs[0, critic_current_index.item(), :2]
        ), (
            node_inputs[0, current_index.item()],
            critic_node_inputs[0, critic_current_index.item()],
        )
        assert torch.all(
            torch.gather(node_inputs, 1, current_edge.repeat(1, 1, 2))
            == torch.gather(critic_node_inputs, 1, critic_current_edge.repeat(1, 1, 2))
        )

    def save_action(self, action_index):
        self.episode_buffer[6] += action_index.reshape(1, 1, 1)

    def save_reward_done(self, reward, done):
        self.episode_buffer[7] += (
            torch.FloatTensor([reward]).reshape(1, 1, 1).to(self.device)
        )
        self.episode_buffer[8] += (
            torch.tensor([int(done)]).reshape(1, 1, 1).to(self.device)
        )

    def save_next_observations(self, observation, ground_truth_observation):
        (
            node_inputs,
            node_padding_mask,
            edge_mask,
            current_index,
            current_edge,
            edge_padding_mask,
        ) = observation
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
            critic_current_index,
            critic_current_edge,
            critic_edge_padding_mask,
        ) = ground_truth_observation
        self.episode_buffer[21] += critic_node_inputs
        self.episode_buffer[22] += critic_node_padding_mask.bool()
        self.episode_buffer[23] += critic_edge_mask.bool()
        self.episode_buffer[24] += critic_current_index
        self.episode_buffer[25] += critic_current_edge
        self.episode_buffer[26] += critic_edge_padding_mask.bool()


if __name__ == "__main__":
    import numpy as np

    from model import PolicyNet

    torch.manual_seed(4777)
    np.random.seed(4777)
    model = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM)
    worker = Worker(0, model, 77, save_image=False)
    worker.run_episode()
