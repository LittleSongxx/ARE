from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from agent import Agent
from env import Env
from ground_truth_node_manager import GroundTruthNodeManager
from model import PolicyNet
from parameter import EMBEDDING_DIM, MAX_EPISODE_STEP, NODE_INPUT_DIM, eval_path
from utils import make_gif


def _copy_state_dict_to_device(state_dict, device):
    return {key: value.detach().to(device) for key, value in state_dict.items()}


def evaluate_policy(
    policy_state_dict,
    output_dir: str | Path | None = None,
    episodes: int = 1,
    start_episode: int = 0,
    greedy: bool = True,
    device: str | torch.device = "cpu",
    max_episode_step: int = MAX_EPISODE_STEP,
):
    eval_root = Path(output_dir or eval_path)
    eval_root.mkdir(parents=True, exist_ok=True)
    eval_device = torch.device(device)

    policy_net = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM).to(eval_device)
    policy_net.load_state_dict(_copy_state_dict_to_device(policy_state_dict, eval_device))
    policy_net.eval()

    results = []
    for episode_offset in range(episodes):
        episode_id = start_episode + episode_offset
        episode_dir = eval_root / f"episode_{episode_id:06d}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        env = Env(episode_id, plot=True)
        agent = Agent(policy_net, device=eval_device, plot=False)
        ground_truth_node_manager = GroundTruthNodeManager(
            agent.node_manager,
            env.ground_truth_info,
            device=eval_device,
            plot=False,
        )

        success = False
        episode_return = 0.0
        steps_taken = 0

        agent.update_planning_state(env.belief_info, env.robot_location)
        observation = agent.get_observation()
        ground_truth_node_manager.get_ground_truth_observation(env.robot_location)
        agent.plot_env()
        ground_truth_node_manager.plot_ground_truth_env(env.robot_location)
        env.plot_env(0, output_dir=str(episode_dir))

        if agent.utility.sum() == 0:
            success = True

        for step in range(1, max_episode_step + 1):
            next_location, _ = agent.select_next_waypoint(observation, greedy=greedy)
            reward = env.step(next_location)
            agent.update_planning_state(env.belief_info, env.robot_location)
            observation = agent.get_observation()
            ground_truth_node_manager.get_ground_truth_observation(env.robot_location)

            steps_taken = step
            if agent.utility.sum() == 0:
                success = True
                reward += 20
            episode_return += float(reward)

            agent.plot_env()
            ground_truth_node_manager.plot_ground_truth_env(env.robot_location)
            env.plot_env(step, output_dir=str(episode_dir))

            if success:
                break

        png_path = Path(env.frame_files[-1]) if env.frame_files else episode_dir / "final.png"
        make_gif(str(episode_dir), episode_id, env.frame_files, env.explored_rate)
        gif_path = episode_dir / f"{episode_id}_explored_rate_{env.explored_rate:.4g}.gif"
        results.append(
            {
                "episode": episode_id,
                "explored_rate": env.explored_rate,
                "travel_dist": env.travel_dist,
                "success": success,
                "episode_return": episode_return,
                "steps_taken": steps_taken,
                "gif_path": str(gif_path),
                "png_path": str(png_path),
            }
        )
    return results


def summarize_eval_results(results: list[dict[str, object]]) -> dict[str, float | int]:
    if not results:
        return {
            "episodes": 0,
            "explored_rate": 0.0,
            "travel_dist": 0.0,
            "success_rate": 0.0,
            "episode_return": 0.0,
            "steps_taken": 0.0,
            "best_explored_rate": 0.0,
            "worst_explored_rate": 0.0,
        }

    explored = np.array([float(result["explored_rate"]) for result in results], dtype=float)
    travel = np.array([float(result["travel_dist"]) for result in results], dtype=float)
    success = np.array([float(result["success"]) for result in results], dtype=float)
    episode_return = np.array([float(result["episode_return"]) for result in results], dtype=float)
    steps = np.array([float(result["steps_taken"]) for result in results], dtype=float)
    return {
        "episodes": len(results),
        "explored_rate": float(np.mean(explored)),
        "travel_dist": float(np.mean(travel)),
        "success_rate": float(np.mean(success)),
        "episode_return": float(np.mean(episode_return)),
        "steps_taken": float(np.mean(steps)),
        "best_explored_rate": float(np.max(explored)),
        "worst_explored_rate": float(np.min(explored)),
    }
