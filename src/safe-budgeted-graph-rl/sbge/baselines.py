from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import SBGEConfig
from .env import SafeBudgetedGraphEnv
from .graph import SafeGraphBuilder
from .results import config_to_dict, write_csv, write_json
from .types import EpisodeMetrics, Observation


def select_nearest_safe(observation: Observation, current_position: np.ndarray) -> int:
    slots = observation.safe_action_slots()
    if len(slots) == 0:
        return int(observation.fallback_action_slot)
    distances = [
        np.linalg.norm(observation.node_positions[observation.neighbor_indices[slot]] - current_position)
        for slot in slots
    ]
    return int(slots[int(np.argmin(distances))])


def select_utility_safe(observation: Observation) -> int:
    slots = observation.safe_action_slots()
    if len(slots) == 0:
        return int(observation.fallback_action_slot)
    utilities = [float(observation.node_features[observation.neighbor_indices[slot], 2]) for slot in slots]
    return int(slots[int(np.argmax(utilities))])


def select_utility_unshielded(observation: Observation) -> int:
    real_slots = [slot for slot, idx in enumerate(observation.neighbor_indices) if idx != observation.current_index]
    if not real_slots:
        return int(observation.fallback_action_slot)
    utilities = [float(observation.node_features[observation.neighbor_indices[slot], 2]) for slot in real_slots]
    return int(real_slots[int(np.argmax(utilities))])


def select_expert(observation: Observation) -> int:
    if observation.expert_action_slot is not None:
        return int(observation.expert_action_slot)
    return select_utility_safe(observation)


def run_baseline_episode(config: SBGEConfig, episode_index: int, policy: str) -> EpisodeMetrics:
    env = SafeBudgetedGraphEnv(config, seed=config.seed)
    graph_builder = SafeGraphBuilder(config)
    env.reset(episode_index)
    metrics = EpisodeMetrics()
    for _ in range(config.max_episode_steps):
        obs = graph_builder.build(env)
        if policy == "nearest":
            slot = select_nearest_safe(obs, env.robot_cell)
        elif policy == "utility":
            slot = select_utility_safe(obs)
        elif policy == "utility_unshielded":
            slot = select_utility_unshielded(obs)
        elif policy == "expert":
            slot = select_expert(obs)
        else:
            raise ValueError(f"Unknown baseline policy: {policy}")
        next_idx = int(obs.neighbor_indices[slot])
        return_cost = graph_builder.edge_return_cost_for_action(obs, slot)
        result = env.step(obs.node_positions[next_idx], return_cost_m=return_cost)
        unsafe_selected = int(obs.action_mask[slot])
        metrics.episode_steps += 1
        metrics.episode_return += result.reward
        metrics.episode_cost += result.cost
        metrics.collision_count += int(result.info["collision"])
        metrics.near_miss_count += int(result.info["near_miss"])
        metrics.risk_integral += float(result.info["risk"])
        metrics.budget_violation = max(metrics.budget_violation, int(result.info["budget_violation"]))
        metrics.return_success = max(metrics.return_success, int(result.info["return_success"]))
        metrics.shield_interventions += 0 if policy == "utility_unshielded" else unsafe_selected
        metrics.unsafe_action_proposals += unsafe_selected
        if result.done:
            break
    metrics.explored_rate = float(env.explored_rate)
    metrics.travel_dist = float(env.travel_dist)
    metrics.success_rate = float(metrics.budget_violation == 0 and env.explored_rate >= config.target_explored_rate)
    return metrics


def run_baselines(config: SBGEConfig, episodes: int, output_dir: str | Path | None = None) -> dict[str, object]:
    output_path = Path(output_dir) if output_dir is not None else config.result_dir / "sbge_baselines"
    output_path.mkdir(parents=True, exist_ok=True)
    write_json(output_path / "config.json", config_to_dict(config))
    rows = []
    policies = ["nearest", "utility", "utility_unshielded", "expert"]
    for policy in policies:
        for episode in range(int(episodes)):
            metrics = run_baseline_episode(config, episode, policy)
            rows.append({"policy": policy, "episode": episode, **metrics.to_dict()})
            print(
                f"baseline={policy} episode={episode + 1}/{episodes} "
                f"explored={metrics.explored_rate:.3f} cost={metrics.episode_cost:.3f}"
            )
    summary = {
        "episodes": int(episodes),
        "policies": policies,
        "rows": rows,
    }
    write_json(output_path / "baseline_summary.json", {key: value for key, value in summary.items() if key != "rows"})
    write_json(output_path / "baseline_results.json", summary)
    write_csv(output_path / "baseline_results.csv", rows)
    return summary
