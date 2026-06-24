from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import SBGEConfig
from .env import SafeBudgetedGraphEnv
from .graph import SafeGraphBuilder
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
        else:
            raise ValueError(f"Unknown baseline policy: {policy}")
        next_idx = int(obs.neighbor_indices[slot])
        return_cost = graph_builder.edge_return_cost_for_action(obs, slot)
        result = env.step(obs.node_positions[next_idx], return_cost_m=return_cost)
        metrics.episode_steps += 1
        metrics.episode_return += result.reward
        metrics.episode_cost += result.cost
        metrics.collision_count += int(result.info["collision"])
        metrics.near_miss_count += int(result.info["near_miss"])
        metrics.risk_integral += float(result.info["risk"])
        metrics.budget_violation = max(metrics.budget_violation, int(result.info["budget_violation"]))
        metrics.return_success = max(metrics.return_success, int(result.info["return_success"]))
        metrics.shield_interventions += int(obs.action_mask[slot])
        if result.done:
            break
    metrics.explored_rate = float(env.explored_rate)
    metrics.travel_dist = float(env.travel_dist)
    metrics.success_rate = float(metrics.budget_violation == 0 and env.explored_rate >= config.target_explored_rate)
    return metrics


def run_baselines(config: SBGEConfig, episodes: int, output_dir: str | Path | None = None) -> dict[str, object]:
    output_path = Path(output_dir) if output_dir is not None else config.result_dir / "sbge_baselines"
    output_path.mkdir(parents=True, exist_ok=True)
    rows = []
    for policy in ("nearest", "utility"):
        for episode in range(int(episodes)):
            metrics = run_baseline_episode(config, episode, policy)
            rows.append({"policy": policy, "episode": episode, **metrics.to_dict()})
            print(
                f"baseline={policy} episode={episode + 1}/{episodes} "
                f"explored={metrics.explored_rate:.3f} cost={metrics.episode_cost:.3f}"
            )
    summary = {
        "episodes": int(episodes),
        "policies": ["nearest", "utility"],
        "rows": rows,
    }
    (output_path / "baseline_results.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary
