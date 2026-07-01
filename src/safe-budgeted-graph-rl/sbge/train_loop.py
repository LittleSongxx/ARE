from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from .algorithms import ConstrainedSACAgent, ReplayBuffer, transition_from_observations
from .config import SBGEConfig
from .env import SafeBudgetedGraphEnv
from .graph import SafeGraphBuilder
from .results import config_to_dict, write_csv, write_json
from .types import EpisodeMetrics


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_episode(
    env: SafeBudgetedGraphEnv,
    graph_builder: SafeGraphBuilder,
    agent: ConstrainedSACAgent,
    episode_index: int,
    replay: ReplayBuffer | None = None,
    train: bool = True,
    greedy: bool = False,
) -> EpisodeMetrics:
    env.reset(episode_index)
    metrics = EpisodeMetrics()
    done = False
    for _ in range(env.config.max_episode_steps):
        observation = graph_builder.build(env)
        decision = agent.select_action(observation, greedy=greedy)
        next_idx = int(observation.neighbor_indices[decision.action_slot])
        next_cell = observation.node_positions[next_idx]
        return_cost = graph_builder.edge_return_cost_for_action(observation, decision.action_slot)
        result = env.step(next_cell, return_cost_m=return_cost)
        next_observation = graph_builder.build(env)
        if replay is not None and train:
            replay.append(
                transition_from_observations(
                    observation,
                    decision.action_slot,
                    result.reward,
                    result.cost,
                    result.done,
                    next_observation,
                )
            )
        metrics.episode_steps += 1
        metrics.episode_return += result.reward
        metrics.episode_cost += result.cost
        metrics.collision_count += int(result.info["collision"])
        metrics.near_miss_count += int(result.info["near_miss"])
        metrics.risk_integral += float(result.info["risk"])
        metrics.budget_violation = max(metrics.budget_violation, int(result.info["budget_violation"]))
        metrics.return_success = max(metrics.return_success, int(result.info["return_success"]))
        metrics.shield_interventions += decision.shield_intervention
        metrics.unsafe_action_proposals += decision.unsafe_action_proposal
        done = bool(result.done)
        if done:
            break

    metrics.explored_rate = float(env.explored_rate)
    metrics.travel_dist = float(env.travel_dist)
    metrics.success_rate = float(done and metrics.budget_violation == 0)
    return metrics


def train(config: SBGEConfig, episodes: int, output_dir: str | Path | None = None) -> dict[str, object]:
    set_global_seeds(config.seed)
    agent = ConstrainedSACAgent(config)
    replay = ReplayBuffer(config.replay_size)
    env = SafeBudgetedGraphEnv(config, seed=config.seed)
    graph_builder = SafeGraphBuilder(config)
    output_path = Path(output_dir) if output_dir is not None else config.result_dir / "sbge_train"
    output_path.mkdir(parents=True, exist_ok=True)
    write_json(output_path / "config.json", config_to_dict(config))

    episode_rows: list[dict[str, float | int]] = []
    train_rows: list[dict[str, float | int]] = []
    total_steps = 0
    for episode in range(int(episodes)):
        metrics = run_episode(env, graph_builder, agent, episode, replay=replay, train=True, greedy=False)
        total_steps += metrics.episode_steps
        row = {"episode": episode, **metrics.to_dict()}
        episode_rows.append(row)
        latest_train: dict[str, float] = {}
        if len(replay) >= config.warmup_steps:
            for _ in range(config.train_updates_per_step * max(metrics.episode_steps, 1)):
                latest_train = agent.train_step(replay.sample(config.batch_size))
                train_rows.append({"episode": episode, "step": len(train_rows), **latest_train})
        print(
            "episode "
            f"{episode + 1}/{episodes} "
            f"explored={metrics.explored_rate:.3f} "
            f"return={metrics.episode_return:.3f} "
            f"cost={metrics.episode_cost:.3f} "
            f"travel={metrics.travel_dist:.2f} "
            f"shield={metrics.shield_interventions} "
            f"lambda={latest_train.get('lambda', float(agent.lagrange_lambda.detach().item())):.3f}"
        )

    checkpoint_path = output_path / "checkpoint.pt"
    agent.save(checkpoint_path)
    write_json(output_path / "episodes.json", episode_rows)
    write_csv(output_path / "episodes.csv", episode_rows)
    write_json(output_path / "train_updates.json", train_rows)
    write_csv(output_path / "train_updates.csv", train_rows)
    summary = {
        "episodes": int(episodes),
        "total_steps": int(total_steps),
        "checkpoint": str(checkpoint_path.resolve()),
        "mean_explored_rate": float(np.mean([row["explored_rate"] for row in episode_rows])) if episode_rows else 0.0,
        "mean_return_success": float(np.mean([row["return_success"] for row in episode_rows])) if episode_rows else 0.0,
        "mean_budget_violation": float(np.mean([row["budget_violation"] for row in episode_rows])) if episode_rows else 0.0,
    }
    write_json(output_path / "summary.json", summary)
    return summary
