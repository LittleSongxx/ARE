from __future__ import annotations

from typing import Final


EPSILON: Final[float] = 1e-8

SUMMARY_BENCHMARK_FIELDS: tuple[str, ...] = (
    "distance_efficiency",
    "step_efficiency",
    "time_efficiency",
    "episode_wall_time_sec",
    "planning_time_sec",
    "graph_update_time_sec",
    "observation_time_sec",
    "policy_inference_time_sec",
    "env_step_time_sec",
    "node_manager_update_time_sec",
    "dense_graph_build_time_sec",
    "graph_rarefaction_time_sec",
    "mean_step_wall_time_ms",
    "mean_planning_time_ms",
    "mean_graph_update_time_ms",
    "mean_observation_time_ms",
    "mean_policy_inference_time_ms",
    "mean_env_step_time_ms",
    "mean_node_manager_update_time_ms",
    "mean_dense_graph_build_time_ms",
    "mean_graph_rarefaction_time_ms",
)


def safe_div(numerator: float | int | None, denominator: float | int | None) -> float:
    if numerator is None or denominator is None:
        return 0.0
    denominator = float(denominator)
    if abs(denominator) < EPSILON:
        return 0.0
    return float(numerator) / denominator


def _mean_time_ms(total_time_sec: float, n_steps: int) -> float:
    return 1000.0 * safe_div(total_time_sec, max(int(n_steps), 1))


def compute_benchmark_metrics(
    *,
    explored_rate: float,
    travel_dist: float,
    steps_taken: int,
    total_free_cells: int | float | None = None,
    cell_size: float | None = None,
    episode_wall_time_sec: float | None = None,
    graph_update_time_sec: float = 0.0,
    observation_time_sec: float = 0.0,
    policy_inference_time_sec: float = 0.0,
    env_step_time_sec: float = 0.0,
    node_manager_update_time_sec: float = 0.0,
    dense_graph_build_time_sec: float = 0.0,
    graph_rarefaction_time_sec: float = 0.0,
) -> dict[str, float]:
    explored_rate = float(explored_rate)
    travel_dist = float(travel_dist)
    steps_taken = max(int(steps_taken), 0)

    total_free_cells_value = float(total_free_cells) if total_free_cells is not None else 0.0
    cell_area = float(cell_size) * float(cell_size) if cell_size is not None else 0.0
    explored_free_cells = explored_rate * total_free_cells_value
    total_free_area = total_free_cells_value * cell_area
    explored_area = explored_free_cells * cell_area

    graph_update_time_sec = float(graph_update_time_sec)
    observation_time_sec = float(observation_time_sec)
    policy_inference_time_sec = float(policy_inference_time_sec)
    env_step_time_sec = float(env_step_time_sec)
    node_manager_update_time_sec = float(node_manager_update_time_sec)
    dense_graph_build_time_sec = float(dense_graph_build_time_sec)
    graph_rarefaction_time_sec = float(graph_rarefaction_time_sec)

    planning_time_sec = graph_update_time_sec + observation_time_sec + policy_inference_time_sec
    episode_wall_time_value = (
        float(episode_wall_time_sec)
        if episode_wall_time_sec is not None
        else planning_time_sec + env_step_time_sec
    )

    exploration_quantity = explored_area if explored_area > 0.0 else explored_rate

    return {
        "total_free_cells": total_free_cells_value,
        "explored_free_cells": explored_free_cells,
        "total_free_area": total_free_area,
        "explored_area": explored_area,
        "distance_efficiency": safe_div(exploration_quantity, travel_dist),
        "step_efficiency": safe_div(exploration_quantity, max(steps_taken, 1)),
        "time_efficiency": safe_div(exploration_quantity, episode_wall_time_value),
        "episode_wall_time_sec": episode_wall_time_value,
        "planning_time_sec": planning_time_sec,
        "graph_update_time_sec": graph_update_time_sec,
        "observation_time_sec": observation_time_sec,
        "policy_inference_time_sec": policy_inference_time_sec,
        "env_step_time_sec": env_step_time_sec,
        "node_manager_update_time_sec": node_manager_update_time_sec,
        "dense_graph_build_time_sec": dense_graph_build_time_sec,
        "graph_rarefaction_time_sec": graph_rarefaction_time_sec,
        "mean_step_wall_time_ms": _mean_time_ms(episode_wall_time_value, steps_taken),
        "mean_planning_time_ms": _mean_time_ms(planning_time_sec, steps_taken),
        "mean_graph_update_time_ms": _mean_time_ms(graph_update_time_sec, steps_taken),
        "mean_observation_time_ms": _mean_time_ms(observation_time_sec, steps_taken),
        "mean_policy_inference_time_ms": _mean_time_ms(policy_inference_time_sec, steps_taken),
        "mean_env_step_time_ms": _mean_time_ms(env_step_time_sec, steps_taken),
        "mean_node_manager_update_time_ms": _mean_time_ms(node_manager_update_time_sec, steps_taken),
        "mean_dense_graph_build_time_ms": _mean_time_ms(dense_graph_build_time_sec, steps_taken),
        "mean_graph_rarefaction_time_ms": _mean_time_ms(graph_rarefaction_time_sec, steps_taken),
    }
