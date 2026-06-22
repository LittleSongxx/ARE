from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .parameter import FREE, SENSOR_RANGE
from .utils import get_cell_position_from_coords


@dataclass(frozen=True)
class OracleNodeTargets:
    oracle_utility: np.ndarray
    expert_potential: np.ndarray


def compute_state_exploration_potential(robot_belief: np.ndarray, ground_truth: np.ndarray) -> float:
    belief = np.asarray(robot_belief)
    truth = np.asarray(ground_truth)
    free_mask = truth == FREE
    total_free = max(int(np.count_nonzero(free_mask)), 1)
    explored_free = int(np.count_nonzero((belief == FREE) & free_mask))
    return float(explored_free) / float(total_free)


def potential_based_shaping(
    previous_potential: float,
    next_potential: float,
    gamma: float,
    weight: float,
) -> float:
    return max(float(weight), 0.0) * (float(gamma) * float(next_potential) - float(previous_potential))


def apply_expert_reward(
    reward: float,
    previous_potential: float,
    next_potential: float,
    gamma: float,
    shaping_weight: float,
    oracle_gain_weight: float = 0.0,
) -> float:
    shaped = float(reward) + potential_based_shaping(previous_potential, next_potential, gamma, shaping_weight)
    if oracle_gain_weight > 0:
        shaped += float(oracle_gain_weight) * max(float(next_potential) - float(previous_potential), 0.0)
    return float(shaped)


def compute_oracle_node_targets(
    node_coords: np.ndarray,
    ground_truth_map_info,
    belief_map_info,
    sensor_range: float = SENSOR_RANGE,
) -> OracleNodeTargets:
    node_coords = np.asarray(node_coords, dtype=np.float64).reshape(-1, 2)
    n_nodes = node_coords.shape[0]
    if n_nodes == 0:
        empty = np.zeros(0, dtype=np.float32)
        return OracleNodeTargets(empty, empty)

    truth = np.asarray(ground_truth_map_info.map)
    belief = np.asarray(belief_map_info.map) if belief_map_info is not None else np.zeros_like(truth)
    radius_cells = max(int(round(float(sensor_range) / max(float(ground_truth_map_info.cell_size), 1e-6))), 1)
    yy, xx = np.ogrid[-radius_cells : radius_cells + 1, -radius_cells : radius_cells + 1]
    circle_mask = xx * xx + yy * yy <= radius_cells * radius_cells
    circle_total = max(int(np.count_nonzero(circle_mask)), 1)

    cells = get_cell_position_from_coords(node_coords, ground_truth_map_info, check_negative=False).reshape(-1, 2)
    unseen_free_ratio = np.zeros(n_nodes, dtype=np.float32)
    for index, (cell_x, cell_y) in enumerate(cells):
        x0 = max(int(cell_x) - radius_cells, 0)
        y0 = max(int(cell_y) - radius_cells, 0)
        x1 = min(int(cell_x) + radius_cells + 1, truth.shape[1])
        y1 = min(int(cell_y) + radius_cells + 1, truth.shape[0])
        if x0 >= x1 or y0 >= y1:
            continue
        mask_x0 = x0 - (int(cell_x) - radius_cells)
        mask_y0 = y0 - (int(cell_y) - radius_cells)
        mask_x1 = mask_x0 + (x1 - x0)
        mask_y1 = mask_y0 + (y1 - y0)
        local_truth = truth[y0:y1, x0:x1]
        local_belief = belief[y0:y1, x0:x1]
        local_mask = circle_mask[mask_y0:mask_y1, mask_x0:mask_x1]
        unseen_free = (local_truth == FREE) & (local_belief != FREE) & local_mask
        unseen_free_ratio[index] = float(np.count_nonzero(unseen_free)) / float(circle_total)

    if np.max(unseen_free_ratio) > 1e-6:
        oracle_utility = unseen_free_ratio / float(np.max(unseen_free_ratio))
    else:
        oracle_utility = unseen_free_ratio
    expert_potential = np.clip(0.7 * oracle_utility + 0.3 * unseen_free_ratio, 0.0, 1.0)
    return OracleNodeTargets(
        oracle_utility=np.clip(oracle_utility, 0.0, 1.0).astype(np.float32),
        expert_potential=expert_potential.astype(np.float32),
    )