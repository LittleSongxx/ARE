from __future__ import annotations

import numpy as np

from .parameter import CELL_SIZE, FREE, NODE_RESOLUTION
from .utils import MapInfo, get_cell_position_from_coords


CORRIDOR_HORIZONTAL = "horizontal"
CORRIDOR_VERTICAL = "vertical"


def _run_length(grid: np.ndarray, x: int, y: int, dx: int, dy: int) -> int:
    length = 0
    x += dx
    y += dy
    while 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]:
        if int(grid[y, x]) != FREE:
            break
        length += 1
        x += dx
        y += dy
    return length


def detect_corridor_axis(
    map_info: MapInfo | None,
    coords,
    max_width: float,
    min_length: float,
) -> str | None:
    if map_info is None:
        return None

    cell = get_cell_position_from_coords(np.asarray(coords, dtype=float), map_info, check_negative=False)
    x = int(cell[0])
    y = int(cell[1])
    if not (0 <= x < map_info.map.shape[1] and 0 <= y < map_info.map.shape[0]):
        return None
    if int(map_info.map[y, x]) != FREE:
        return None

    horizontal_span = 1 + _run_length(map_info.map, x, y, -1, 0) + _run_length(map_info.map, x, y, 1, 0)
    vertical_span = 1 + _run_length(map_info.map, x, y, 0, -1) + _run_length(map_info.map, x, y, 0, 1)

    max_width_cells = max(int(round(float(max_width) / map_info.cell_size)), 1)
    min_length_cells = max(int(round(float(min_length) / map_info.cell_size)), 1)

    if vertical_span >= min_length_cells and horizontal_span <= max_width_cells and vertical_span > horizontal_span:
        return CORRIDOR_VERTICAL
    if horizontal_span >= min_length_cells and vertical_span <= max_width_cells and horizontal_span > vertical_span:
        return CORRIDOR_HORIZONTAL
    return None


def _axis_vectors(axis: str) -> tuple[np.ndarray, np.ndarray]:
    if axis == CORRIDOR_VERTICAL:
        return np.array([0.0, 1.0], dtype=float), np.array([1.0, 0.0], dtype=float)
    return np.array([1.0, 0.0], dtype=float), np.array([0.0, 1.0], dtype=float)


def refine_neighbor_indices(
    node_coords,
    current_index: int,
    neighbor_indices,
    map_info: MapInfo | None,
    *,
    enable_edge_pruning: bool = False,
    enable_graph_compression: bool = False,
    corridor_max_width: float = 1.5 * NODE_RESOLUTION,
    corridor_min_length: float = 2.0 * NODE_RESOLUTION,
) -> np.ndarray:
    if not enable_edge_pruning and not enable_graph_compression:
        return np.asarray(neighbor_indices, dtype=int).reshape(-1)

    node_coords = np.asarray(node_coords, dtype=float).reshape(-1, 2)
    neighbor_indices = np.asarray(neighbor_indices, dtype=int).reshape(-1)
    if neighbor_indices.size == 0:
        return neighbor_indices

    current_coords = node_coords[int(current_index)]
    axis = detect_corridor_axis(map_info, current_coords, corridor_max_width, corridor_min_length)
    if axis is None:
        return neighbor_indices

    axis_unit, cross_unit = _axis_vectors(axis)
    alignment_tol = max(CELL_SIZE, 0.25 * NODE_RESOLUTION)

    keep_indices = []
    by_bucket = {}

    for neighbor_index in neighbor_indices:
        neighbor_index = int(neighbor_index)
        if neighbor_index == int(current_index):
            keep_indices.append(neighbor_index)
            continue

        delta = node_coords[neighbor_index] - current_coords
        along = float(np.dot(delta, axis_unit))
        cross = float(np.dot(delta, cross_unit))

        if enable_edge_pruning and abs(cross) > alignment_tol:
            continue

        if enable_graph_compression:
            sign = 1 if along > alignment_tol else -1 if along < -alignment_tol else 0
            step_bucket = max(int(round(abs(along) / max(NODE_RESOLUTION, CELL_SIZE))), 1)
            bucket = (sign, step_bucket)
            score = (abs(cross), -abs(along), neighbor_index)
            best = by_bucket.get(bucket)
            if best is None or score < best[0]:
                by_bucket[bucket] = (score, neighbor_index)
        else:
            keep_indices.append(neighbor_index)

    if enable_graph_compression:
        keep_indices.extend(item[1] for item in sorted(by_bucket.values(), key=lambda item: item[0]))

    keep_indices = np.asarray(sorted(set(int(index) for index in keep_indices)), dtype=int)
    non_self = keep_indices[keep_indices != int(current_index)]
    if non_self.size == 0:
        return neighbor_indices
    return keep_indices


def build_refined_adjacency_matrix(
    node_coords,
    adjacency_matrix,
    map_info: MapInfo | None,
    *,
    enable_edge_pruning: bool = False,
    enable_graph_compression: bool = False,
    corridor_max_width: float = 1.5 * NODE_RESOLUTION,
    corridor_min_length: float = 2.0 * NODE_RESOLUTION,
) -> np.ndarray:
    adjacency_matrix = np.asarray(adjacency_matrix, dtype=int)
    if not enable_edge_pruning and not enable_graph_compression:
        return adjacency_matrix

    n_nodes = adjacency_matrix.shape[0]
    refined_sets = []
    for node_index in range(n_nodes):
        raw_neighbors = np.argwhere(adjacency_matrix[node_index] == 0).reshape(-1)
        refined = refine_neighbor_indices(
            node_coords,
            node_index,
            raw_neighbors,
            map_info,
            enable_edge_pruning=enable_edge_pruning,
            enable_graph_compression=enable_graph_compression,
            corridor_max_width=corridor_max_width,
            corridor_min_length=corridor_min_length,
        )
        refined_sets.append(set(int(value) for value in refined.tolist()))

    refined_adjacency = np.ones_like(adjacency_matrix)
    np.fill_diagonal(refined_adjacency, 0)
    for i in range(n_nodes):
        for j in refined_sets[i]:
            if i == j:
                continue
            if i in refined_sets[j]:
                refined_adjacency[i, j] = 0
                refined_adjacency[j, i] = 0

    for i in range(n_nodes):
        if np.count_nonzero(refined_adjacency[i] == 0) <= 1:
            raw_neighbors = np.argwhere(adjacency_matrix[i] == 0).reshape(-1)
            for j in raw_neighbors:
                refined_adjacency[i, j] = 0
                refined_adjacency[j, i] = 0

    return refined_adjacency


def compute_smoothness_penalty(
    previous_position,
    current_position,
    next_position,
    *,
    turn_penalty_weight: float,
    lateral_penalty_weight: float,
) -> float:
    if previous_position is None:
        return 0.0

    previous_position = np.asarray(previous_position, dtype=float).reshape(2)
    current_position = np.asarray(current_position, dtype=float).reshape(2)
    next_position = np.asarray(next_position, dtype=float).reshape(2)

    previous_step = current_position - previous_position
    next_step = next_position - current_position
    previous_norm = float(np.linalg.norm(previous_step))
    next_norm = float(np.linalg.norm(next_step))
    if previous_norm <= 1e-6 or next_norm <= 1e-6:
        return 0.0

    previous_dir = previous_step / previous_norm
    next_dir = next_step / next_norm
    cosine = float(np.clip(np.dot(previous_dir, next_dir), -1.0, 1.0))
    turn_penalty = turn_penalty_weight * (1.0 - cosine) / 2.0

    projected_next = float(np.dot(next_step, previous_dir))
    lateral_component = next_step - projected_next * previous_dir
    lateral_ratio = float(np.linalg.norm(lateral_component) / max(next_norm, 1e-6))
    lateral_penalty = lateral_penalty_weight * lateral_ratio
    return float(turn_penalty + lateral_penalty)
