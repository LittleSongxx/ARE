from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


def stable_id(x: float, y: float, resolution: float = 0.1, namespace: int = 0) -> int:
    qx, qy = int(round(x / resolution)), int(round(y / resolution))
    raw = hashlib.blake2b(f"{namespace}:{qx}:{qy}".encode(), digest_size=8).digest()
    return int.from_bytes(raw, "little") & ((1 << 63) - 1)


def _frontier_mask(grid: np.ndarray) -> np.ndarray:
    free = grid == 0
    unknown = grid < 0
    adjacent_free = np.zeros_like(free)
    adjacent_free[1:] |= free[:-1]
    adjacent_free[:-1] |= free[1:]
    adjacent_free[:, 1:] |= free[:, :-1]
    adjacent_free[:, :-1] |= free[:, 1:]
    return unknown & adjacent_free


def _line_is_free(grid: np.ndarray, start: np.ndarray, end: np.ndarray) -> bool:
    delta = end - start
    count = int(max(abs(delta[0]), abs(delta[1]))) + 1
    if count <= 1:
        return True
    x = np.rint(np.linspace(start[0], end[0], count)).astype(int)
    y = np.rint(np.linspace(start[1], end[1], count)).astype(int)
    inside = (x >= 0) & (x < grid.shape[1]) & (y >= 0) & (y < grid.shape[0])
    if not inside.all():
        return False
    return bool(np.all(grid[y, x] == 0))


@dataclass
class GraphInput:
    feeds: dict[str, np.ndarray]
    candidate_xy: np.ndarray
    candidate_ids: np.ndarray
    candidate_events: np.ndarray
    node_ids: np.ndarray


class OccupancyGraphBuilder:
    def __init__(
        self,
        *,
        nodes: int = 224,
        candidates: int = 25,
        node_resolution_m: float = 4.0,
        sensor_range_m: float = 16.0,
        local_budget: int = 192,
        region_size_m: float = 16.0,
    ) -> None:
        self.nodes = int(nodes)
        self.candidates = int(candidates)
        self.node_resolution_m = float(node_resolution_m)
        self.sensor_range_m = float(sensor_range_m)
        self.local_budget = min(int(local_budget), self.nodes)
        self.region_size_m = float(region_size_m)
        self.visited: set[int] = set()
        self.previous_utility: dict[int, float] = {}
        self.previous_neighbors: dict[int, set[int]] = {}

    def reset(self) -> None:
        self.visited.clear()
        self.previous_utility.clear()
        self.previous_neighbors.clear()

    def mark_visited(self, x: float, y: float) -> None:
        self.visited.add(stable_id(x, y))

    def build(
        self,
        grid: np.ndarray,
        resolution: float,
        origin_xy: tuple[float, float],
        robot_xy: tuple[float, float],
    ) -> GraphInput:
        grid = np.asarray(grid, dtype=np.int16)
        stride = max(1, int(round(self.node_resolution_m / resolution)))
        free_y, free_x = np.nonzero(grid == 0)
        keep = (free_x % stride == 0) & (free_y % stride == 0)
        cells = np.column_stack((free_x[keep], free_y[keep])).astype(np.int32)
        robot_cell = np.rint(
            (np.asarray(robot_xy, dtype=np.float32) - np.asarray(origin_xy, dtype=np.float32)) / resolution
        ).astype(np.int32)
        if cells.size == 0:
            cells = robot_cell.reshape(1, 2)
        distance_cells = np.linalg.norm(cells - robot_cell, axis=1)
        nearest = int(np.argmin(distance_cells))
        cells[[0, nearest]] = cells[[nearest, 0]]
        cells[0] = robot_cell
        world = cells.astype(np.float32) * resolution + np.asarray(origin_xy, dtype=np.float32)
        world[0] = np.asarray(robot_xy, dtype=np.float32)

        frontier_cells = np.column_stack(np.nonzero(_frontier_mask(grid))[::-1]).astype(np.float32)
        utility = np.zeros(len(cells), dtype=np.float32)
        if len(frontier_cells):
            radius_cells = self.sensor_range_m / resolution
            for index, cell in enumerate(cells):
                utility[index] = np.count_nonzero(np.linalg.norm(frontier_cells - cell, axis=1) <= radius_cells)

        candidate_pool = []
        for index in np.argsort(np.linalg.norm(world - world[0], axis=1))[1:]:
            distance = np.linalg.norm(world[index] - world[0])
            if distance > self.node_resolution_m * 2.5:
                break
            if _line_is_free(grid, robot_cell, cells[index]):
                candidate_pool.append(int(index))
            if len(candidate_pool) >= self.candidates:
                break
        if not candidate_pool and len(cells) > 1:
            candidate_pool = [int(np.argsort(np.linalg.norm(world - world[0], axis=1))[1])]

        protected = [0] + candidate_pool
        score = utility - 0.01 * np.linalg.norm(world - world[0], axis=1)
        ranked = [int(index) for index in np.argsort(score)[::-1] if int(index) not in protected]
        local_old = protected + ranked[: max(0, self.local_budget - len(protected))]
        local_old = list(dict.fromkeys(local_old))
        local_lookup = {old: new for new, old in enumerate(local_old)}
        remote = [index for index in range(len(cells)) if index not in local_lookup]
        groups: dict[tuple[int, int], list[int]] = {}
        for index in remote:
            key = tuple(np.floor(world[index] / self.region_size_m).astype(int))
            groups.setdefault(key, []).append(index)
        region_budget = self.nodes - len(local_old)
        region_items = sorted(groups.items(), key=lambda item: (utility[item[1]].sum(), len(item[1])), reverse=True)[
            :region_budget
        ]

        feature_dim = 4
        utility_normalizer = max(1.0, np.pi * self.sensor_range_m / (2 * resolution))
        node_features = np.zeros((self.nodes, feature_dim), dtype=np.float32)
        node_mask = np.zeros(self.nodes, dtype=np.bool_)
        adjacency = np.zeros((self.nodes, self.nodes), dtype=np.bool_)
        node_world = np.zeros((self.nodes, 2), dtype=np.float32)
        local_world = world[local_old]
        node_world[: len(local_old)] = local_world
        node_features[: len(local_old), :2] = (local_world - world[0]) / 160.0
        node_features[: len(local_old), 2] = utility[local_old] / utility_normalizer
        node_features[: len(local_old), 3] = [stable_id(*xy) in self.visited for xy in local_world]
        node_mask[: len(local_old)] = True

        old_to_region = {}
        for offset, (region_key, members) in enumerate(region_items):
            new_index = len(local_old) + offset
            node_world[new_index] = world[members].mean(axis=0)
            node_features[new_index, :2] = (node_world[new_index] - world[0]) / 160.0
            node_features[new_index, 2] = utility[members].mean() / utility_normalizer
            node_features[new_index, 3] = np.mean([stable_id(*world[item]) in self.visited for item in members])
            node_mask[new_index] = True
            for member in members:
                old_to_region[member] = new_index

        selected = set(local_old) | set(old_to_region)
        connection_radius = self.node_resolution_m * 2.5
        # Spatial hashing keeps this near-linear even when a selected remote
        # region contains thousands of raw grid samples.
        spatial: dict[tuple[int, int], list[int]] = {}
        for old_index in selected:
            key = tuple(np.floor(world[old_index] / connection_radius).astype(int))
            spatial.setdefault(key, []).append(old_index)
        checked = set()
        for key, members in spatial.items():
            nearby = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nearby.extend(spatial.get((key[0] + dx, key[1] + dy), ()))
            for left in members:
                left_new = local_lookup.get(left, old_to_region.get(left))
                for right in nearby:
                    pair = (min(left, right), max(left, right))
                    if pair in checked:
                        continue
                    checked.add(pair)
                    right_new = local_lookup.get(right, old_to_region.get(right))
                    if left_new is None or right_new is None or left_new == right_new:
                        continue
                    distance = np.linalg.norm(world[left] - world[right])
                    if distance <= connection_radius and _line_is_free(grid, cells[left], cells[right]):
                        adjacency[left_new, right_new] = True
                        adjacency[right_new, left_new] = True
        valid_count = int(node_mask.sum())
        adjacency[np.arange(valid_count), np.arange(valid_count)] = True

        candidate_indices = np.zeros(self.candidates, dtype=np.int64)
        candidate_mask = np.zeros(self.candidates, dtype=np.bool_)
        candidate_xy = np.zeros((self.candidates, 2), dtype=np.float32)
        candidate_ids = np.zeros(self.candidates, dtype=np.int64)
        edge_features = np.zeros((self.candidates, 4), dtype=np.float32)
        candidate_events = np.zeros(self.candidates, dtype=np.int16)
        stable_ids = np.zeros(valid_count, dtype=np.int64)
        for index in range(valid_count):
            stable_ids[index] = stable_id(*node_world[index], namespace=int(index >= len(local_old)))
        neighbor_sets = {
            int(stable_ids[index]): {int(stable_ids[target]) for target in np.flatnonzero(adjacency[index])}
            for index in range(valid_count)
        }
        for slot, old_index in enumerate(candidate_pool[: self.candidates]):
            new_index = local_lookup[old_index]
            candidate_indices[slot] = new_index
            candidate_mask[slot] = True
            candidate_xy[slot] = world[old_index]
            candidate_ids[slot] = stable_id(*world[old_index])
            delta = world[old_index] - world[0]
            distance = np.linalg.norm(delta)
            edge_features[slot] = [delta[0] / 80.0, delta[1] / 80.0, distance / 80.0, distance / 80.0]
            candidate_id = int(candidate_ids[slot])
            previous_utility = self.previous_utility.get(candidate_id)
            if previous_utility is not None and previous_utility != float(node_features[new_index, 2]):
                candidate_events[slot] |= 1
            new_index_neighbors = neighbor_sets.get(int(stable_ids[new_index]), set())
            previous_neighbors = self.previous_neighbors.get(candidate_id)
            if previous_neighbors is not None and previous_neighbors != new_index_neighbors:
                candidate_events[slot] |= 2
            if candidate_id in self.visited:
                candidate_events[slot] |= 16

        feeds = {
            "node_features": node_features[None],
            "node_mask": node_mask[None],
            "adjacency": adjacency[None],
            "current_index": np.asarray([0], dtype=np.int64),
            "candidate_indices": candidate_indices[None],
            "candidate_mask": candidate_mask[None],
            "edge_features": edge_features[None],
            "posterior_mean": np.zeros((1, self.candidates), dtype=np.float32),
            "posterior_variance": np.ones((1, self.candidates), dtype=np.float32),
        }
        for index in range(valid_count):
            key = int(stable_ids[index])
            self.previous_utility[key] = float(node_features[index, 2])
            self.previous_neighbors[key] = neighbor_sets[key]
        return GraphInput(feeds, candidate_xy, candidate_ids, candidate_events, stable_ids)
