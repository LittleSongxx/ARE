from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from .config import SBGEConfig
from .env import SafeBudgetedGraphEnv
from .map_utils import FREE, OCCUPIED, UNKNOWN, bresenham_cells, in_bounds
from .types import Observation

try:
    from scipy.ndimage import distance_transform_edt
except Exception:  # pragma: no cover
    distance_transform_edt = None


@dataclass
class GraphBuildInfo:
    edge_risk: np.ndarray
    return_cost_m: np.ndarray
    edge_dist_m: np.ndarray


class SafeGraphBuilder:
    def __init__(self, config: SBGEConfig):
        self.config = config
        self.last_info: GraphBuildInfo | None = None

    def build(self, env: SafeBudgetedGraphEnv) -> Observation:
        assert env.belief is not None and env.robot_cell is not None and env.start_cell is not None
        positions = self._sample_node_positions(env)
        current_index = self._ensure_position(positions, env.robot_cell)
        start_index = self._ensure_position(positions, env.start_cell)
        if len(positions) > self.config.max_nodes:
            positions = self._trim_positions(positions, env.robot_cell, env.start_cell)
            current_index = self._ensure_position(positions, env.robot_cell)
            start_index = self._ensure_position(positions, env.start_cell)

        n_nodes = len(positions)
        edge_mask = np.ones((n_nodes, n_nodes), dtype=bool)
        edge_risk = np.ones((n_nodes, n_nodes), dtype=np.float32)
        edge_dist_m = np.full((n_nodes, n_nodes), np.inf, dtype=np.float32)
        np.fill_diagonal(edge_mask, False)
        np.fill_diagonal(edge_risk, 1.0)
        np.fill_diagonal(edge_dist_m, 0.0)

        max_edge_cells = self.config.node_stride_cells * 1.55
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                dist_cells = float(np.linalg.norm(positions[i] - positions[j]))
                if dist_cells > max_edge_cells:
                    continue
                if not env.can_traverse_belief(positions[i], positions[j]):
                    continue
                stats = env.edge_stats(positions[i], positions[j])
                risk = max(float(stats["static_risk"]), float(stats["dynamic_risk"]))
                dist_m = dist_cells * self.config.cell_size_m
                edge_mask[i, j] = False
                edge_mask[j, i] = False
                edge_risk[i, j] = risk
                edge_risk[j, i] = risk
                edge_dist_m[i, j] = dist_m
                edge_dist_m[j, i] = dist_m

        return_cost_m = self._dijkstra(edge_mask, edge_dist_m, start_index)
        node_features, critic_node_features = self._node_features(env, positions, current_index, return_cost_m)
        neighbor_indices, action_mask, fallback_slot = self._current_actions(
            positions,
            current_index,
            edge_mask,
            edge_risk,
            edge_dist_m,
            return_cost_m,
            env.remaining_budget,
        )
        expert_slot = self._expert_action_slot(node_features, action_mask, neighbor_indices, edge_dist_m, current_index)
        self.last_info = GraphBuildInfo(edge_risk=edge_risk, return_cost_m=return_cost_m, edge_dist_m=edge_dist_m)
        return Observation(
            node_features=node_features,
            critic_node_features=critic_node_features,
            edge_mask=edge_mask,
            action_mask=action_mask,
            current_index=current_index,
            neighbor_indices=neighbor_indices,
            node_positions=positions,
            fallback_action_slot=fallback_slot,
            expert_action_slot=expert_slot,
        )

    def _sample_node_positions(self, env: SafeBudgetedGraphEnv) -> np.ndarray:
        assert env.belief is not None and env.robot_cell is not None and env.start_cell is not None
        yx = np.argwhere(env.belief == FREE)
        stride = max(int(self.config.node_stride_cells), 1)
        offset_x = int(round(env.start_cell[0])) % stride
        offset_y = int(round(env.start_cell[1])) % stride
        sampled = yx[((yx[:, 1] - offset_x) % stride == 0) & ((yx[:, 0] - offset_y) % stride == 0)]
        if len(sampled) == 0:
            sampled = yx[:: max(len(yx) // max(self.config.max_nodes // 2, 1), 1)]
        positions = [np.array([x, y], dtype=float) for y, x in sampled]
        positions.append(env.robot_cell.astype(float))
        positions.append(env.start_cell.astype(float))
        frontier_yx = np.argwhere(env.frontier_mask())
        if len(frontier_yx) > 0:
            step = max(len(frontier_yx) // 24, 1)
            for y, x in frontier_yx[::step][:24]:
                positions.append(np.array([x, y], dtype=float))
        return self._unique_positions(np.array(positions, dtype=float))

    def _trim_positions(self, positions: np.ndarray, robot_cell: np.ndarray, start_cell: np.ndarray) -> np.ndarray:
        d_robot = np.linalg.norm(positions - robot_cell, axis=1)
        d_start = np.linalg.norm(positions - start_cell, axis=1)
        score = d_robot + 0.25 * d_start
        keep = np.argsort(score)[: self.config.max_nodes - 2]
        trimmed = positions[keep]
        trimmed = np.vstack([trimmed, robot_cell.astype(float), start_cell.astype(float)])
        return self._unique_positions(trimmed)[: self.config.max_nodes]

    def _unique_positions(self, positions: np.ndarray) -> np.ndarray:
        seen: set[tuple[int, int]] = set()
        unique = []
        for pos in positions:
            key = (int(round(pos[0])), int(round(pos[1])))
            if key not in seen:
                seen.add(key)
                unique.append(np.array(key, dtype=float))
        return np.array(unique, dtype=float)

    def _ensure_position(self, positions: np.ndarray, target: np.ndarray) -> int:
        keys = np.round(positions).astype(int)
        key = np.round(target).astype(int)
        matches = np.flatnonzero((keys[:, 0] == key[0]) & (keys[:, 1] == key[1]))
        if len(matches) > 0:
            return int(matches[0])
        raise ValueError("required graph position is missing")

    def _dijkstra(self, edge_mask: np.ndarray, edge_dist_m: np.ndarray, source: int) -> np.ndarray:
        n = edge_mask.shape[0]
        dist = np.full(n, np.inf, dtype=np.float32)
        dist[source] = 0.0
        heap = [(0.0, source)]
        while heap:
            cost, node = heapq.heappop(heap)
            if cost > dist[node]:
                continue
            for nxt in np.flatnonzero(~edge_mask[node]):
                if nxt == node:
                    continue
                alt = cost + float(edge_dist_m[node, nxt])
                if alt < dist[nxt]:
                    dist[nxt] = alt
                    heapq.heappush(heap, (alt, int(nxt)))
        return dist

    def _observed_clearance(self, belief: np.ndarray) -> np.ndarray:
        free_or_unknown = belief != OCCUPIED
        if distance_transform_edt is not None:
            return distance_transform_edt(free_or_unknown).astype(np.float32)
        return np.where(free_or_unknown, self.config.sensor_range_cells, 0).astype(np.float32)

    def _node_features(
        self,
        env: SafeBudgetedGraphEnv,
        positions: np.ndarray,
        current_index: int,
        return_cost_m: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert env.belief is not None and env.static_risk_map is not None and env.ground_truth_free is not None
        observed_clearance = self._observed_clearance(env.belief)
        frontiers = np.argwhere(env.frontier_mask())
        diag = max(env.map_diagonal_m, 1e-6)
        current = positions[current_index]
        actor_rows = []
        critic_rows = []
        for pos in positions:
            x, y = np.round(pos).astype(int)
            x = int(np.clip(x, 0, env.shape[1] - 1))
            y = int(np.clip(y, 0, env.shape[0] - 1))
            rel = (pos - current) * self.config.cell_size_m / diag
            if len(frontiers) == 0:
                utility = 0.0
            else:
                frontier_xy = np.stack([frontiers[:, 1], frontiers[:, 0]], axis=1)
                utility = float(np.sum(np.linalg.norm(frontier_xy - pos, axis=1) <= self.config.utility_radius_cells))
            utility_norm = utility / max((np.pi * self.config.utility_radius_cells**2) ** 0.5, 1.0)
            visited = float((int(round(pos[0])), int(round(pos[1]))) in env.visited_cells)
            local_entropy = self._local_unknown_fraction(env.belief, x, y)
            clearance = float(observed_clearance[y, x])
            clearance_norm = min(clearance / max(self.config.sensor_range_cells, 1), 1.0)
            static_risk = float(env.static_risk_map[y, x])
            dynamic_risk = float(env.dynamic_risk_at(pos))
            return_cost_norm = 1.0 if not np.isfinite(return_cost_m[self._position_index(positions, pos)]) else min(
                float(return_cost_m[self._position_index(positions, pos)]) / diag,
                1.0,
            )
            remaining_norm = min(max(env.remaining_budget / max(env.initial_budget, 1e-6), 0.0), 1.0)
            budget_feasible = float(env.remaining_budget + 1e-6 >= float(return_cost_m[self._position_index(positions, pos)]))
            actor = np.array(
                [
                    rel[0],
                    rel[1],
                    utility_norm,
                    visited,
                    local_entropy,
                    clearance_norm,
                    static_risk,
                    dynamic_risk,
                    return_cost_norm,
                    remaining_norm,
                    budget_feasible,
                ],
                dtype=np.float32,
            )
            true_free_ratio = self._local_true_free_ratio(env.ground_truth_free, x, y)
            true_return_feasible = budget_feasible
            critic = np.concatenate(
                [
                    actor,
                    np.array([true_free_ratio, float(env.static_risk_map[y, x]), true_return_feasible], dtype=np.float32),
                ]
            )
            actor_rows.append(actor)
            critic_rows.append(critic)
        return np.stack(actor_rows), np.stack(critic_rows)

    def _position_index(self, positions: np.ndarray, pos: np.ndarray) -> int:
        matches = np.flatnonzero(np.all(np.round(positions).astype(int) == np.round(pos).astype(int), axis=1))
        return int(matches[0])

    def _local_unknown_fraction(self, belief: np.ndarray, x: int, y: int) -> float:
        radius = self.config.local_patch_radius_cells
        patch = belief[max(0, y - radius) : y + radius + 1, max(0, x - radius) : x + radius + 1]
        return float(np.mean(patch == UNKNOWN)) if patch.size else 1.0

    def _local_true_free_ratio(self, free_mask: np.ndarray, x: int, y: int) -> float:
        radius = self.config.local_patch_radius_cells
        patch = free_mask[max(0, y - radius) : y + radius + 1, max(0, x - radius) : x + radius + 1]
        return float(np.mean(patch)) if patch.size else 0.0

    def _current_actions(
        self,
        positions: np.ndarray,
        current_index: int,
        edge_mask: np.ndarray,
        edge_risk: np.ndarray,
        edge_dist_m: np.ndarray,
        return_cost_m: np.ndarray,
        remaining_budget: float,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        neighbors = [idx for idx in np.flatnonzero(~edge_mask[current_index]) if idx != current_index]
        neighbors = sorted(neighbors, key=lambda idx: float(edge_dist_m[current_index, idx]))
        neighbors = neighbors[: self.config.max_neighbors]
        padded = np.full(self.config.max_neighbors, current_index, dtype=np.int64)
        action_mask = np.ones(self.config.max_neighbors, dtype=bool)
        fallback_slot = 0
        best_fallback_score = np.inf
        for slot, idx in enumerate(neighbors):
            padded[slot] = int(idx)
            edge_budget_feasible = remaining_budget - float(edge_dist_m[current_index, idx]) + 1e-6 >= float(return_cost_m[idx])
            safe = bool(edge_risk[current_index, idx] <= self.config.risk_threshold and edge_budget_feasible)
            action_mask[slot] = not safe
            fallback_score = float(edge_risk[current_index, idx]) + 0.1 * float(edge_dist_m[current_index, idx])
            if fallback_score < best_fallback_score:
                fallback_slot = slot
                best_fallback_score = fallback_score
        return padded, action_mask, int(fallback_slot)

    def _expert_action_slot(
        self,
        node_features: np.ndarray,
        action_mask: np.ndarray,
        neighbor_indices: np.ndarray,
        edge_dist_m: np.ndarray,
        current_index: int,
    ) -> int | None:
        safe_slots = np.flatnonzero(~action_mask)
        if len(safe_slots) == 0:
            return None
        scores = []
        for slot in safe_slots:
            idx = neighbor_indices[slot]
            gain = float(node_features[idx, 2]) + 0.5 * float(node_features[idx, 4])
            cost = float(edge_dist_m[current_index, idx])
            risk = float(node_features[idx, 6]) + float(node_features[idx, 7])
            scores.append(gain - 0.05 * cost - risk)
        return int(safe_slots[int(np.argmax(scores))])

    def edge_return_cost_for_action(self, observation: Observation, action_slot: int) -> float:
        if self.last_info is None:
            return 0.0
        idx = int(observation.neighbor_indices[action_slot])
        return float(self.last_info.return_cost_m[idx])
