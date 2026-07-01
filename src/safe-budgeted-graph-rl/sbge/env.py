from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import SBGEConfig
from .map_utils import FREE, OCCUPIED, UNKNOWN, bresenham_cells, in_bounds, load_large_drl_map
from .splits import load_map_files_for_config
from .types import StepResult

try:
    from scipy.ndimage import binary_dilation, distance_transform_edt
except Exception:  # pragma: no cover - scipy is available in rosclaw, fallback is defensive.
    binary_dilation = None
    distance_transform_edt = None


class SafeBudgetedGraphEnv:
    def __init__(self, config: SBGEConfig, seed: int | None = None, forced_map_path: str | Path | None = None):
        self.config = config
        self.seed = config.seed if seed is None else int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.map_files = load_map_files_for_config(config)
        self.forced_map_path = Path(forced_map_path).expanduser().resolve() if forced_map_path else None

        self.map_path: Path | None = None
        self.ground_truth_free: np.ndarray | None = None
        self.belief: np.ndarray | None = None
        self.start_cell: np.ndarray | None = None
        self.robot_cell: np.ndarray | None = None
        self.travel_dist: float = 0.0
        self.remaining_budget: float = 0.0
        self.initial_budget: float = 0.0
        self.step_count: int = 0
        self.explored_rate: float = 0.0
        self.static_risk_map: np.ndarray | None = None
        self.true_clearance_cells: np.ndarray | None = None
        self.dynamic_obstacles: list[dict[str, np.ndarray | float]] = []
        self.known_free_count: int = 0
        self.unknown_count: int = 0
        self.visited_cells: set[tuple[int, int]] = set()

    @property
    def shape(self) -> tuple[int, int]:
        assert self.ground_truth_free is not None
        return self.ground_truth_free.shape

    @property
    def map_diagonal_m(self) -> float:
        h, w = self.shape
        return float(np.hypot(w, h) * self.config.cell_size_m)

    def reset(self, episode_index: int = 0) -> None:
        self.rng = np.random.default_rng(self.seed + int(episode_index) * 9973)
        self.map_path = self.forced_map_path or self.map_files[int(episode_index) % len(self.map_files)]
        self.ground_truth_free, self.start_cell = load_large_drl_map(self.map_path)
        self.robot_cell = self.start_cell.astype(float).copy()
        self.belief = np.full(self.shape, UNKNOWN, dtype=np.int16)
        self.travel_dist = 0.0
        budget_ratio = self.rng.uniform(self.config.budget_ratio_min, self.config.budget_ratio_max)
        self.initial_budget = float(budget_ratio * self.map_diagonal_m)
        self.remaining_budget = self.initial_budget
        self.step_count = 0
        self.explored_rate = 0.0
        self.visited_cells = {self._cell_key(self.robot_cell)}
        self.true_clearance_cells = self._compute_clearance(self.ground_truth_free)
        self.static_risk_map = self._make_static_risk_map()
        self.dynamic_obstacles = self._make_dynamic_obstacles()
        self._sense()
        self.known_free_count = int(np.sum(self.belief == FREE))
        self.unknown_count = int(np.sum(self.belief == UNKNOWN))
        self._update_explored_rate()

    def _cell_key(self, cell_xy: np.ndarray) -> tuple[int, int]:
        x, y = np.round(cell_xy).astype(int)
        return int(x), int(y)

    def _compute_clearance(self, free_mask: np.ndarray) -> np.ndarray:
        if distance_transform_edt is not None:
            return distance_transform_edt(free_mask).astype(np.float32)
        obstacle_yx = np.argwhere(~free_mask)
        clearance = np.zeros(free_mask.shape, dtype=np.float32)
        free_yx = np.argwhere(free_mask)
        for y, x in free_yx:
            if obstacle_yx.size == 0:
                clearance[y, x] = max(free_mask.shape)
            else:
                clearance[y, x] = float(np.min(np.sum((obstacle_yx - [y, x]) ** 2, axis=1)) ** 0.5)
        return clearance

    def _make_static_risk_map(self) -> np.ndarray:
        assert self.true_clearance_cells is not None
        clearance_risk = np.exp(-self.true_clearance_cells / max(self.config.clearance_risk_scale_cells, 1e-6))
        risk = np.where(self.ground_truth_free, clearance_risk, 1.0).astype(np.float32)
        h, w = self.shape
        for _ in range(self.config.negative_obstacle_count):
            cx = self.rng.integers(0, w)
            cy = self.rng.integers(0, h)
            radius = max(int(self.config.negative_obstacle_radius_cells), 1)
            yy, xx = np.ogrid[:h, :w]
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
            risk[mask & self.ground_truth_free] = np.maximum(risk[mask & self.ground_truth_free], 0.85)
        return risk

    def _make_dynamic_obstacles(self) -> list[dict[str, np.ndarray | float]]:
        h, w = self.shape
        obstacles: list[dict[str, np.ndarray | float]] = []
        free_yx = np.argwhere(self.ground_truth_free)
        if free_yx.size == 0:
            return obstacles
        for _ in range(self.config.dynamic_obstacle_count):
            y, x = free_yx[self.rng.integers(0, len(free_yx))]
            angle = self.rng.uniform(0, 2 * np.pi)
            speed = self.rng.uniform(0.8, 1.8)
            obstacles.append(
                {
                    "position": np.array([x, y], dtype=float),
                    "velocity": np.array([np.cos(angle) * speed, np.sin(angle) * speed], dtype=float),
                    "radius": float(self.config.dynamic_obstacle_radius_cells),
                }
            )
        return obstacles

    def _sense(self) -> None:
        assert self.belief is not None and self.ground_truth_free is not None and self.robot_cell is not None
        angles = np.linspace(0, 2 * np.pi, self.config.sensor_rays, endpoint=False)
        for angle in angles:
            end = self.robot_cell + np.array(
                [np.cos(angle) * self.config.sensor_range_cells, np.sin(angle) * self.config.sensor_range_cells]
            )
            cells = bresenham_cells(self.robot_cell, end)
            cells = cells[in_bounds(cells, self.shape)]
            for x, y in cells:
                if self.ground_truth_free[y, x]:
                    observed = FREE
                    if self.config.sensor_false_occupied_rate > 0 and self.rng.random() < self.config.sensor_false_occupied_rate:
                        observed = OCCUPIED
                    self.belief[y, x] = observed
                else:
                    observed = OCCUPIED
                    if self.config.sensor_false_free_rate > 0 and self.rng.random() < self.config.sensor_false_free_rate:
                        observed = FREE
                    self.belief[y, x] = observed
                    break

    def _update_explored_rate(self) -> None:
        assert self.belief is not None and self.ground_truth_free is not None
        free_total = max(int(np.sum(self.ground_truth_free)), 1)
        self.explored_rate = float(np.sum((self.belief == FREE) & self.ground_truth_free) / free_total)

    def frontier_mask(self) -> np.ndarray:
        assert self.belief is not None
        free = self.belief == FREE
        unknown = self.belief == UNKNOWN
        if binary_dilation is not None:
            unknown_neighbor = binary_dilation(unknown, structure=np.ones((3, 3), dtype=bool))
            return free & unknown_neighbor
        padded = np.pad(unknown, 1, constant_values=False)
        neighbor = np.zeros_like(unknown, dtype=bool)
        for dy in range(3):
            for dx in range(3):
                neighbor |= padded[dy : dy + unknown.shape[0], dx : dx + unknown.shape[1]]
        return free & neighbor

    def dynamic_risk_at(self, xy: np.ndarray, horizon: int | None = None) -> float:
        horizon = self.config.dynamic_risk_horizon if horizon is None else int(horizon)
        risk = 0.0
        for obs in self.dynamic_obstacles:
            position = obs["position"]
            velocity = obs["velocity"]
            radius = float(obs["radius"])
            assert isinstance(position, np.ndarray) and isinstance(velocity, np.ndarray)
            for t in range(horizon + 1):
                pred = position + velocity * t
                dist = float(np.linalg.norm(xy - pred))
                risk = max(risk, float(np.exp(-(dist**2) / (2 * max(radius, 1e-6) ** 2))))
        return min(risk, 1.0)

    def _advance_dynamic_obstacles(self) -> None:
        h, w = self.shape
        for obs in self.dynamic_obstacles:
            position = obs["position"]
            velocity = obs["velocity"]
            assert isinstance(position, np.ndarray) and isinstance(velocity, np.ndarray)
            position += velocity
            if position[0] < 0 or position[0] >= w:
                velocity[0] *= -1
                position[0] = np.clip(position[0], 0, w - 1)
            if position[1] < 0 or position[1] >= h:
                velocity[1] *= -1
                position[1] = np.clip(position[1], 0, h - 1)

    def edge_stats(self, start_xy: np.ndarray, end_xy: np.ndarray) -> dict[str, float | int]:
        assert self.ground_truth_free is not None and self.static_risk_map is not None and self.true_clearance_cells is not None
        cells = bresenham_cells(start_xy, end_xy)
        cells = cells[in_bounds(cells, self.shape)]
        if len(cells) == 0:
            return {"collision": 1, "near_miss": 1, "static_risk": 1.0, "dynamic_risk": 1.0, "min_clearance": 0.0}
        xs = cells[:, 0]
        ys = cells[:, 1]
        collision = int(np.any(~self.ground_truth_free[ys, xs]))
        static_risk = float(np.max(self.static_risk_map[ys, xs]))
        dynamic_risk = max(self.dynamic_risk_at(cell.astype(float)) for cell in cells[:: max(len(cells) // 8, 1)])
        min_clearance = float(np.min(self.true_clearance_cells[ys, xs]))
        near_miss = int(min_clearance < self.config.near_miss_clearance_cells)
        return {
            "collision": collision,
            "near_miss": near_miss,
            "static_risk": static_risk,
            "dynamic_risk": float(dynamic_risk),
            "min_clearance": min_clearance,
        }

    def can_traverse_belief(self, start_xy: np.ndarray, end_xy: np.ndarray) -> bool:
        assert self.belief is not None
        cells = bresenham_cells(start_xy, end_xy)
        cells = cells[in_bounds(cells, self.shape)]
        if len(cells) == 0:
            return False
        xs = cells[:, 0]
        ys = cells[:, 1]
        return bool(np.all(self.belief[ys, xs] != OCCUPIED))

    def step(self, next_cell_xy: np.ndarray, return_cost_m: float | None = None) -> StepResult:
        assert self.robot_cell is not None and self.belief is not None
        old_free = int(np.sum(self.belief == FREE))
        old_unknown = int(np.sum(self.belief == UNKNOWN))
        old_frontiers = int(np.sum(self.frontier_mask()))
        start = self.robot_cell.copy()
        edge = self.edge_stats(start, next_cell_xy)
        dist_m = float(np.linalg.norm(next_cell_xy - start) * self.config.cell_size_m)
        self.robot_cell = np.array(next_cell_xy, dtype=float)
        self.travel_dist += dist_m
        self.remaining_budget -= dist_m
        self.step_count += 1
        self.visited_cells.add(self._cell_key(self.robot_cell))
        self._advance_dynamic_obstacles()
        self._sense()
        self._update_explored_rate()

        new_free = int(np.sum(self.belief == FREE))
        new_unknown = int(np.sum(self.belief == UNKNOWN))
        new_frontiers = int(np.sum(self.frontier_mask()))
        newly_free = max(new_free - old_free, 0)
        entropy_gain = max(old_unknown - new_unknown, 0)
        frontier_gain = max(old_frontiers - new_frontiers, 0)
        risk = max(float(edge["static_risk"]), float(edge["dynamic_risk"]))
        budget_violation = int(self.remaining_budget < 0)
        return_feasible = 1
        if return_cost_m is not None:
            return_feasible = int(self.remaining_budget + 1e-6 >= return_cost_m)
            budget_violation = int(budget_violation or not return_feasible)

        done = bool(self.explored_rate >= self.config.target_explored_rate or budget_violation)
        return_success = int(done and self.explored_rate >= self.config.target_explored_rate and return_feasible)
        reward = (
            newly_free / max(self.config.sensor_range_cells, 1)
            + 0.05 * entropy_gain / max(self.config.sensor_range_cells, 1)
            + 0.1 * frontier_gain
            - 2.0 * dist_m / max(self.map_diagonal_m, 1e-6)
            - 1.5 * risk
        )
        if self.explored_rate >= self.config.target_explored_rate:
            reward += self.config.reward_completion_bonus
        if return_success:
            reward += self.config.reward_return_bonus

        cost = float(
            0.5 * float(edge["static_risk"])
            + 0.5 * float(edge["dynamic_risk"])
            + float(edge["collision"])
            + float(budget_violation)
        )
        info = {
            "distance_m": dist_m,
            "newly_free": newly_free,
            "entropy_gain": entropy_gain,
            "frontier_gain": frontier_gain,
            "collision": int(edge["collision"]),
            "near_miss": int(edge["near_miss"]),
            "static_risk": float(edge["static_risk"]),
            "dynamic_risk": float(edge["dynamic_risk"]),
            "risk": risk,
            "budget_violation": budget_violation,
            "return_feasible": return_feasible,
            "return_success": return_success,
            "remaining_budget": float(self.remaining_budget),
        }
        return StepResult(float(reward), cost, done, info)
