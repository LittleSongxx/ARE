from __future__ import annotations

import numpy as np

from .belief_state import PredictionResult
from .parameter import FREE, SENSOR_RANGE, UNKNOWN
from .utils import get_cell_position_from_coords


def _safe_normalize(values: np.ndarray, denominator: float | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if denominator is None:
        finite = values[np.isfinite(values)]
        denominator = float(np.max(np.abs(finite))) if finite.size else 1.0
    denominator = max(float(denominator), 1.0)
    return np.clip(values / denominator, 0.0, 1.0).astype(np.float32)


class HeuristicMapPredictor:
    def __init__(self, sensor_range: float = SENSOR_RANGE, risk_weight: float = 0.35):
        self.sensor_range = max(float(sensor_range), 1e-6)
        self.risk_weight = max(float(risk_weight), 0.0)

    def _local_unknown_density(self, node_coords: np.ndarray, map_info) -> np.ndarray:
        node_coords = np.asarray(node_coords, dtype=np.float64).reshape(-1, 2)
        output = np.zeros(node_coords.shape[0], dtype=np.float32)
        if map_info is None or getattr(map_info, "map", None) is None or node_coords.size == 0:
            return output

        grid = np.asarray(map_info.map)
        radius_cells = max(int(round(self.sensor_range / max(float(map_info.cell_size), 1e-6))), 1)
        yy, xx = np.ogrid[-radius_cells : radius_cells + 1, -radius_cells : radius_cells + 1]
        circle_mask = xx * xx + yy * yy <= radius_cells * radius_cells
        circle_total = max(int(np.count_nonzero(circle_mask)), 1)

        cells = get_cell_position_from_coords(node_coords, map_info, check_negative=False).reshape(-1, 2)
        for index, (cell_x, cell_y) in enumerate(cells):
            x0 = max(int(cell_x) - radius_cells, 0)
            y0 = max(int(cell_y) - radius_cells, 0)
            x1 = min(int(cell_x) + radius_cells + 1, grid.shape[1])
            y1 = min(int(cell_y) + radius_cells + 1, grid.shape[0])
            if x0 >= x1 or y0 >= y1:
                continue
            mask_x0 = x0 - (int(cell_x) - radius_cells)
            mask_y0 = y0 - (int(cell_y) - radius_cells)
            mask_x1 = mask_x0 + (x1 - x0)
            mask_y1 = mask_y0 + (y1 - y0)
            local = grid[y0:y1, x0:x1]
            local_mask = circle_mask[mask_y0:mask_y1, mask_x0:mask_x1]
            output[index] = float(np.count_nonzero((local == UNKNOWN) & local_mask)) / float(circle_total)
        return np.clip(output, 0.0, 1.0).astype(np.float32)

    def _frontier_proximity(self, node_coords: np.ndarray, frontier) -> np.ndarray:
        node_coords = np.asarray(node_coords, dtype=np.float64).reshape(-1, 2)
        if frontier is None or len(frontier) == 0 or node_coords.size == 0:
            return np.zeros(node_coords.shape[0], dtype=np.float32)
        frontier_coords = np.asarray(list(frontier), dtype=np.float64).reshape(-1, 2)
        dists = np.linalg.norm(node_coords[:, None, :] - frontier_coords[None, :, :], axis=-1)
        nearest = np.min(dists, axis=1)
        return np.exp(-nearest / self.sensor_range).astype(np.float32)

    def predict(
        self,
        node_coords: np.ndarray,
        utility: np.ndarray,
        guidepost: np.ndarray,
        map_info,
        frontier=None,
        utility_normalizer: float | None = None,
    ) -> PredictionResult:
        node_coords = np.asarray(node_coords, dtype=np.float64).reshape(-1, 2)
        n_nodes = node_coords.shape[0]
        if n_nodes == 0:
            empty = np.zeros(0, dtype=np.float32)
            return PredictionResult(empty, empty, empty)

        utility = np.asarray(utility, dtype=np.float32).reshape(n_nodes)
        guidepost = np.asarray(guidepost, dtype=np.float32).reshape(n_nodes)
        observed_utility = _safe_normalize(np.maximum(utility, 0.0), utility_normalizer)
        unknown_density = self._local_unknown_density(node_coords, map_info)
        frontier_score = self._frontier_proximity(node_coords, frontier)
        prediction_from_unknown = unknown_density * (0.35 + 0.65 * np.maximum(frontier_score, observed_utility))
        predicted_utility = np.maximum(observed_utility, prediction_from_unknown)
        predicted_utility = predicted_utility * (1.0 - 0.25 * np.clip(guidepost, 0.0, 1.0))

        free_density = np.zeros_like(predicted_utility)
        if map_info is not None and getattr(map_info, "map", None) is not None:
            free_density.fill(float(np.mean(np.asarray(map_info.map) == FREE)))
        uncertainty = np.clip(0.75 * unknown_density + 0.25 * (1.0 - frontier_score) * (1.0 - free_density), 0.0, 1.0)
        risk_aware = np.clip(predicted_utility - self.risk_weight * uncertainty, -1.0, 1.0)
        return PredictionResult(
            predicted_utility=predicted_utility.astype(np.float32),
            uncertainty=uncertainty.astype(np.float32),
            risk_aware_utility=risk_aware.astype(np.float32),
        )