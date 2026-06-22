from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PredictionResult:
    predicted_utility: np.ndarray
    uncertainty: np.ndarray
    risk_aware_utility: np.ndarray

    def as_feature_matrix(self) -> np.ndarray:
        return np.stack(
            (
                self.predicted_utility,
                self.uncertainty,
                self.risk_aware_utility,
            ),
            axis=-1,
        ).astype(np.float32)


@dataclass(frozen=True)
class BeliefNodeFeatures:
    predicted_utility: np.ndarray
    uncertainty: np.ndarray
    risk_aware_utility: np.ndarray
    cluster_prior: np.ndarray

    def as_feature_matrix(self) -> np.ndarray:
        return np.stack(
            (
                self.predicted_utility,
                self.uncertainty,
                self.risk_aware_utility,
                self.cluster_prior,
            ),
            axis=-1,
        ).astype(np.float32)


def zero_belief_features(n_nodes: int) -> np.ndarray:
    return np.zeros((max(int(n_nodes), 0), 4), dtype=np.float32)


def _coord_key(coords: np.ndarray, digits: int = 3) -> tuple[float, float]:
    coords = np.asarray(coords, dtype=np.float64).reshape(-1)
    return tuple(np.round(coords[:2], digits).tolist())


class BeliefStateTracker:
    def __init__(self, alpha: float = 0.35, risk_weight: float = 0.35, digits: int = 3):
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.risk_weight = max(float(risk_weight), 0.0)
        self.digits = int(digits)
        self._state: dict[tuple[float, float], tuple[float, float]] = {}

    def reset(self) -> None:
        self._state.clear()

    def update(
        self,
        node_coords: np.ndarray,
        prediction: PredictionResult,
        cluster_prior: np.ndarray | None = None,
    ) -> BeliefNodeFeatures:
        node_coords = np.asarray(node_coords, dtype=np.float64).reshape(-1, 2)
        n_nodes = node_coords.shape[0]
        predicted = np.asarray(prediction.predicted_utility, dtype=np.float32).reshape(n_nodes)
        input_uncertainty = np.asarray(prediction.uncertainty, dtype=np.float32).reshape(n_nodes)
        if cluster_prior is None:
            cluster_prior = np.zeros(n_nodes, dtype=np.float32)
        cluster_prior = np.asarray(cluster_prior, dtype=np.float32).reshape(n_nodes)

        means = np.zeros(n_nodes, dtype=np.float32)
        variances = np.zeros(n_nodes, dtype=np.float32)
        for index, coords in enumerate(node_coords):
            key = _coord_key(coords, self.digits)
            old_mean, old_var = self._state.get(key, (float(predicted[index]), float(input_uncertainty[index] ** 2)))
            delta = float(predicted[index]) - old_mean
            mean = old_mean + self.alpha * delta
            variance = max((1.0 - self.alpha) * (old_var + self.alpha * delta * delta), 0.0)
            self._state[key] = (mean, variance)
            means[index] = mean
            variances[index] = variance

        uncertainty = np.maximum(input_uncertainty, np.sqrt(variances)).astype(np.float32)
        predicted_utility = np.clip(means, 0.0, 1.0)
        uncertainty = np.clip(uncertainty, 0.0, 1.0)
        risk_aware = np.clip(predicted_utility - self.risk_weight * uncertainty + 0.15 * cluster_prior, -1.0, 1.0)
        return BeliefNodeFeatures(
            predicted_utility=predicted_utility,
            uncertainty=uncertainty,
            risk_aware_utility=risk_aware.astype(np.float32),
            cluster_prior=np.clip(cluster_prior, 0.0, 1.0).astype(np.float32),
        )