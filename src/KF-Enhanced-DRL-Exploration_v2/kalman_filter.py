"""Lightweight scalar Kalman filter for RL training signals.

Implements ideas from:
- KRPO (arXiv:2505.07527): KF-enhanced advantage estimation
- KARNet (arXiv:2305.14644): KF-augmented state prediction
- Sim-to-Real DRL UAV (arXiv:2303.07243): KF denoising for observations

Each KF instance tracks a single scalar quantity with online mean and
variance estimation, requiring no learned parameters.
"""

from __future__ import annotations

import numpy as np


class ScalarKalmanFilter:
    """1D Kalman filter for tracking a slowly-varying scalar signal.

    State model:  x_{t+1} = x_t + w_t,  w_t ~ N(0, process_noise)
    Obs. model:   z_t = x_t + v_t,      v_t ~ N(0, measurement_noise)
    """

    def __init__(
        self,
        initial_state: float = 0.0,
        initial_variance: float = 1.0,
        process_noise: float = 0.01,
        measurement_noise: float = 1.0,
    ):
        self.state = float(initial_state)
        self.variance = float(initial_variance)
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
        self.n_updates = 0

    def predict(self) -> tuple[float, float]:
        """Predict step: propagate state forward."""
        predicted_state = self.state
        predicted_variance = self.variance + self.process_noise
        return predicted_state, predicted_variance

    def update(self, measurement: float) -> float:
        """Full predict-then-update cycle. Returns the filtered state."""
        pred_state, pred_var = self.predict()

        innovation = measurement - pred_state
        innovation_var = pred_var + self.measurement_noise
        kalman_gain = pred_var / max(innovation_var, 1e-10)

        self.state = pred_state + kalman_gain * innovation
        self.variance = (1.0 - kalman_gain) * pred_var
        self.n_updates += 1
        return self.state

    def get_state(self) -> float:
        return self.state

    def get_uncertainty(self) -> float:
        return float(np.sqrt(max(self.variance, 0.0)))

    def reset(
        self,
        initial_state: float = 0.0,
        initial_variance: float = 1.0,
    ) -> None:
        self.state = float(initial_state)
        self.variance = float(initial_variance)
        self.n_updates = 0


class RewardBaselineKF:
    """Kalman filter for dynamically estimating the reward baseline.

    Inspired by KRPO (arXiv:2505.07527): replaces naive group-mean reward
    baseline with KF-tracked baseline that adapts to non-stationary reward
    distributions during training.
    """

    def __init__(
        self,
        process_noise: float = 0.005,
        measurement_noise: float = 0.5,
    ):
        self.kf = ScalarKalmanFilter(
            initial_state=0.0,
            initial_variance=1.0,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
        )

    def update_and_normalize(self, reward: float) -> float:
        """Update the baseline with a new reward and return the advantage.

        advantage = reward - kf_estimated_baseline
        """
        baseline = self.kf.update(reward)
        advantage = reward - baseline
        uncertainty = self.kf.get_uncertainty()
        if uncertainty > 1e-6:
            advantage = advantage / (uncertainty + 1e-6)
        return advantage

    def get_baseline(self) -> float:
        return self.kf.get_state()

    def get_uncertainty(self) -> float:
        return self.kf.get_uncertainty()


class TargetQSoftTracker:
    """Kalman-filter-based soft tracking for target Q-network values.

    Inspired by LKTD (arXiv:2403.13178): instead of hard-copying target
    network weights every N steps, this tracks the Q-value estimates with
    a KF that provides uncertainty-aware soft updates.
    """

    def __init__(
        self,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
    ):
        self.kf = ScalarKalmanFilter(
            initial_state=0.0,
            initial_variance=1.0,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
        )

    def update(self, q_value: float) -> float:
        return self.kf.update(q_value)

    def get_tracked_value(self) -> float:
        return self.kf.get_state()

    def get_uncertainty(self) -> float:
        return self.kf.get_uncertainty()


class PositionKF:
    """2D Kalman filter for robot position denoising.

    Inspired by Sim-to-Real DRL UAV (arXiv:2303.07243): filters noisy
    sensor-derived position estimates before feeding them to the RL agent.
    """

    def __init__(
        self,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
    ):
        self.kf_x = ScalarKalmanFilter(0.0, 1.0, process_noise, measurement_noise)
        self.kf_y = ScalarKalmanFilter(0.0, 1.0, process_noise, measurement_noise)

    def update(self, position: tuple[float, float]) -> tuple[float, float]:
        x = self.kf_x.update(position[0])
        y = self.kf_y.update(position[1])
        return (x, y)

    def get_position(self) -> tuple[float, float]:
        return (self.kf_x.get_state(), self.kf_y.get_state())

    def get_uncertainty(self) -> tuple[float, float]:
        return (self.kf_x.get_uncertainty(), self.kf_y.get_uncertainty())

    def reset(self, position: tuple[float, float]) -> None:
        self.kf_x.reset(position[0], 1.0)
        self.kf_y.reset(position[1], 1.0)
