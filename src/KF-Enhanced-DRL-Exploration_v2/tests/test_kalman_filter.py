"""Tests for the kalman_filter module."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kalman_filter import (
    PositionKF,
    RewardBaselineKF,
    ScalarKalmanFilter,
    TargetQSoftTracker,
)


class TestScalarKalmanFilter:
    def test_initial_state(self):
        kf = ScalarKalmanFilter(initial_state=5.0)
        assert kf.get_state() == pytest.approx(5.0)

    def test_converges_to_constant(self):
        kf = ScalarKalmanFilter(0.0, 1.0, 0.01, 0.1)
        for _ in range(100):
            kf.update(10.0)
        assert kf.get_state() == pytest.approx(10.0, abs=0.5)

    def test_tracks_linear_signal(self):
        kf = ScalarKalmanFilter(0.0, 1.0, 1.0, 0.1)
        for t in range(50):
            kf.update(float(t))
        assert kf.get_state() == pytest.approx(49.0, abs=3.0)

    def test_uncertainty_decreases(self):
        kf = ScalarKalmanFilter(0.0, 10.0, 0.01, 1.0)
        initial_unc = kf.get_uncertainty()
        for _ in range(20):
            kf.update(5.0)
        assert kf.get_uncertainty() < initial_unc

    def test_reset(self):
        kf = ScalarKalmanFilter(0.0, 1.0, 0.01, 1.0)
        for _ in range(10):
            kf.update(5.0)
        kf.reset(0.0, 1.0)
        assert kf.get_state() == pytest.approx(0.0)
        assert kf.n_updates == 0

    def test_predict_does_not_change_state(self):
        kf = ScalarKalmanFilter(5.0, 1.0, 0.1, 1.0)
        state_before = kf.get_state()
        pred_state, pred_var = kf.predict()
        assert kf.get_state() == state_before
        assert pred_state == state_before


class TestRewardBaselineKF:
    def test_initial_baseline(self):
        kf = RewardBaselineKF()
        assert kf.get_baseline() == pytest.approx(0.0)

    def test_update_changes_baseline(self):
        kf = RewardBaselineKF()
        kf.update(10.0)
        assert kf.get_baseline() != 0.0

    def test_baseline_tracks_reward(self):
        kf = RewardBaselineKF()
        for _ in range(200):
            kf.update(5.0)
        assert kf.get_baseline() == pytest.approx(5.0, abs=1.0)

    def test_normalization_factor_floor(self):
        kf = RewardBaselineKF()
        assert kf.get_normalization_factor() >= 1.0
        kf.update(0.1)
        assert kf.get_normalization_factor() >= 1.0

    def test_reward_std_positive(self):
        kf = RewardBaselineKF()
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            kf.update(v)
        assert kf.get_reward_std() >= 0.0


class TestTargetQSoftTracker:
    def test_tracks_constant(self):
        tracker = TargetQSoftTracker()
        for _ in range(100):
            tracker.update(3.0)
        assert tracker.get_tracked_value() == pytest.approx(3.0, abs=0.5)

    def test_uncertainty_decreases(self):
        tracker = TargetQSoftTracker()
        initial_unc = tracker.get_uncertainty()
        for _ in range(50):
            tracker.update(1.0)
        assert tracker.get_uncertainty() < initial_unc


class TestPositionKF:
    def test_tracks_constant_position(self):
        kf = PositionKF()
        for _ in range(100):
            kf.update((10.0, 20.0))
        pos = kf.get_position()
        assert pos[0] == pytest.approx(10.0, abs=0.5)
        assert pos[1] == pytest.approx(20.0, abs=0.5)

    def test_filters_noisy_position(self):
        kf = PositionKF(process_noise=0.01, measurement_noise=0.5)
        rng = np.random.RandomState(42)
        errors = []
        for _ in range(200):
            noisy = (5.0 + rng.normal(0, 0.3), 5.0 + rng.normal(0, 0.3))
            kf.update(noisy)
            pos = kf.get_position()
            errors.append(np.sqrt((pos[0] - 5.0) ** 2 + (pos[1] - 5.0) ** 2))
        assert np.mean(errors[-20:]) < 0.5

    def test_reset(self):
        kf = PositionKF()
        for _ in range(10):
            kf.update((10.0, 20.0))
        kf.reset((0.0, 0.0))
        pos = kf.get_position()
        assert pos[0] == pytest.approx(0.0)
        assert pos[1] == pytest.approx(0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
