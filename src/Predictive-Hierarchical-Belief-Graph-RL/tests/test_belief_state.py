from __future__ import annotations

import unittest

import numpy as np

from hpbg_rl.belief_state import BeliefStateTracker, PredictionResult, zero_belief_features


class BeliefStateTests(unittest.TestCase):
    def test_tracker_outputs_stable_four_column_features(self):
        tracker = BeliefStateTracker(alpha=0.5, risk_weight=0.25)
        coords = np.asarray([[0.0, 0.0], [4.0, 0.0], [8.0, 0.0]], dtype=float)
        prediction = PredictionResult(
            predicted_utility=np.asarray([0.2, 0.8, 0.4], dtype=np.float32),
            uncertainty=np.asarray([0.1, 0.5, 0.0], dtype=np.float32),
            risk_aware_utility=np.asarray([0.1, 0.4, 0.4], dtype=np.float32),
        )
        cluster_prior = np.asarray([0.0, 1.0, 0.5], dtype=np.float32)

        first = tracker.update(coords, prediction, cluster_prior).as_feature_matrix()
        second = tracker.update(coords, prediction, cluster_prior).as_feature_matrix()

        self.assertEqual(first.shape, (3, 4))
        self.assertEqual(second.shape, (3, 4))
        self.assertTrue(np.isfinite(second).all())
        self.assertTrue(np.all(second[:, 0] >= 0.0))
        self.assertTrue(np.all(second[:, 0] <= 1.0))
        self.assertTrue(np.all(second[:, 1] >= 0.0))
        self.assertTrue(np.all(second[:, 1] <= 1.0))
        self.assertTrue(np.all(second[:, 3] >= 0.0))
        self.assertTrue(np.all(second[:, 3] <= 1.0))

    def test_zero_features_keep_fixed_actor_extension_width(self):
        features = zero_belief_features(5)
        self.assertEqual(features.shape, (5, 4))
        self.assertTrue(np.all(features == 0.0))


if __name__ == "__main__":
    unittest.main()