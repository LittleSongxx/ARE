from __future__ import annotations

import unittest

import numpy as np

from hpbg_rl.map_prediction import HeuristicMapPredictor
from hpbg_rl.parameter import FREE, UNKNOWN
from hpbg_rl.utils import MapInfo


class MapPredictionTests(unittest.TestCase):
    def test_heuristic_prediction_shape_range_and_finiteness(self):
        grid = np.full((21, 21), UNKNOWN, dtype=int)
        grid[8:13, 8:13] = FREE
        map_info = MapInfo(grid, -4.0, -4.0, 0.4)
        node_coords = np.asarray([[0.0, 0.0], [2.0, 0.0], [20.0, 20.0]], dtype=float)
        utility = np.asarray([3.0, 0.0, -1.0], dtype=float)
        guidepost = np.asarray([0.0, 1.0, 0.0], dtype=float)
        frontier = {(0.4, 0.0), (0.8, 0.0)}

        prediction = HeuristicMapPredictor(sensor_range=2.0, risk_weight=0.25).predict(
            node_coords,
            utility,
            guidepost,
            map_info,
            frontier=frontier,
            utility_normalizer=10.0,
        )

        for values in (prediction.predicted_utility, prediction.uncertainty, prediction.risk_aware_utility):
            self.assertEqual(values.shape, (3,))
            self.assertTrue(np.isfinite(values).all())
        self.assertTrue(np.all(prediction.predicted_utility >= 0.0))
        self.assertTrue(np.all(prediction.predicted_utility <= 1.0))
        self.assertTrue(np.all(prediction.uncertainty >= 0.0))
        self.assertTrue(np.all(prediction.uncertainty <= 1.0))
        self.assertTrue(np.all(prediction.risk_aware_utility >= -1.0))
        self.assertTrue(np.all(prediction.risk_aware_utility <= 1.0))


if __name__ == "__main__":
    unittest.main()