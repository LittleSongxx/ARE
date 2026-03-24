import unittest

import numpy as np

from ARiADNE_Wavelet.sensor import collision_check


class SensorNumpyCompatTests(unittest.TestCase):
    def test_collision_check_accepts_float_coords_without_itemset(self):
        ground_truth = np.full((16, 16), 255, dtype=np.int16)
        ground_truth[8, 8] = 1
        robot_belief = np.full((16, 16), 127, dtype=np.int16)

        updated = collision_check(1.2, 1.8, 10.4, 10.6, ground_truth, robot_belief)
        self.assertEqual(updated.shape, robot_belief.shape)
        self.assertTrue(np.any(updated != 127))


if __name__ == "__main__":
    unittest.main()
