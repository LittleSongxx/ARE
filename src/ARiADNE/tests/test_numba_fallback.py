import unittest

import numpy as np

from ARiADNE.utils import MapInfo, check_collision


class NumbaFallbackTests(unittest.TestCase):
    def test_check_collision_works_without_numba_dependency(self):
        grid = np.array(
            [
                [255, 255, 255],
                [255, 1, 255],
                [255, 255, 255],
            ],
            dtype=np.int16,
        )
        map_info = MapInfo(grid, 0.0, 0.0, 1.0)
        self.assertFalse(check_collision(np.array([0.0, 0.0]), np.array([0.0, 2.0]), map_info))
        self.assertTrue(check_collision(np.array([0.0, 0.0]), np.array([2.0, 2.0]), map_info))


if __name__ == "__main__":
    unittest.main()
