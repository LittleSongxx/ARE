import unittest

import numpy as np

from ARiADNE.parameter import RuntimeConfig
from ARiADNE.utils import compute_wavelet_maps, wavelet_feature_at_coords


class WaveletMapsTests(unittest.TestCase):
    def setUp(self):
        self.grid = np.array(
            [
                [255, 255, 127, 1, 1, 1],
                [255, 127, 127, 1, 1, 1],
                [255, 255, 255, 1, 127, 127],
                [1, 1, 127, 127, 255, 255],
                [1, 1, 1, 255, 255, 255],
                [127, 127, 1, 255, 255, 255],
            ],
            dtype=np.int16,
        )
        self.map_info = type("MapInfo", (), {"map_origin_x": 0.0, "map_origin_y": 0.0, "cell_size": 1.0})()

    def test_scales_orient_mode_shapes_and_range(self):
        config = RuntimeConfig(
            use_wavelet_feature=True,
            wavelet_feature_mode="scales_orient",
            wavelet_scales_auto=False,
            wavelet_scales=(1, 2),
            wavelet_norm_method="percentile",
        )
        wavelet_maps = compute_wavelet_maps(self.grid, config)
        self.assertEqual(wavelet_maps.scalar_map.shape, self.grid.shape)
        self.assertEqual(len(wavelet_maps.scale_maps), 2)
        self.assertEqual(len(wavelet_maps.orient_maps), 2)
        self.assertGreaterEqual(float(wavelet_maps.scalar_map.min()), 0.0)
        self.assertLessEqual(float(wavelet_maps.scalar_map.max()), 1.0)

        feature = wavelet_feature_at_coords(np.array([2.0, 2.0]), self.map_info, wavelet_maps, config)
        self.assertEqual(feature.shape, (6,))
        self.assertTrue(np.all(feature >= 0.0))
        self.assertTrue(np.all(feature <= 1.0))

    def test_local_pool_changes_feature_value(self):
        base_config = RuntimeConfig(
            use_wavelet_feature=True,
            wavelet_feature_mode="scales",
            wavelet_scales_auto=False,
            wavelet_scales=(1, 2),
            wavelet_local_pool="none",
        )
        pooled_config = base_config.with_overrides(
            wavelet_local_pool="mean",
            wavelet_local_pool_radius_cells=1,
        )
        base_maps = compute_wavelet_maps(self.grid, base_config)
        pooled_maps = compute_wavelet_maps(self.grid, pooled_config)
        coords = np.array([2.0, 2.0])
        base_feature = wavelet_feature_at_coords(coords, self.map_info, base_maps, base_config)
        pooled_feature = wavelet_feature_at_coords(coords, self.map_info, pooled_maps, pooled_config)
        self.assertEqual(base_feature.shape, pooled_feature.shape)
        self.assertFalse(np.allclose(base_feature, pooled_feature))


if __name__ == "__main__":
    unittest.main()
