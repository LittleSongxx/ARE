import unittest

from ARiADNE.parameter import (
    RuntimeConfig,
    checkpoint_model_config,
    ensure_checkpoint_compatible,
    runtime_config_from_checkpoint,
)


class CheckpointConfigTests(unittest.TestCase):
    def test_runtime_config_roundtrip_from_checkpoint_snapshot(self):
        config = RuntimeConfig(
            use_wavelet_feature=True,
            wavelet_feature_mode="scales_orient",
            wavelet_scales_auto=False,
            wavelet_scales=(1, 2),
            use_wavelet_attn_bias=True,
        )
        payload = {"config_snapshot": checkpoint_model_config(config)}
        restored = runtime_config_from_checkpoint(payload, RuntimeConfig(run_session="x"))
        self.assertTrue(restored.use_wavelet_feature)
        self.assertEqual(restored.wavelet_feature_mode, "scales_orient")
        self.assertEqual(restored.wavelet_scales, (1, 2))
        self.assertTrue(restored.use_wavelet_attn_bias)

    def test_incompatible_checkpoint_raises(self):
        config = RuntimeConfig(use_wavelet_feature=True, wavelet_feature_mode="scales")
        payload = {"config_snapshot": checkpoint_model_config(config)}
        with self.assertRaises(ValueError):
            ensure_checkpoint_compatible(payload, RuntimeConfig(use_wavelet_feature=False))


if __name__ == "__main__":
    unittest.main()
