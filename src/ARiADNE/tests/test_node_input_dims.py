import unittest

from ARiADNE.parameter import (
    BASE_NODE_INPUT_DIM,
    CRITIC_EXTRA_INPUT_DIM,
    RuntimeConfig,
    get_critic_node_input_dim,
    get_node_input_dim,
    resolve_wavelet_scales,
)


class NodeInputDimTests(unittest.TestCase):
    def test_wavelet_disabled_reverts_to_original_dims(self):
        config = RuntimeConfig(use_wavelet_feature=False)
        self.assertEqual(get_node_input_dim(config), BASE_NODE_INPUT_DIM)
        self.assertEqual(get_critic_node_input_dim(config), BASE_NODE_INPUT_DIM + CRITIC_EXTRA_INPUT_DIM)

    def test_scales_orient_adds_three_channels_per_scale(self):
        config = RuntimeConfig(
            use_wavelet_feature=True,
            wavelet_feature_mode="scales_orient",
            wavelet_scales_auto=False,
            wavelet_scales=(1, 2, 4),
        )
        self.assertEqual(get_node_input_dim(config), BASE_NODE_INPUT_DIM + len(resolve_wavelet_scales(config)) * 3)
        self.assertEqual(
            get_critic_node_input_dim(config),
            BASE_NODE_INPUT_DIM + CRITIC_EXTRA_INPUT_DIM + len(resolve_wavelet_scales(config)) * 3,
        )


if __name__ == "__main__":
    unittest.main()
