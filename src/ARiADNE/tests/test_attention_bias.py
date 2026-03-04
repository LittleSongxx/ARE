import unittest

import torch

from ARiADNE.model import MultiHeadAttention
from ARiADNE.parameter import RuntimeConfig
from ARiADNE.utils import build_wavelet_attn_bias


class AttentionBiasTests(unittest.TestCase):
    def test_attn_bias_shape_and_masking(self):
        config = RuntimeConfig(
            use_wavelet_attn_bias=True,
            wavelet_attn_bias_type="sim_exp",
            wavelet_attn_bias_apply_on_masked_edges_only=True,
        )
        wavelet_feat = torch.tensor(
            [[[0.1, 0.2], [0.3, 0.1], [0.9, 0.7], [0.95, 0.6]]],
            dtype=torch.float32,
        )
        edge_mask = torch.tensor(
            [[[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]],
            dtype=torch.int64,
        )
        attn_bias = build_wavelet_attn_bias(wavelet_feat, runtime_config=config, edge_mask=edge_mask)
        self.assertEqual(tuple(attn_bias.shape), (1, 1, 4, 4))
        masked_entries = attn_bias[:, 0].masked_select(edge_mask == 1)
        self.assertTrue(torch.all(masked_entries == 0))

    def test_multi_head_attention_accepts_explicit_bias(self):
        torch.manual_seed(0)
        attention = MultiHeadAttention(embedding_dim=8, n_heads=2)
        qkv = torch.randn(1, 4, 8)
        attn_mask = torch.tensor([[[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]], dtype=torch.int64)
        wavelet_feat = torch.tensor([[[0.0], [0.2], [0.8], [1.0]]], dtype=torch.float32)
        config = RuntimeConfig(use_wavelet_attn_bias=True, wavelet_attn_bias_type="neg_l2")
        attn_bias = build_wavelet_attn_bias(wavelet_feat, runtime_config=config, edge_mask=attn_mask)

        _, weights = attention(qkv, attn_mask=attn_mask, attn_bias=attn_bias)
        masked_weights = weights.masked_select(attn_mask.unsqueeze(0).bool())
        self.assertTrue(torch.all(masked_weights < 1e-9))

    def test_new_bias_modes_and_clamp(self):
        wavelet_feat = torch.tensor(
            [[[0.0, 0.5], [0.5, 0.5], [1.0, 0.0]]],
            dtype=torch.float32,
        )
        for bias_type in ("diff", "product", "rbf"):
            config = RuntimeConfig(
                use_wavelet_attn_bias=True,
                wavelet_attn_bias_type=bias_type,
                wavelet_attn_bias_beta=2.0,
                wavelet_attn_bias_sigma=0.5,
                wavelet_attn_bias_clamp=0.25,
                wavelet_attn_bias_apply_on_masked_edges_only=False,
            )
            attn_bias = build_wavelet_attn_bias(wavelet_feat, runtime_config=config)
            self.assertEqual(tuple(attn_bias.shape), (1, 1, 3, 3))
            self.assertTrue(torch.all(attn_bias <= 0.25 + 1e-6))
            self.assertTrue(torch.all(attn_bias >= -0.25 - 1e-6))

    def test_zero_bias_scale_matches_no_bias(self):
        torch.manual_seed(0)
        attention = MultiHeadAttention(embedding_dim=8, n_heads=2)
        qkv = torch.randn(1, 4, 8)
        attn_mask = torch.zeros((1, 4, 4), dtype=torch.int64)
        wavelet_feat = torch.tensor([[[0.0], [0.2], [0.8], [1.0]]], dtype=torch.float32)
        zero_bias_config = RuntimeConfig(
            use_wavelet_attn_bias=True,
            wavelet_attn_bias_type="product",
            wavelet_attn_bias_beta=0.0,
        )
        attn_bias = build_wavelet_attn_bias(wavelet_feat, runtime_config=zero_bias_config, edge_mask=attn_mask)

        out_no_bias, weights_no_bias = attention(qkv, attn_mask=attn_mask, attn_bias=None)
        out_zero_bias, weights_zero_bias = attention(qkv, attn_mask=attn_mask, attn_bias=attn_bias)
        self.assertTrue(torch.allclose(out_no_bias, out_zero_bias, atol=1e-7))
        self.assertTrue(torch.allclose(weights_no_bias, weights_zero_bias, atol=1e-7))


if __name__ == "__main__":
    unittest.main()
