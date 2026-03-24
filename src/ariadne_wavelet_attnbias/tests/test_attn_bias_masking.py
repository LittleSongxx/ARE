import unittest

import torch

from ariadne_wavelet_attnbias import parameter as attn_parameter
from ariadne_wavelet_attnbias.model import MultiHeadAttention
from ariadne_wavelet_attnbias.parameter import configure_attention_bias
from ariadne_wavelet_attnbias.utils import make_attn_bias_from_node_inputs


class AttentionBiasMaskingTests(unittest.TestCase):
    def setUp(self):
        self.original_config = (
            attn_parameter.USE_ATTENTION_BIAS,
            attn_parameter.ATTN_BIAS_BETA,
            attn_parameter.ATTN_BIAS_MODE,
            attn_parameter.ATTN_BIAS_APPLY_ENCODER,
            attn_parameter.ATTN_BIAS_APPLY_DECODER,
        )
        configure_attention_bias(
            use_attention_bias=True,
            attn_bias_beta=0.5,
            attn_bias_mode="hybrid",
            attn_bias_apply_encoder=True,
            attn_bias_apply_decoder=False,
        )

    def tearDown(self):
        configure_attention_bias(
            use_attention_bias=self.original_config[0],
            attn_bias_beta=self.original_config[1],
            attn_bias_mode=self.original_config[2],
            attn_bias_apply_encoder=self.original_config[3],
            attn_bias_apply_decoder=self.original_config[4],
        )

    def test_masked_edges_stay_zero_with_attention_bias(self):
        torch.manual_seed(0)
        attention = MultiHeadAttention(embedding_dim=8, n_heads=2)
        qkv = torch.randn(1, 4, 8)
        node_inputs = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.2, 0.0, 0.0],
                    [0.1, 0.0, 0.3, 0.0, 0.25],
                    [0.2, 0.0, 0.4, 0.0, 0.75],
                    [0.3, 0.0, 0.5, 0.0, 1.0],
                ]
            ],
            dtype=torch.float32,
        )
        attn_mask = torch.tensor(
            [
                [
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 1],
                ]
            ],
            dtype=torch.int64,
        )
        attn_bias = make_attn_bias_from_node_inputs(node_inputs)

        _, weights = attention(qkv, attn_mask=attn_mask, attn_bias=attn_bias)

        masked_weights = weights.masked_select(attn_mask.unsqueeze(0).bool())
        self.assertTrue(torch.all(masked_weights < 1e-9))

        allowed_weights = weights.masked_fill(attn_mask.unsqueeze(0).bool(), 0.0)
        row_sums = allowed_weights.sum(dim=-1)
        torch.testing.assert_close(row_sums, torch.ones_like(row_sums), atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
