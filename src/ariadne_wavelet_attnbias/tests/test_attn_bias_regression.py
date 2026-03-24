import unittest

import torch

from ariadne_wavelet_attnbias import parameter as attn_parameter
from ariadne_wavelet_attnbias.model_stage1_baseline import PolicyNet as Stage1PolicyNet
from ariadne_wavelet_attnbias.model_stage1_baseline import QNet as Stage1QNet
from ariadne_wavelet_attnbias.model import PolicyNet as Stage2PolicyNet
from ariadne_wavelet_attnbias.model import QNet as Stage2QNet
from ariadne_wavelet_attnbias.parameter import EMBEDDING_DIM, NODE_INPUT_DIM
from ariadne_wavelet_attnbias.parameter import configure_attention_bias


class AttentionBiasRegressionTests(unittest.TestCase):
    def setUp(self):
        self.original_config = (
            attn_parameter.USE_ATTENTION_BIAS,
            attn_parameter.ATTN_BIAS_BETA,
            attn_parameter.ATTN_BIAS_MODE,
            attn_parameter.ATTN_BIAS_APPLY_ENCODER,
            attn_parameter.ATTN_BIAS_APPLY_DECODER,
        )
        configure_attention_bias(use_attention_bias=False)

    def tearDown(self):
        configure_attention_bias(
            use_attention_bias=self.original_config[0],
            attn_bias_beta=self.original_config[1],
            attn_bias_mode=self.original_config[2],
            attn_bias_apply_encoder=self.original_config[3],
            attn_bias_apply_decoder=self.original_config[4],
        )

    @staticmethod
    def _policy_inputs():
        node_inputs = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.1, 0.0, 0.2],
                    [0.1, -0.1, 0.2, 1.0, 0.4],
                    [0.2, 0.1, 0.3, 0.0, 0.8],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ],
            dtype=torch.float32,
        )
        node_padding_mask = torch.tensor([[[0, 0, 0, 1]]], dtype=torch.bool)
        edge_mask = torch.tensor(
            [
                [
                    [0, 0, 0, 1],
                    [0, 0, 0, 1],
                    [0, 0, 0, 1],
                    [1, 1, 1, 1],
                ]
            ],
            dtype=torch.bool,
        )
        current_index = torch.tensor([[[1]]], dtype=torch.long)
        current_edge = torch.tensor([[[0], [1], [2]]], dtype=torch.long)
        edge_padding_mask = torch.tensor([[[0, 1, 0]]], dtype=torch.bool)
        return node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask

    @staticmethod
    def _critic_inputs():
        node_inputs = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.1, 0.0, 0.0, 0.2],
                    [0.1, -0.1, 0.2, 1.0, 1.0, 0.4],
                    [0.2, 0.1, 0.3, 0.0, 0.0, 0.8],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ],
            dtype=torch.float32,
        )
        node_padding_mask = torch.tensor([[[0, 0, 0, 1]]], dtype=torch.bool)
        edge_mask = torch.tensor(
            [
                [
                    [0, 0, 0, 1],
                    [0, 0, 0, 1],
                    [0, 0, 0, 1],
                    [1, 1, 1, 1],
                ]
            ],
            dtype=torch.bool,
        )
        current_index = torch.tensor([[[1]]], dtype=torch.long)
        current_edge = torch.tensor([[[0], [1], [2]]], dtype=torch.long)
        edge_padding_mask = torch.tensor([[[0, 1, 0]]], dtype=torch.bool)
        return node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask

    def test_policy_net_matches_stage1_when_bias_disabled(self):
        torch.manual_seed(0)
        stage1_model = Stage1PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM).eval()
        stage2_model = Stage2PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM).eval()
        stage2_model.load_state_dict(stage1_model.state_dict(), strict=True)

        inputs = self._policy_inputs()
        with torch.no_grad():
            stage1_encoded = stage1_model.encode_graph(inputs[0], inputs[1], inputs[2])
            stage2_encoded = stage2_model.encode_graph(inputs[0], inputs[1], inputs[2])
            stage1_output = stage1_model(*inputs)
            stage2_output = stage2_model(*inputs)

        torch.testing.assert_close(stage2_encoded, stage1_encoded, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(stage2_output, stage1_output, atol=1e-6, rtol=1e-6)

    def test_q_net_matches_stage1_when_bias_disabled(self):
        torch.manual_seed(0)
        stage1_model = Stage1QNet(NODE_INPUT_DIM + 1, EMBEDDING_DIM).eval()
        stage2_model = Stage2QNet(NODE_INPUT_DIM + 1, EMBEDDING_DIM).eval()
        stage2_model.load_state_dict(stage1_model.state_dict(), strict=True)

        inputs = self._critic_inputs()
        with torch.no_grad():
            stage1_encoded = stage1_model.encode_graph(inputs[0], inputs[1], inputs[2])
            stage2_encoded = stage2_model.encode_graph(inputs[0], inputs[1], inputs[2])
            stage1_output = stage1_model(*inputs)
            stage2_output = stage2_model(*inputs)

        torch.testing.assert_close(stage2_encoded, stage1_encoded, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(stage2_output, stage1_output, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
