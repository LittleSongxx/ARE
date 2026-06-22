from __future__ import annotations

import unittest

import torch

from hpbg_rl.model import PolicyNet, QNet
from hpbg_rl.parameter import CRITIC_NODE_INPUT_DIM, EMBEDDING_DIM, K_SIZE, NODE_INPUT_DIM


def build_observation(batch_size=2, n_nodes=12):
    node_inputs = torch.randn(batch_size, n_nodes, NODE_INPUT_DIM)
    node_padding_mask = torch.zeros(batch_size, 1, n_nodes, dtype=torch.bool)
    edge_mask = torch.zeros(batch_size, n_nodes, n_nodes, dtype=torch.bool)
    current_index = torch.zeros(batch_size, 1, 1, dtype=torch.long)
    current_edge = torch.zeros(batch_size, K_SIZE, 1, dtype=torch.long)
    for b in range(batch_size):
        current_edge[b, :, 0] = torch.arange(K_SIZE) % n_nodes
    edge_padding_mask = torch.zeros(batch_size, 1, K_SIZE, dtype=torch.bool)
    edge_padding_mask[:, :, -2:] = True
    return [node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask]


class ModelWaveletEncoderTests(unittest.TestCase):
    def test_baseline_shapes_with_return_hidden(self):
        obs = build_observation()
        policy = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM)
        critic = QNet(CRITIC_NODE_INPUT_DIM, EMBEDDING_DIM)

        logp, hidden = policy(*obs, return_hidden=True)
        critic_inputs = obs.copy()
        critic_inputs[0] = torch.randn(obs[0].size(0), obs[0].size(1), CRITIC_NODE_INPUT_DIM)
        q_values, critic_hidden = critic(*critic_inputs, return_hidden=True)

        self.assertEqual(logp.shape, (obs[0].size(0), K_SIZE))
        self.assertEqual(hidden.shape, (obs[0].size(0), obs[0].size(1), EMBEDDING_DIM))
        self.assertEqual(q_values.shape, (obs[0].size(0), K_SIZE, 1))
        self.assertEqual(critic_hidden.shape, hidden.shape)

    def test_lf_attention_hf_residual_forward_backward(self):
        obs = build_observation()
        policy = PolicyNet(
            NODE_INPUT_DIM,
            EMBEDDING_DIM,
            use_lf_attention_hf_residual=True,
            wavelet_scales=(1, 2, 4),
            wavelet_fuse_dim=128,
            wavelet_lf_qk=True,
        )
        logp, hidden = policy(*obs, return_hidden=True)
        loss = -(logp.mean() + hidden.mean())
        loss.backward()

        self.assertEqual(logp.shape, (obs[0].size(0), K_SIZE))
        self.assertEqual(hidden.shape[-1], EMBEDDING_DIM)
        self.assertTrue(torch.isfinite(logp).all())
        self.assertTrue(any(param.grad is not None for param in policy.parameters()))


if __name__ == "__main__":
    unittest.main()
