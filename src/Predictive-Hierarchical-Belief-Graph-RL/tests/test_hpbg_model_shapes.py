from __future__ import annotations

import unittest

import torch

from hpbg_rl.model import PolicyNet, QNet
from hpbg_rl.parameter import CRITIC_NODE_INPUT_DIM, EMBEDDING_DIM, K_SIZE, NODE_INPUT_DIM, HPBG_CRITIC_PRIVILEGED_DIM


def build_actor_observation(batch_size=2, n_nodes=10):
    node_inputs = torch.randn(batch_size, n_nodes, NODE_INPUT_DIM)
    node_inputs[..., -4:] = torch.sigmoid(node_inputs[..., -4:])
    node_padding_mask = torch.zeros(batch_size, 1, n_nodes, dtype=torch.bool)
    node_padding_mask[:, :, -1] = True
    edge_mask = torch.zeros(batch_size, n_nodes, n_nodes, dtype=torch.bool)
    edge_mask[:, -1, :] = True
    edge_mask[:, :, -1] = True
    current_index = torch.zeros(batch_size, 1, 1, dtype=torch.long)
    current_edge = torch.zeros(batch_size, K_SIZE, 1, dtype=torch.long)
    for batch_index in range(batch_size):
        current_edge[batch_index, :, 0] = torch.arange(K_SIZE) % (n_nodes - 1)
    edge_padding_mask = torch.zeros(batch_size, 1, K_SIZE, dtype=torch.bool)
    edge_padding_mask[:, :, -3:] = True
    return [node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask]


class HPBGModelShapeTests(unittest.TestCase):
    def test_actor_critic_forward_shapes_and_hidden_contract(self):
        actor_obs = build_actor_observation()
        critic_obs = actor_obs.copy()
        critic_obs[0] = torch.randn(actor_obs[0].size(0), actor_obs[0].size(1), CRITIC_NODE_INPUT_DIM)
        critic_obs[0][..., -HPBG_CRITIC_PRIVILEGED_DIM:] = torch.sigmoid(
            critic_obs[0][..., -HPBG_CRITIC_PRIVILEGED_DIM:]
        )

        policy = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM, use_hierarchical_context=True)
        critic = QNet(CRITIC_NODE_INPUT_DIM, EMBEDDING_DIM)

        logp = policy(*actor_obs)
        logp_with_hidden, actor_hidden = policy(*actor_obs, return_hidden=True)
        q_values = critic(*critic_obs)
        q_values_with_hidden, critic_hidden = critic(*critic_obs, return_hidden=True)
        belief_targets = policy.predict_belief_targets(actor_hidden)

        self.assertEqual(logp.shape, (actor_obs[0].size(0), K_SIZE))
        self.assertEqual(logp_with_hidden.shape, logp.shape)
        self.assertEqual(actor_hidden.shape, (actor_obs[0].size(0), actor_obs[0].size(1), EMBEDDING_DIM))
        self.assertEqual(q_values.shape, (actor_obs[0].size(0), K_SIZE, 1))
        self.assertEqual(q_values_with_hidden.shape, q_values.shape)
        self.assertEqual(critic_hidden.shape, actor_hidden.shape)
        self.assertEqual(belief_targets.shape, (actor_obs[0].size(0), actor_obs[0].size(1), HPBG_CRITIC_PRIVILEGED_DIM))
        self.assertTrue(torch.isfinite(logp).all())
        self.assertTrue(torch.isfinite(q_values).all())
        self.assertTrue(torch.isfinite(belief_targets).all())

    def test_hierarchical_context_is_part_of_actor_decision_path(self):
        actor_obs = build_actor_observation(batch_size=1, n_nodes=8)
        policy = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM, use_hierarchical_context=True)
        logp, hidden = policy(*actor_obs, return_hidden=True)
        loss = -(logp.mean() + hidden.mean())
        loss.backward()

        context_params = list(policy.hierarchical_context.parameters())
        self.assertTrue(any(param.grad is not None for param in context_params))
    def test_coarse_adjacency_changes_hierarchical_context(self):
        policy = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM, use_hierarchical_context=True)
        hidden = torch.randn(1, 4, EMBEDDING_DIM)
        node_inputs = torch.zeros(1, 4, NODE_INPUT_DIM)
        node_inputs[0, :, :2] = torch.tensor(
            [
                [0.00, 0.00],
                [0.03, 0.00],
                [0.20, 0.00],
                [0.23, 0.00],
            ]
        )
        node_inputs[0, :, -1] = torch.tensor([0.2, 0.4, 0.6, 0.8])
        node_padding_mask = torch.zeros(1, 1, 4, dtype=torch.bool)

        disconnected_edge_mask = torch.ones(1, 4, 4, dtype=torch.bool)
        connected_edge_mask = torch.ones(1, 4, 4, dtype=torch.bool)
        for edge_mask in (disconnected_edge_mask, connected_edge_mask):
            edge_mask[:, torch.arange(4), torch.arange(4)] = False
            edge_mask[:, 0, 1] = False
            edge_mask[:, 1, 0] = False
            edge_mask[:, 2, 3] = False
            edge_mask[:, 3, 2] = False
        connected_edge_mask[:, 1, 2] = False
        connected_edge_mask[:, 2, 1] = False

        disconnected = policy.hierarchical_context(hidden, node_inputs, node_padding_mask, disconnected_edge_mask)
        connected = policy.hierarchical_context(hidden, node_inputs, node_padding_mask, connected_edge_mask)

        self.assertFalse(torch.allclose(disconnected, connected))


if __name__ == "__main__":
    unittest.main()