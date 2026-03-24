import unittest

import torch

from ARiADNE_Wavelet.model import PolicyNet, QNet, compute_wavelet_utility_aux_loss
from ARiADNE_Wavelet.parameter import EMBEDDING_DIM, HISTORY_INPUT_DIM, HISTORY_LEN, K_SIZE, NODE_INPUT_DIM


class ModelWaveletFeatureTests(unittest.TestCase):
    def _dummy_obs(self, batch_size=2, n_nodes=12):
        node_inputs = torch.randn(batch_size, n_nodes, NODE_INPUT_DIM)
        node_padding_mask = torch.zeros(batch_size, 1, n_nodes, dtype=torch.bool)
        edge_mask = torch.zeros(batch_size, n_nodes, n_nodes, dtype=torch.bool)
        current_index = torch.zeros(batch_size, 1, 1, dtype=torch.long)
        current_edge = torch.zeros(batch_size, K_SIZE, 1, dtype=torch.long)
        edge_padding_mask = torch.ones(batch_size, 1, K_SIZE, dtype=torch.bool)
        edge_padding_mask[:, :, : min(8, K_SIZE)] = 0
        history_inputs = torch.randn(batch_size, HISTORY_LEN, HISTORY_INPUT_DIM)
        return [
            node_inputs,
            node_padding_mask,
            edge_mask,
            current_index,
            current_edge,
            edge_padding_mask,
            history_inputs,
        ]

    def test_policy_forward_without_history(self):
        obs = self._dummy_obs()
        model = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM, enable_wavelet_history=False)
        out = model(*obs)
        self.assertEqual(tuple(out.shape), (obs[0].size(0), K_SIZE))

    def test_policy_forward_with_history_split_mode(self):
        obs = self._dummy_obs()
        model = PolicyNet(
            NODE_INPUT_DIM,
            EMBEDDING_DIM,
            enable_wavelet_history=True,
            history_input_dim=HISTORY_INPUT_DIM,
            history_embed_dim=64,
            history_wavelet_levels=2,
            history_encoder_mode="wavelet_split",
        )
        out = model(*obs)
        self.assertEqual(tuple(out.shape), (obs[0].size(0), K_SIZE))

    def test_qnet_aux_shape_is_candidate_level(self):
        obs = self._dummy_obs(batch_size=2, n_nodes=16)
        qnet = QNet(
            NODE_INPUT_DIM,
            EMBEDDING_DIM,
            enable_wavelet_history=True,
            history_input_dim=HISTORY_INPUT_DIM,
            history_embed_dim=64,
            history_wavelet_levels=2,
            history_encoder_mode="wavelet_split",
            enable_wavelet_utility_loss=True,
        )
        q_values, utility_pred = qnet(*obs, return_aux=True)
        self.assertEqual(tuple(q_values.shape), (2, K_SIZE, 1))
        self.assertEqual(tuple(utility_pred.shape), (2, K_SIZE, 1))

    def test_utility_aux_loss_basic_and_spatial2d(self):
        batch_size = 2
        pred = torch.randn(batch_size, K_SIZE, 1)
        target = torch.randn(batch_size, K_SIZE, 1)
        valid_mask = torch.ones(batch_size, K_SIZE, dtype=torch.bool)
        valid_mask[:, -3:] = False
        supervision_mask = torch.ones(batch_size, K_SIZE, dtype=torch.bool)
        supervision_mask[:, 1::3] = False
        target[:, 1::3, 0] = float("nan")
        coords = torch.randn(batch_size, K_SIZE, 2)

        total_basic, base_basic, wave_basic = compute_wavelet_utility_aux_loss(
            pred,
            target,
            valid_mask,
            candidate_coords=coords,
            supervision_mask=supervision_mask,
            loss_mode="basic",
            loss_weight=1.0,
            loss_type="smoothl1",
            base_weight=1.0,
            wavelet_weight=0.1,
            patch_size=5,
            patch_sigma=0.5,
            wavelet_levels=2,
            wavelet_rho=1.0,
        )
        self.assertTrue(torch.isfinite(total_basic))
        self.assertTrue(torch.isfinite(base_basic))
        self.assertTrue(torch.isfinite(wave_basic))
        self.assertAlmostEqual(float(wave_basic.item()), 0.0, places=6)

        total_spatial, base_spatial, wave_spatial = compute_wavelet_utility_aux_loss(
            pred,
            target,
            valid_mask,
            candidate_coords=coords,
            supervision_mask=supervision_mask,
            loss_mode="spatial2d",
            loss_weight=1.0,
            loss_type="smoothl1",
            base_weight=1.0,
            wavelet_weight=0.2,
            patch_size=5,
            patch_sigma=0.5,
            wavelet_levels=2,
            wavelet_rho=1.0,
        )
        self.assertTrue(torch.isfinite(total_spatial))
        self.assertTrue(torch.isfinite(base_spatial))
        self.assertTrue(torch.isfinite(wave_spatial))


if __name__ == "__main__":
    unittest.main()
