import unittest

import torch

from ARiADNE_Wavelet.model import PolicyNet, QNet, compute_wavelet_utility_aux_loss
from ARiADNE_Wavelet.parameter import EMBEDDING_DIM, HISTORY_INPUT_DIM, HISTORY_LEN, K_SIZE, NODE_INPUT_DIM


class SmokeTrainStepTests(unittest.TestCase):
    def _build_obs(self, batch_size=2, n_nodes=20):
        node_inputs = torch.randn(batch_size, n_nodes, NODE_INPUT_DIM)
        node_padding_mask = torch.zeros(batch_size, 1, n_nodes, dtype=torch.bool)
        edge_mask = torch.zeros(batch_size, n_nodes, n_nodes, dtype=torch.bool)
        current_index = torch.zeros(batch_size, 1, 1, dtype=torch.long)

        current_edge = torch.zeros(batch_size, K_SIZE, 1, dtype=torch.long)
        for b in range(batch_size):
            candidates = torch.arange(K_SIZE, dtype=torch.long) % n_nodes
            current_edge[b, :, 0] = candidates

        edge_padding_mask = torch.zeros(batch_size, 1, K_SIZE, dtype=torch.bool)
        edge_padding_mask[:, :, -3:] = True
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

    def _run_single_train_step(self, enable_history: bool, enable_aux: bool) -> None:
        torch.manual_seed(13)
        obs = self._build_obs()
        next_obs = self._build_obs()

        policy = PolicyNet(
            NODE_INPUT_DIM,
            EMBEDDING_DIM,
            enable_wavelet_history=enable_history,
            history_input_dim=HISTORY_INPUT_DIM,
            history_embed_dim=64,
            history_wavelet_levels=2,
            history_encoder_mode="wavelet_split",
        )
        q1 = QNet(
            NODE_INPUT_DIM,
            EMBEDDING_DIM,
            enable_wavelet_history=enable_history,
            history_input_dim=HISTORY_INPUT_DIM,
            history_embed_dim=64,
            history_wavelet_levels=2,
            history_encoder_mode="wavelet_split",
            enable_wavelet_utility_loss=enable_aux,
        )
        q2 = QNet(
            NODE_INPUT_DIM,
            EMBEDDING_DIM,
            enable_wavelet_history=enable_history,
            history_input_dim=HISTORY_INPUT_DIM,
            history_embed_dim=64,
            history_wavelet_levels=2,
            history_encoder_mode="wavelet_split",
            enable_wavelet_utility_loss=enable_aux,
        )

        policy_opt = torch.optim.Adam(policy.parameters(), lr=1e-4)
        q1_opt = torch.optim.Adam(q1.parameters(), lr=1e-4)

        action = torch.zeros(obs[0].size(0), 1, 1, dtype=torch.long)
        reward = torch.randn(obs[0].size(0), 1, 1)
        done = torch.zeros(obs[0].size(0), 1, 1)

        with torch.no_grad():
            q_min = torch.min(q1(*obs), q2(*obs))

        logp = policy(*obs)
        policy_loss = torch.sum(logp.exp().unsqueeze(2) * (-q_min.detach()), dim=1).mean()
        policy_opt.zero_grad()
        policy_loss.backward()
        policy_opt.step()

        with torch.no_grad():
            next_value = torch.min(q1(*next_obs), q2(*next_obs)).max(dim=1, keepdim=True).values
            target_q = reward + (1 - done) * next_value

        q_values, aux_pred = q1(*obs, return_aux=True)
        q_loss = torch.nn.functional.mse_loss(torch.gather(q_values, 1, action), target_q)

        total_loss = q_loss
        if enable_aux:
            candidate_valid_mask = ~obs[5].squeeze(1).bool()
            candidate_indices = obs[4].squeeze(-1).long()
            candidate_coords = torch.gather(obs[0][..., :2], 1, candidate_indices.unsqueeze(-1).expand(-1, -1, 2))
            td_target = q_values.detach()
            aux_loss, _, _ = compute_wavelet_utility_aux_loss(
                aux_pred,
                td_target,
                candidate_valid_mask,
                candidate_coords=candidate_coords,
                supervision_mask=candidate_valid_mask,
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
            total_loss = total_loss + aux_loss

        q1_opt.zero_grad()
        total_loss.backward()
        q1_opt.step()

        self.assertTrue(torch.isfinite(policy_loss))
        self.assertTrue(torch.isfinite(total_loss))

    def test_smoke_train_step_all_core_toggle_combinations(self):
        combos = [
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ]
        for enable_history, enable_aux in combos:
            with self.subTest(enable_history=enable_history, enable_aux=enable_aux):
                self._run_single_train_step(enable_history=enable_history, enable_aux=enable_aux)


if __name__ == "__main__":
    unittest.main()
