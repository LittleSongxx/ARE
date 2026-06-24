from __future__ import annotations

import torch
import torch.nn as nn


class GraphPolicyNet(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.context = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU())
        self.logit_head = nn.Linear(hidden_dim * 2, 1)

    def forward(self, node_features, node_mask, current_index, neighbor_indices, action_mask):
        encoded = self.node_encoder(node_features)
        batch_size, _, hidden_dim = encoded.shape
        current = torch.gather(encoded, 1, current_index.view(batch_size, 1, 1).repeat(1, 1, hidden_dim)).squeeze(1)
        global_context = masked_mean(encoded, node_mask, dim=1)
        state = self.context(torch.cat([current, global_context], dim=-1))
        neighbors = gather_nodes(encoded, neighbor_indices)
        logits = self.logit_head(torch.cat([state.unsqueeze(1).expand_as(neighbors), neighbors], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(action_mask, -1e8)
        return torch.log_softmax(logits, dim=-1)


class GraphQNet(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.state_encoder = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU())
        self.q_head = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, node_features, node_mask, current_index, neighbor_indices):
        encoded = self.node_encoder(node_features)
        batch_size, _, hidden_dim = encoded.shape
        current = torch.gather(encoded, 1, current_index.view(batch_size, 1, 1).repeat(1, 1, hidden_dim)).squeeze(1)
        global_context = masked_mean(encoded, node_mask, dim=1)
        state = self.state_encoder(torch.cat([current, global_context], dim=-1))
        neighbors = gather_nodes(encoded, neighbor_indices)
        q_values = self.q_head(torch.cat([state.unsqueeze(1).expand_as(neighbors), neighbors], dim=-1)).squeeze(-1)
        return q_values


def gather_nodes(encoded: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    hidden_dim = encoded.shape[-1]
    return torch.gather(encoded, 1, indices.unsqueeze(-1).repeat(1, 1, hidden_dim))


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    valid = (~mask).float().unsqueeze(-1)
    total = torch.sum(values * valid, dim=dim)
    denom = torch.clamp(torch.sum(valid, dim=dim), min=1.0)
    return total / denom
