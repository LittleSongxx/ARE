from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

from ac_pbgrl.state import ExplorationState, PolicyOutput

from .attention import GraphEncoder, MaskedMultiHeadAttention, PointerAttention
from .diffusion import DiffusionInputEmbedding
from .potential import ActionConditionedPotentialHead


def _batch_gather(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    width = values.shape[-1]
    return torch.gather(values, 1, indices.unsqueeze(-1).expand(-1, -1, width))


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weights = mask.to(values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    return (values * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)


def _masked_standardize(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype)
    count = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (values * weights).sum(dim=1, keepdim=True) / count
    variance = ((values - mean).square() * weights).sum(dim=1, keepdim=True) / count
    return (values - mean) / variance.sqrt().clamp_min(1.0e-4)


class ACPolicyNetwork(nn.Module):
    def __init__(
        self,
        node_feature_dim: int = 4,
        edge_feature_dim: int = 4,
        embedding_dim: int = 128,
        heads: int = 8,
        layers: int = 6,
        dropout: float = 0.0,
        *,
        use_potential: bool = True,
        use_diffusion: bool = True,
        fuse_uncertainty: bool = True,
        logvar_min: float = -8.0,
        logvar_max: float = 4.0,
    ) -> None:
        super().__init__()
        self.use_potential = use_potential
        self.use_diffusion = use_diffusion
        self.fuse_uncertainty = fuse_uncertainty
        self.edge_feature_dim = edge_feature_dim
        self.input_embedding = DiffusionInputEmbedding(node_feature_dim, embedding_dim, enabled=use_diffusion)
        self.encoder = GraphEncoder(embedding_dim, heads, layers, dropout=dropout)
        self.current_attention = MaskedMultiHeadAttention(embedding_dim, heads)
        self.current_projection = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.pointer = PointerAttention(embedding_dim)
        if use_potential:
            self.potential_head = ActionConditionedPotentialHead(
                embedding_dim, edge_feature_dim, logvar_min=logvar_min, logvar_max=logvar_max
            )
            uncertainty_dim = 2 if fuse_uncertainty else 1
            self.fusion = nn.Sequential(
                nn.Linear(embedding_dim * 2 + edge_feature_dim + uncertainty_dim, embedding_dim),
                nn.GELU(),
                nn.Linear(embedding_dim, 1),
            )

    def encode(self, state: ExplorationState) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded, _ = self.input_embedding(state.node_features, state.adjacency, state.node_mask)
        encoded = self.encoder(embedded, state.node_mask, state.adjacency)
        batch = encoded.shape[0]
        current = _batch_gather(encoded, state.current_index.unsqueeze(1)).squeeze(1)
        query_mask = torch.ones((batch, 1), dtype=torch.bool, device=encoded.device)
        attended = self.current_attention(
            current.unsqueeze(1),
            encoded,
            encoded,
            query_mask=query_mask,
            key_mask=state.node_mask,
        ).squeeze(1)
        current_state = self.current_projection(torch.cat((current, attended), dim=-1))
        candidates = _batch_gather(encoded, state.candidate_indices)
        global_context = _masked_mean(encoded, state.node_mask, dim=1)
        return encoded, current_state, candidates, global_context

    def _score(
        self,
        current: torch.Tensor,
        candidates: torch.Tensor,
        edge_features: torch.Tensor,
        mask: torch.Tensor,
        action_mean: Optional[torch.Tensor],
        action_log_variance: Optional[torch.Tensor],
        posterior_mean: Optional[torch.Tensor],
        posterior_variance: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base_logits = self.pointer(current, candidates, mask)
        if not self.use_potential:
            return base_logits, base_logits
        mean = action_mean if posterior_mean is None else posterior_mean
        variance = action_log_variance.exp() if posterior_variance is None else posterior_variance
        mean_feature = _masked_standardize(mean, mask).unsqueeze(-1)
        inputs = [
            current.unsqueeze(1).expand_as(candidates),
            candidates,
            edge_features,
            mean_feature,
        ]
        if self.fuse_uncertainty:
            inputs.append(variance.clamp_min(1.0e-8).log().unsqueeze(-1))
        residual = self.fusion(torch.cat(inputs, dim=-1)).squeeze(-1)
        logits = (base_logits + residual).masked_fill(~mask, -1.0e4)
        return logits, base_logits

    def forward(
        self,
        state: ExplorationState,
        posterior_mean: Optional[torch.Tensor] = None,
        posterior_variance: Optional[torch.Tensor] = None,
    ) -> PolicyOutput:
        state.validate()
        _, current, candidates, global_context = self.encode(state)
        action_mean = action_log_variance = region_mean = region_log_variance = None
        if self.use_potential:
            action_mean, action_log_variance, region_mean, region_log_variance = self.potential_head(
                current, candidates, global_context, state.edge_features
            )
        if posterior_mean is None:
            posterior_mean = state.posterior_mean
        if posterior_variance is None:
            posterior_variance = state.posterior_variance
        logits, base_logits = self._score(
            current,
            candidates,
            state.edge_features,
            state.candidate_mask,
            action_mean,
            action_log_variance,
            posterior_mean,
            posterior_variance,
        )
        log_probs = torch.log_softmax(logits.float(), dim=-1).to(logits.dtype)
        probabilities = log_probs.exp() * state.candidate_mask.to(log_probs.dtype)
        return PolicyOutput(
            logits=logits,
            log_probs=log_probs,
            probabilities=probabilities,
            base_logits=base_logits,
            action_mean=action_mean,
            action_log_variance=action_log_variance,
            region_mean=region_mean,
            region_log_variance=region_log_variance,
            posterior_mean=posterior_mean,
            posterior_variance=posterior_variance,
            candidate_embeddings=candidates,
        )


class PrivilegedQNetwork(nn.Module):
    def __init__(
        self,
        node_feature_dim: int = 5,
        edge_feature_dim: int = 4,
        embedding_dim: int = 128,
        heads: int = 8,
        layers: int = 6,
        dropout: float = 0.0,
        use_diffusion: bool = False,
    ) -> None:
        super().__init__()
        self.input_embedding = DiffusionInputEmbedding(node_feature_dim, embedding_dim, enabled=use_diffusion)
        self.encoder = GraphEncoder(embedding_dim, heads, layers, dropout=dropout)
        self.current_attention = MaskedMultiHeadAttention(embedding_dim, heads)
        self.q_head = nn.Sequential(
            nn.Linear(embedding_dim * 3 + edge_feature_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Linear(embedding_dim * 2, 1),
        )

    def forward(self, state: ExplorationState) -> torch.Tensor:
        state.validate()
        embedded, _ = self.input_embedding(state.node_features, state.adjacency, state.node_mask)
        encoded = self.encoder(embedded, state.node_mask, state.adjacency)
        batch = encoded.shape[0]
        current = _batch_gather(encoded, state.current_index.unsqueeze(1)).squeeze(1)
        query_mask = torch.ones((batch, 1), dtype=torch.bool, device=encoded.device)
        global_context = self.current_attention(
            current.unsqueeze(1), encoded, encoded, query_mask=query_mask, key_mask=state.node_mask
        ).squeeze(1)
        candidates = _batch_gather(encoded, state.candidate_indices)
        count = candidates.shape[1]
        action_features = torch.cat(
            (
                current.unsqueeze(1).expand(-1, count, -1),
                global_context.unsqueeze(1).expand(-1, count, -1),
                candidates,
                state.edge_features,
            ),
            dim=-1,
        )
        q_values = self.q_head(action_features).squeeze(-1)
        return q_values.masked_fill(~state.candidate_mask, -1.0e4)
