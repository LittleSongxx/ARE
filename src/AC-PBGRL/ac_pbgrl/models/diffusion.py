from __future__ import annotations

import torch
from torch import nn


def random_walk_matrix(adjacency: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    valid = node_mask[:, :, None] & node_mask[:, None, :]
    # Every environment/adapter materializes valid self-loops before batching.
    # Avoid constructing an EyeLike/diag_embed node here because the pinned
    # Python-3.8 ONNX Runtime cannot execute the former and PyTorch 2.4 cannot
    # export the latter.
    connected = adjacency.to(torch.bool) & valid
    transition = connected.to(torch.float32)
    transition = transition / transition.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return transition


def multiscale_diffusion_features(
    features: torch.Tensor,
    adjacency: torch.Tensor,
    node_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    transition = random_walk_matrix(adjacency, node_mask).to(features.dtype)
    smooth1 = torch.bmm(transition, features)
    smooth2 = torch.bmm(transition, smooth1)
    smooth4 = torch.bmm(transition, torch.bmm(transition, smooth2))
    detail1 = features - smooth1
    detail2 = smooth1 - smooth2
    detail4 = smooth2 - smooth4
    concatenated = torch.cat((features, smooth4, detail1, detail2, detail4), dim=-1)
    concatenated = concatenated * node_mask.unsqueeze(-1).to(concatenated.dtype)
    return concatenated, {
        "low_frequency": smooth4,
        "detail_1": detail1,
        "detail_2": detail2,
        "detail_4": detail4,
    }


class DiffusionInputEmbedding(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int, enabled: bool = True) -> None:
        super().__init__()
        self.enabled = enabled
        multiplier = 5 if enabled else 1
        self.projection = nn.Sequential(
            nn.Linear(input_dim * multiplier, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(
        self, features: torch.Tensor, adjacency: torch.Tensor, node_mask: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.enabled:
            features, components = multiscale_diffusion_features(features, adjacency, node_mask)
        else:
            components = {}
        embedded = self.projection(features)
        return embedded * node_mask.unsqueeze(-1).to(embedded.dtype), components
