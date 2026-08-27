from __future__ import annotations

import torch
from torch import nn


class ActionConditionedPotentialHead(nn.Module):
    """Decomposes persistent regional potential from action-specific residual."""

    def __init__(
        self,
        embedding_dim: int,
        edge_feature_dim: int,
        logvar_min: float = -8.0,
        logvar_max: float = 4.0,
    ) -> None:
        super().__init__()
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        self.region = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, 2),
        )
        self.residual = nn.Sequential(
            nn.Linear(embedding_dim * 4 + edge_feature_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Linear(embedding_dim * 2, 2),
        )

    def forward(
        self,
        current: torch.Tensor,
        candidates: torch.Tensor,
        global_context: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        count = candidates.shape[1]
        global_repeat = global_context.unsqueeze(1).expand(-1, count, -1)
        current_repeat = current.unsqueeze(1).expand(-1, count, -1)
        region_values = self.region(torch.cat((candidates, global_repeat), dim=-1))
        residual_values = self.residual(
            torch.cat((current_repeat, candidates, global_repeat, candidates - current_repeat, edge_features), dim=-1)
        )
        region_mean = region_values[..., 0]
        region_log_variance = region_values[..., 1].clamp(self.logvar_min, self.logvar_max)
        action_mean = region_mean + residual_values[..., 0]
        residual_log_variance = residual_values[..., 1].clamp(self.logvar_min, self.logvar_max)
        # For independent Gaussian components, variances add.  Summing their
        # logarithms would multiply variances and can make the purported total
        # uncertainty smaller than the persistent region uncertainty.
        # Bounds make the explicit exp/sum/log form numerically safe, while
        # keeping it exportable by the pinned PyTorch 2.4 ONNX exporter (which
        # has no symbolic implementation for aten::logaddexp).
        action_log_variance = torch.log(
            torch.exp(region_log_variance) + torch.exp(residual_log_variance)
        ).clamp(self.logvar_min, self.logvar_max)
        return action_mean, action_log_variance, region_mean, region_log_variance
