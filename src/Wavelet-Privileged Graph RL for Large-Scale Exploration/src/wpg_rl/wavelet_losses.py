from __future__ import annotations

import torch
import torch.nn as nn


def masked_l1_mean(
    prediction: torch.Tensor,
    target: torch.Tensor,
    node_padding_mask: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if valid_mask is not None:
        if valid_mask.dim() != 2:
            raise ValueError(f"valid_mask must be [B, N], got {tuple(valid_mask.shape)}")
        valid = valid_mask.unsqueeze(-1).to(dtype=prediction.dtype)
    elif node_padding_mask is not None:
        if node_padding_mask.dim() == 3:
            valid = (~node_padding_mask.squeeze(1).bool()).unsqueeze(-1).to(dtype=prediction.dtype)
        elif node_padding_mask.dim() == 2:
            valid = (~node_padding_mask.bool()).unsqueeze(-1).to(dtype=prediction.dtype)
        else:
            raise ValueError(f"Unsupported node_padding_mask shape: {tuple(node_padding_mask.shape)}")
    else:
        raise ValueError("Either node_padding_mask or valid_mask must be provided")

    denominator = valid.sum().clamp_min(1.0)
    return (torch.abs(prediction - target) * valid).sum() / denominator


def compute_distill_weight(
    update_step: int,
    base_weight: float,
    warmup_updates: int,
    ramp_updates: int,
) -> float:
    base_weight = max(float(base_weight), 0.0)
    update_step = max(int(update_step), 0)
    warmup_updates = max(int(warmup_updates), 0)
    ramp_updates = max(int(ramp_updates), 0)

    if base_weight == 0.0:
        return 0.0
    if update_step < warmup_updates:
        return 0.0
    if ramp_updates == 0:
        return base_weight
    progress = min(max(update_step - warmup_updates, 0) / float(ramp_updates), 1.0)
    return base_weight * progress


class WaveletDistillationLoss(nn.Module):
    def __init__(
        self,
        lf_weight: float,
        hf_weight: float,
        base_weight: float,
        warmup_updates: int,
        ramp_updates: int,
    ):
        super().__init__()
        self.lf_weight = max(float(lf_weight), 0.0)
        self.hf_weight = max(float(hf_weight), 0.0)
        self.base_weight = max(float(base_weight), 0.0)
        self.warmup_updates = max(int(warmup_updates), 0)
        self.ramp_updates = max(int(ramp_updates), 0)

    def effective_weight(self, update_step: int) -> float:
        return compute_distill_weight(
            update_step,
            self.base_weight,
            self.warmup_updates,
            self.ramp_updates,
        )

    def forward(
        self,
        actor_lf: torch.Tensor,
        actor_hf: torch.Tensor,
        critic_lf: torch.Tensor,
        critic_hf: torch.Tensor,
        overlap_valid_mask: torch.Tensor,
        update_step: int,
    ) -> dict[str, torch.Tensor | float]:
        loss_lf = masked_l1_mean(actor_lf, critic_lf, valid_mask=overlap_valid_mask)
        loss_hf = masked_l1_mean(actor_hf, critic_hf, valid_mask=overlap_valid_mask)
        loss = self.lf_weight * loss_lf + self.hf_weight * loss_hf
        lambda_eff = self.effective_weight(update_step)
        weighted_loss = loss * lambda_eff
        return {
            "loss_lf": loss_lf,
            "loss_hf": loss_hf,
            "loss": loss,
            "weighted_loss": weighted_loss,
            "lambda_eff": lambda_eff,
        }
