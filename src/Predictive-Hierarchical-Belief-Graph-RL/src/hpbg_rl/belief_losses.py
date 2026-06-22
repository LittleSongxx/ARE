from __future__ import annotations

import torch
import torch.nn.functional as F


class BeliefDistillationLoss:
    def __init__(self, weight: float = 0.05, warmup_updates: int = 1000, ramp_updates: int = 2000):
        self.weight = max(float(weight), 0.0)
        self.warmup_updates = max(int(warmup_updates), 0)
        self.ramp_updates = max(int(ramp_updates), 0)

    def lambda_eff(self, update_step: int) -> float:
        if self.weight <= 0:
            return 0.0
        step = max(int(update_step), 0)
        if step < self.warmup_updates:
            return 0.0
        if self.ramp_updates <= 0:
            return self.weight
        progress = min((step - self.warmup_updates + 1) / float(self.ramp_updates), 1.0)
        return self.weight * progress

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        update_step: int,
    ) -> dict[str, torch.Tensor | float]:
        if prediction.shape != target.shape:
            raise ValueError(f"belief prediction/target shape mismatch: {prediction.shape} vs {target.shape}")
        if prediction.size(-1) < 3:
            raise ValueError(f"belief distillation expects at least 3 target channels, got {prediction.size(-1)}")

        node_mask = valid_mask.bool()
        if node_mask.dim() == prediction.dim():
            node_mask = node_mask[..., 0]
        elif node_mask.dim() != prediction.dim() - 1:
            raise ValueError(f"valid_mask shape mismatch: {tuple(valid_mask.shape)} for {tuple(prediction.shape)}")
        if not torch.any(node_mask):
            zero = prediction.sum() * 0.0
            return {
                "loss": zero,
                "weighted_loss": zero,
                "lambda_eff": 0.0,
                "explored_loss": zero,
                "oracle_loss": zero,
                "potential_loss": zero,
            }

        target = target.detach()
        explored_pred = prediction[..., 0].clamp(1e-6, 1.0 - 1e-6)
        explored_target = target[..., 0].clamp(0.0, 1.0)
        explored_loss = F.binary_cross_entropy(
            explored_pred[node_mask],
            explored_target[node_mask],
            reduction="mean",
        )

        oracle_loss = F.smooth_l1_loss(
            prediction[..., 1][node_mask],
            target[..., 1].clamp(0.0, 1.0)[node_mask],
            reduction="mean",
        )
        potential_loss = F.smooth_l1_loss(
            prediction[..., 2][node_mask],
            target[..., 2].clamp(0.0, 1.0)[node_mask],
            reduction="mean",
        )
        raw = (explored_loss + oracle_loss + potential_loss) / 3.0
        lam = self.lambda_eff(update_step)
        return {
            "loss": raw,
            "weighted_loss": raw * lam,
            "lambda_eff": lam,
            "explored_loss": explored_loss,
            "oracle_loss": oracle_loss,
            "potential_loss": potential_loss,
        }