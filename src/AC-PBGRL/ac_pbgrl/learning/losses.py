from __future__ import annotations

import torch
from torch.nn import functional as F


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over valid values without letting padded NaNs poison the sum."""

    valid = mask.bool() & torch.isfinite(values)
    weights = valid.to(values.dtype)
    safe_values = torch.where(valid, values, torch.zeros_like(values))
    return safe_values.sum() / weights.sum().clamp_min(1.0)


def heteroscedastic_gaussian_nll(
    mean: torch.Tensor,
    log_variance: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    logvar_min: float = -8.0,
    logvar_max: float = 4.0,
) -> torch.Tensor:
    log_variance = log_variance.float().clamp(logvar_min, logvar_max)
    mean = mean.float()
    target = target.float()
    valid = mask.bool() & torch.isfinite(mean) & torch.isfinite(log_variance) & torch.isfinite(target)
    safe_target = torch.where(valid, target, mean.detach())
    residual = safe_target - mean
    loss = 0.5 * (torch.exp(-log_variance) * residual.square() + log_variance)
    return masked_mean(loss, valid)


def masked_regression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    kind: str = "mse",
) -> torch.Tensor:
    valid = mask.bool() & torch.isfinite(prediction) & torch.isfinite(target)
    safe_target = torch.where(valid, target, prediction.detach())
    if kind == "mse":
        loss = (prediction.float() - safe_target.float()).square()
    elif kind in {"smooth_l1", "huber"}:
        loss = F.smooth_l1_loss(prediction.float(), safe_target.float(), reduction="none")
    else:
        raise ValueError(f"unknown regression loss: {kind}")
    return masked_mean(loss, valid)


def ranknet_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    tie_delta: float = 0.0,
    max_pairs_per_state: int = 64,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    losses = []
    for batch_index in range(scores.shape[0]):
        finite = torch.isfinite(scores[batch_index]) & torch.isfinite(targets[batch_index])
        valid = torch.nonzero(mask[batch_index].bool() & finite, as_tuple=False).flatten()
        if valid.numel() < 2:
            continue
        i_grid, j_grid = torch.triu_indices(valid.numel(), valid.numel(), offset=1, device=scores.device)
        left = valid.index_select(0, i_grid)
        right = valid.index_select(0, j_grid)
        target_delta = targets[batch_index, left] - targets[batch_index, right]
        informative = target_delta.abs() > tie_delta
        left, right, target_delta = left[informative], right[informative], target_delta[informative]
        if left.numel() == 0:
            continue
        if left.numel() > max_pairs_per_state:
            selection = torch.randperm(left.numel(), device=left.device, generator=generator)[:max_pairs_per_state]
            left, right, target_delta = left[selection], right[selection], target_delta[selection]
        sign = target_delta.sign()
        score_delta = scores[batch_index, left] - scores[batch_index, right]
        losses.append(F.softplus(-sign * score_delta))
    if not losses:
        return scores.sum() * 0.0
    return torch.cat(losses).mean()


def auxiliary_weight(step: int, target_weight: float, warmup_steps: int, ramp_steps: int) -> float:
    if step < warmup_steps:
        return 0.0
    if ramp_steps <= 0:
        return float(target_weight)
    progress = min(1.0, max(0.0, (step - warmup_steps) / float(ramp_steps)))
    return float(target_weight) * progress


def uncertainty_diagnostics(
    mean: torch.Tensor,
    log_variance: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    valid = mask.bool() & torch.isfinite(mean) & torch.isfinite(log_variance) & torch.isfinite(target)
    valid_mean = mean[valid].float()
    valid_target = target[valid].float()
    valid_variance = log_variance[valid].float().exp().clamp_min(1.0e-8)
    if valid_mean.numel() == 0:
        return {key: float("nan") for key in ("rmse", "mae", "nll", "coverage_90", "coverage_95")}
    residual = valid_target - valid_mean
    standardized = residual.abs() / valid_variance.sqrt()
    return {
        "rmse": float(residual.square().mean().sqrt()),
        "mae": float(residual.abs().mean()),
        "nll": float((0.5 * (residual.square() / valid_variance + valid_variance.log())).mean()),
        "coverage_90": float((standardized <= 1.6448536269514722).float().mean()),
        "coverage_95": float((standardized <= 1.959963984540054).float().mean()),
    }
