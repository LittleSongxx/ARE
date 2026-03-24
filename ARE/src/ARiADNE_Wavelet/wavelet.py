from __future__ import annotations

import math

import torch


_SQRT2 = math.sqrt(2.0)


def _pad_last_to_even(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Pad one element by repeating the last slice when length is odd."""
    dim = dim % x.dim()
    if x.size(dim) % 2 == 0:
        return x
    pad_index = [slice(None)] * x.dim()
    pad_index[dim] = slice(x.size(dim) - 1, x.size(dim))
    last = x[tuple(pad_index)]
    return torch.cat((x, last), dim=dim)


def _haar_once_last_dim(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = _pad_last_to_even(x, dim=-1)
    even = x[..., 0::2]
    odd = x[..., 1::2]
    low = (even + odd) / _SQRT2
    high = (even - odd) / _SQRT2
    return low, high


def haar_decompose_last_dim(x: torch.Tensor, levels: int = 2) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Multi-level Haar decomposition along the last dim.

    Returns:
        low: final low-frequency tensor after `levels` decimations
        highs: high-frequency tensors from each level in order [lvl1, lvl2, ...]
    """
    levels = max(int(levels), 1)
    current = x
    highs: list[torch.Tensor] = []
    for _ in range(levels):
        current, high = _haar_once_last_dim(current)
        highs.append(high)
    return current, highs


def haar_decompose_time(x: torch.Tensor, levels: int = 2) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Multi-level Haar decomposition along time axis for [B, T, C] tensors."""
    if x.dim() != 3:
        raise ValueError(f"Expected [B, T, C], got shape={tuple(x.shape)}")
    y = x.transpose(1, 2)  # [B, C, T]
    low, highs = haar_decompose_last_dim(y, levels=levels)
    return low.transpose(1, 2), [high.transpose(1, 2) for high in highs]


def haar_decompose_vector(x: torch.Tensor, levels: int = 2) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Haar decomposition for 1D vectors [N]."""
    if x.dim() != 1:
        raise ValueError(f"Expected 1D tensor, got shape={tuple(x.shape)}")
    low, highs = haar_decompose_last_dim(x.reshape(1, 1, -1), levels=levels)
    return low.reshape(-1), [high.reshape(-1) for high in highs]


def _pad_last_two_to_even(x: torch.Tensor) -> torch.Tensor:
    if x.dim() < 2:
        raise ValueError(f"Expected tensor with at least 2 dims, got shape={tuple(x.shape)}")
    y = x
    if y.size(-2) % 2 != 0:
        y = torch.cat((y, y[..., -1:, :]), dim=-2)
    if y.size(-1) % 2 != 0:
        y = torch.cat((y, y[..., :, -1:]), dim=-1)
    return y


def _haar_once_2d(x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    y = _pad_last_two_to_even(x)

    even_rows = y[..., 0::2, :]
    odd_rows = y[..., 1::2, :]
    low_rows = (even_rows + odd_rows) / _SQRT2
    high_rows = (even_rows - odd_rows) / _SQRT2

    ll = (low_rows[..., 0::2] + low_rows[..., 1::2]) / _SQRT2
    lh = (low_rows[..., 0::2] - low_rows[..., 1::2]) / _SQRT2
    hl = (high_rows[..., 0::2] + high_rows[..., 1::2]) / _SQRT2
    hh = (high_rows[..., 0::2] - high_rows[..., 1::2]) / _SQRT2
    return ll, (lh, hl, hh)


def haar_decompose_2d(
    x: torch.Tensor,
    levels: int = 1,
) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    """Multi-level 2D Haar decomposition on the last two dims [H, W]."""
    levels = max(int(levels), 1)
    current = x
    highs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for _ in range(levels):
        current, high = _haar_once_2d(current)
        highs.append(high)
    return current, highs
