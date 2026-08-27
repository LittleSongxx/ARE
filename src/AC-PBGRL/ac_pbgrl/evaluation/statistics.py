from __future__ import annotations

import numpy as np
from scipy import stats


def paired_bootstrap_ci(
    differences,
    *,
    samples: int = 10000,
    confidence: float = 0.95,
    seed: int = 4777,
) -> tuple[float, float, float]:
    values = np.asarray(differences, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(values.mean()), float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def cliffs_delta(left, right) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    comparisons = np.sign(left[:, None] - right[None, :])
    return float(comparisons.mean())


def holm_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    adjusted = np.full_like(values, np.nan)
    valid_indices = np.flatnonzero(np.isfinite(values))
    if not len(valid_indices):
        return adjusted.tolist()
    order = valid_indices[np.argsort(values[valid_indices])]
    running = 0.0
    count = len(order)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def paired_comparison(left, right, *, bootstrap_samples: int = 10000) -> dict[str, float]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    left, right = left[valid], right[valid]
    if not len(left):
        return {"n": 0, "difference": np.nan, "ci_low": np.nan, "ci_high": np.nan, "wilcoxon_p": np.nan, "cliffs_delta": np.nan}
    difference, low, high = paired_bootstrap_ci(left - right, samples=bootstrap_samples)
    if np.allclose(left, right):
        wilcoxon = 1.0
    else:
        try:
            wilcoxon = float(stats.wilcoxon(left, right).pvalue)
        except ValueError:
            wilcoxon = 1.0
    if not np.isfinite(wilcoxon):
        wilcoxon = 1.0
    return {
        "n": int(len(left)),
        "difference": difference,
        "ci_low": low,
        "ci_high": high,
        "wilcoxon_p": wilcoxon,
        "cliffs_delta": cliffs_delta(left, right),
    }
