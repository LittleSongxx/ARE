from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy import stats


def ranking_metrics(prediction, target, mask) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.bool_) & np.isfinite(prediction) & np.isfinite(target)
    if mask.sum() < 2:
        return {"spearman": np.nan, "kendall": np.nan, "top1_regret": np.nan, "pairwise_accuracy": np.nan}
    pred, truth = prediction[mask], target[mask]
    spearman = stats.spearmanr(pred, truth).statistic
    kendall = stats.kendalltau(pred, truth).statistic
    selected = int(np.argmax(pred))
    top1_regret = float(np.max(truth) - truth[selected])
    correct, total = 0, 0
    for left in range(len(pred)):
        for right in range(left + 1, len(pred)):
            truth_delta = truth[left] - truth[right]
            if abs(truth_delta) < 1.0e-12:
                continue
            correct += int(np.sign(pred[left] - pred[right]) == np.sign(truth_delta))
            total += 1
    return {
        "spearman": float(spearman),
        "kendall": float(kendall),
        "top1_regret": top1_regret,
        "pairwise_accuracy": float(correct / total) if total else np.nan,
    }


def uncertainty_metrics(mean, variance, target, mask) -> dict[str, float]:
    mean = np.asarray(mean, dtype=np.float64)
    variance = np.asarray(variance, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.bool_) & np.isfinite(mean) & np.isfinite(variance) & np.isfinite(target)
    if not mask.any():
        return {key: np.nan for key in ("rmse", "mae", "nll", "coverage_50", "coverage_80", "coverage_90", "coverage_95")}
    residual = target[mask] - mean[mask]
    variance = np.clip(variance[mask], 1.0e-8, None)
    standardized = np.abs(residual) / np.sqrt(variance)
    thresholds = {"50": 0.67448975, "80": 1.28155157, "90": 1.64485363, "95": 1.95996398}
    result = {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "nll": float(np.mean(0.5 * (residual**2 / variance + np.log(variance) + np.log(2 * np.pi)))),
    }
    result.update({f"coverage_{name}": float(np.mean(standardized <= threshold)) for name, threshold in thresholds.items()})
    return result


def path_behavior(node_ids: Iterable[int]) -> dict[str, float]:
    nodes = list(node_ids)
    edges = list(zip(nodes[:-1], nodes[1:]))
    repeated = len(edges) - len(set(edges))
    backtracks = sum(int(index >= 1 and nodes[index + 1] == nodes[index - 1]) for index in range(1, len(nodes) - 1))
    return {"repeated_edges": float(repeated), "backtracks": float(backtracks)}
