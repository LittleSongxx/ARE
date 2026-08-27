from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ac_pbgrl.config import Config
from ac_pbgrl.data.labels import LabelDataset
from ac_pbgrl.evaluation.evaluator import load_actor
from ac_pbgrl.evaluation.metrics import uncertainty_metrics
from ac_pbgrl.models.temporal import VarianceCalibrator
from ac_pbgrl.utils import atomic_write_json, sha256_file


def calibration_sample_indices(dataset_size: int, samples: int, seed: int) -> np.ndarray:
    """Select a deterministic held-out subset without duplicate states."""
    dataset_size = int(dataset_size)
    samples = int(samples)
    if dataset_size <= 0:
        raise ValueError("calibration label split is empty")
    if samples <= 0:
        raise ValueError("calibration sample count must be positive")
    count = min(samples, dataset_size)
    return np.random.default_rng(int(seed)).choice(dataset_size, size=count, replace=False)


def default_calibration_path(config: Config) -> Path:
    return (
        Path(config.project.data_root)
        / "calibration"
        / str(config.project.experiment)
        / f"seed_{int(config.project.seed)}.json"
    )


def resolve_calibration_path(config: Config) -> Path:
    configured = str(config.filter.get("calibration_path", "auto"))
    return default_calibration_path(config) if configured == "auto" else Path(configured)


@torch.no_grad()
def fit_variance_calibration(
    config: Config,
    checkpoint: str | Path,
    label_root: str | Path,
    *,
    split: str = "validation",
    samples: int = 2048,
    batch_size: int = 32,
    device: str | torch.device = "cpu",
) -> dict:
    checkpoint = Path(checkpoint)
    dataset = LabelDataset(label_root, split)
    if len(dataset) <= 0:
        raise ValueError("calibration label split is empty")
    if int(batch_size) <= 0:
        raise ValueError("calibration batch size must be positive")
    actor = load_actor(config, checkpoint, device)
    sampling_seed = 1907 + int(config.project.seed)
    sample_indices = calibration_sample_indices(len(dataset), samples, sampling_seed)
    means, action_variances, region_variances, targets = [], [], [], []
    for start in range(0, len(sample_indices), int(batch_size)):
        batch = dataset.batch(
            sample_indices[start : start + int(batch_size)],
            hierarchy=bool(config.method.hierarchy),
            local_budget=int(config.graph_context.local_budget),
            region_budget=int(config.graph_context.region_budget),
            region_size_m=float(config.graph_context.region_size_m),
        ).to(device)
        output = actor(batch.state)
        mask = batch.future_gain_mask & batch.state.candidate_mask & torch.isfinite(batch.future_gain)
        means.append(output.action_mean[mask].float().cpu().numpy())
        action_variances.append(output.action_log_variance[mask].float().exp().cpu().numpy())
        region_variances.append(output.region_log_variance[mask].float().exp().cpu().numpy())
        targets.append(batch.future_gain[mask].float().cpu().numpy())
    mean = np.concatenate(means)
    action_variance = np.concatenate(action_variances)
    region_variance = np.concatenate(region_variances)
    target = np.concatenate(targets)
    if not len(target):
        raise ValueError("validation labels contain no finite candidate future-gain targets")
    residual = target - mean
    action_calibrator = VarianceCalibrator()
    action_temperature = action_calibrator.fit(action_variance, residual)
    region_calibrator = VarianceCalibrator()
    unconstrained_region_temperature = region_calibrator.fit(region_variance, residual)
    # The head models action variance as region variance plus a non-negative
    # residual variance.  Keeping the region temperature no larger than the
    # action temperature preserves that ordering after calibration, so temporal
    # recomposition never needs a negative uncertainty component.
    region_temperature = min(unconstrained_region_temperature, action_temperature)
    valid = np.ones_like(target, dtype=np.bool_)
    manifest_path = Path(label_root) / f"manifest_{split}.json"
    return {
        "version": 1,
        "method": str(config.project.experiment),
        "seed": int(config.project.seed),
        "split": split,
        "states": int(len(sample_indices)),
        "action_targets": int(len(target)),
        "sampling": "deterministic_without_replacement",
        "sampling_seed": sampling_seed,
        "sample_indices_sha256": hashlib.sha256(
            np.asarray(sample_indices, dtype="<i8").tobytes()
        ).hexdigest(),
        "temperature": float(region_temperature),
        "region_temperature": float(region_temperature),
        "region_temperature_unconstrained": float(unconstrained_region_temperature),
        "action_temperature": float(action_temperature),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "label_manifest": str(manifest_path.resolve()),
        "label_manifest_sha256": sha256_file(manifest_path),
        "action_before": uncertainty_metrics(mean, action_variance, target, valid),
        "action_after": uncertainty_metrics(mean, action_variance * action_temperature, target, valid),
        "region_before": uncertainty_metrics(mean, region_variance, target, valid),
        "region_after": uncertainty_metrics(mean, region_variance * region_temperature, target, valid),
    }


def save_calibration(report: dict, path: str | Path) -> Path:
    path = Path(path)
    atomic_write_json(path, report)
    return path


def load_calibrator(path: str | Path) -> VarianceCalibrator:
    region, _ = load_variance_temperatures(path)
    return VarianceCalibrator(region)


def load_variance_temperatures(path: str | Path) -> tuple[float, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    region = float(payload.get("region_temperature", payload.get("temperature", 1.0)))
    action = float(payload.get("action_temperature", region))
    return min(region, action), action
