from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from ac_pbgrl.config import Config
from ac_pbgrl.data.map_splits import load_split_paths
from ac_pbgrl.envs.ariadne.adapter import AriadneExplorationEnv
from ac_pbgrl.envs.synthetic import SyntheticGraphExplorationEnv
from ac_pbgrl.learning.rollout import EpisodeCollector
from ac_pbgrl.evaluation.metrics import ranking_metrics, uncertainty_metrics
from ac_pbgrl.models.policy import ACPolicyNetwork
from ac_pbgrl.utils import atomic_write_json, seed_everything


def load_actor(config: Config, checkpoint: str | Path, device: str | torch.device) -> ACPolicyNetwork:
    actor = ACPolicyNetwork(
        int(config.environment.node_feature_dim),
        int(config.environment.edge_feature_dim),
        int(config.model.embedding_dim),
        int(config.model.heads),
        int(config.model.encoder_layers),
        float(config.model.dropout),
        use_potential=bool(config.method.potential),
        use_diffusion=bool(config.method.graph_diffusion),
        fuse_uncertainty=bool(config.method.fuse_uncertainty),
        logvar_min=float(config.model.logvar_min),
        logvar_max=float(config.model.logvar_max),
    ).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    container = payload.get("learner", payload)
    actor.load_state_dict(container.get("actor", container.get("policy_model")), strict=True)
    return actor.eval()


def _environment(config: Config, seed: int):
    if str(config.environment.backend) == "synthetic":
        return SyntheticGraphExplorationEnv(
            int(config.environment.node_padding), int(config.environment.candidate_padding), seed
        )
    return AriadneExplorationEnv(
        maps_dir=config.project.maps_dir,
        node_padding=int(config.environment.node_padding),
        critic_node_padding=int(config.environment.critic_node_padding),
        candidate_padding=int(config.environment.candidate_padding),
        max_episode_steps=int(config.environment.max_episode_steps),
        completion_threshold=float(config.environment.completion_threshold),
        terminal_reward=float(config.environment.terminal_reward),
        hierarchy=bool(config.method.hierarchy),
        local_budget=int(config.graph_context.local_budget),
        region_budget=int(config.graph_context.region_budget),
        region_size_m=float(config.graph_context.region_size_m),
        seed=seed,
    )


def evaluate_checkpoint(
    config: Config,
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    split_manifest: str | Path | None = None,
    split: str = "iid_test",
    seeds: list[int] | None = None,
    map_limit: int | None = None,
    device: str = "cpu",
) -> list[dict]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = seeds or [0, 1, 2, 3, 4]
    actor = load_actor(config, checkpoint, device)
    if str(config.environment.backend) == "synthetic":
        maps = [None]
    elif split_manifest is not None:
        maps = load_split_paths(split_manifest, split)
    else:
        maps = sorted(Path(config.project.maps_dir).glob("*.png"))
    if map_limit is not None:
        maps = maps[: int(map_limit)]
    rows = []
    path_rows = []
    potential_samples = {"mean": [], "variance": [], "target": [], "group": []}
    potential_group = 0
    labeler = None
    if bool(config.method.potential) and bool(config.evaluation.get("potential_diagnostics", True)):
        from ac_pbgrl.learning.train import build_labeler

        labeler = build_labeler(
            config,
            smoke=str(config.environment.backend) == "synthetic",
            device=torch.device(device),
        )
    action_temperature = 1.0
    if bool(config.method.potential):
        from ac_pbgrl.learning.calibration import load_variance_temperatures, resolve_calibration_path

        calibration_path = resolve_calibration_path(config)
        if calibration_path.is_file():
            _, action_temperature = load_variance_temperatures(calibration_path)
    for seed in seeds:
        seed_everything(seed)
        environment = _environment(config, seed)
        collector = EpisodeCollector(
            environment,
            actor,
            config,
            device=device,
            labeler=labeler,
            label_budget=int(config.evaluation.get("potential_diagnostic_steps", 16)),
            label_interval=int(config.evaluation.get("potential_diagnostic_interval", 4)),
            greedy=True,
        )
        for map_index, map_path in enumerate(maps):
            if str(device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(torch.device(device))
            started = time.perf_counter()
            transitions, metrics = collector.collect(
                episode=seed * max(1, len(maps)) + map_index,
                map_path=map_path,
            )
            metrics["system/episode_wall_time_s"] = time.perf_counter() - started
            if str(device).startswith("cuda"):
                metrics["system/peak_gpu_memory_mib"] = torch.cuda.max_memory_allocated(torch.device(device)) / 1024**2
            state_rankings = []
            episode_mean, episode_variance, episode_target = [], [], []
            if bool(config.method.potential):
                with torch.no_grad():
                    for transition in transitions:
                        if transition.future_gain is None or transition.future_gain_mask is None:
                            continue
                        output = actor(transition.state.to(device))
                        mask = (
                            transition.future_gain_mask
                            & transition.state.candidate_mask
                            & torch.isfinite(transition.future_gain)
                        )
                        mean = output.action_mean.detach().cpu().numpy()[0]
                        variance = output.action_log_variance.detach().float().exp().cpu().numpy()[0] * action_temperature
                        target = transition.future_gain.detach().cpu().numpy()[0]
                        valid = mask.detach().cpu().numpy()[0]
                        state_rankings.append(ranking_metrics(mean, target, valid))
                        episode_mean.extend(mean[valid])
                        episode_variance.extend(variance[valid])
                        episode_target.extend(target[valid])
                        potential_samples["mean"].extend(mean[valid])
                        potential_samples["variance"].extend(variance[valid])
                        potential_samples["target"].extend(target[valid])
                        potential_samples["group"].extend([potential_group] * int(valid.sum()))
                        potential_group += 1
            if episode_mean:
                diagnostic = uncertainty_metrics(
                    episode_mean,
                    episode_variance,
                    episode_target,
                    np.ones(len(episode_mean), dtype=np.bool_),
                )
                metrics.update({f"potential/{key}": value for key, value in diagnostic.items()})
                for key in ("spearman", "kendall", "top1_regret", "pairwise_accuracy"):
                    metrics[f"potential/{key}"] = float(np.nanmean([item[key] for item in state_rankings]))
            method = str(config.project.experiment)
            map_id = "synthetic" if map_path is None else Path(map_path).name
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "split": split,
                    "map_id": map_id,
                    **metrics,
                }
            )
            path_rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "split": split,
                    "map_id": map_id,
                    "xy": collector.last_trajectory,
                }
            )
    jsonl = output_dir / "episodes.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            serializable = {
                key: (None if isinstance(value, float) and not math.isfinite(value) else value)
                for key, value in row.items()
            }
            handle.write(json.dumps(serializable, sort_keys=True, allow_nan=False) + "\n")
    if rows:
        with (output_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
            writer.writeheader()
            writer.writerows(
                {
                    key: ("" if isinstance(value, float) and not math.isfinite(value) else value)
                    for key, value in row.items()
                }
                for row in rows
            )
    with (output_dir / "paths.jsonl").open("w", encoding="utf-8") as handle:
        for row in path_rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    if potential_samples["target"]:
        np.savez_compressed(
            output_dir / "potential_samples.npz",
            **{key: np.asarray(value) for key, value in potential_samples.items()},
        )
    summary = {}
    if rows:
        for key in rows[0]:
            if not key.startswith(("episode/", "system/", "graph/", "potential/", "kf/")):
                continue
            values = np.asarray([row.get(key, np.nan) for row in rows], dtype=np.float64)
            finite = values[np.isfinite(values)]
            summary[key] = float(finite.mean()) if len(finite) else None
    atomic_write_json(output_dir / "summary.json", summary)
    return rows
