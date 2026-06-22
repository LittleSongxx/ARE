from __future__ import annotations

import json
import re
from pathlib import Path
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from . import parameter as parameter_module
from .agent import Agent
from .env import Env, list_map_files
from .ground_truth_node_manager import GroundTruthNodeManager
from .map_splits import (
    manifest_entry_by_path,
    materialize_split_manifest,
    normalize_split,
    resolve_map_split_paths,
    split_count_for_eval,
)
from .model import PolicyNet
from .parameter import (
    EMBEDDING_DIM,
    NODE_INPUT_DIM,
    RuntimeConfig,
    get_result_eval_gifs_path,
    get_result_test_gifs_path,
    get_result_validation_gifs_path,
)
from .utils import ensure_episode_bucket_dir, make_gif


CORE_EVAL_METRICS = (
    ("explored_rate", "Explored Rate"),
    ("success_rate", "Success Rate"),
    ("completion_steps", "Completion Steps"),
    ("completion_travel_dist", "Completion Travel Distance"),
)


def _copy_state_dict_to_device(state_dict, device: torch.device) -> dict:
    return {key: value.detach().to(device) for key, value in state_dict.items()}


def _normalize_eval_device(device: str | torch.device) -> torch.device:
    resolved = torch.device(device)
    if resolved.type != "cuda":
        return resolved
    if resolved.index is not None:
        return resolved
    try:
        cuda_index = int(torch.cuda.current_device())
    except (AssertionError, RuntimeError):
        cuda_index = 0
    return torch.device(f"cuda:{cuda_index}")


def _prepare_eval_device(device: torch.device) -> None:
    if device.type != "cuda":
        return
    cuda_index = 0 if device.index is None else int(device.index)
    torch.cuda.set_device(cuda_index)
    # Materialize the primary context once so the first evaluation forward pass
    # does not lazily initialize cuBLAS in the middle of a layer call.
    torch.empty(1, device=device)


def _sanitize_artifact_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return token.strip("_") or "map"


def get_eval_map_name(map_number: int) -> str:
    return f"map_{max(int(map_number), 1)}"


def ensure_eval_map_dir(base_dir: str | Path, episode_number: int, map_number: int, bucket_size: int) -> Path:
    episode_dir = ensure_episode_bucket_dir(base_dir, episode_number, bucket_size)
    map_dir = episode_dir / get_eval_map_name(map_number)
    map_dir.mkdir(parents=True, exist_ok=True)
    return map_dir


def get_eval_raw_dir(output_dir: str | Path) -> Path:
    return Path(output_dir) / "raw"


def resolve_fixed_eval_maps(eval_map_count: int, maps_dir: str | Path | None = None) -> list[Path]:
    eval_map_count = max(int(eval_map_count), 0)
    if eval_map_count <= 0:
        return []

    target_maps_dir = maps_dir if maps_dir is not None else parameter_module.MAPS_DIR
    map_files = list_map_files(target_maps_dir)
    return map_files[: min(eval_map_count, len(map_files))]


def resolve_eval_maps(output_config: RuntimeConfig, split: str = "val", count: int | None = None) -> list[Path]:
    split = normalize_split(split)
    eval_count = split_count_for_eval(output_config, split) if count is None else max(int(count), 0)
    if eval_count <= 0:
        return []
    return resolve_map_split_paths(
        output_config,
        split,
        count=eval_count,
        allow_train_eval=output_config.allow_train_split_eval,
    )


def _normalize_result_entry(result: dict[str, object], fallback_index: int) -> dict[str, object]:
    normalized = dict(result)
    map_number = max(int(result.get("map_slot", fallback_index)), 1)
    normalized["map_slot"] = map_number
    normalized["map_label"] = str(result.get("map_label") or get_eval_map_name(map_number))
    return normalized


def _normalize_detail_payload(payload: dict[str, object], detail_path: Path) -> dict[str, object] | None:
    if "episode" not in payload:
        return None
    results = payload.get("results", [])
    if not isinstance(results, list):
        results = []
    normalized_results = [
        _normalize_result_entry(result, index)
        for index, result in enumerate(results, start=1)
        if isinstance(result, dict)
    ]
    return {
        "episode": int(payload["episode"]),
        "summary": payload.get("summary", {}),
        "results": normalized_results,
        "detail_path": str(detail_path.resolve()),
    }


def load_evaluation_detail_history(output_dir: str | Path) -> list[dict[str, object]]:
    output_dir = Path(output_dir)
    detail_by_episode: dict[int, dict[str, object]] = {}
    search_roots = [
        output_dir,
        get_eval_raw_dir(output_dir),
    ]
    for root in search_roots:
        if not root.exists():
            continue
        for detail_path in sorted(root.glob("episode_*/eval_episode_*.json")):
            try:
                payload = json.loads(detail_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            normalized = _normalize_detail_payload(payload, detail_path)
            if normalized is None:
                continue
            detail_by_episode[int(normalized["episode"])] = normalized
    return [detail_by_episode[key] for key in sorted(detail_by_episode.keys())]


def load_evaluation_history(output_dir: str | Path) -> list[dict[str, object]]:
    history_path = Path(output_dir) / "evaluation_history.json"
    if not history_path.exists():
        return []
    try:
        payload = json.loads(history_path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    history = payload.get("evaluations", [])
    history = [item for item in history if isinstance(item, dict) and "episode" in item]
    history.sort(key=lambda item: int(item["episode"]))
    return history


def get_evaluated_episodes(output_dir: str | Path) -> set[int]:
    detail_history = load_evaluation_detail_history(output_dir)
    if detail_history:
        return {int(item["episode"]) for item in detail_history}
    return {int(item["episode"]) for item in load_evaluation_history(output_dir)}


def save_evaluation_history_csv(history: list[dict[str, object]], output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "evaluation_history.csv"
    fieldnames = [
        "episode",
        "split",
        "evaluated_maps",
        "explored_rate",
        "explored_rate_std",
        "success_rate",
        "success_rate_std",
        "travel_dist",
        "travel_dist_std",
        "steps_taken",
        "steps_taken_std",
        "completion_steps",
        "completion_steps_std",
        "completion_travel_dist",
        "completion_travel_dist_std",
        "normalized_exploration_efficiency",
        "exploration_auc",
        "best_explored_rate",
        "worst_explored_rate",
        "manifest_hash",
        "detail_path",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in history:
            writer.writerow({key: item.get(key) for key in fieldnames})
    return csv_path


def save_per_map_results_csv(detail_history: list[dict[str, object]], output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "per_map_results.csv"
    fieldnames = [
        "episode",
        "split",
        "map_slot",
        "map_label",
        "map_name",
        "map_path",
        "size_bucket",
        "free_area",
        "free_ratio",
        "explored_rate",
        "success",
        "travel_dist",
        "steps_taken",
        "completion_steps",
        "completion_travel_dist",
        "normalized_exploration_efficiency",
        "exploration_auc",
        "episode_return",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in detail_history:
            summary = item.get("summary", {}) if isinstance(item.get("summary"), dict) else {}
            split = summary.get("split") or item.get("split")
            for result in item.get("results", []):
                if not isinstance(result, dict):
                    continue
                row = {key: result.get(key) for key in fieldnames}
                row["episode"] = item.get("episode")
                row["split"] = result.get("split") or split
                writer.writerow(row)
    return csv_path


def _metric_value(result: dict[str, object], metric_name: str) -> float:
    value = result.get(metric_name)
    if value is None:
        return float("nan")
    return float(value)


def save_per_map_metric_plots(
    detail_history: list[dict[str, object]],
    output_dir: str | Path,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = np.asarray([int(item["episode"]) for item in detail_history], dtype=float)
    map_numbers = sorted(
        {
            int(result["map_slot"])
            for item in detail_history
            for result in item.get("results", [])
            if isinstance(result, dict) and "map_slot" in result
        }
    )
    plot_paths: dict[str, Path] = {}
    color_map = matplotlib.colormaps.get_cmap("tab10")

    for metric_name, metric_title in CORE_EVAL_METRICS:
        metric_dir = output_dir / metric_name
        metric_dir.mkdir(parents=True, exist_ok=True)
        plot_path = metric_dir / f"{metric_name}_history.png"

        fig, ax = plt.subplots(figsize=(10, 5))
        has_data = False
        for color_index, map_number in enumerate(map_numbers):
            values = []
            for item in detail_history:
                result_by_slot = {
                    int(result["map_slot"]): result
                    for result in item.get("results", [])
                    if isinstance(result, dict) and "map_slot" in result
                }
                result = result_by_slot.get(map_number)
                values.append(_metric_value(result, metric_name) if result is not None else float("nan"))

            values_array = np.asarray(values, dtype=float)
            finite_mask = np.isfinite(values_array)
            if not np.any(finite_mask):
                continue
            has_data = True
            ax.plot(
                episodes[finite_mask],
                values_array[finite_mask],
                marker="o",
                linewidth=2.0,
                label=get_eval_map_name(map_number),
                color=color_map(color_index / max(len(map_numbers) - 1, 1)),
            )

        if has_data:
            ax.legend(loc="best", fontsize=8)
            if metric_name.endswith("_rate"):
                ax.set_ylim(-0.05, 1.05)
        else:
            ax.text(0.5, 0.5, "No evaluation data yet", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(metric_title)
        ax.set_xlabel("Train Episode")
        ax.set_ylabel(metric_title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plot_paths[metric_name] = plot_path

    return plot_paths


def save_evaluation_summary(
    output_dir: str | Path,
    episode_number: int,
    results: list[dict[str, object]],
    summary: dict[str, object],
    bucket_size: int,
    protocol: dict[str, object] | None = None,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    normalized_results = [
        _normalize_result_entry(result, index)
        for index, result in enumerate(results, start=1)
        if isinstance(result, dict)
    ]
    protocol = dict(protocol or {})
    if "split" in summary and "split" not in protocol:
        protocol["split"] = summary["split"]
    if "manifest_hash" in summary and "manifest_hash" not in protocol:
        protocol["manifest_hash"] = summary["manifest_hash"]

    raw_dir = get_eval_raw_dir(output_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    detail_dir = ensure_episode_bucket_dir(raw_dir, episode_number, bucket_size)
    detail_path = detail_dir / f"eval_episode_{int(episode_number):05d}.json"
    detail_payload = {
        "episode": int(episode_number),
        "protocol": protocol,
        "summary": summary,
        "results": normalized_results,
    }
    detail_path.write_text(json.dumps(detail_payload, indent=2))

    history = load_evaluation_history(output_dir)
    history_entry = {"episode": int(episode_number), **summary, "detail_path": str(detail_path.resolve())}
    history = [item for item in history if int(item["episode"]) != int(episode_number)]
    history.append(history_entry)
    history.sort(key=lambda item: int(item["episode"]))

    history_path = output_dir / "evaluation_history.json"
    history_path.write_text(json.dumps({"protocol": protocol, "evaluations": history}, indent=2))

    maps_manifest_path = output_dir / "fixed_eval_maps.txt"
    maps_manifest_lines = [
        f"{result['map_label']}\t{result.get('split', summary.get('split', 'unknown'))}\t{result['map_name']}\t{result['map_path']}\n"
        for result in normalized_results
    ]
    maps_manifest_path.write_text("".join(maps_manifest_lines))

    csv_path = save_evaluation_history_csv(history, output_dir)
    detail_history = load_evaluation_detail_history(output_dir)
    per_map_csv_path = save_per_map_results_csv(detail_history, output_dir)
    metric_plot_paths = save_per_map_metric_plots(detail_history, output_dir)
    legacy_plot_path = output_dir / "evaluation_metrics.png"
    if legacy_plot_path.exists():
        legacy_plot_path.unlink()
    return {
        "detail_path": detail_path,
        "history_path": history_path,
        "history_csv_path": csv_path,
        "per_map_csv_path": per_map_csv_path,
        "maps_manifest_path": maps_manifest_path,
        "metric_plot_paths": metric_plot_paths,
    }


def evaluate_policy(
    policy_state_dict,
    output_config: RuntimeConfig,
    episode_number: int,
    device: str | torch.device = "cpu",
    greedy: bool | None = None,
    max_episode_step: int | None = None,
    split: str = "val",
    map_count: int | None = None,
) -> list[dict[str, object]]:
    split = normalize_split(split)
    eval_device = _normalize_eval_device(device)
    _prepare_eval_device(eval_device)
    greedy = output_config.auto_eval_greedy if greedy is None else bool(greedy)
    eval_maps = resolve_eval_maps(output_config, split=split, count=map_count)
    if not eval_maps:
        return []

    split_manifest = materialize_split_manifest(output_config)
    manifest_hash = split_manifest.content_hash()
    entry_by_path = manifest_entry_by_path(split_manifest)
    eval_gifs_root = get_result_test_gifs_path(output_config) if split == "test" else get_result_validation_gifs_path(output_config)
    eval_bucket_size = max(int(output_config.auto_eval_interval), 1)

    policy_net = PolicyNet(
        NODE_INPUT_DIM,
        EMBEDDING_DIM,
        use_lf_attention_hf_residual=output_config.use_lf_attention_hf_residual,
        use_privileged_wavelet_distillation=output_config.use_hpbg
        and output_config.use_privileged_wavelet_distillation,
        wavelet_scales=output_config.wavelet_scales,
        wavelet_fuse_dim=output_config.wavelet_fuse_dim,
        wavelet_lf_qk=output_config.wavelet_lf_qk,
        use_hierarchical_context=output_config.use_hpbg and output_config.use_hierarchical_graph,
    ).to(eval_device)
    policy_net.load_state_dict(_copy_state_dict_to_device(policy_state_dict, eval_device))
    policy_net.eval()

    results = []
    for map_index, map_path in enumerate(eval_maps, start=1):
        map_key = str(map_path.expanduser().resolve())
        map_entry = entry_by_path.get(map_key)
        map_dir = ensure_eval_map_dir(
            eval_gifs_root,
            episode_number,
            map_index,
            eval_bucket_size,
        )
        artifact_stem = (
            f"{split}_episode_{int(episode_number):05d}_{get_eval_map_name(map_index)}_{_sanitize_artifact_token(map_path.stem)}"
        )
        env = Env(
            episode_number,
            plot=True,
            gifs_dir=map_dir,
            forced_map_path=map_path,
            artifact_prefix=artifact_stem,
            runtime_config=output_config,
        )
        robot = Agent(policy_net, device=eval_device, plot=True, runtime_config=output_config)
        ground_truth_node_manager = GroundTruthNodeManager(
            robot.node_manager,
            env.ground_truth_info,
            device=eval_device,
            plot=True,
            runtime_config=output_config,
        )

        episode_return = 0.0
        done = False
        steps_taken = 0
        explored_trace = [float(env.explored_rate)]

        robot.update_planning_state(env.belief_info, env.robot_location)
        observation = robot.get_observation()
        ground_truth_node_manager.get_ground_truth_observation(env.robot_location, env.belief_info)

        robot.plot_env()
        ground_truth_node_manager.plot_ground_truth_env(env.robot_location)
        env.plot_env(0)

        if robot.utility.sum() == 0:
            done = True

        step_limit = output_config.max_episode_step if max_episode_step is None else max_episode_step
        for step in range(max(int(step_limit), 1)):
            if done:
                break

            next_location, _ = robot.select_next_waypoint(observation, greedy=greedy)
            reward = env.step(next_location)

            robot.update_planning_state(env.belief_info, env.robot_location)
            observation = robot.get_observation()
            ground_truth_node_manager.get_ground_truth_observation(env.robot_location, env.belief_info)
            steps_taken = step + 1

            if robot.utility.sum() == 0:
                done = True
                reward += 20
            episode_return += float(reward)
            explored_trace.append(float(env.explored_rate))

            robot.plot_env()
            ground_truth_node_manager.plot_ground_truth_env(env.robot_location)
            env.plot_env(steps_taken)

        gif_path = make_gif(map_dir, artifact_stem, env.frame_files, env.explored_rate)
        last_frame = env.frame_files[-1] if env.frame_files else None
        efficiency = float(env.explored_rate) / max(float(env.travel_dist), 1e-6)
        exploration_auc = float(np.trapz(np.asarray(explored_trace, dtype=float)) / max(len(explored_trace) - 1, 1))
        results.append(
            {
                "episode": int(episode_number),
                "split": split,
                "manifest_hash": manifest_hash,
                "map_slot": int(map_index),
                "map_label": get_eval_map_name(map_index),
                "map_name": map_path.name,
                "map_path": str(map_path.resolve()),
                "map_sha256": map_entry.sha256 if map_entry is not None else None,
                "width": int(map_entry.width) if map_entry is not None else None,
                "height": int(map_entry.height) if map_entry is not None else None,
                "free_area": int(map_entry.free_area) if map_entry is not None else None,
                "free_ratio": float(map_entry.free_ratio) if map_entry is not None else None,
                "size_bucket": map_entry.size_bucket if map_entry is not None else "unknown",
                "explored_rate": float(env.explored_rate),
                "travel_dist": float(env.travel_dist),
                "success": bool(done),
                "episode_return": float(episode_return),
                "steps_taken": int(steps_taken),
                "completion_steps": int(steps_taken) if done else None,
                "completion_travel_dist": float(env.travel_dist) if done else None,
                "normalized_exploration_efficiency": efficiency,
                "exploration_auc": exploration_auc,
                "explored_trace": explored_trace,
                "step_budget": int(step_limit),
                "gif_path": str(Path(gif_path).resolve()) if gif_path is not None else None,
                "last_frame_path": str(Path(last_frame).resolve()) if last_frame is not None else None,
            }
        )
    return results


def _mean_std(values: list[float] | np.ndarray) -> tuple[float | None, float | None]:
    if len(values) == 0:
        return None, None
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return None, None
    return float(np.mean(finite)), float(np.std(finite))


def _bucket_summary(results: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for result in results:
        grouped.setdefault(str(result.get("size_bucket") or "unknown"), []).append(result)

    summary: dict[str, dict[str, object]] = {}
    for bucket, items in grouped.items():
        explored = [float(item["explored_rate"]) for item in items]
        success = [float(bool(item["success"])) for item in items]
        travel = [float(item["travel_dist"]) for item in items]
        efficiency = [float(item.get("normalized_exploration_efficiency", 0.0)) for item in items]
        auc = [float(item.get("exploration_auc", 0.0)) for item in items]
        explored_mean, explored_std = _mean_std(explored)
        success_mean, success_std = _mean_std(success)
        travel_mean, travel_std = _mean_std(travel)
        efficiency_mean, efficiency_std = _mean_std(efficiency)
        auc_mean, auc_std = _mean_std(auc)
        summary[bucket] = {
            "maps": len(items),
            "explored_rate": explored_mean,
            "explored_rate_std": explored_std,
            "success_rate": success_mean,
            "success_rate_std": success_std,
            "travel_dist": travel_mean,
            "travel_dist_std": travel_std,
            "normalized_exploration_efficiency": efficiency_mean,
            "normalized_exploration_efficiency_std": efficiency_std,
            "exploration_auc": auc_mean,
            "exploration_auc_std": auc_std,
        }
    return summary


def summarize_eval_results(results: list[dict[str, object]]) -> dict[str, object]:
    if not results:
        return {
            "split": "unknown",
            "manifest_hash": None,
            "evaluated_maps": 0,
            "explored_rate": 0.0,
            "explored_rate_std": 0.0,
            "travel_dist": 0.0,
            "travel_dist_std": 0.0,
            "success_rate": 0.0,
            "success_rate_std": 0.0,
            "episode_return": 0.0,
            "episode_return_std": 0.0,
            "steps_taken": 0.0,
            "steps_taken_std": 0.0,
            "completion_steps": None,
            "completion_steps_std": None,
            "completion_travel_dist": None,
            "completion_travel_dist_std": None,
            "normalized_exploration_efficiency": 0.0,
            "normalized_exploration_efficiency_std": 0.0,
            "exploration_auc": 0.0,
            "exploration_auc_std": 0.0,
            "best_explored_rate": 0.0,
            "worst_explored_rate": 0.0,
            "by_size_bucket": {},
        }

    explored = np.array([float(result["explored_rate"]) for result in results], dtype=float)
    travel = np.array([float(result["travel_dist"]) for result in results], dtype=float)
    success = np.array([float(bool(result["success"])) for result in results], dtype=float)
    episode_return = np.array([float(result["episode_return"]) for result in results], dtype=float)
    steps = np.array([float(result["steps_taken"]) for result in results], dtype=float)
    efficiency = np.array(
        [float(result.get("normalized_exploration_efficiency", 0.0)) for result in results],
        dtype=float,
    )
    exploration_auc = np.array([float(result.get("exploration_auc", 0.0)) for result in results], dtype=float)
    completion_steps = [float(result["completion_steps"]) for result in results if result["completion_steps"] is not None]
    completion_travel = [
        float(result["completion_travel_dist"])
        for result in results
        if result["completion_travel_dist"] is not None
    ]
    completion_steps_mean, completion_steps_std = _mean_std(completion_steps)
    completion_travel_mean, completion_travel_std = _mean_std(completion_travel)
    split_values = sorted({str(result.get("split") or "unknown") for result in results})
    manifest_values = sorted({str(result.get("manifest_hash")) for result in results if result.get("manifest_hash")})
    return {
        "split": split_values[0] if len(split_values) == 1 else ",".join(split_values),
        "manifest_hash": manifest_values[0] if len(manifest_values) == 1 else ",".join(manifest_values),
        "evaluated_maps": len(results),
        "explored_rate": float(np.mean(explored)),
        "explored_rate_std": float(np.std(explored)),
        "travel_dist": float(np.mean(travel)),
        "travel_dist_std": float(np.std(travel)),
        "success_rate": float(np.mean(success)),
        "success_rate_std": float(np.std(success)),
        "episode_return": float(np.mean(episode_return)),
        "episode_return_std": float(np.std(episode_return)),
        "steps_taken": float(np.mean(steps)),
        "steps_taken_std": float(np.std(steps)),
        "completion_steps": completion_steps_mean,
        "completion_steps_std": completion_steps_std,
        "completion_travel_dist": completion_travel_mean,
        "completion_travel_dist_std": completion_travel_std,
        "normalized_exploration_efficiency": float(np.mean(efficiency)),
        "normalized_exploration_efficiency_std": float(np.std(efficiency)),
        "exploration_auc": float(np.mean(exploration_auc)),
        "exploration_auc_std": float(np.std(exploration_auc)),
        "best_explored_rate": float(np.max(explored)),
        "worst_explored_rate": float(np.min(explored)),
        "by_size_bucket": _bucket_summary(results),
    }
