from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import parameter as parameter_module
from agent import Agent
from benchmark_metrics import SUMMARY_BENCHMARK_FIELDS, compute_benchmark_metrics
from env import Env, list_map_files
from ground_truth_node_manager import GroundTruthNodeManager
from model import PolicyNet
from parameter import (
    EMBEDDING_DIM,
    MAX_EPISODE_STEP,
    NODE_INPUT_DIM,
    RuntimeConfig,
    get_result_eval_gifs_path,
    get_result_eval_path,
)
from runtime_utils import derive_episode_seed, set_global_seeds
from utils import ensure_episode_bucket_dir, make_gif


CORE_EVAL_METRICS = (
    ("explored_rate", "Explored Rate"),
    ("success_rate", "Success Rate"),
    ("completion_steps", "Completion Steps"),
    ("completion_travel_dist", "Completion Travel Distance"),
    ("distance_efficiency", "Distance Efficiency"),
    ("time_efficiency", "Time Efficiency"),
    ("mean_planning_time_ms", "Mean Planning Time (ms)"),
)

SUMMARY_MEAN_FIELDS = (
    "explored_rate",
    "travel_dist",
    "success_rate",
    "episode_return",
    "steps_taken",
    *SUMMARY_BENCHMARK_FIELDS,
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
    torch.empty(1, device=device)


def _sanitize_artifact_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return token.strip("_") or "map"


def get_eval_map_name(map_number: int) -> str:
    return f"map_{max(int(map_number), 1)}"


def ensure_eval_map_dir(
    base_dir: str | Path, episode_number: int, map_number: int, bucket_size: int
) -> Path:
    episode_dir = ensure_episode_bucket_dir(base_dir, episode_number, bucket_size)
    map_dir = episode_dir / get_eval_map_name(map_number)
    map_dir.mkdir(parents=True, exist_ok=True)
    return map_dir


def get_eval_raw_dir(output_dir: str | Path) -> Path:
    return Path(output_dir) / "raw"


def resolve_fixed_eval_maps(
    eval_map_count: int, maps_dir: str | Path | None = None
) -> list[Path]:
    eval_map_count = max(int(eval_map_count), 0)
    if eval_map_count <= 0:
        return []

    target_maps_dir = maps_dir if maps_dir is not None else parameter_module.MAPS_DIR
    map_files = list_map_files(target_maps_dir)
    return map_files[: min(eval_map_count, len(map_files))]


def _normalize_result_entry(
    result: dict[str, object], fallback_index: int
) -> dict[str, object]:
    normalized = dict(result)
    map_number = max(int(result.get("map_slot", fallback_index)), 1)
    normalized["map_slot"] = map_number
    normalized["map_label"] = str(
        result.get("map_label") or get_eval_map_name(map_number)
    )
    return normalized


def _normalize_detail_payload(
    payload: dict[str, object], detail_path: Path
) -> dict[str, object] | None:
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
    search_roots = [output_dir, get_eval_raw_dir(output_dir)]
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


def save_evaluation_history_csv(
    history: list[dict[str, object]], output_dir: str | Path
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "evaluation_history.csv"
    fieldnames = [
        "episode",
        "evaluated_maps",
        "explored_rate",
        "success_rate",
        "travel_dist",
        "steps_taken",
        "completion_steps",
        "completion_travel_dist",
        "episode_return",
        *SUMMARY_BENCHMARK_FIELDS,
        "best_explored_rate",
        "worst_explored_rate",
        "detail_path",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in history:
            writer.writerow({key: item.get(key) for key in fieldnames})
    return csv_path


def _metric_value(result: dict[str, object], metric_name: str) -> float:
    if result is None:
        return float("nan")
    value = result.get(metric_name)
    if value is None:
        return float("nan")
    return float(value)


def save_per_map_metric_plots(
    detail_history: list[dict[str, object]], output_dir: str | Path
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = np.asarray(
        [int(item["episode"]) for item in detail_history], dtype=float
    )
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
                values.append(
                    _metric_value(result_by_slot.get(map_number), metric_name)
                )

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
            ax.text(
                0.5,
                0.5,
                "No evaluation data yet",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
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
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    normalized_results = [
        _normalize_result_entry(result, index)
        for index, result in enumerate(results, start=1)
        if isinstance(result, dict)
    ]

    raw_dir = get_eval_raw_dir(output_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    detail_dir = ensure_episode_bucket_dir(raw_dir, episode_number, bucket_size)
    detail_path = detail_dir / f"eval_episode_{int(episode_number):05d}.json"
    detail_payload = {
        "episode": int(episode_number),
        "summary": summary,
        "results": normalized_results,
    }
    detail_path.write_text(json.dumps(detail_payload, indent=2))

    history = load_evaluation_history(output_dir)
    history_entry = {
        "episode": int(episode_number),
        **summary,
        "detail_path": str(detail_path.resolve()),
    }
    history = [item for item in history if int(item["episode"]) != int(episode_number)]
    history.append(history_entry)
    history.sort(key=lambda item: int(item["episode"]))

    history_path = output_dir / "evaluation_history.json"
    history_path.write_text(json.dumps({"evaluations": history}, indent=2))

    maps_manifest_path = output_dir / "fixed_eval_maps.txt"
    maps_manifest_lines = [
        f"{result['map_label']}\t{result['map_name']}\t{result['map_path']}\n"
        for result in normalized_results
    ]
    maps_manifest_path.write_text("".join(maps_manifest_lines))

    csv_path = save_evaluation_history_csv(history, output_dir)
    detail_history = load_evaluation_detail_history(output_dir)
    metric_plot_paths = save_per_map_metric_plots(detail_history, output_dir)
    legacy_plot_path = output_dir / "evaluation_metrics.png"
    if legacy_plot_path.exists():
        legacy_plot_path.unlink()
    return {
        "detail_path": detail_path,
        "history_path": history_path,
        "history_csv_path": csv_path,
        "maps_manifest_path": maps_manifest_path,
        "metric_plot_paths": metric_plot_paths,
    }


def evaluate_policy(
    policy_state_dict,
    output_config: RuntimeConfig,
    episode_number: int,
    device: str | torch.device = "cpu",
    greedy: bool | None = None,
    max_episode_step: int = MAX_EPISODE_STEP,
) -> list[dict[str, object]]:
    eval_device = _normalize_eval_device(device)
    _prepare_eval_device(eval_device)
    greedy = output_config.auto_eval_greedy if greedy is None else bool(greedy)
    eval_maps = resolve_fixed_eval_maps(output_config.auto_eval_map_count)
    if not eval_maps:
        return []

    eval_gifs_root = get_result_eval_gifs_path(output_config)
    eval_bucket_size = max(int(output_config.auto_eval_interval), 1)

    policy_net = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM).to(eval_device)
    policy_net.load_state_dict(
        _copy_state_dict_to_device(policy_state_dict, eval_device)
    )
    policy_net.eval()

    results = []
    for map_index, map_path in enumerate(eval_maps, start=1):
        eval_seed = derive_episode_seed(
            output_config.seed,
            episode_number,
            meta_agent_id=map_index,
            offset=output_config.eval_seed_offset,
        )
        set_global_seeds(eval_seed)
        map_dir = ensure_eval_map_dir(
            eval_gifs_root, episode_number, map_index, eval_bucket_size
        )
        artifact_stem = f"eval_episode_{int(episode_number):05d}_{get_eval_map_name(map_index)}_{_sanitize_artifact_token(map_path.stem)}"
        env = Env(
            episode_number,
            plot=True,
            gifs_dir=map_dir,
            forced_map_path=map_path,
            artifact_prefix=artifact_stem,
        )
        robot = Agent(policy_net, device=eval_device, plot=True)
        ground_truth_node_manager = GroundTruthNodeManager(
            robot.node_manager,
            env.ground_truth_info,
            device=eval_device,
            plot=True,
        )

        episode_return = 0.0
        done = False
        steps_taken = 0
        timing_totals = {
            "graph_update_time_sec": 0.0,
            "observation_time_sec": 0.0,
            "policy_inference_time_sec": 0.0,
            "env_step_time_sec": 0.0,
            "node_manager_update_time_sec": 0.0,
            "dense_graph_build_time_sec": 0.0,
            "graph_rarefaction_time_sec": 0.0,
        }

        planning_start = time.perf_counter()
        robot.update_planning_state(env.belief_info, env.robot_location)
        initial_graph_time = time.perf_counter() - planning_start
        timing_totals["graph_update_time_sec"] += float(
            robot.last_planning_profile.get("planning_total_sec", initial_graph_time)
        )
        timing_totals["node_manager_update_time_sec"] += float(
            robot.last_planning_profile.get("node_manager_update_sec", 0.0)
        )
        timing_totals["dense_graph_build_time_sec"] += float(
            robot.last_planning_profile.get("dense_graph_build_sec", 0.0)
        )
        timing_totals["graph_rarefaction_time_sec"] += float(
            robot.last_planning_profile.get("graph_rarefaction_sec", 0.0)
        )
        observation_start = time.perf_counter()
        observation = robot.get_observation()
        timing_totals["observation_time_sec"] += time.perf_counter() - observation_start
        ground_truth_node_manager.get_ground_truth_observation(env.robot_location)

        robot.plot_env()
        ground_truth_node_manager.plot_ground_truth_env(env.robot_location)
        env.plot_env(0)

        if robot.utility.sum() == 0:
            done = True

        for step in range(max(int(max_episode_step), 1)):
            if done:
                break

            policy_start = time.perf_counter()
            next_location, _ = robot.select_next_waypoint(observation, greedy=greedy)
            timing_totals["policy_inference_time_sec"] += (
                time.perf_counter() - policy_start
            )
            env_step_start = time.perf_counter()
            reward = env.step(next_location)
            timing_totals["env_step_time_sec"] += time.perf_counter() - env_step_start

            planning_start = time.perf_counter()
            robot.update_planning_state(env.belief_info, env.robot_location)
            graph_update_time = time.perf_counter() - planning_start
            timing_totals["graph_update_time_sec"] += float(
                robot.last_planning_profile.get("planning_total_sec", graph_update_time)
            )
            timing_totals["node_manager_update_time_sec"] += float(
                robot.last_planning_profile.get("node_manager_update_sec", 0.0)
            )
            timing_totals["dense_graph_build_time_sec"] += float(
                robot.last_planning_profile.get("dense_graph_build_sec", 0.0)
            )
            timing_totals["graph_rarefaction_time_sec"] += float(
                robot.last_planning_profile.get("graph_rarefaction_sec", 0.0)
            )
            observation_start = time.perf_counter()
            observation = robot.get_observation()
            timing_totals["observation_time_sec"] += (
                time.perf_counter() - observation_start
            )
            ground_truth_node_manager.get_ground_truth_observation(env.robot_location)
            steps_taken = step + 1

            if robot.utility.sum() == 0:
                done = True
                reward += 20
            episode_return += float(reward)

            robot.plot_env()
            ground_truth_node_manager.plot_ground_truth_env(env.robot_location)
            env.plot_env(steps_taken)

        gif_path = make_gif(map_dir, artifact_stem, env.frame_files, env.explored_rate)
        last_frame = env.frame_files[-1] if env.frame_files else None
        benchmark_metrics = compute_benchmark_metrics(
            explored_rate=env.explored_rate,
            travel_dist=env.travel_dist,
            steps_taken=steps_taken,
            total_free_cells=int(np.sum(env.ground_truth == parameter_module.FREE)),
            cell_size=float(env.cell_size),
            episode_wall_time_sec=sum(
                timing_totals[key]
                for key in (
                    "graph_update_time_sec",
                    "observation_time_sec",
                    "policy_inference_time_sec",
                    "env_step_time_sec",
                )
            ),
            graph_update_time_sec=timing_totals["graph_update_time_sec"],
            observation_time_sec=timing_totals["observation_time_sec"],
            policy_inference_time_sec=timing_totals["policy_inference_time_sec"],
            env_step_time_sec=timing_totals["env_step_time_sec"],
            node_manager_update_time_sec=timing_totals["node_manager_update_time_sec"],
            dense_graph_build_time_sec=timing_totals["dense_graph_build_time_sec"],
            graph_rarefaction_time_sec=timing_totals["graph_rarefaction_time_sec"],
        )
        results.append(
            {
                "episode": int(episode_number),
                "map_slot": int(map_index),
                "map_label": get_eval_map_name(map_index),
                "map_name": map_path.name,
                "map_path": str(map_path.resolve()),
                "eval_seed": int(eval_seed),
                "explored_rate": float(env.explored_rate),
                "travel_dist": float(env.travel_dist),
                "success": bool(done),
                "success_rate": float(bool(done)),
                "episode_return": float(episode_return),
                "steps_taken": int(steps_taken),
                "completion_steps": int(steps_taken) if done else None,
                "completion_travel_dist": float(env.travel_dist) if done else None,
                **benchmark_metrics,
                "gif_path": (
                    str(Path(gif_path).resolve()) if gif_path is not None else None
                ),
                "last_frame_path": (
                    str(Path(last_frame).resolve()) if last_frame is not None else None
                ),
            }
        )
    return results


def summarize_eval_results(results: list[dict[str, object]]) -> dict[str, object]:
    if not results:
        return {
            "evaluated_maps": 0,
            "explored_rate": 0.0,
            "travel_dist": 0.0,
            "success_rate": 0.0,
            "episode_return": 0.0,
            "steps_taken": 0.0,
            "completion_steps": None,
            "completion_travel_dist": None,
            **{field: 0.0 for field in SUMMARY_BENCHMARK_FIELDS},
            "best_explored_rate": 0.0,
            "worst_explored_rate": 0.0,
        }

    explored = np.array(
        [float(result["explored_rate"]) for result in results], dtype=float
    )
    completion_steps = [
        float(result["completion_steps"])
        for result in results
        if result["completion_steps"] is not None
    ]
    completion_travel = [
        float(result["completion_travel_dist"])
        for result in results
        if result["completion_travel_dist"] is not None
    ]
    summary = {
        "evaluated_maps": len(results),
        "completion_steps": (
            float(np.mean(completion_steps)) if completion_steps else None
        ),
        "completion_travel_dist": (
            float(np.mean(completion_travel)) if completion_travel else None
        ),
        "best_explored_rate": float(np.max(explored)),
        "worst_explored_rate": float(np.min(explored)),
    }
    for field in SUMMARY_MEAN_FIELDS:
        values = np.array(
            [
                float(result[field])
                for result in results
                if result.get(field) is not None
            ],
            dtype=float,
        )
        summary[field] = float(np.mean(values)) if values.size > 0 else 0.0
    return summary
