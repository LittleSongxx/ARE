from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .agent import InferenceAgent
from .env import Env, list_map_files
from .model import PolicyNet
from .parameter import (
    CLUSTER_RANGE,
    EMBEDDING_DIM,
    MAPS_DIR,
    MAX_EPISODE_STEP,
    NODE_INPUT_DIM,
    RuntimeConfig,
    ensure_result_dirs,
    get_gifs_path,
)
from .utils import (
    build_artifact_stem,
    ensure_bucket_dir,
    finalize_episode_artifacts,
    get_cell_position_from_coords,
)


def _copy_state_dict_to_device(state_dict, device):
    return {key: value.detach().to(device) for key, value in state_dict.items()}


def resolve_benchmark_eval_maps(output_config: RuntimeConfig) -> list[Path]:
    map_files = list_map_files(MAPS_DIR)
    benchmark_names = tuple(name for name in output_config.eval_benchmark_maps if str(name).strip())
    if benchmark_names:
        available = {path.name: path for path in map_files}
        missing = [name for name in benchmark_names if name not in available]
        if missing:
            raise FileNotFoundError(
                "Benchmark eval maps not found in ariadne_wavelet_attnbias/maps: "
                + ", ".join(missing)
            )
        return [available[name] for name in benchmark_names]
    return map_files[:1]


def save_evaluation_metrics_plot(results: list[dict[str, object]], output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "evaluation_metrics.png"

    metrics = [
        ("explored_rate", "Explored Rate"),
        ("travel_dist", "Travel Distance"),
        ("success", "Success"),
        ("episode_return", "Episode Return"),
        ("steps_taken", "Steps Taken"),
    ]
    episodes = [int(result["episode"]) for result in results]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(9, 3 * len(metrics)))
    axes = np.atleast_1d(axes)

    for axis, (key, title) in zip(axes, metrics):
        values = [float(result[key]) for result in results]
        if len(values) <= 1:
            axis.bar([episodes[0] if episodes else 0], values or [0.0], width=0.6, color="#2f6db3")
        else:
            axis.plot(episodes, values, marker="o", linewidth=2.0, color="#2f6db3")
        axis.set_title(title)
        axis.set_xlabel("Episode")
        axis.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def save_eval_frame(output_dir, artifact_stem, env, agent, episode_id, step, trajectory, frame_files):
    plt.switch_backend("agg")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    belief_ax, graph_ax = axes
    belief_ax.imshow(env.robot_belief, cmap="gray")
    belief_ax.axis("off")
    belief_ax.set_title("Belief + Trajectory")
    belief_xy = np.array(trajectory)
    if belief_xy.size:
        belief_ax.plot(
            (belief_xy[:, 0] - env.belief_origin_x) / env.cell_size,
            (belief_xy[:, 1] - env.belief_origin_y) / env.cell_size,
            "b",
            linewidth=2,
        )
    belief_ax.plot(
        (env.robot_location[0] - env.belief_origin_x) / env.cell_size,
        (env.robot_location[1] - env.belief_origin_y) / env.cell_size,
        "mo",
        markersize=5,
    )

    graph_ax.imshow(agent.map_info.map, cmap="gray")
    graph_ax.axis("off")
    graph_ax.set_title("Rarefied Graph + Clusters")
    key_nodes = get_cell_position_from_coords(agent.key_node_coords, agent.map_info).reshape(-1, 2)
    graph_ax.scatter(key_nodes[:, 0], key_nodes[:, 1], c=agent.key_utility, zorder=2)
    robot = get_cell_position_from_coords(env.robot_location, agent.map_info)
    graph_ax.plot(robot[0], robot[1], "mo", markersize=6, zorder=5)
    if len(agent.frontier) > 0:
        frontiers = get_cell_position_from_coords(np.array(list(agent.frontier)), agent.map_info).reshape(-1, 2)
        graph_ax.scatter(frontiers[:, 0], frontiers[:, 1], c="r", s=2)
    for coords in agent.key_node_coords:
        node = agent.node_manager.key_node_dict[(coords[0], coords[1])]
        for neighbor_coords in node.neighbor_set:
            end = (np.array(neighbor_coords) - coords) / 2 + coords
            graph_ax.plot(
                (np.array([coords[0], end[0]]) - agent.map_info.map_origin_x) / agent.map_info.cell_size,
                (np.array([coords[1], end[1]]) - agent.map_info.map_origin_y) / agent.map_info.cell_size,
                "tan",
                zorder=1,
            )

    cluster_centers = [value[0].coords for value in agent.node_manager.cluster_center_node_dict.values()]
    if cluster_centers:
        cluster_centers = np.array(cluster_centers).reshape(-1, 2)
        cluster_cells = get_cell_position_from_coords(cluster_centers, agent.map_info).reshape(-1, 2)
        graph_ax.scatter(
            cluster_cells[:, 0],
            cluster_cells[:, 1],
            s=160,
            facecolors="none",
            edgecolors="cyan",
            linewidths=1.5,
            zorder=4,
        )
        radius = CLUSTER_RANGE / agent.map_info.cell_size
        for center_cell in cluster_cells:
            graph_ax.add_patch(
                plt.Circle(
                    (center_cell[0], center_cell[1]),
                    radius=radius,
                    fill=False,
                    edgecolor="cyan",
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.7,
                    zorder=3,
                )
            )

    fig.suptitle(f"Episode {episode_id} Step {step}  explored={env.explored_rate:.4f} dist={env.travel_dist:.2f}")
    fig.tight_layout()
    frame_dir = output_dir / f".{artifact_stem}_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frame_path = frame_dir / f"{artifact_stem}_{step:03d}.png"
    fig.savefig(frame_path, dpi=150)
    plt.close(fig)
    frame_files.append(str(frame_path))


def evaluate_policy(
    policy_state_dict,
    output_config: RuntimeConfig | None = None,
    episodes: int = 1,
    start_episode: int = 0,
    greedy: bool = False,
    device: str | torch.device = "cpu",
    result_bucket_episodes: int = 100,
    max_episode_step: int = MAX_EPISODE_STEP,
):
    output_config = output_config or RuntimeConfig()
    ensure_result_dirs(output_config)
    eval_gifs_path = get_gifs_path(output_config)
    eval_device = torch.device(device)
    benchmark_maps = resolve_benchmark_eval_maps(output_config) if output_config.use_fixed_eval_maps else None

    policy_net = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM).to(eval_device)
    policy_net.load_state_dict(_copy_state_dict_to_device(policy_state_dict, eval_device))
    policy_net.eval()

    results = []
    for episode_offset in range(episodes):
        episode_id = start_episode + episode_offset
        artifact_stem = build_artifact_stem(episode_id, prefix="eval_episode")
        output_dir = ensure_bucket_dir(eval_gifs_path, episode_id, result_bucket_episodes)
        forced_map_path = None
        if benchmark_maps:
            forced_map_path = benchmark_maps[episode_offset % len(benchmark_maps)]
        env = Env(
            episode_id,
            plot=False,
            runtime_config=output_config,
            curriculum_override=output_config.rl_options.use_curriculum_in_eval,
            forced_map_path=forced_map_path,
        )
        agent = InferenceAgent(policy_net, device=eval_device, plot=False)
        trajectory = [env.robot_location.copy()]
        frame_files = []
        episode_return = 0.0
        steps_taken = 0
        success = False

        agent.update_planning_state(env.belief_info, env.robot_location)
        save_eval_frame(output_dir, artifact_stem, env, agent, episode_id, 0, trajectory, frame_files)

        if agent.utility.sum() == 0:
            success = True

        for step in range(1, max_episode_step + 1):
            observation = agent.get_observation(env.robot_location)
            next_location, _ = agent.select_next_waypoint(observation, greedy=greedy)
            reward = env.step(next_location)
            trajectory.append(env.robot_location.copy())
            agent.update_planning_state(env.belief_info, env.robot_location)
            steps_taken = step
            if agent.utility.sum() == 0:
                success = True
                reward = env.apply_terminal_bonus(reward)
            episode_return += float(reward)
            save_eval_frame(output_dir, artifact_stem, env, agent, episode_id, step, trajectory, frame_files)
            if success:
                break

        gif_path, png_path = finalize_episode_artifacts(output_dir, artifact_stem, frame_files)
        results.append(
            {
                "episode": episode_id,
                "explored_rate": env.explored_rate,
                "travel_dist": env.travel_dist,
                "success": success,
                "episode_return": episode_return,
                "steps_taken": steps_taken,
                "gif_path": str(Path(gif_path)),
                "png_path": str(Path(png_path)),
            }
        )
    return results


def summarize_eval_results(results: list[dict[str, object]]) -> dict[str, float | int]:
    if not results:
        return {
            "episodes": 0,
            "explored_rate": 0.0,
            "travel_dist": 0.0,
            "success_rate": 0.0,
            "episode_return": 0.0,
            "steps_taken": 0.0,
            "best_explored_rate": 0.0,
            "worst_explored_rate": 0.0,
        }

    explored = np.array([float(result["explored_rate"]) for result in results], dtype=float)
    travel = np.array([float(result["travel_dist"]) for result in results], dtype=float)
    success = np.array([float(result["success"]) for result in results], dtype=float)
    episode_return = np.array([float(result["episode_return"]) for result in results], dtype=float)
    steps = np.array([float(result["steps_taken"]) for result in results], dtype=float)
    return {
        "episodes": len(results),
        "explored_rate": float(np.mean(explored)),
        "travel_dist": float(np.mean(travel)),
        "success_rate": float(np.mean(success)),
        "episode_return": float(np.mean(episode_return)),
        "steps_taken": float(np.mean(steps)),
        "best_explored_rate": float(np.max(explored)),
        "worst_explored_rate": float(np.min(explored)),
    }
