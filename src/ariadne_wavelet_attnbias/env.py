from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy  # kept for non-array uses
from fnmatch import fnmatch
from skimage import io
from skimage.measure import block_reduce

from pathlib import Path

from . import parameter
from .parameter import CELL_SIZE, FREE, FRONTIER_CELL_SIZE, SENSOR_RANGE, UNKNOWN, UPDATING_MAP_SIZE, RuntimeConfig
from .sensor import sensor_work
from .utils import MapInfo, get_frontier_in_map


def list_map_files(map_dir: str | Path) -> list[Path]:
    directory = Path(map_dir)
    if not directory.exists():
        raise FileNotFoundError(
            f"MAPS_DIR does not exist: {directory}. "
            "Place training maps under ariadne_wavelet_attnbias/maps or set ARIADNE_MAPS_DIR."
        )
    return sorted(path for path in directory.iterdir() if path.is_file())


def resolve_curriculum_map_files(
    maps_dir: str | Path,
    episode_index: int,
    rl_options: parameter.RLOptions,
    curriculum_override: bool | None = None,
) -> tuple[list[Path], str | None, int | None]:
    maps_dir = Path(maps_dir)
    base_map_files = list_map_files(maps_dir)
    use_curriculum = rl_options.use_curriculum if curriculum_override is None else bool(curriculum_override)
    if not use_curriculum:
        return base_map_files, None, None

    level = parameter.get_curriculum_level(episode_index, rl_options)
    level_index = parameter.get_curriculum_level_index(episode_index, rl_options)

    if rl_options.curriculum_mode == "dir":
        bucket_name = rl_options.curriculum_dirs_map().get(level, level)
        bucket_dir = maps_dir / bucket_name
        bucket_files = list_map_files(bucket_dir) if bucket_dir.exists() else []
        if bucket_files:
            return bucket_files, level, level_index
        return base_map_files, level, level_index

    patterns = rl_options.curriculum_patterns_map().get(level, ())
    matched_files = []
    seen_paths = set()
    for path in base_map_files:
        if any(fnmatch(path.name, pattern) for pattern in patterns):
            resolved = path.resolve()
            if resolved not in seen_paths:
                seen_paths.add(resolved)
                matched_files.append(path)
    if matched_files:
        return matched_files, level, level_index
    return base_map_files, level, level_index


def select_map_path_for_episode(
    maps_dir: str | Path,
    episode_index: int,
    rl_options: parameter.RLOptions,
    curriculum_override: bool | None = None,
) -> tuple[Path, str | None, int | None]:
    map_files, curriculum_level, curriculum_level_index = resolve_curriculum_map_files(
        maps_dir,
        episode_index,
        rl_options,
        curriculum_override=curriculum_override,
    )
    if not map_files:
        raise FileNotFoundError(
            f"No map files found in MAPS_DIR: {maps_dir}. "
            "Place training maps under ariadne_wavelet_attnbias/maps or set ARIADNE_MAPS_DIR."
        )
    map_index = int(episode_index) % len(map_files)
    return map_files[map_index], curriculum_level, curriculum_level_index


class Env:
    def __init__(
        self,
        episode_index,
        plot=False,
        output_dir=None,
        artifact_stem=None,
        runtime_config: RuntimeConfig | None = None,
        curriculum_override: bool | None = None,
        forced_map_path: str | Path | None = None,
    ):
        self.episode_index = episode_index
        self.plot = plot
        self.runtime_config = runtime_config or RuntimeConfig()
        self.rl_options = self.runtime_config.rl_options
        self.curriculum_override = curriculum_override
        self.forced_map_path = Path(forced_map_path).resolve() if forced_map_path is not None else None
        self.maps_dir = parameter.MAPS_DIR
        self.curriculum_level = None
        self.curriculum_level_index = None
        self.last_reward_components: dict[str, float] = {}
        self.ground_truth, self.robot_cell = self.import_ground_truth(episode_index)
        self.ground_truth_size = np.shape(self.ground_truth)
        self.cell_size = CELL_SIZE

        self.robot_location = np.array([0.0, 0.0])
        self.robot_belief = np.ones(self.ground_truth_size) * UNKNOWN
        self.belief_origin_x = -np.round(self.robot_cell[0] * self.cell_size, 1)
        self.belief_origin_y = -np.round(self.robot_cell[1] * self.cell_size, 1)

        self.global_frontiers = set()
        self.sensor_range = SENSOR_RANGE
        self.travel_dist = 0
        self.explored_rate = 0

        self.robot_belief = sensor_work(
            self.robot_cell,
            self.sensor_range / self.cell_size,
            self.robot_belief,
            self.ground_truth,
        )
        self.old_belief = self.robot_belief.copy()
        self.belief_info = MapInfo(self.robot_belief, self.belief_origin_x, self.belief_origin_y, self.cell_size)
        self.ground_truth_info = MapInfo(
            self.ground_truth,
            self.belief_origin_x,
            self.belief_origin_y,
            self.cell_size,
        )

        if self.plot:
            self.output_dir = Path(output_dir) if output_dir is not None else None
            self.artifact_stem = artifact_stem or f"episode_{episode_index:06d}"
            self.frame_files = []
            self.trajectory_x = [self.robot_location[0]]
            self.trajectory_y = [self.robot_location[1]]

    def import_ground_truth(self, episode_index):
        if self.forced_map_path is not None:
            map_path = self.forced_map_path
            curriculum_level = None
            curriculum_level_index = None
        else:
            map_path, curriculum_level, curriculum_level_index = select_map_path_for_episode(
                self.maps_dir,
                episode_index,
                self.rl_options,
                curriculum_override=self.curriculum_override,
            )
        self.curriculum_level = curriculum_level
        self.curriculum_level_index = curriculum_level_index
        ground_truth = (io.imread(map_path, as_gray=True) * 255).astype(int)
        ground_truth = block_reduce(ground_truth, 2, np.min)

        robot_cell = np.nonzero(ground_truth == 208)
        robot_cell = np.array([np.array(robot_cell)[1, 10], np.array(robot_cell)[0, 10]])

        ground_truth = (ground_truth > 150) | ((ground_truth <= 80) & (ground_truth >= 50))
        ground_truth = ground_truth * (FREE - 1) + 1
        return ground_truth, robot_cell

    def update_robot_location(self, robot_location):
        self.robot_location = robot_location
        self.robot_cell = np.array(
            [
                round((robot_location[0] - self.belief_origin_x) / self.cell_size),
                round((robot_location[1] - self.belief_origin_y) / self.cell_size),
            ]
        )
        if self.plot:
            self.trajectory_x.append(self.robot_location[0])
            self.trajectory_y.append(self.robot_location[1])

    def update_robot_belief(self):
        self.robot_belief = sensor_work(
            self.robot_cell,
            round(self.sensor_range / self.cell_size),
            self.robot_belief,
            self.ground_truth,
        )

    def calculate_reward(self, dist):
        r_dist = -dist / UPDATING_MAP_SIZE * 5

        global_frontiers = get_frontier_in_map(self.belief_info)
        if len(global_frontiers) == 0:
            delta_num = len(self.global_frontiers)
        else:
            observed_frontiers = self.global_frontiers - global_frontiers
            delta_num = len(observed_frontiers)

        r_info = delta_num / (SENSOR_RANGE * 3.14 // FRONTIER_CELL_SIZE)
        r_safe = 0.0

        if self.rl_options.use_reward_decomposition:
            reward = (
                self.rl_options.r_info_w * r_info
                + self.rl_options.r_dist_w * r_dist
                + self.rl_options.r_safe_w * r_safe
            )
            self.last_reward_components = {
                "r_info": float(self.rl_options.r_info_w * r_info),
                "r_dist": float(self.rl_options.r_dist_w * r_dist),
                "r_safe": float(self.rl_options.r_safe_w * r_safe),
                "r_terminal": 0.0,
                "total": float(reward),
            }
        else:
            reward = r_dist + r_info
            self.last_reward_components = {}

        self.global_frontiers = global_frontiers
        self.old_belief = self.robot_belief.copy()
        return reward

    def apply_terminal_bonus(self, reward: float) -> float:
        reward += self.rl_options.r_terminal_bonus
        if self.rl_options.use_reward_decomposition:
            self.last_reward_components.setdefault("r_info", 0.0)
            self.last_reward_components.setdefault("r_dist", 0.0)
            self.last_reward_components.setdefault("r_safe", 0.0)
            self.last_reward_components["r_terminal"] = (
                self.last_reward_components.get("r_terminal", 0.0) + float(self.rl_options.r_terminal_bonus)
            )
            self.last_reward_components["total"] = self.last_reward_components.get("total", 0.0) + float(
                self.rl_options.r_terminal_bonus
            )
        return reward

    def evaluate_exploration_rate(self):
        self.explored_rate = np.sum(self.robot_belief == FREE) / np.sum(self.ground_truth == FREE)

    def step(self, next_waypoint):
        dist = np.linalg.norm(self.robot_location - next_waypoint)
        self.update_robot_location(next_waypoint)
        self.update_robot_belief()
        self.travel_dist += dist
        self.evaluate_exploration_rate()
        return self.calculate_reward(dist)

    def plot_env(self, step):
        if self.output_dir is None:
            raise ValueError("output_dir must be set when plot=True")
        plt.subplot(1, 3, 1)
        plt.imshow(self.robot_belief, cmap="gray")
        plt.axis("off")
        plt.plot(
            (self.robot_location[0] - self.belief_origin_x) / self.cell_size,
            (self.robot_location[1] - self.belief_origin_y) / self.cell_size,
            "mo",
            markersize=4,
            zorder=5,
        )
        plt.plot(
            (np.array(self.trajectory_x) - self.belief_origin_x) / self.cell_size,
            (np.array(self.trajectory_y) - self.belief_origin_y) / self.cell_size,
            "b",
            linewidth=2,
            zorder=1,
        )
        plt.suptitle(
            f"Explored ratio: {self.explored_rate:.4g}  Travel distance: {self.travel_dist:.4g}"
        )
        plt.tight_layout()
        frame_dir = self.output_dir / f".{self.artifact_stem}_frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        frame_path = frame_dir / f"{self.artifact_stem}_{step:03d}.png"
        plt.savefig(frame_path, dpi=150)
        plt.close()
        self.frame_files.append(str(frame_path))
