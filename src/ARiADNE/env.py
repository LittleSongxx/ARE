from __future__ import annotations

import json
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy  # kept for non-array uses
from functools import lru_cache
from skimage import io
from skimage.measure import block_reduce

from pathlib import Path

from .parameter import (
    CELL_SIZE,
    FREE,
    FRONTIER_CELL_SIZE,
    MAPS_DIR,
    SENSOR_RANGE,
    UNKNOWN,
    UPDATING_MAP_SIZE,
    RuntimeConfig,
    get_curriculum_level,
    get_curriculum_level_index,
)
from .sensor import sensor_work
from .utils import MapInfo, get_frontier_in_map


def list_map_files(map_dir: str | Path) -> list[Path]:
    directory = Path(map_dir)
    if not directory.exists():
        raise FileNotFoundError(
            f"MAPS_DIR does not exist: {directory}. "
            "Place training maps under ARiADNE/maps or src/maps, or set ARIADNE_MAPS_DIR."
        )
    return sorted(path for path in directory.iterdir() if path.is_file())


@lru_cache(maxsize=8)
def _load_curriculum_manifest_bucket_map(manifest_path: str) -> dict[str, tuple[str, ...]]:
    manifest = Path(manifest_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"Curriculum manifest is missing a 'records' list: {manifest}")

    bucket_map: dict[str, set[str]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        bucket = str(record.get("difficulty_bucket", "")).strip()
        map_path = str(record.get("map_path", "")).strip()
        if not bucket or not map_path:
            continue
        resolved_path = Path(map_path).expanduser().resolve()
        if not resolved_path.is_file():
            continue
        bucket_map.setdefault(bucket, set()).add(str(resolved_path))

    return {
        bucket: tuple(sorted(paths, key=lambda value: (Path(value).name, value)))
        for bucket, paths in bucket_map.items()
    }


def _resolve_curriculum_source_paths(curriculum_source: str | Path) -> tuple[Path | None, Path | None]:
    source_path = Path(curriculum_source).expanduser().resolve()
    if source_path.is_file():
        if source_path.name != "difficulty_manifest.json":
            raise ValueError(
                "Curriculum source file must be difficulty_manifest.json, "
                f"got: {source_path.name}"
            )
        bucket_root = source_path.parent / "buckets"
        return source_path, bucket_root if bucket_root.is_dir() else None

    if not source_path.exists():
        raise FileNotFoundError(f"Curriculum source does not exist: {source_path}")

    manifest_path = source_path / "difficulty_manifest.json"
    if manifest_path.is_file():
        bucket_root = source_path / "buckets"
        return manifest_path, bucket_root if bucket_root.is_dir() else None

    if source_path.name == "buckets":
        return None, source_path

    if any(path.is_dir() for path in source_path.iterdir()):
        return None, source_path

    raise ValueError(
        "Curriculum source must be a difficulty_manifest.json file, "
        "a directory containing difficulty_manifest.json, or a buckets directory: "
        f"{source_path}"
    )


def _bucket_files_from_manifest(manifest_path: Path, bucket_name: str) -> list[Path]:
    bucket_map = _load_curriculum_manifest_bucket_map(str(manifest_path))
    return [Path(path) for path in bucket_map.get(bucket_name, ())]


def _bucket_files_from_directory(bucket_root: Path, bucket_name: str) -> list[Path]:
    bucket_dir = bucket_root / bucket_name
    if not bucket_dir.is_dir():
        return []

    files = []
    for path in sorted(bucket_dir.iterdir(), key=lambda item: (item.name, str(item))):
        resolved = path.resolve()
        if resolved.is_file():
            files.append(resolved)
    return files


def _interleave_curriculum_candidates(
    candidate_groups: list[list[tuple[Path, str, int]]],
) -> list[tuple[Path, str, int]]:
    if not candidate_groups:
        return []
    merged: list[tuple[Path, str, int]] = []
    max_len = max(len(group) for group in candidate_groups)
    for offset in range(max_len):
        for group in candidate_groups:
            if offset < len(group):
                merged.append(group[offset])
    return merged


def _resolve_curriculum_candidates(
    maps_dir: Path,
    episode_index: int,
    runtime_config: RuntimeConfig,
    curriculum_override: bool | None = None,
) -> tuple[list[tuple[Path, str, int]], str | None, int | None]:
    base_map_files = list_map_files(maps_dir)
    use_curriculum = runtime_config.enable_curriculum if curriculum_override is None else bool(curriculum_override)
    if not use_curriculum:
        return [], None, None
    if runtime_config.curriculum_source is None:
        raise ValueError("Curriculum is enabled but curriculum_source is not set")

    stage_level = get_curriculum_level(episode_index, runtime_config)
    stage_level_index = get_curriculum_level_index(episode_index, runtime_config)
    manifest_path, bucket_root = _resolve_curriculum_source_paths(runtime_config.curriculum_source)

    level_specs = [(stage_level, stage_level_index)]
    if stage_level_index > 0 and runtime_config.curriculum_mix_window > 0:
        stage_start = int(runtime_config.curriculum_milestones[stage_level_index])
        if int(episode_index) < stage_start + int(runtime_config.curriculum_mix_window):
            previous_level_index = stage_level_index - 1
            previous_level = runtime_config.curriculum_levels[previous_level_index]
            level_specs = [(previous_level, previous_level_index), (stage_level, stage_level_index)]

    candidate_groups: list[list[tuple[Path, str, int]]] = []
    for level_name, level_index in level_specs:
        bucket_files: list[Path] = []
        if manifest_path is not None:
            bucket_files = _bucket_files_from_manifest(manifest_path, level_name)
        if not bucket_files and bucket_root is not None:
            bucket_files = _bucket_files_from_directory(bucket_root, level_name)
        if bucket_files:
            candidate_groups.append([(path, level_name, level_index) for path in bucket_files])

    if not candidate_groups:
        return [(path, stage_level, stage_level_index) for path in base_map_files], stage_level, stage_level_index
    return _interleave_curriculum_candidates(candidate_groups), stage_level, stage_level_index


def resolve_curriculum_map_files(
    maps_dir: str | Path,
    episode_index: int,
    runtime_config: RuntimeConfig,
    curriculum_override: bool | None = None,
) -> tuple[list[Path], str | None, int | None]:
    maps_dir = Path(maps_dir)
    candidates, curriculum_level, curriculum_level_index = _resolve_curriculum_candidates(
        maps_dir,
        episode_index,
        runtime_config,
        curriculum_override=curriculum_override,
    )
    if curriculum_level is None:
        return list_map_files(maps_dir), None, None
    return [path for path, _, _ in candidates], curriculum_level, curriculum_level_index


def select_map_path_for_episode(
    maps_dir: str | Path,
    episode_index: int,
    runtime_config: RuntimeConfig,
    curriculum_override: bool | None = None,
) -> tuple[Path, str | None, int | None]:
    candidates, curriculum_level, curriculum_level_index = _resolve_curriculum_candidates(
        maps_dir,
        episode_index,
        runtime_config,
        curriculum_override=curriculum_override,
    )
    if curriculum_level is None:
        map_files = list_map_files(maps_dir)
        if not map_files:
            raise FileNotFoundError(
                f"No map files found in MAPS_DIR: {maps_dir}. "
                "Place training maps under ARiADNE/maps or src/maps, or set ARIADNE_MAPS_DIR."
            )
        map_index = int(episode_index) % len(map_files)
        return map_files[map_index], None, None
    if not candidates:
        raise FileNotFoundError(
            f"No map files found in MAPS_DIR: {maps_dir}. "
            "Place training maps under ARiADNE/maps or src/maps, or set ARIADNE_MAPS_DIR."
        )
    map_index = int(episode_index) % len(candidates)
    map_path, selected_level, selected_level_index = candidates[map_index]
    return map_path, selected_level, selected_level_index


class Env:
    def __init__(
        self,
        episode_index,
        plot=False,
        output_dir=None,
        artifact_stem=None,
        forced_map_path=None,
        runtime_config: RuntimeConfig | None = None,
        curriculum_override: bool | None = None,
    ):
        self.episode_index = episode_index
        self.plot = plot
        self.runtime_config = runtime_config or RuntimeConfig()
        self.curriculum_override = curriculum_override
        self.forced_map_path = Path(forced_map_path).resolve() if forced_map_path is not None else None
        self.maps_dir = MAPS_DIR
        self.curriculum_level = None
        self.curriculum_level_index = None
        self.curriculum_stage_level = None
        self.curriculum_stage_level_index = None
        self.selected_map_path: Path | None = None
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
        self.last_reward_components: dict[str, float] = {}

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
            if not map_path.is_file():
                raise FileNotFoundError(f"Forced map path does not exist: {map_path}")
            curriculum_level = None
            curriculum_level_index = None
            curriculum_stage_level = None
            curriculum_stage_level_index = None
        else:
            map_path, curriculum_level, curriculum_level_index = select_map_path_for_episode(
                self.maps_dir,
                episode_index,
                self.runtime_config,
                curriculum_override=self.curriculum_override,
            )
            curriculum_active = (
                self.runtime_config.enable_curriculum
                if self.curriculum_override is None
                else bool(self.curriculum_override)
            )
            if curriculum_active:
                curriculum_stage_level = get_curriculum_level(episode_index, self.runtime_config)
                curriculum_stage_level_index = get_curriculum_level_index(episode_index, self.runtime_config)
            else:
                curriculum_stage_level = None
                curriculum_stage_level_index = None
        self.curriculum_level = curriculum_level
        self.curriculum_level_index = curriculum_level_index
        self.curriculum_stage_level = curriculum_stage_level
        self.curriculum_stage_level_index = curriculum_stage_level_index
        self.selected_map_path = map_path
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
        # Preserve the decomposition schema even though ARiADNE does not yet expose
        # a separate safety penalty in its scalar reward.
        r_safe = 0.0

        if self.runtime_config.enable_reward_decomposition:
            reward = (
                self.runtime_config.reward_info_weight * r_info
                + self.runtime_config.reward_dist_weight * r_dist
                + self.runtime_config.reward_safe_weight * r_safe
            )
            self.last_reward_components = {
                "r_info": float(self.runtime_config.reward_info_weight * r_info),
                "r_dist": float(self.runtime_config.reward_dist_weight * r_dist),
                "r_safe": float(self.runtime_config.reward_safe_weight * r_safe),
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
        reward += self.runtime_config.reward_terminal_bonus
        if self.runtime_config.enable_reward_decomposition:
            self.last_reward_components.setdefault("r_info", 0.0)
            self.last_reward_components.setdefault("r_dist", 0.0)
            self.last_reward_components.setdefault("r_safe", 0.0)
            self.last_reward_components["r_terminal"] = (
                self.last_reward_components.get("r_terminal", 0.0) + float(self.runtime_config.reward_terminal_bonus)
            )
            self.last_reward_components["total"] = (
                self.last_reward_components.get("total", 0.0) + float(self.runtime_config.reward_terminal_bonus)
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
