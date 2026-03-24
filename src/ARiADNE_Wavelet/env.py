from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

from .parameter import CELL_SIZE, FREE, FRONTIER_CELL_SIZE, MAPS_DIR, SENSOR_RANGE, UNKNOWN, UPDATING_MAP_SIZE
from .sensor import sensor_work
from .utils import MapInfo, get_frontier_in_map


def list_map_files(map_dir: str | Path) -> list[Path]:
    directory = Path(map_dir)
    if not directory.exists():
        raise FileNotFoundError(
            f"MAPS_DIR does not exist: {directory}. "
            "Place training maps under ARiADNE_Wavelet/maps or set ARIADNE_MAPS_DIR."
        )
    return sorted(path for path in directory.iterdir() if path.is_file())


class Env:
    def __init__(self, episode_index, plot=False, output_dir=None, artifact_stem=None, forced_map_path=None):
        self.episode_index = episode_index
        self.plot = plot
        self.ground_truth, self.robot_cell = self.import_ground_truth(episode_index, forced_map_path=forced_map_path)
        self.ground_truth_size = np.shape(self.ground_truth)
        self.cell_size = CELL_SIZE

        self.robot_location = np.array([0.0, 0.0])
        self.robot_belief = np.ones(self.ground_truth_size) * UNKNOWN
        self.belief_origin_x = -np.round(self.robot_cell[0] * self.cell_size, 1)
        self.belief_origin_y = -np.round(self.robot_cell[1] * self.cell_size, 1)

        self.global_frontiers = set()
        self.sensor_range = SENSOR_RANGE
        self.travel_dist = 0.0
        self.explored_rate = 0.0

        self.robot_belief = sensor_work(
            self.robot_cell,
            self.sensor_range / self.cell_size,
            self.robot_belief,
            self.ground_truth,
        )
        self.old_belief = deepcopy(self.robot_belief)
        self.belief_info = MapInfo(self.robot_belief, self.belief_origin_x, self.belief_origin_y, self.cell_size)

        if self.plot:
            self.output_dir = Path(output_dir) if output_dir is not None else None
            self.artifact_stem = artifact_stem or f"episode_{episode_index:06d}"
            self.frame_files = []
            self.trajectory_x = [self.robot_location[0]]
            self.trajectory_y = [self.robot_location[1]]

    def import_ground_truth(self, episode_index, forced_map_path=None):
        if forced_map_path is not None:
            map_path = Path(forced_map_path)
            if not map_path.is_file():
                raise FileNotFoundError(f"Forced map path does not exist: {map_path}")
        else:
            map_list = list_map_files(MAPS_DIR)
            if not map_list:
                raise FileNotFoundError(f"No map files found in MAPS_DIR: {MAPS_DIR}")
            map_index = episode_index % len(map_list)
            map_path = map_list[map_index]

        ground_truth = imageio.imread(map_path)
        if ground_truth.ndim == 3:
            # RGB/BGR -> grayscale
            ground_truth = (
                0.2989 * ground_truth[..., 0]
                + 0.5870 * ground_truth[..., 1]
                + 0.1140 * ground_truth[..., 2]
            )
        ground_truth = np.asarray(ground_truth, dtype=np.float32)
        if ground_truth.max() <= 1.0:
            ground_truth = ground_truth * 255.0
        ground_truth = _min_pool_2x2(ground_truth).astype(int)

        robot_candidates = np.argwhere(ground_truth == 208)
        if robot_candidates.size == 0:
            # Fallback for maps without explicit robot marker.
            free_candidates = np.argwhere(ground_truth > 150)
            if free_candidates.size == 0:
                robot_cell = np.array([ground_truth.shape[1] // 2, ground_truth.shape[0] // 2])
            else:
                pick_idx = min(10, free_candidates.shape[0] - 1)
                pick = free_candidates[pick_idx]
                robot_cell = np.array([pick[1], pick[0]])
        else:
            pick_idx = min(10, robot_candidates.shape[0] - 1)
            pick = robot_candidates[pick_idx]
            robot_cell = np.array([pick[1], pick[0]])

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
        reward = -dist / UPDATING_MAP_SIZE * 5

        global_frontiers = get_frontier_in_map(self.belief_info)
        if len(global_frontiers) == 0:
            delta_num = len(self.global_frontiers)
        else:
            observed_frontiers = self.global_frontiers - global_frontiers
            delta_num = len(observed_frontiers)

        reward += delta_num / (SENSOR_RANGE * 3.14 // FRONTIER_CELL_SIZE)

        self.global_frontiers = global_frontiers
        self.old_belief = deepcopy(self.robot_belief)

        return reward

    def evaluate_exploration_rate(self):
        self.explored_rate = np.sum(self.robot_belief == FREE) / np.sum(self.ground_truth == FREE)

    def step(self, next_waypoint):
        dist = np.linalg.norm(self.robot_location - next_waypoint)
        self.update_robot_location(next_waypoint)
        self.update_robot_belief()

        self.travel_dist += dist
        self.evaluate_exploration_rate()

        reward = self.calculate_reward(dist)

        return reward

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
        plt.suptitle(f"Explored ratio: {self.explored_rate:.4g}  Travel distance: {self.travel_dist:.4g}")
        plt.tight_layout()
        frame_dir = self.output_dir / f".{self.artifact_stem}_frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        frame_path = frame_dir / f"{self.artifact_stem}_{step:03d}.png"
        plt.savefig(frame_path, dpi=150)
        plt.close()
        self.frame_files.append(str(frame_path))


def _min_pool_2x2(array: np.ndarray) -> np.ndarray:
    """2x2 min-pooling using NumPy only (drop last row/col when odd)."""
    h, w = array.shape[:2]
    h2 = (h // 2) * 2
    w2 = (w // 2) * 2
    pooled_input = array[:h2, :w2]
    if pooled_input.size == 0:
        return array
    return pooled_input.reshape(h2 // 2, 2, w2 // 2, 2).min(axis=(1, 3))
