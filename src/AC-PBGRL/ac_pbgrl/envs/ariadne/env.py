from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
from skimage import io
from skimage.measure import block_reduce

from .parameter import CELL_SIZE, FRONTIER_CELL_SIZE, FREE, MAPS_DIR, SENSOR_RANGE, UPDATING_MAP_SIZE, gifs_path
from .sensor import sensor_work
from .utils import MapInfo, get_frontier_in_map


def list_map_files(maps_dir: str | Path | None = None) -> list[Path]:
    target_dir = Path(maps_dir) if maps_dir is not None else MAPS_DIR
    map_files = sorted(path for path in target_dir.iterdir() if path.is_file())
    if not map_files:
        raise FileNotFoundError(f"No map files found in {target_dir}")
    return map_files


class Env:
    def __init__(
        self,
        episode_index,
        plot=False,
        gifs_dir: str | Path | None = None,
        forced_map_path: str | Path | None = None,
        artifact_prefix: str | None = None,
    ):
        self.episode_index = episode_index
        self.plot = plot
        self.gifs_dir = Path(gifs_dir) if gifs_dir is not None else Path(gifs_path)
        self.forced_map_path = Path(forced_map_path).expanduser().resolve() if forced_map_path is not None else None
        self.artifact_prefix = str(artifact_prefix or episode_index)
        self.ground_truth, self.robot_cell, self.map_path = self.import_ground_truth(episode_index)
        self.ground_truth_size = np.shape(self.ground_truth)
        self.cell_size = CELL_SIZE

        self.robot_location = np.array([0.0, 0.0])
        self.robot_belief = np.ones(self.ground_truth_size) * 127
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
        self.old_belief = deepcopy(self.robot_belief)

        self.belief_info = MapInfo(self.robot_belief, self.belief_origin_x, self.belief_origin_y, self.cell_size)
        self.ground_truth_info = MapInfo(self.ground_truth, self.belief_origin_x, self.belief_origin_y, self.cell_size)

        if self.plot:
            self.frame_files = []
            self.trajectory_x = [self.robot_location[0]]
            self.trajectory_y = [self.robot_location[1]]

    def import_ground_truth(self, episode_index):
        if self.forced_map_path is not None:
            map_path = self.forced_map_path
        else:
            map_list = list_map_files(MAPS_DIR)
            map_index = episode_index % np.size(map_list)
            map_path = map_list[map_index]

        ground_truth = (io.imread(str(map_path), 1) * 255).astype(int)
        ground_truth = block_reduce(ground_truth, 2, np.min)

        robot_cell = np.nonzero(ground_truth == 208)
        robot_cell = np.array([np.array(robot_cell)[1, 10], np.array(robot_cell)[0, 10]])

        ground_truth = (ground_truth > 150) | ((ground_truth <= 80) & (ground_truth >= 50))
        ground_truth = ground_truth * 254 + 1
        return ground_truth, robot_cell, map_path

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
        reward = 0
        reward -= dist / UPDATING_MAP_SIZE * 5

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
        import matplotlib.pyplot as plt

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
        self.gifs_dir.mkdir(parents=True, exist_ok=True)
        frame = str(self.gifs_dir / f"{self.artifact_prefix}_{step}_samples.png")
        plt.savefig(frame, dpi=150)
        plt.close()
        self.frame_files.append(frame)
