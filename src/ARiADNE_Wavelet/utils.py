from __future__ import annotations

import os
import shutil
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from .parameter import (
    CELL_SIZE,
    FREE,
    FRONTIER_CELL_SIZE,
    NODE_RESOLUTION,
    OCCUPIED,
    RESULT_BUCKET_EPISODES,
    UNKNOWN,
)


def get_cell_position_from_coords(coords, map_info, check_negative=True):
    single_cell = False
    if coords.flatten().shape[0] == 2:
        single_cell = True

    coords = coords.reshape(-1, 2)
    coords_x = coords[:, 0]
    coords_y = coords[:, 1]
    cell_x = (coords_x - map_info.map_origin_x) / map_info.cell_size
    cell_y = (coords_y - map_info.map_origin_y) / map_info.cell_size

    cell_position = np.around(np.stack((cell_x, cell_y), axis=-1)).astype(int)

    if check_negative:
        assert np.all(cell_position >= 0), (
            cell_position,
            coords,
            map_info.map_origin_x,
            map_info.map_origin_y,
        )
    if single_cell:
        return cell_position[0]
    return cell_position


def get_coords_from_cell_position(cell_position, map_info):
    cell_position = cell_position.reshape(-1, 2)
    cell_x = cell_position[:, 0]
    cell_y = cell_position[:, 1]
    coords_x = cell_x * map_info.cell_size + map_info.map_origin_x
    coords_y = cell_y * map_info.cell_size + map_info.map_origin_y
    coords = np.stack((coords_x, coords_y), axis=-1)
    coords = np.around(coords, 1)
    if coords.shape[0] == 1:
        return coords[0]
    return coords


def get_free_area_coords(map_info):
    free_indices = np.where(map_info.map == FREE)
    free_cells = np.asarray([free_indices[1], free_indices[0]]).T
    return get_coords_from_cell_position(free_cells, map_info)


def get_free_and_connected_map(location, map_info):
    free = map_info.map == FREE
    cell = get_cell_position_from_coords(location, map_info)
    cell_x, cell_y = int(cell[0]), int(cell[1])
    if not (0 <= cell_y < free.shape[0] and 0 <= cell_x < free.shape[1]):
        return np.zeros_like(free, dtype=bool)
    if not free[cell_y, cell_x]:
        return np.zeros_like(free, dtype=bool)

    connected = np.zeros_like(free, dtype=bool)
    stack = [(cell_x, cell_y)]
    while stack:
        x, y = stack.pop()
        if connected[y, x] or not free[y, x]:
            continue
        connected[y, x] = True
        for ny in (y - 1, y, y + 1):
            for nx in (x - 1, x, x + 1):
                if 0 <= ny < free.shape[0] and 0 <= nx < free.shape[1]:
                    if not connected[ny, nx] and free[ny, nx]:
                        stack.append((nx, ny))
    return connected


def get_updating_node_coords(location, updating_map_info, check_connectivity=True):
    x_min = updating_map_info.map_origin_x
    y_min = updating_map_info.map_origin_y
    x_max = updating_map_info.map_origin_x + (updating_map_info.map.shape[1] - 1) * CELL_SIZE
    y_max = updating_map_info.map_origin_y + (updating_map_info.map.shape[0] - 1) * CELL_SIZE

    if x_min % NODE_RESOLUTION != 0:
        x_min = (x_min // NODE_RESOLUTION + 1) * NODE_RESOLUTION
    if x_max % NODE_RESOLUTION != 0:
        x_max = x_max // NODE_RESOLUTION * NODE_RESOLUTION
    if y_min % NODE_RESOLUTION != 0:
        y_min = (y_min // NODE_RESOLUTION + 1) * NODE_RESOLUTION
    if y_max % NODE_RESOLUTION != 0:
        y_max = y_max // NODE_RESOLUTION * NODE_RESOLUTION

    x_coords = np.arange(x_min, x_max + 0.1, NODE_RESOLUTION)
    y_coords = np.arange(y_min, y_max + 0.1, NODE_RESOLUTION)
    t1, t2 = np.meshgrid(x_coords, y_coords)
    nodes = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
    nodes = np.around(nodes, 1)

    free_connected_map = None

    if not check_connectivity:
        indices = []
        nodes_cells = get_cell_position_from_coords(nodes, updating_map_info).reshape(-1, 2)
        for i, cell in enumerate(nodes_cells):
            assert 0 <= cell[1] < updating_map_info.map.shape[0]
            assert 0 <= cell[0] < updating_map_info.map.shape[1]
            if updating_map_info.map[cell[1], cell[0]] == FREE:
                indices.append(i)
        nodes = nodes[np.array(indices)].reshape(-1, 2)
    else:
        free_connected_map = np.asarray(get_free_and_connected_map(location, updating_map_info))
        indices = []
        nodes_cells = get_cell_position_from_coords(nodes, updating_map_info).reshape(-1, 2)
        for i, cell in enumerate(nodes_cells):
            assert 0 <= cell[1] < free_connected_map.shape[0]
            assert 0 <= cell[0] < free_connected_map.shape[1]
            if free_connected_map[cell[1], cell[0]] == 1:
                indices.append(i)
        nodes = nodes[np.array(indices)].reshape(-1, 2)

    return nodes, free_connected_map


def get_frontier_in_map(map_info):
    x_len = map_info.map.shape[1]
    y_len = map_info.map.shape[0]
    unknown = (map_info.map == UNKNOWN).astype(int)
    unknown = np.pad(unknown, ((1, 1), (1, 1)), "constant", constant_values=0)
    unknown_neighbor = (
        unknown[2:][:, 1 : x_len + 1]
        + unknown[:y_len][:, 1 : x_len + 1]
        + unknown[1 : y_len + 1][:, 2:]
        + unknown[1 : y_len + 1][:, :x_len]
        + unknown[:y_len][:, 2:]
        + unknown[2:][:, :x_len]
        + unknown[2:][:, 2:]
        + unknown[:y_len][:, :x_len]
    )
    free_cell_indices = np.where(map_info.map.ravel(order="F") == FREE)[0]
    frontier_cell_1 = np.where(1 < unknown_neighbor.ravel(order="F"))[0]
    frontier_cell_2 = np.where(unknown_neighbor.ravel(order="F") < 8)[0]
    frontier_cell_indices = np.intersect1d(frontier_cell_1, frontier_cell_2)
    frontier_cell_indices = np.intersect1d(free_cell_indices, frontier_cell_indices)

    x = np.linspace(0, x_len - 1, x_len)
    y = np.linspace(0, y_len - 1, y_len)
    t1, t2 = np.meshgrid(x, y)
    cells = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
    frontier_cell = cells[frontier_cell_indices]

    frontier_coords = get_coords_from_cell_position(frontier_cell, map_info).reshape(-1, 2)
    if frontier_cell.shape[0] > 0 and FRONTIER_CELL_SIZE != CELL_SIZE:
        frontier_coords = frontier_down_sample(frontier_coords)
    else:
        frontier_coords = set(map(tuple, frontier_coords))
    return frontier_coords


def frontier_down_sample(data, voxel_size=FRONTIER_CELL_SIZE):
    voxel_indices = np.array(data / voxel_size, dtype=int).reshape(-1, 2)

    voxel_dict = {}
    for i, point in enumerate(data):
        voxel_index = tuple(voxel_indices[i])

        if voxel_index not in voxel_dict:
            voxel_dict[voxel_index] = point
        else:
            current_point = voxel_dict[voxel_index]
            if np.linalg.norm(point - np.array(voxel_index) * voxel_size) < np.linalg.norm(
                current_point - np.array(voxel_index) * voxel_size
            ):
                voxel_dict[voxel_index] = point

    return set(map(tuple, voxel_dict.values()))


def check_collision(start, end, map_info):
    start_cell = get_cell_position_from_coords(start, map_info)
    end_cell = get_cell_position_from_coords(end, map_info)
    grid = map_info.map

    x0 = int(start_cell[0])
    y0 = int(start_cell[1])
    x1 = int(end_cell[0])
    y1 = int(end_cell[1])
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    x, y = x0, y0
    error = dx - dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2

    while 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]:
        cell_value = grid.item(int(y), int(x))
        if x == x1 and y == y1:
            break
        if cell_value == OCCUPIED or cell_value == UNKNOWN:
            return True
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx
    return False


def make_gif(path, n, frame_files, rate):
    path = os.fspath(path)
    with imageio.get_writer(f"{path}/{n}_explored_rate_{rate:.4g}.gif", mode="I", duration=0.5) as writer:
        for frame in frame_files:
            image = imageio.imread(frame)
            writer.append_data(image)

    for filename in frame_files[:-1]:
        os.remove(filename)


def get_bucket_name(episode, bucket_size=RESULT_BUCKET_EPISODES):
    episode = max(int(episode), 1)
    bucket_size = max(int(bucket_size), 1)
    end = ((episode - 1) // bucket_size + 1) * bucket_size
    return f"episodes_{end}"


def ensure_bucket_dir(base_dir, episode, bucket_size=RESULT_BUCKET_EPISODES):
    bucket_dir = Path(base_dir) / get_bucket_name(episode, bucket_size)
    bucket_dir.mkdir(parents=True, exist_ok=True)
    return bucket_dir


def build_artifact_stem(episode, prefix="episode"):
    if prefix == "episode":
        return f"ep{int(episode):04d}"
    if prefix == "eval_episode":
        return f"eval{int(episode):04d}"
    return f"{prefix}{int(episode):04d}"


def finalize_episode_artifacts(output_dir, artifact_stem, frame_files, frame_rate=2.0):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = output_dir / f"{artifact_stem}.gif"
    final_png_path = output_dir / f"{artifact_stem}.png"

    with imageio.get_writer(gif_path, mode="I", duration=1.0 / max(float(frame_rate), 1e-3)) as writer:
        for frame in frame_files:
            writer.append_data(imageio.imread(frame))

    if frame_files:
        shutil.move(frame_files[-1], final_png_path)
        for frame in frame_files[:-1]:
            os.remove(frame)

    frame_dir = output_dir / f".{artifact_stem}_frames"
    if frame_dir.exists() and not any(frame_dir.iterdir()):
        frame_dir.rmdir()

    return gif_path, final_png_path


class MapInfo:
    def __init__(self, map, map_origin_x, map_origin_y, cell_size):
        self.map = map
        self.map_origin_x = map_origin_x
        self.map_origin_y = map_origin_y
        self.cell_size = cell_size

    def update_map_info(self, map, map_origin_x, map_origin_y):
        self.map = map
        self.map_origin_x = map_origin_x
        self.map_origin_y = map_origin_y
