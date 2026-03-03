from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from skimage.morphology import label

from . import parameter, quads
from .parameter import (
    CELL_SIZE,
    FREE,
    FRONTIER_CELL_SIZE,
    NODE_RESOLUTION,
    OCCUPIED,
    UNKNOWN,
    WAVELET_SCALES,
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


def get_quad_tree_box(coords, box_size):
    min_x = np.round(coords[0] - box_size / 2, 1)
    min_y = np.round(coords[1] - box_size / 2, 1)
    max_x = np.round(coords[0] + box_size / 2, 1)
    max_y = np.round(coords[1] + box_size / 2, 1)
    return quads.BoundingBox(min_x, min_y, max_x, max_y)


def get_free_and_connected_map(location, map_info):
    free = (map_info.map == FREE).astype(float)
    labeled_free = label(free, connectivity=2)
    cell = get_cell_position_from_coords(location, map_info)
    label_number = labeled_free[cell[1], cell[0]]
    return labeled_free == label_number


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


def is_free(location, map_info):
    cell = get_cell_position_from_coords(location, map_info)
    return map_info.map[cell[1], cell[0]] == FREE


try:
    import numba as nb
except ImportError:  # pragma: no cover - exercised in ros_conda runtime
    nb = None


def _bresenham_collision_py(x0, y0, x1, y1, grid, occupied, unknown):
    """Return True if the Bresenham line hits an OCCUPIED or UNKNOWN cell."""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    error = dx - dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2
    h, w = grid.shape[0], grid.shape[1]
    while 0 <= x < w and 0 <= y < h:
        cv = grid[y, x]
        if x == x1 and y == y1:
            break
        if cv == occupied or cv == unknown:
            return True
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx
    return False


def _bresenham_collision_type_py(x0, y0, x1, y1, grid, occupied, unknown, free):
    """Return the first collision cell type, or FREE if none."""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    error = dx - dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2
    h, w = grid.shape[0], grid.shape[1]
    while 0 <= x < w and 0 <= y < h:
        cv = grid[y, x]
        if x == x1 and y == y1:
            break
        if cv == occupied:
            return occupied
        if cv == unknown:
            return unknown
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx
    return free


if nb is not None:  # pragma: no branch - small runtime dispatch
    _bresenham_collision = nb.njit(cache=True)(_bresenham_collision_py)
    _bresenham_collision_type = nb.njit(cache=True)(_bresenham_collision_type_py)
else:  # pragma: no cover - exercised in environments without numba
    _bresenham_collision = _bresenham_collision_py
    _bresenham_collision_type = _bresenham_collision_type_py


def check_collision(start, end, map_info):
    start_cell = get_cell_position_from_coords(start, map_info)
    end_cell = get_cell_position_from_coords(end, map_info)
    grid = map_info.map
    return _bresenham_collision(
        int(start_cell[0]), int(start_cell[1]),
        int(end_cell[0]), int(end_cell[1]),
        grid, OCCUPIED, UNKNOWN,
    )


def check_collision_type(start, end, map_info):
    start_cell = get_cell_position_from_coords(start, map_info)
    end_cell = get_cell_position_from_coords(end, map_info)
    grid = map_info.map.astype(np.int32)
    return _bresenham_collision_type(
        int(start_cell[0]), int(start_cell[1]),
        int(end_cell[0]), int(end_cell[1]),
        grid, OCCUPIED, UNKNOWN, FREE,
    )


def make_gif(path, n, frame_files, rate, cleanup=True):
    path = os.fspath(path)
    with imageio.get_writer(f"{path}/{n}_explored_rate_{rate:.4g}.gif", mode="I", duration=0.5) as writer:
        for frame in frame_files:
            image = imageio.imread(frame)
            writer.append_data(image)

    if cleanup:
        for filename in frame_files[:-1]:
            os.remove(filename)


def get_bucket_name(episode, bucket_size):
    episode = max(int(episode), 1)
    bucket_size = max(int(bucket_size), 1)
    end = ((episode - 1) // bucket_size + 1) * bucket_size
    return f"episodes_{end}"


def ensure_bucket_dir(base_dir, episode, bucket_size):
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

    with imageio.get_writer(gif_path, mode="I", duration=1.0 / frame_rate) as writer:
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


def normalize_ros_occupancy_if_needed(map_array):
    map_array = np.asarray(map_array)
    unique = np.unique(map_array)
    if unique.size == 0:
        return map_array.astype(np.int16)

    ros_like = np.any(unique < 0) or np.all(unique <= 100)
    if not ros_like:
        return map_array.astype(np.int16)

    normalized = np.full(map_array.shape, OCCUPIED, dtype=np.int16)
    normalized[map_array == -1] = UNKNOWN
    normalized[map_array == 0] = FREE
    normalized[map_array > 0] = OCCUPIED
    return normalized


def occupancy_to_wavelet_float(map_array):
    normalized = normalize_ros_occupancy_if_needed(map_array)
    mapped = np.empty(normalized.shape, dtype=np.float32)
    mapped.fill(1.0)
    mapped[normalized == FREE] = 0.0
    mapped[normalized == UNKNOWN] = 0.5
    mapped[normalized == OCCUPIED] = 1.0
    return mapped


def _normalize_01(array):
    array = array.astype(np.float32, copy=False)
    min_value = float(np.min(array))
    max_value = float(np.max(array))
    if max_value - min_value < 1e-8:
        return np.zeros_like(array, dtype=np.float32)
    return (array - min_value) / (max_value - min_value)


def average_pool_map(map_array, scale):
    map_array = np.asarray(map_array, dtype=np.float32)
    if scale <= 1:
        return map_array

    height, width = map_array.shape
    pad_h = (scale - height % scale) % scale
    pad_w = (scale - width % scale) % scale
    padded = np.pad(map_array, ((0, pad_h), (0, pad_w)), mode="edge")
    pooled = padded.reshape(
        padded.shape[0] // scale,
        scale,
        padded.shape[1] // scale,
        scale,
    ).mean(axis=(1, 3))
    return pooled.astype(np.float32)


def haar_detail_energy_map(map_array):
    map_array = np.asarray(map_array, dtype=np.float32)
    height, width = map_array.shape
    pad_h = height % 2
    pad_w = width % 2
    padded = np.pad(map_array, ((0, pad_h), (0, pad_w)), mode="edge")

    a = padded[0::2, 0::2]
    b = padded[0::2, 1::2]
    c = padded[1::2, 0::2]
    d = padded[1::2, 1::2]

    lh = 0.5 * (a + b - c - d)
    hl = 0.5 * (a - b + c - d)
    hh = 0.5 * (a - b - c + d)
    energy = np.sqrt(lh * lh + hl * hl + hh * hh)
    energy = np.repeat(np.repeat(energy, 2, axis=0), 2, axis=1)
    return _normalize_01(energy[:height, :width])


def multiscale_haar_energy_map(map_array, scales=WAVELET_SCALES):
    base_map = occupancy_to_wavelet_float(map_array)
    accumulated = np.zeros(base_map.shape, dtype=np.float32)

    for scale in scales:
        pooled = average_pool_map(base_map, scale)
        energy = haar_detail_energy_map(pooled)
        upsampled = np.repeat(np.repeat(energy, scale, axis=0), scale, axis=1)
        upsampled = upsampled[: base_map.shape[0], : base_map.shape[1]]
        accumulated += _normalize_01(upsampled)

    return _normalize_01(accumulated)


def wavelet_score_at_coords(coords, map_info, wavelet_map):
    cell = get_cell_position_from_coords(np.asarray(coords), map_info, check_negative=False)
    cell_x = int(np.clip(cell[0], 0, wavelet_map.shape[1] - 1))
    cell_y = int(np.clip(cell[1], 0, wavelet_map.shape[0] - 1))
    return float(wavelet_map[cell_y, cell_x])


def _get_valid_node_mask(node_padding_mask, batch_size, num_nodes, device):
    if node_padding_mask is None:
        return torch.ones((batch_size, num_nodes), dtype=torch.bool, device=device)

    if node_padding_mask.dim() == 3 and node_padding_mask.size(1) == 1:
        mask = node_padding_mask[:, 0, :]
    elif node_padding_mask.dim() == 2:
        mask = node_padding_mask
    else:
        raise ValueError(f"Unsupported node_padding_mask shape: {tuple(node_padding_mask.shape)}")

    if mask.shape != (batch_size, num_nodes):
        raise ValueError(
            f"node_padding_mask shape {tuple(mask.shape)} does not match expected {(batch_size, num_nodes)}"
        )

    if mask.dtype == torch.bool:
        return ~mask
    return mask == 0


def build_attention_bias(w, mode=None, node_padding_mask=None):
    if w.dim() == 3:
        if w.size(-1) != 1:
            raise ValueError(f"Wavelet feature tensor must end with size 1, got {tuple(w.shape)}")
        w = w.squeeze(-1)
    elif w.dim() != 2:
        raise ValueError(f"Wavelet feature tensor must have shape [B, N] or [B, N, 1], got {tuple(w.shape)}")

    mode = (mode or "diff").strip().lower()
    if mode not in {"diff", "open", "hybrid"}:
        raise ValueError(f"Unsupported attention bias mode: {mode}")

    w = w.to(dtype=torch.float32).clamp(0.0, 1.0)
    wi = w.unsqueeze(-1)
    wj = w.unsqueeze(-2)

    diff_bias = -(wi - wj).abs()
    open_bias = (1.0 - wi) * (1.0 - wj)

    if mode == "diff":
        bias = diff_bias
    elif mode == "open":
        bias = open_bias
    else:
        bias = 0.5 * diff_bias + 0.5 * open_bias

    valid_nodes = _get_valid_node_mask(node_padding_mask, w.size(0), w.size(1), w.device)
    valid_pairs = valid_nodes.unsqueeze(-1) & valid_nodes.unsqueeze(-2)
    pair_weight = valid_pairs.to(dtype=bias.dtype)
    pair_count = pair_weight.sum(dim=(-1, -2), keepdim=True).clamp_min(1.0)
    mean = (bias * pair_weight).sum(dim=(-1, -2), keepdim=True) / pair_count
    centered = (bias - mean) * pair_weight
    scale = centered.abs().amax(dim=(-1, -2), keepdim=True).clamp_min(1e-6)
    normalized = (centered / scale).clamp(-1.0, 1.0)
    normalized = normalized.masked_fill(~valid_pairs, 0.0)
    return normalized.unsqueeze(1).to(dtype=torch.float32)


def make_attn_bias_from_node_inputs(node_inputs, wavelet_col=-1, node_padding_mask=None):
    if node_inputs.dim() != 3:
        raise ValueError(f"node_inputs must have shape [B, N, D], got {tuple(node_inputs.shape)}")
    wavelet_feature = node_inputs[..., wavelet_col]
    return build_attention_bias(
        wavelet_feature,
        mode=parameter.ATTN_BIAS_MODE,
        node_padding_mask=node_padding_mask,
    )


@dataclass
class MapInfo:
    map: np.ndarray
    map_origin_x: float
    map_origin_y: float
    cell_size: float

    def update_map_info(self, map_array, map_origin_x, map_origin_y):
        self.map = map_array
        self.map_origin_x = map_origin_x
        self.map_origin_y = map_origin_y
