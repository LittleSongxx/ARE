from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import shutil

import imageio.v2 as imageio
import numpy as np
from skimage.morphology import label
import torch

from . import quads
from .parameter import (
    CELL_SIZE,
    FREE,
    FRONTIER_CELL_SIZE,
    NODE_RESOLUTION,
    OCCUPIED,
    UNKNOWN,
    RuntimeConfig,
    WAVELET_SCALES,
    resolve_wavelet_scales,
)


try:
    import numba as _numba
except Exception:  # pragma: no cover - environment dependent
    _numba = None


COLLISION_CHECK_COUNTER = 0


@dataclass
class WaveletMaps:
    scalar_map: np.ndarray
    scale_maps: tuple[np.ndarray, ...]
    orient_maps: tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]
    scalar_proxy_map: np.ndarray
    scales: tuple[int, ...]


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


def reset_collision_counter():
    global COLLISION_CHECK_COUNTER
    COLLISION_CHECK_COUNTER = 0


def get_collision_counter():
    return int(COLLISION_CHECK_COUNTER)


def _bresenham_collision_py(x0, y0, x1, y1, grid, occupied, unknown):
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
        cell_value = grid[y, x]
        if x == x1 and y == y1:
            break
        if cell_value == occupied or cell_value == unknown:
            return True
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx
    return False


def _bresenham_collision_type_py(x0, y0, x1, y1, grid, occupied, unknown, free):
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
        cell_value = grid[y, x]
        if x == x1 and y == y1:
            break
        if cell_value == occupied:
            return occupied
        if cell_value == unknown:
            return unknown
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx
    return free


if _numba is not None:  # pragma: no branch - small runtime dispatch
    _bresenham_collision = _numba.njit(cache=True)(_bresenham_collision_py)
    _bresenham_collision_type = _numba.njit(cache=True)(_bresenham_collision_type_py)
else:  # pragma: no cover - exercised in environments without numba
    _bresenham_collision = _bresenham_collision_py
    _bresenham_collision_type = _bresenham_collision_type_py


def check_collision(start, end, map_info):
    global COLLISION_CHECK_COUNTER
    COLLISION_CHECK_COUNTER += 1
    start_cell = get_cell_position_from_coords(start, map_info)
    end_cell = get_cell_position_from_coords(end, map_info)
    grid = map_info.map
    return _bresenham_collision(
        int(start_cell[0]),
        int(start_cell[1]),
        int(end_cell[0]),
        int(end_cell[1]),
        grid,
        OCCUPIED,
        UNKNOWN,
    )


def check_collision_type(start, end, map_info):
    start_cell = get_cell_position_from_coords(start, map_info)
    end_cell = get_cell_position_from_coords(end, map_info)
    grid = map_info.map.astype(np.int32)
    return _bresenham_collision_type(
        int(start_cell[0]),
        int(start_cell[1]),
        int(end_cell[0]),
        int(end_cell[1]),
        grid,
        OCCUPIED,
        UNKNOWN,
        FREE,
    )


def make_gif(path, n, frame_files, rate, cleanup=True):
    path = os.fspath(path)
    with imageio.get_writer(f"{path}/{n}_explored_rate_{rate:.4g}.gif", mode="I", duration=0.5) as writer:
        for frame in frame_files:
            writer.append_data(imageio.imread(frame))
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


def _normalize_minmax(array, eps):
    array = np.asarray(array, dtype=np.float32)
    min_value = float(np.min(array))
    max_value = float(np.max(array))
    if max_value - min_value < eps:
        return np.zeros_like(array, dtype=np.float32)
    return (array - min_value) / (max_value - min_value)


def _normalize_wavelet_channel(array, runtime_config: RuntimeConfig):
    eps = runtime_config.wavelet_eps
    method = runtime_config.wavelet_norm_method
    array = np.asarray(array, dtype=np.float32)

    if method == "minmax":
        return _normalize_minmax(array, eps)

    if method == "percentile":
        clip_value = float(np.percentile(array, runtime_config.wavelet_clip_percentile))
        clip_value = max(clip_value, eps)
        return np.clip(array, 0.0, clip_value) / clip_value

    if method == "log_percentile":
        array = np.log1p(array)
        clip_value = float(np.percentile(array, runtime_config.wavelet_clip_percentile))
        clip_value = max(clip_value, eps)
        return np.clip(array, 0.0, clip_value) / clip_value

    clip_value = max(runtime_config.wavelet_fixed_clip_value, eps)
    return np.clip(array, 0.0, clip_value) / clip_value


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


def _haar_detail_components(map_array):
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

    def _upsample(detail):
        expanded = np.repeat(np.repeat(np.abs(detail), 2, axis=0), 2, axis=1)
        return expanded[:height, :width].astype(np.float32)

    return _upsample(lh), _upsample(hl), _upsample(hh)


def _upsample_to_base(map_array, scale, output_shape):
    upsampled = np.repeat(np.repeat(map_array, scale, axis=0), scale, axis=1)
    return upsampled[: output_shape[0], : output_shape[1]]


def compute_wavelet_maps(occ_map, runtime_config: RuntimeConfig | None = None) -> WaveletMaps:
    runtime_config = runtime_config or RuntimeConfig()
    base_map = occupancy_to_wavelet_float(occ_map)
    scales = resolve_wavelet_scales(runtime_config)

    scale_maps = []
    orient_maps = []
    scalar_accumulator = np.zeros(base_map.shape, dtype=np.float32)
    proxy_channels = []

    for scale in scales:
        pooled = average_pool_map(base_map, scale)
        lh, hl, hh = _haar_detail_components(pooled)
        lh = _normalize_wavelet_channel(_upsample_to_base(lh, scale, base_map.shape), runtime_config)
        hl = _normalize_wavelet_channel(_upsample_to_base(hl, scale, base_map.shape), runtime_config)
        hh = _normalize_wavelet_channel(_upsample_to_base(hh, scale, base_map.shape), runtime_config)
        energy = np.sqrt(lh * lh + hl * hl + hh * hh).astype(np.float32)
        energy = _normalize_wavelet_channel(energy, runtime_config)

        scale_maps.append(energy)
        orient_maps.append((lh, hl, hh))
        scalar_accumulator += energy

        if runtime_config.wavelet_feature_mode == "scales":
            proxy_channels.append(energy)
        elif runtime_config.wavelet_feature_mode == "scales_orient":
            proxy_channels.extend((lh, hl, hh))

    scalar_map = _normalize_wavelet_channel(scalar_accumulator, runtime_config)
    if runtime_config.wavelet_feature_mode == "scalar" or not proxy_channels:
        scalar_proxy_map = scalar_map
    else:
        scalar_proxy_map = np.mean(np.stack(proxy_channels, axis=0), axis=0).astype(np.float32)
        scalar_proxy_map = _normalize_wavelet_channel(scalar_proxy_map, runtime_config)

    return WaveletMaps(
        scalar_map=scalar_map.astype(np.float32),
        scale_maps=tuple(scale_maps),
        orient_maps=tuple(orient_maps),
        scalar_proxy_map=scalar_proxy_map.astype(np.float32),
        scales=scales,
    )


def _extract_patch_value(channel_map, cell_x, cell_y, runtime_config: RuntimeConfig):
    if runtime_config.wavelet_local_pool == "none" or runtime_config.wavelet_local_pool_radius_cells <= 0:
        return float(channel_map[cell_y, cell_x])

    radius = runtime_config.wavelet_local_pool_radius_cells
    min_x = max(cell_x - radius, 0)
    max_x = min(cell_x + radius + 1, channel_map.shape[1])
    min_y = max(cell_y - radius, 0)
    max_y = min(cell_y + radius + 1, channel_map.shape[0])
    patch = channel_map[min_y:max_y, min_x:max_x]
    if runtime_config.wavelet_local_pool == "max":
        return float(np.max(patch))
    return float(np.mean(patch))


def wavelet_feature_at_coords(
    coords,
    map_info,
    wavelet_maps: WaveletMaps,
    runtime_config: RuntimeConfig | None = None,
) -> np.ndarray:
    runtime_config = runtime_config or RuntimeConfig()
    cell = get_cell_position_from_coords(np.asarray(coords), map_info, check_negative=False)
    cell_x = int(np.clip(cell[0], 0, wavelet_maps.scalar_map.shape[1] - 1))
    cell_y = int(np.clip(cell[1], 0, wavelet_maps.scalar_map.shape[0] - 1))

    if runtime_config.wavelet_feature_mode == "scalar":
        values = [_extract_patch_value(wavelet_maps.scalar_map, cell_x, cell_y, runtime_config)]
    elif runtime_config.wavelet_feature_mode == "scales":
        values = [
            _extract_patch_value(channel_map, cell_x, cell_y, runtime_config)
            for channel_map in wavelet_maps.scale_maps
        ]
    else:
        values = []
        for orient_triplet in wavelet_maps.orient_maps:
            values.extend(
                _extract_patch_value(channel_map, cell_x, cell_y, runtime_config)
                for channel_map in orient_triplet
            )
    return np.asarray(values, dtype=np.float32)


def wavelet_scalar_at_coords(coords, map_info, wavelet_maps: WaveletMaps) -> float:
    cell = get_cell_position_from_coords(np.asarray(coords), map_info, check_negative=False)
    cell_x = int(np.clip(cell[0], 0, wavelet_maps.scalar_proxy_map.shape[1] - 1))
    cell_y = int(np.clip(cell[1], 0, wavelet_maps.scalar_proxy_map.shape[0] - 1))
    return float(wavelet_maps.scalar_proxy_map[cell_y, cell_x])


def multiscale_haar_energy_map(map_array, scales=WAVELET_SCALES):
    runtime_config = RuntimeConfig(
        use_wavelet_feature=True,
        wavelet_feature_mode="scalar",
        wavelet_scales_auto=False,
        wavelet_scales=tuple(scales),
        wavelet_norm_method="minmax",
    )
    return compute_wavelet_maps(map_array, runtime_config).scalar_map


def wavelet_score_at_coords(coords, map_info, wavelet_map):
    cell = get_cell_position_from_coords(np.asarray(coords), map_info, check_negative=False)
    cell_x = int(np.clip(cell[0], 0, wavelet_map.shape[1] - 1))
    cell_y = int(np.clip(cell[1], 0, wavelet_map.shape[0] - 1))
    return float(wavelet_map[cell_y, cell_x])


def build_wavelet_attn_bias(
    wavelet_feature,
    runtime_config: RuntimeConfig | None = None,
    edge_mask=None,
    node_padding_mask=None,
):
    runtime_config = runtime_config or RuntimeConfig()
    feature = torch.as_tensor(wavelet_feature, dtype=torch.float32)
    if feature.dim() == 2:
        feature = feature.unsqueeze(-1)
    if feature.dim() != 3:
        raise ValueError(f"wavelet_feature must have shape [B, N, D] or [B, N], got {tuple(feature.shape)}")

    pairwise_diff = feature.unsqueeze(-2) - feature.unsqueeze(-3)
    if runtime_config.wavelet_attn_bias_type == "sim_exp":
        dist = torch.linalg.norm(pairwise_diff, ord=2, dim=-1)
        bias = runtime_config.wavelet_attn_bias_beta * torch.exp(
            -dist / runtime_config.wavelet_attn_bias_sigma
        )
    elif runtime_config.wavelet_attn_bias_type == "neg_l1":
        bias = -runtime_config.wavelet_attn_bias_beta * pairwise_diff.abs().sum(dim=-1)
    else:
        bias = -runtime_config.wavelet_attn_bias_beta * torch.linalg.norm(pairwise_diff, ord=2, dim=-1)

    if node_padding_mask is not None:
        node_padding_mask = torch.as_tensor(node_padding_mask, device=bias.device)
        if node_padding_mask.dim() == 3 and node_padding_mask.size(1) == 1:
            valid_nodes = ~node_padding_mask[:, 0, :].bool()
        elif node_padding_mask.dim() == 2:
            valid_nodes = ~node_padding_mask.bool()
        else:
            raise ValueError(f"Unsupported node_padding_mask shape: {tuple(node_padding_mask.shape)}")
        valid_pairs = valid_nodes.unsqueeze(-1) & valid_nodes.unsqueeze(-2)
        bias = bias.masked_fill(~valid_pairs, 0.0)

    if edge_mask is not None and runtime_config.wavelet_attn_bias_apply_on_masked_edges_only:
        edge_mask = torch.as_tensor(edge_mask, device=bias.device)
        if edge_mask.dim() == 3:
            allowed_edges = edge_mask == 0
        elif edge_mask.dim() == 4 and edge_mask.size(1) == 1:
            allowed_edges = edge_mask[:, 0, :, :] == 0
        else:
            raise ValueError(f"Unsupported edge_mask shape: {tuple(edge_mask.shape)}")
        bias = bias * allowed_edges.to(dtype=bias.dtype)

    return bias.unsqueeze(1).to(dtype=torch.float32)
