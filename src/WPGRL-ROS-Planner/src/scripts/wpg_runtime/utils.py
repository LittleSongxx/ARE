from __future__ import annotations

import numpy as np
from skimage.morphology import label

from .parameter import (
    CELL_SIZE,
    FRONTIER_CELL_SIZE,
    FREE,
    NODE_RESOLUTION,
    OCCUPIED,
    UNKNOWN,
)

try:
    from wpg_rl_planner.collision_checker import check_collision_type as _cc_native
    _HAS_NATIVE_CC = True
except ImportError:
    _HAS_NATIVE_CC = False


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
        assert np.all(cell_position.flatten() >= 0), (
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
            assert 0 <= cell[1] < updating_map_info.map.shape[0] and 0 <= cell[0] < updating_map_info.map.shape[1]
            if updating_map_info.map[cell[1], cell[0]] == FREE:
                indices.append(i)
        nodes = nodes[np.array(indices)].reshape(-1, 2)
    else:
        free_connected_map = np.array(get_free_and_connected_map(location, updating_map_info))
        indices = []
        nodes_cells = get_cell_position_from_coords(nodes, updating_map_info).reshape(-1, 2)
        for i, cell in enumerate(nodes_cells):
            assert 0 <= cell[1] < free_connected_map.shape[0] and 0 <= cell[0] < free_connected_map.shape[1]
            if free_connected_map[cell[1], cell[0]] == 1:
                indices.append(i)
        nodes = nodes[np.array(indices)].reshape(-1, 2)

    return nodes, free_connected_map


def get_frontier_in_map(map_info):
    x_len = map_info.map.shape[1]
    y_len = map_info.map.shape[0]
    unknown = (map_info.map == UNKNOWN) * 1
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
        frontier_coords = frontier_down_sample(frontier_coords.reshape(-1, 2))
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
    assert start[0] >= map_info.map_origin_x
    assert start[1] >= map_info.map_origin_y
    assert end[0] >= map_info.map_origin_x
    assert end[1] >= map_info.map_origin_y
    assert start[0] <= map_info.map_origin_x + map_info.cell_size * map_info.map.shape[1]
    assert start[1] <= map_info.map_origin_y + map_info.cell_size * map_info.map.shape[0]
    assert end[0] <= map_info.map_origin_x + map_info.cell_size * map_info.map.shape[1]
    assert end[1] <= map_info.map_origin_y + map_info.cell_size * map_info.map.shape[0]
    collision = False

    start_cell = get_cell_position_from_coords(start, map_info)
    end_cell = get_cell_position_from_coords(end, map_info)
    grid = map_info.map

    x0 = start_cell[0]
    y0 = start_cell[1]
    x1 = end_cell[0]
    y1 = end_cell[1]
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
        if cell_value in {OCCUPIED, UNKNOWN}:
            collision = True
            break
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx
    return collision


def is_free(coords, map_info):
    cell = get_cell_position_from_coords(coords, map_info)
    if 0 <= cell[0] < map_info.map.shape[1] and 0 <= cell[1] < map_info.map.shape[0]:
        return map_info.map[cell[1], cell[0]] == FREE
    return False


def check_collision_type(start, end, map_info):
    start_cell = get_cell_position_from_coords(start, map_info)
    end_cell = get_cell_position_from_coords(end, map_info)

    if _HAS_NATIVE_CC:
        grid = np.ascontiguousarray(map_info.map, dtype=np.int32)
        return _cc_native(
            grid, int(FREE), int(OCCUPIED), int(UNKNOWN),
            (int(start_cell[0]), int(start_cell[1])),
            (int(end_cell[0]), int(end_cell[1])),
        )

    grid = map_info.map.astype(np.int32)
    x0, y0 = start_cell[0], start_cell[1]
    x1, y1 = end_cell[0], end_cell[1]
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    x, y = x0, y0
    error = dx - dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2

    while 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]:
        k = grid.item(int(y), int(x))
        if x == x1 and y == y1:
            break
        if k == OCCUPIED:
            return OCCUPIED
        if k == UNKNOWN:
            return UNKNOWN
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx
    return FREE


# ============================================================
# Wavelet energy helpers (for adaptive distance threshold)
# ============================================================

def _normalize_minmax(array, eps=1e-6):
    array = np.asarray(array, dtype=np.float32)
    mn, mx = float(np.min(array)), float(np.max(array))
    if mx - mn < eps:
        return np.zeros_like(array, dtype=np.float32)
    return (array - mn) / (mx - mn)


def _average_pool_map(map_array, scale):
    map_array = np.asarray(map_array, dtype=np.float32)
    if scale <= 1:
        return map_array
    h, w = map_array.shape
    pad_h = (scale - h % scale) % scale
    pad_w = (scale - w % scale) % scale
    padded = np.pad(map_array, ((0, pad_h), (0, pad_w)), mode="edge")
    return padded.reshape(
        padded.shape[0] // scale, scale,
        padded.shape[1] // scale, scale,
    ).mean(axis=(1, 3)).astype(np.float32)


def _haar_detail_components(map_array):
    map_array = np.asarray(map_array, dtype=np.float32)
    h, w = map_array.shape
    padded = np.pad(map_array, ((0, h % 2), (0, w % 2)), mode="edge")
    a, b = padded[0::2, 0::2], padded[0::2, 1::2]
    c, d = padded[1::2, 0::2], padded[1::2, 1::2]
    lh = 0.5 * (a + b - c - d)
    hl = 0.5 * (a - b + c - d)
    hh = 0.5 * (a - b - c + d)

    def _up(detail):
        return np.repeat(np.repeat(np.abs(detail), 2, axis=0), 2, axis=1)[:h, :w].astype(np.float32)

    return _up(lh), _up(hl), _up(hh)


def _upsample_to_base(arr, scale, shape):
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)[:shape[0], :shape[1]]


def _occupancy_to_wavelet_float(map_array):
    map_array = np.asarray(map_array)
    mapped = np.full(map_array.shape, 0.5, dtype=np.float32)
    mapped[map_array == FREE] = 0.0
    mapped[map_array == OCCUPIED] = 1.0
    return mapped


def compute_wavelet_energy_map(map_array, scales=None):
    from .parameter import NODE_RESOLUTION, WAVELET_DTH_SCALE_MULTS
    if scales is None:
        base_scale = max(1, int(round(NODE_RESOLUTION / max(CELL_SIZE, 1e-6))))
        scales = tuple(sorted({max(1, int(round(base_scale * m))) for m in WAVELET_DTH_SCALE_MULTS}))
    if not scales:
        return np.zeros_like(map_array, dtype=np.float32)

    base_map = _occupancy_to_wavelet_float(map_array)
    acc = np.zeros(base_map.shape, dtype=np.float32)
    for s in scales:
        pooled = _average_pool_map(base_map, s)
        lh, hl, hh = _haar_detail_components(pooled)
        lh = _normalize_minmax(_upsample_to_base(lh, s, base_map.shape))
        hl = _normalize_minmax(_upsample_to_base(hl, s, base_map.shape))
        hh = _normalize_minmax(_upsample_to_base(hh, s, base_map.shape))
        acc += _normalize_minmax(np.sqrt(lh * lh + hl * hl + hh * hh))
    return _normalize_minmax(acc)


def wavelet_scalar_at_coords(coords, map_info, wavelet_map):
    if wavelet_map is None or np.size(wavelet_map) == 0:
        return 0.0
    cell = get_cell_position_from_coords(np.asarray(coords), map_info, check_negative=False)
    cx = int(np.clip(cell[0], 0, wavelet_map.shape[1] - 1))
    cy = int(np.clip(cell[1], 0, wavelet_map.shape[0] - 1))
    return float(wavelet_map[cy, cx])


def extract_local_map_info(location, map_info, window_size_m):
    window_size_m = float(window_size_m)
    if window_size_m <= 0.0:
        return map_info
    half = window_size_m / 2.0
    ox = max(location[0] - half, map_info.map_origin_x)
    oy = max(location[1] - half, map_info.map_origin_y)
    tx = min(location[0] + half,
             map_info.map_origin_x + (map_info.map.shape[1] - 1) * map_info.cell_size)
    ty = min(location[1] + half,
             map_info.map_origin_y + (map_info.map.shape[0] - 1) * map_info.cell_size)
    if tx <= ox or ty <= oy:
        return map_info
    cs = map_info.cell_size
    ox = np.round((ox // cs + 1) * cs, 1)
    oy = np.round((oy // cs + 1) * cs, 1)
    tx = np.round((tx // cs) * cs, 1)
    ty = np.round((ty // cs) * cs, 1)
    if tx <= ox or ty <= oy:
        return map_info
    o_cell = get_cell_position_from_coords(np.array([ox, oy]), map_info, check_negative=False)
    t_cell = get_cell_position_from_coords(np.array([tx, ty]), map_info, check_negative=False)
    x0 = int(np.clip(o_cell[0], 0, map_info.map.shape[1] - 1))
    y0 = int(np.clip(o_cell[1], 0, map_info.map.shape[0] - 1))
    x1 = int(np.clip(t_cell[0], 0, map_info.map.shape[1] - 1))
    y1 = int(np.clip(t_cell[1], 0, map_info.map.shape[0] - 1))
    if x1 <= x0 or y1 <= y0:
        return map_info
    return MapInfo(map_info.map[y0:y1 + 1, x0:x1 + 1], ox, oy, cs)


class MapInfo:
    def __init__(self, map, map_origin_x, map_origin_y, cell_size):
        self.map = map
        self.map_origin_x = map_origin_x
        self.map_origin_y = map_origin_y
        self.cell_size = cell_size
