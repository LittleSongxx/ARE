from __future__ import annotations

import csv
import json
import math
import shutil
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.ndimage import convolve, distance_transform_edt
from skimage import io
from skimage.measure import block_reduce, label
from skimage.morphology import skeletonize

from .parameter import FREE, MAPS_DIR, PACKAGE_ROOT, RuntimeConfig
from .utils import compute_wavelet_maps


DEFAULT_BUCKET_NAMES = ("easy", "medium", "hard")
DEFAULT_COMPONENT_WEIGHTS = {
    "path_extent_score": 0.30,
    "topology_score": 0.25,
    "narrowness_score": 0.20,
    "texture_score": 0.15,
    "boundary_score": 0.10,
}
_NORMALIZED_METRICS = (
    "max_geodesic_norm",
    "junction_ratio",
    "dead_end_ratio",
    "corridor_ratio",
    "obstacle_ratio",
    "wavelet_mean",
    "wavelet_p90",
    "boundary_density",
)


def default_output_dir() -> Path:
    timestamp = datetime.utcnow().strftime("%Y_%m%d_%H%M%S")
    return PACKAGE_ROOT / "result" / "map_difficulty" / timestamp


def list_map_files(map_dir: str | Path = MAPS_DIR) -> list[Path]:
    directory = Path(map_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Map directory does not exist: {directory}")
    return sorted(path for path in directory.iterdir() if path.is_file())


def load_training_map(map_path: str | Path, downsample_factor: int = 2) -> tuple[np.ndarray, np.ndarray]:
    map_path = Path(map_path)
    if not map_path.is_file():
        raise FileNotFoundError(f"Map file does not exist: {map_path}")

    raw_map = (io.imread(map_path, as_gray=True) * 255).astype(np.int16)
    factor = max(int(downsample_factor), 1)
    if factor > 1:
        raw_map = block_reduce(raw_map, (factor, factor), np.min)

    free_mask = (raw_map > 150) | ((raw_map <= 80) & (raw_map >= 50))
    occupancy_map = free_mask.astype(np.int16) * (FREE - 1) + 1
    robot_cell = _resolve_robot_cell(raw_map, free_mask)
    robot_cell = _snap_to_nearest_free(robot_cell, free_mask)
    return occupancy_map.astype(np.int16), robot_cell.astype(np.int64)


def analyze_map_file(
    map_path: str | Path,
    corridor_clearance_cells: float = 3.0,
    downsample_factor: int = 2,
) -> dict[str, float | int | str]:
    occupancy_map, robot_cell = load_training_map(map_path, downsample_factor=downsample_factor)
    return analyze_occupancy_map(
        occupancy_map,
        robot_cell,
        map_path=Path(map_path),
        corridor_clearance_cells=corridor_clearance_cells,
    )


def analyze_occupancy_map(
    occupancy_map: np.ndarray,
    robot_cell: np.ndarray,
    map_path: str | Path | None = None,
    corridor_clearance_cells: float = 3.0,
) -> dict[str, float | int | str]:
    occupancy_map = np.asarray(occupancy_map, dtype=np.int16)
    free_mask = occupancy_map == FREE
    reachable_mask = _reachable_free_mask(free_mask, robot_cell)
    reachable_free = int(np.sum(reachable_mask))
    total_free = int(np.sum(free_mask))
    total_cells = int(occupancy_map.size)
    if reachable_free <= 0:
        raise ValueError("Map has no reachable free space from the selected robot cell")

    obstacle_ratio = float(np.sum(~free_mask)) / float(total_cells)
    reachable_free_ratio = float(reachable_free) / float(total_cells)
    connected_free_ratio = float(reachable_free) / float(max(total_free, 1))

    clearance_map = distance_transform_edt(free_mask)
    corridor_ratio = float(
        np.mean(clearance_map[reachable_mask] <= max(float(corridor_clearance_cells), 1e-6))
    )

    occupied_neighbors = convolve(
        (~free_mask).astype(np.int16),
        np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.int16),
        mode="constant",
        cval=1,
    )
    boundary_density = float(np.mean(occupied_neighbors[reachable_mask] > 0))

    skeleton_mask = skeletonize(reachable_mask)
    dead_end_ratio = 0.0
    junction_ratio = 0.0
    skeleton_density = 0.0
    if np.any(skeleton_mask):
        neighbor_count = convolve(
            skeleton_mask.astype(np.int16),
            np.ones((3, 3), dtype=np.int16),
            mode="constant",
            cval=0,
        ) - skeleton_mask.astype(np.int16)
        skeleton_neighbors = neighbor_count[skeleton_mask]
        dead_end_ratio = float(np.mean(skeleton_neighbors <= 1))
        junction_ratio = float(np.mean(skeleton_neighbors >= 3))
        skeleton_density = float(np.sum(skeleton_mask)) / float(reachable_free)

    max_geodesic = _max_geodesic_distance(reachable_mask, robot_cell)
    max_geodesic_norm = float(max_geodesic) / float(max(math.sqrt(reachable_free), 1.0))

    wavelet_config = RuntimeConfig(
        use_wavelet_feature=True,
        wavelet_feature_mode="scalar",
        wavelet_scales_auto=False,
        wavelet_scales=(1, 2, 4),
        wavelet_norm_method="minmax",
    )
    wavelet_maps = compute_wavelet_maps(occupancy_map, wavelet_config)
    wavelet_values = wavelet_maps.scalar_map[reachable_mask]
    wavelet_mean = float(np.mean(wavelet_values))
    wavelet_std = float(np.std(wavelet_values))
    wavelet_p90 = float(np.percentile(wavelet_values, 90.0))

    return {
        "map_path": str(Path(map_path).resolve()) if map_path is not None else "",
        "map_name": Path(map_path).name if map_path is not None else "<memory>",
        "robot_col": int(robot_cell[0]),
        "robot_row": int(robot_cell[1]),
        "total_cells": total_cells,
        "total_free_cells": total_free,
        "reachable_free_cells": reachable_free,
        "reachable_free_ratio": reachable_free_ratio,
        "connected_free_ratio": connected_free_ratio,
        "obstacle_ratio": obstacle_ratio,
        "corridor_ratio": corridor_ratio,
        "boundary_density": boundary_density,
        "dead_end_ratio": dead_end_ratio,
        "junction_ratio": junction_ratio,
        "skeleton_density": skeleton_density,
        "max_geodesic": float(max_geodesic),
        "max_geodesic_norm": max_geodesic_norm,
        "wavelet_mean": wavelet_mean,
        "wavelet_std": wavelet_std,
        "wavelet_p90": wavelet_p90,
    }


def score_map_records(
    records: list[dict[str, float | int | str]],
    component_weights: dict[str, float] | None = None,
) -> list[dict[str, float | int | str]]:
    if not records:
        return []

    component_weights = dict(component_weights or DEFAULT_COMPONENT_WEIGHTS)
    normalized = {
        key: _normalize_series([float(record[key]) for record in records]) for key in _NORMALIZED_METRICS
    }

    scored_records: list[dict[str, float | int | str]] = []
    for index, record in enumerate(records):
        enriched = dict(record)
        for key, values in normalized.items():
            enriched[f"{key}_norm"] = float(values[index])

        enriched["path_extent_score"] = float(enriched["max_geodesic_norm_norm"])
        enriched["topology_score"] = float(
            0.5 * enriched["junction_ratio_norm"] + 0.5 * enriched["dead_end_ratio_norm"]
        )
        enriched["narrowness_score"] = float(
            0.7 * enriched["corridor_ratio_norm"] + 0.3 * enriched["obstacle_ratio_norm"]
        )
        enriched["texture_score"] = float(
            0.5 * enriched["wavelet_mean_norm"] + 0.5 * enriched["wavelet_p90_norm"]
        )
        enriched["boundary_score"] = float(enriched["boundary_density_norm"])
        enriched["difficulty_score"] = float(
            sum(component_weights[key] * float(enriched[key]) for key in component_weights)
        )
        scored_records.append(enriched)

    return sorted(scored_records, key=lambda record: (float(record["difficulty_score"]), str(record["map_name"])))


def assign_difficulty_buckets(
    records: list[dict[str, float | int | str]],
    bucket_names: tuple[str, ...] = DEFAULT_BUCKET_NAMES,
) -> list[dict[str, float | int | str]]:
    if not records:
        return []

    ordered_records = [
        dict(record) for record in sorted(records, key=lambda item: (float(item["difficulty_score"]), str(item["map_name"])))
    ]
    bucket_count = max(len(bucket_names), 1)
    total = len(ordered_records)
    bucket_sizes = [total // bucket_count] * bucket_count
    for index in range(total % bucket_count):
        bucket_sizes[index] += 1

    cursor = 0
    for bucket_name, bucket_size in zip(bucket_names, bucket_sizes):
        for record in ordered_records[cursor : cursor + bucket_size]:
            record["difficulty_bucket"] = bucket_name
        cursor += bucket_size

    for record in ordered_records[cursor:]:
        record["difficulty_bucket"] = bucket_names[-1]

    return ordered_records


def write_difficulty_outputs(
    records: list[dict[str, float | int | str]],
    output_dir: str | Path,
    link_mode: str = "symlink",
    clear_output: bool = False,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not clear_output:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. Pass clear_output=True or choose a new path."
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    bucket_root = output_dir / "buckets"
    bucket_root.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "difficulty_scores.csv"
    json_path = output_dir / "difficulty_manifest.json"
    summary_path = output_dir / "summary.json"

    fieldnames = sorted({key for record in records for key in record})
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    manifest = {
        "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "records": records,
    }
    json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    bucket_counts: dict[str, int] = {}
    for record in records:
        bucket = str(record.get("difficulty_bucket", "unassigned"))
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        bucket_dir = bucket_root / bucket
        bucket_dir.mkdir(parents=True, exist_ok=True)
        _materialize_bucket_entry(Path(str(record["map_path"])), bucket_dir / str(record["map_name"]), link_mode)

    summary = {
        "map_count": len(records),
        "bucket_counts": bucket_counts,
        "score_min": float(min(float(record["difficulty_score"]) for record in records)) if records else 0.0,
        "score_max": float(max(float(record["difficulty_score"]) for record in records)) if records else 0.0,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "output_dir": output_dir,
        "csv_path": csv_path,
        "json_path": json_path,
        "summary_path": summary_path,
        "bucket_root": bucket_root,
    }


def build_difficulty_dataset(
    maps_dir: str | Path = MAPS_DIR,
    corridor_clearance_cells: float = 3.0,
    downsample_factor: int = 2,
    bucket_names: tuple[str, ...] = DEFAULT_BUCKET_NAMES,
    max_maps: int | None = None,
) -> list[dict[str, float | int | str]]:
    map_files = list_map_files(maps_dir)
    if max_maps is not None:
        map_files = map_files[: max(int(max_maps), 0)]
    records = [
        analyze_map_file(path, corridor_clearance_cells=corridor_clearance_cells, downsample_factor=downsample_factor)
        for path in map_files
    ]
    return assign_difficulty_buckets(score_map_records(records), bucket_names=bucket_names)


def _resolve_robot_cell(raw_map: np.ndarray, free_mask: np.ndarray) -> np.ndarray:
    start_candidates = np.argwhere(raw_map == 208)
    if len(start_candidates) > 0:
        candidate = start_candidates[min(len(start_candidates) - 1, 10)]
        return np.array([candidate[1], candidate[0]], dtype=np.int64)

    free_candidates = np.argwhere(free_mask)
    if len(free_candidates) == 0:
        raise ValueError("Map does not contain any free cells")
    center = np.array([(raw_map.shape[1] - 1) / 2.0, (raw_map.shape[0] - 1) / 2.0], dtype=np.float64)
    deltas = free_candidates[:, ::-1].astype(np.float64) - center.reshape(1, 2)
    best_index = int(np.argmin(np.sum(deltas * deltas, axis=1)))
    best = free_candidates[best_index]
    return np.array([best[1], best[0]], dtype=np.int64)


def _snap_to_nearest_free(robot_cell: np.ndarray, free_mask: np.ndarray) -> np.ndarray:
    col = int(np.clip(robot_cell[0], 0, free_mask.shape[1] - 1))
    row = int(np.clip(robot_cell[1], 0, free_mask.shape[0] - 1))
    if free_mask[row, col]:
        return np.array([col, row], dtype=np.int64)

    free_candidates = np.argwhere(free_mask)
    if len(free_candidates) == 0:
        raise ValueError("Map does not contain any free cells")
    deltas = free_candidates[:, ::-1].astype(np.float64) - np.array([col, row], dtype=np.float64).reshape(1, 2)
    best_index = int(np.argmin(np.sum(deltas * deltas, axis=1)))
    best = free_candidates[best_index]
    return np.array([best[1], best[0]], dtype=np.int64)


def _reachable_free_mask(free_mask: np.ndarray, robot_cell: np.ndarray) -> np.ndarray:
    free_mask = np.asarray(free_mask, dtype=bool)
    labels = label(free_mask, connectivity=1)
    row = int(np.clip(robot_cell[1], 0, free_mask.shape[0] - 1))
    col = int(np.clip(robot_cell[0], 0, free_mask.shape[1] - 1))
    component_id = int(labels[row, col])
    if component_id == 0:
        snapped = _snap_to_nearest_free(np.array([col, row], dtype=np.int64), free_mask)
        component_id = int(labels[int(snapped[1]), int(snapped[0])])
    return labels == component_id


def _max_geodesic_distance(reachable_mask: np.ndarray, robot_cell: np.ndarray) -> int:
    rows, cols = reachable_mask.shape
    start_row = int(np.clip(robot_cell[1], 0, rows - 1))
    start_col = int(np.clip(robot_cell[0], 0, cols - 1))
    if not reachable_mask[start_row, start_col]:
        return 0

    distance = -np.ones((rows, cols), dtype=np.int32)
    queue: deque[tuple[int, int]] = deque()
    queue.append((start_row, start_col))
    distance[start_row, start_col] = 0
    max_distance = 0

    while queue:
        row, col = queue.popleft()
        next_distance = int(distance[row, col]) + 1
        for delta_row, delta_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            new_row = row + delta_row
            new_col = col + delta_col
            if new_row < 0 or new_row >= rows or new_col < 0 or new_col >= cols:
                continue
            if not reachable_mask[new_row, new_col] or distance[new_row, new_col] >= 0:
                continue
            distance[new_row, new_col] = next_distance
            max_distance = max(max_distance, next_distance)
            queue.append((new_row, new_col))
    return max_distance


def _normalize_series(values: list[float]) -> list[float]:
    if not values:
        return []
    min_value = min(values)
    max_value = max(values)
    if math.isclose(min_value, max_value, rel_tol=0.0, abs_tol=1e-12):
        return [0.0 for _ in values]
    scale = max_value - min_value
    return [(value - min_value) / scale for value in values]


def _materialize_bucket_entry(source: Path, destination: Path, link_mode: str) -> None:
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if link_mode == "copy":
        shutil.copy2(source, destination)
        return

    try:
        destination.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, destination)
