from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import ndimage
from skimage import io
from skimage.morphology import skeletonize

from ac_pbgrl.utils import atomic_write_json, sha256_file


@dataclass(frozen=True)
class MapRecord:
    name: str
    sha256: str
    canonical_d4: str
    width: int
    height: int
    free_fraction: float
    component_count: int
    boundary_density: float
    junction_density: float
    complexity: float
    split: str = ""


def _load_binary(path: Path) -> np.ndarray:
    image = io.imread(str(path), as_gray=True)
    values = np.asarray(image, dtype=np.float32)
    if values.max() > 1.5:
        values = values / 255.0
    return values > 0.55


def _d4_variants(image: np.ndarray) -> list[np.ndarray]:
    variants = []
    for rotations in range(4):
        rotated = np.rot90(image, rotations)
        variants.append(rotated)
        variants.append(np.fliplr(rotated))
    return variants


def canonical_d4_hash(image: np.ndarray) -> str:
    digests = []
    for variant in _d4_variants(image):
        packed = np.packbits(np.ascontiguousarray(variant), axis=None)
        header = f"{variant.shape[0]}x{variant.shape[1]}:".encode("ascii")
        digests.append(hashlib.sha256(header + packed.tobytes()).hexdigest())
    return min(digests)


def map_complexity(path: Path) -> MapRecord:
    free = _load_binary(path)
    _, components = ndimage.label(free)
    eroded = ndimage.binary_erosion(free)
    boundary = free ^ eroded
    skeleton = skeletonize(free)
    neighbor_count = ndimage.convolve(skeleton.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)) - skeleton
    junctions = skeleton & (neighbor_count >= 3)
    free_count = max(int(free.sum()), 1)
    free_fraction = float(free.mean())
    component_score = min(float(components) / 20.0, 1.0)
    boundary_density = float(boundary.sum() / free_count)
    junction_density = float(junctions.sum() / max(int(skeleton.sum()), 1))
    # Fixed, pre-registered score; no outcome-dependent tuning.
    complexity = (
        0.20 * (1.0 - abs(free_fraction - 0.5) * 2.0)
        + 0.20 * component_score
        + 0.30 * min(boundary_density / 0.5, 1.0)
        + 0.30 * min(junction_density / 0.2, 1.0)
    )
    return MapRecord(
        name=path.name,
        sha256=sha256_file(path),
        canonical_d4=canonical_d4_hash(free),
        width=int(free.shape[1]),
        height=int(free.shape[0]),
        free_fraction=free_fraction,
        component_count=int(components),
        boundary_density=boundary_density,
        junction_density=junction_density,
        complexity=float(complexity),
    )


def create_map_splits(
    maps_dir: str | Path,
    output_path: str | Path,
    *,
    seed: int = 4777,
    ood_fraction: float = 0.10,
) -> dict:
    maps_dir = Path(maps_dir)
    paths = sorted(maps_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"no maps found in {maps_dir}")
    records = [map_complexity(path) for path in paths]
    groups: dict[str, list[MapRecord]] = {}
    for record in records:
        groups.setdefault(record.canonical_d4, []).append(record)
    group_items = list(groups.items())
    group_items.sort(key=lambda item: (np.mean([record.complexity for record in item[1]]), item[0]))
    ood_count = max(1, int(round(len(group_items) * ood_fraction)))
    ood_keys = {key for key, _ in group_items[-ood_count:]}
    remaining = group_items[:-ood_count]

    rng = np.random.default_rng(seed)
    # Complexity-stratified allocation by sorting, then shuffling within ten bins.
    bins = np.array_split(np.arange(len(remaining)), min(10, max(1, len(remaining))))
    split_by_group: dict[str, str] = {key: "ood" for key in ood_keys}
    for bin_indices in bins:
        shuffled = np.asarray(bin_indices).copy()
        rng.shuffle(shuffled)
        count = len(shuffled)
        train_end = int(round(count * 0.8))
        validation_end = train_end + int(round(count * 0.1))
        for position, group_index in enumerate(shuffled):
            split = "train" if position < train_end else "validation" if position < validation_end else "iid_test"
            split_by_group[remaining[int(group_index)][0]] = split

    assigned = []
    for record in records:
        payload = asdict(record)
        payload["split"] = split_by_group[record.canonical_d4]
        assigned.append(payload)
    canonical_sets = {
        split: {record["canonical_d4"] for record in assigned if record["split"] == split}
        for split in ("train", "validation", "iid_test", "ood")
    }
    for left, left_values in canonical_sets.items():
        for right, right_values in canonical_sets.items():
            if left < right and left_values & right_values:
                raise AssertionError(f"D4-equivalent map leakage between {left} and {right}")
    manifest = {
        "version": 1,
        "seed": seed,
        "ood_fraction": ood_fraction,
        "maps_dir": str(maps_dir.resolve()),
        "records": assigned,
        "counts": {split: sum(item["split"] == split for item in assigned) for split in canonical_sets},
    }
    encoded_records = json.dumps(assigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest["split_hash"] = hashlib.sha256(encoded_records).hexdigest()
    atomic_write_json(Path(output_path), manifest)
    return manifest


def load_split_paths(manifest_path: str | Path, split: str) -> list[Path]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    maps_dir = Path(manifest["maps_dir"])
    paths = [maps_dir / item["name"] for item in manifest["records"] if item["split"] == split]
    if not paths:
        raise ValueError(f"split {split!r} is empty")
    return paths
