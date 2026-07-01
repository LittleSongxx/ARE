from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .map_utils import list_map_files


SPLIT_NAMES = ("train", "val", "test")


def create_map_split(
    maps_dir: str | Path,
    seed: int = 0,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> dict[str, list[str]]:
    maps_path = Path(maps_dir).expanduser().resolve()
    files = list_map_files(maps_path)
    names = [str(path.relative_to(maps_path)) for path in files]
    rng = np.random.default_rng(int(seed))
    shuffled = list(names)
    rng.shuffle(shuffled)

    n_files = len(shuffled)
    if n_files >= 3:
        n_train = max(1, int(n_files * train_ratio))
        n_val = max(1, int(n_files * val_ratio))
        if n_train + n_val >= n_files:
            n_val = max(1, n_files - n_train - 1)
    elif n_files == 2:
        n_train, n_val = 1, 0
    else:
        n_train, n_val = n_files, 0

    return {
        "train": sorted(shuffled[:n_train]),
        "val": sorted(shuffled[n_train : n_train + n_val]),
        "test": sorted(shuffled[n_train + n_val :]),
    }


def save_map_split(split: dict[str, list[str]], output_path: str | Path) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=2, sort_keys=True))
    return path


def load_map_split(split_file: str | Path) -> dict[str, list[str]]:
    path = Path(split_file).expanduser().resolve()
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Split file must contain a JSON object: {path}")
    split: dict[str, list[str]] = {}
    for name in SPLIT_NAMES:
        values = payload.get(name, [])
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            raise ValueError(f"Split '{name}' must be a list of relative map paths")
        split[name] = list(values)
    return split


def load_map_files_for_config(config: Any) -> list[Path]:
    maps_dir = Path(config.maps_dir).expanduser().resolve()
    if config.split_file is None:
        files = list_map_files(maps_dir)
    else:
        split = load_map_split(config.split_file)
        if config.split_name not in split:
            raise ValueError(f"Unknown split '{config.split_name}'. Expected one of {sorted(split)}")
        files = [(maps_dir / relative_path).resolve() for relative_path in split[config.split_name]]
        missing = [path for path in files if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Split references missing map files: {missing[:3]}")
        if not files:
            raise ValueError(f"Split '{config.split_name}' in {config.split_file} has no maps")

    if config.map_limit is not None:
        limit = max(int(config.map_limit), 0)
        files = files[:limit]
    if not files:
        raise FileNotFoundError("No map files available after split/map_limit filtering")
    return files
