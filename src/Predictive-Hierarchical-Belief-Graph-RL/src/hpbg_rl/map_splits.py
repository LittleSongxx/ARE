from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from skimage import io


VALID_SPLITS = ("train", "val", "test")
MAP_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".pgm", ".tif", ".tiff"}
MANIFEST_VERSION = 1


@dataclass(frozen=True)
class MapSplitEntry:
    path: str
    name: str
    sha256: str
    width: int
    height: int
    free_area: int
    free_ratio: float
    split: str
    size_bucket: str
    source: str


@dataclass(frozen=True)
class MapSplitManifest:
    version: int
    root: str
    seed: int
    entries: tuple[MapSplitEntry, ...]
    generated_from: str

    def split_entries(self, split: str) -> list[MapSplitEntry]:
        split = normalize_split(split)
        return [entry for entry in self.entries if entry.split == split]

    def split_paths(self, split: str, count: int | None = None) -> list[Path]:
        paths = [Path(entry.path).expanduser().resolve() for entry in self.split_entries(split)]
        if count is None:
            return paths
        return paths[: max(int(count), 0)]

    def to_dict(self) -> dict[str, object]:
        return {
            "version": int(self.version),
            "root": self.root,
            "seed": int(self.seed),
            "generated_from": self.generated_from,
            "entries": [asdict(entry) for entry in self.entries],
        }

    def content_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class MapSplitError(ValueError):
    pass


def normalize_split(split: str) -> str:
    normalized = str(split).strip().lower()
    if normalized == "validation":
        normalized = "val"
    if normalized not in VALID_SPLITS:
        raise MapSplitError(f"Unknown map split: {split}")
    return normalized


def _config_value(config: object, name: str, default: object = None) -> object:
    return getattr(config, name, default)


def _as_optional_path(value: object) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text).expanduser().resolve()


def _list_map_files(directory: Path | None) -> list[Path]:
    if directory is None or not directory.exists():
        return []
    return sorted(
        path.resolve()
        for path in directory.iterdir()
        if path.is_file() and (path.suffix.lower() in MAP_EXTENSIONS or not path.suffix)
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_map_metadata(path: Path) -> tuple[int, int, int, float]:
    try:
        image = io.imread(str(path), as_gray=True)
        array = np.asarray(image)
        if array.ndim == 3:
            array = array[..., 0]
        height, width = int(array.shape[0]), int(array.shape[1])
        if array.max(initial=0) <= 1.0:
            scaled = array * 255.0
        else:
            scaled = array.astype(float)
        free_mask = (scaled > 150.0) | ((scaled <= 80.0) & (scaled >= 50.0))
        free_area = int(np.count_nonzero(free_mask))
        total = max(int(width * height), 1)
        return width, height, free_area, float(free_area / total)
    except Exception:
        return 0, 0, 0, 0.0


def _size_bucket(width: int, height: int, free_area: int, min_free_area: int, min_side: int) -> str:
    shortest_side = min(int(width), int(height))
    longest_side = max(int(width), int(height))
    if free_area >= int(min_free_area) and shortest_side >= int(min_side) and (min_free_area > 0 or min_side > 0):
        return "large"
    if longest_side >= 512 or free_area >= 160_000:
        return "large"
    if longest_side >= 256 or free_area >= 40_000:
        return "medium"
    return "small"


def _entry_for_path(
    path: Path,
    split: str,
    source: str,
    min_free_area: int,
    min_side: int,
) -> MapSplitEntry:
    width, height, free_area, free_ratio = _read_map_metadata(path)
    return MapSplitEntry(
        path=str(path.expanduser().resolve()),
        name=path.name,
        sha256=_sha256_file(path),
        width=width,
        height=height,
        free_area=free_area,
        free_ratio=free_ratio,
        split=normalize_split(split),
        size_bucket=_size_bucket(width, height, free_area, min_free_area, min_side),
        source=source,
    )


def _validate_entries(entries: Sequence[MapSplitEntry]) -> tuple[MapSplitEntry, ...]:
    seen: dict[str, str] = {}
    normalized_entries = []
    for entry in entries:
        split = normalize_split(entry.split)
        path = str(Path(entry.path).expanduser().resolve())
        previous_split = seen.get(path)
        if previous_split is not None:
            raise MapSplitError(f"Map appears in multiple split slots: {path} ({previous_split}, {split})")
        seen[path] = split
        normalized_entries.append(
            MapSplitEntry(
                path=path,
                name=entry.name or Path(path).name,
                sha256=entry.sha256,
                width=int(entry.width),
                height=int(entry.height),
                free_area=int(entry.free_area),
                free_ratio=float(entry.free_ratio),
                split=split,
                size_bucket=str(entry.size_bucket or "unknown"),
                source=str(entry.source or "manifest"),
            )
        )
    return tuple(sorted(normalized_entries, key=lambda item: (VALID_SPLITS.index(item.split), item.name, item.path)))


def _manifest_root(payload: dict[str, object], manifest_path: Path) -> Path:
    raw_root = payload.get("root")
    if isinstance(raw_root, str) and raw_root.strip():
        root = Path(raw_root).expanduser()
        if not root.is_absolute():
            root = manifest_path.parent / root
        return root.resolve()
    return manifest_path.parent.resolve()


def _entry_from_manifest_item(
    item: object,
    split: str,
    root: Path,
    min_free_area: int,
    min_side: int,
) -> MapSplitEntry:
    if isinstance(item, str):
        raw_path = item
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = root / path
        return _entry_for_path(path.resolve(), split, "manifest", min_free_area, min_side)

    if not isinstance(item, dict):
        raise MapSplitError(f"Invalid manifest entry for split {split}: {item!r}")

    raw_path = item.get("path")
    if not raw_path:
        raise MapSplitError(f"Manifest entry missing path for split {split}: {item!r}")
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()

    if path.exists():
        measured = _entry_for_path(path, split, str(item.get("source") or "manifest"), min_free_area, min_side)
        sha256 = str(item.get("sha256") or measured.sha256)
        width = int(item.get("width") or measured.width)
        height = int(item.get("height") or measured.height)
        free_area = int(item.get("free_area") or measured.free_area)
        free_ratio = float(item.get("free_ratio") if item.get("free_ratio") is not None else measured.free_ratio)
        size_bucket = str(item.get("size_bucket") or measured.size_bucket)
    else:
        sha256 = str(item.get("sha256") or "")
        width = int(item.get("width") or 0)
        height = int(item.get("height") or 0)
        free_area = int(item.get("free_area") or 0)
        free_ratio = float(item.get("free_ratio") or 0.0)
        size_bucket = str(item.get("size_bucket") or _size_bucket(width, height, free_area, min_free_area, min_side))

    return MapSplitEntry(
        path=str(path),
        name=str(item.get("name") or path.name),
        sha256=sha256,
        width=width,
        height=height,
        free_area=free_area,
        free_ratio=free_ratio,
        split=str(item.get("split") or split),
        size_bucket=size_bucket,
        source=str(item.get("source") or "manifest"),
    )


def load_split_manifest(
    manifest_path: str | Path,
    *,
    min_free_area: int = 0,
    min_side: int = 0,
) -> MapSplitManifest:
    path = Path(manifest_path).expanduser().resolve()
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise MapSplitError(f"Split manifest must be a JSON object: {path}")

    root = _manifest_root(payload, path)
    entries: list[MapSplitEntry] = []
    if isinstance(payload.get("entries"), list):
        for item in payload["entries"]:
            if not isinstance(item, dict):
                raise MapSplitError(f"Invalid manifest entry: {item!r}")
            split = str(item.get("split") or "")
            entries.append(_entry_from_manifest_item(item, split, root, min_free_area, min_side))
    elif isinstance(payload.get("splits"), dict):
        splits = payload["splits"]
        for split in VALID_SPLITS:
            for item in splits.get(split, []) or []:
                entries.append(_entry_from_manifest_item(item, split, root, min_free_area, min_side))
    else:
        raise MapSplitError(f"Split manifest must contain entries or splits: {path}")

    return MapSplitManifest(
        version=int(payload.get("version") or MANIFEST_VERSION),
        root=str(root),
        seed=int(payload.get("seed") or 0),
        generated_from=str(payload.get("generated_from") or path),
        entries=_validate_entries(entries),
    )


def _build_from_split_dirs(config: object) -> MapSplitManifest | None:
    maps_dir = _as_optional_path(_config_value(config, "maps_dir"))
    explicit_dirs = {
        "train": _as_optional_path(_config_value(config, "train_maps_dir")),
        "val": _as_optional_path(_config_value(config, "val_maps_dir")),
        "test": _as_optional_path(_config_value(config, "test_maps_dir")),
    }
    if maps_dir is not None:
        for split in VALID_SPLITS:
            explicit_dirs[split] = explicit_dirs[split] or (maps_dir / split if (maps_dir / split).is_dir() else None)

    if not any(directory is not None and directory.exists() for directory in explicit_dirs.values()):
        return None

    min_free_area = int(_config_value(config, "large_scale_min_free_area", 0) or 0)
    min_side = int(_config_value(config, "large_scale_min_side", 0) or 0)
    entries: list[MapSplitEntry] = []
    for split, directory in explicit_dirs.items():
        for path in _list_map_files(directory):
            entries.append(_entry_for_path(path, split, f"{split}_dir", min_free_area, min_side))

    root = maps_dir or next(directory for directory in explicit_dirs.values() if directory is not None)
    return MapSplitManifest(
        version=MANIFEST_VERSION,
        root=str(root.resolve()),
        seed=int(_config_value(config, "split_seed", 0) or 0),
        generated_from="split_dirs",
        entries=_validate_entries(entries),
    )


def _split_counts(total: int) -> tuple[int, int, int]:
    if total <= 0:
        return 0, 0, 0
    if total == 1:
        return 1, 0, 0
    if total == 2:
        return 1, 1, 0
    test_count = max(1, int(round(total * 0.15)))
    val_count = max(1, int(round(total * 0.15)))
    if test_count + val_count >= total:
        test_count = 1
        val_count = 1
    train_count = total - val_count - test_count
    return train_count, val_count, test_count


def _build_from_single_dir(config: object) -> MapSplitManifest:
    maps_dir = _as_optional_path(_config_value(config, "maps_dir"))
    if maps_dir is None:
        from . import parameter as parameter_module

        maps_dir = Path(parameter_module.MAPS_DIR).expanduser().resolve()

    map_files = _list_map_files(maps_dir)
    if not map_files:
        raise FileNotFoundError(f"No map files found in {maps_dir}")

    seed = int(_config_value(config, "split_seed", 2024) or 2024)
    rng = random.Random(seed)
    shuffled = list(map_files)
    rng.shuffle(shuffled)

    min_free_area = int(_config_value(config, "large_scale_min_free_area", 0) or 0)
    min_side = int(_config_value(config, "large_scale_min_side", 0) or 0)
    train_count, val_count, test_count = _split_counts(len(shuffled))

    measured = [_entry_for_path(path, "train", "single_dir", min_free_area, min_side) for path in shuffled]
    eligible_large = [entry for entry in measured if entry.size_bucket == "large"]
    selected_test_paths: set[str] = set()
    if test_count > 0 and eligible_large:
        selected_test_paths = {entry.path for entry in eligible_large[:test_count]}

    remaining = [entry for entry in measured if entry.path not in selected_test_paths]
    test_entries = [entry for entry in measured if entry.path in selected_test_paths]
    if len(test_entries) < test_count:
        needed = test_count - len(test_entries)
        test_entries.extend(remaining[-needed:])
        remaining = remaining[:-needed]

    val_entries = remaining[-val_count:] if val_count > 0 else []
    train_entries = remaining[:train_count]

    entries = []
    for split, group in (("train", train_entries), ("val", val_entries), ("test", test_entries)):
        for entry in group:
            entries.append(
                MapSplitEntry(
                    path=entry.path,
                    name=entry.name,
                    sha256=entry.sha256,
                    width=entry.width,
                    height=entry.height,
                    free_area=entry.free_area,
                    free_ratio=entry.free_ratio,
                    split=split,
                    size_bucket=entry.size_bucket,
                    source="single_dir",
                )
            )

    return MapSplitManifest(
        version=MANIFEST_VERSION,
        root=str(maps_dir.resolve()),
        seed=seed,
        generated_from="single_dir",
        entries=_validate_entries(entries),
    )


def materialize_split_manifest(config: object) -> MapSplitManifest:
    manifest_path = _as_optional_path(_config_value(config, "split_manifest_path"))
    min_free_area = int(_config_value(config, "large_scale_min_free_area", 0) or 0)
    min_side = int(_config_value(config, "large_scale_min_side", 0) or 0)
    if manifest_path is not None:
        return load_split_manifest(manifest_path, min_free_area=min_free_area, min_side=min_side)

    split_dir_manifest = _build_from_split_dirs(config)
    if split_dir_manifest is not None:
        return split_dir_manifest

    return _build_from_single_dir(config)


def save_split_manifest(manifest: MapSplitManifest, output_path: str | Path) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
    return path


def save_runtime_split_manifest(config: object, protocol_dir: str | Path) -> Path:
    manifest = materialize_split_manifest(config)
    return save_split_manifest(manifest, Path(protocol_dir) / "split_manifest.json")


def manifest_entry_by_path(manifest: MapSplitManifest) -> dict[str, MapSplitEntry]:
    return {str(Path(entry.path).expanduser().resolve()): entry for entry in manifest.entries}


def resolve_map_split_paths(
    config: object,
    split: str,
    *,
    count: int | None = None,
    allow_train_eval: bool = False,
) -> list[Path]:
    split = normalize_split(split)
    if split == "train" and not allow_train_eval:
        raise MapSplitError("Evaluation is not allowed to consume the train split by default")
    manifest = materialize_split_manifest(config)
    return manifest.split_paths(split, count=count)


def split_count_for_eval(config: object, split: str) -> int:
    split = normalize_split(split)
    if split == "val":
        value = _config_value(config, "val_map_count", None)
        if value is not None:
            return max(int(value), 0)
        return max(int(_config_value(config, "auto_eval_map_count", 0) or 0), 0)
    if split == "test":
        value = _config_value(config, "test_map_count", None)
        if value is not None:
            return max(int(value), 0)
        return max(int(_config_value(config, "auto_eval_map_count", 0) or 0), 0)
    return max(int(_config_value(config, "auto_eval_map_count", 0) or 0), 0)