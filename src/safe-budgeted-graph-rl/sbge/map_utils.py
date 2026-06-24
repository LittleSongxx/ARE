from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage import io
from skimage.measure import block_reduce

FREE = 255
OCCUPIED = 1
UNKNOWN = 127


def list_map_files(maps_dir: str | Path) -> list[Path]:
    maps_path = Path(maps_dir).expanduser().resolve()
    files = sorted(path for path in maps_path.iterdir() if path.is_file() and path.suffix.lower() in {".png", ".jpg"})
    if not files:
        raise FileNotFoundError(f"No map image files found in {maps_path}")
    return files


def load_large_drl_map(map_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (free_mask, start_cell_xy) using the large-scale-DRL map convention."""
    path = Path(map_path).expanduser().resolve()
    raw = (io.imread(str(path), as_gray=True) * 255).astype(int)
    raw = block_reduce(raw, 2, np.min)

    marker_yx = np.argwhere(raw == 208)
    if marker_yx.size > 0:
        marker = marker_yx[min(10, len(marker_yx) - 1)]
        start = np.array([marker[1], marker[0]], dtype=float)
    else:
        candidate_yx = np.argwhere((raw > 150) | ((raw <= 80) & (raw >= 50)))
        if candidate_yx.size == 0:
            raise ValueError(f"Map has no free start candidate: {path}")
        marker = candidate_yx[len(candidate_yx) // 2]
        start = np.array([marker[1], marker[0]], dtype=float)

    free_mask = (raw > 150) | ((raw <= 80) & (raw >= 50))
    free_mask[int(start[1]), int(start[0])] = True
    return free_mask.astype(bool), start


def bresenham_cells(start_xy: np.ndarray, end_xy: np.ndarray) -> np.ndarray:
    x0, y0 = np.round(start_xy).astype(int)
    x1, y1 = np.round(end_xy).astype(int)
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    cells = []
    x, y = x0, y0
    while True:
        cells.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
    return np.array(cells, dtype=int)


def in_bounds(cells_xy: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    return (
        (cells_xy[:, 0] >= 0)
        & (cells_xy[:, 0] < w)
        & (cells_xy[:, 1] >= 0)
        & (cells_xy[:, 1] < h)
    )
