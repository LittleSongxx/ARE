from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import SBGEConfig


METRIC_KEYS = [
    "explored_rate",
    "travel_dist",
    "episode_return",
    "episode_cost",
    "collision_count",
    "near_miss_count",
    "risk_integral",
    "budget_violation",
    "return_success",
    "shield_interventions",
    "unsafe_action_proposals",
]


def config_to_dict(config: SBGEConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key in ("maps_dir", "result_dir", "split_file"):
        if payload.get(key) is not None:
            payload[key] = str(payload[key])
    return payload


def write_json(path: str | Path, payload: Any) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    names = fieldnames or (list(rows[0].keys()) if rows else [])
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def aggregate_metric_rows(
    rows: list[dict[str, Any]],
    group_keys: list[str],
    metric_keys: list[str] | None = None,
) -> list[dict[str, Any]]:
    metrics = metric_keys or METRIC_KEYS
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[group_key] for group_key in group_keys)
        groups.setdefault(key, []).append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for key, group_rows in sorted(groups.items(), key=lambda item: item[0]):
        out = {group_key: value for group_key, value in zip(group_keys, key)}
        out["count"] = len(group_rows)
        for metric in metrics:
            values = [float(row[metric]) for row in group_rows if metric in row and row[metric] is not None]
            if values:
                arr = np.asarray(values, dtype=float)
                out[f"{metric}_mean"] = float(np.mean(arr))
                out[f"{metric}_std"] = float(np.std(arr))
                out[f"{metric}_count"] = int(len(values))
            else:
                out[f"{metric}_mean"] = 0.0
                out[f"{metric}_std"] = 0.0
                out[f"{metric}_count"] = 0
        aggregate_rows.append(out)
    return aggregate_rows
