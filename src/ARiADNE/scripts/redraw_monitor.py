#!/usr/bin/env python3
"""
Redraw the latest ARiADNE training monitor:
source /root/miniconda3/etc/profile.d/conda.sh && conda activate ros_conda && cd /root/ros_ws/ARE/src && python -m ARiADNE.scripts.redraw_monitor

Redraw a specific run session:
source /root/miniconda3/etc/profile.d/conda.sh && conda activate ros_conda && cd /root/ros_ws/ARE/src && python -m ARiADNE.scripts.redraw_monitor --run-session 2026_0304_1200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ARiADNE.parameter import RuntimeConfig, get_monitor_path, get_result_root, model_path
from ARiADNE.training_monitor import TrainingMonitor


def get_monitor_history_path(monitor_dir: str | Path) -> Path:
    return Path(monitor_dir).expanduser().resolve() / ".state" / "training_history.json"


def find_latest_monitor_dir(result_root: str | Path | None = None) -> Path:
    result_root = Path(result_root or get_result_root()).expanduser().resolve()
    if not result_root.exists():
        raise FileNotFoundError(f"Result root does not exist: {result_root}")

    candidates = []
    for session_dir in result_root.iterdir():
        if not session_dir.is_dir():
            continue
        monitor_dir = session_dir / "train" / "monitor"
        history_file = get_monitor_history_path(monitor_dir)
        if history_file.is_file():
            candidates.append(monitor_dir)

    if not candidates:
        raise FileNotFoundError(f"No training monitor history found under {result_root}")
    return max(candidates, key=lambda path: path.parents[1].name)


def _run_session_from_checkpoint(checkpoint_path: str | Path) -> str:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    model_root = model_path.resolve()
    try:
        relative = checkpoint_path.relative_to(model_root)
    except ValueError as exc:
        raise ValueError(f"Checkpoint must be inside {model_root}") from exc

    if len(relative.parts) != 2:
        raise ValueError(f"Checkpoint must match model/<run_session>/<checkpoint_name>.pth: {checkpoint_path}")
    return relative.parts[0]


def resolve_monitor_dir(args: argparse.Namespace) -> Path:
    if args.monitor_dir:
        return Path(args.monitor_dir).expanduser().resolve()
    if args.run_session:
        return get_monitor_path(RuntimeConfig(run_session=args.run_session)).resolve()
    if args.checkpoint:
        run_session = _run_session_from_checkpoint(args.checkpoint)
        return get_monitor_path(RuntimeConfig(run_session=run_session)).resolve()
    return find_latest_monitor_dir(args.result_root)


def redraw_monitor(monitor_dir: str | Path) -> tuple[Path, Path]:
    monitor_dir = Path(monitor_dir).expanduser().resolve()
    history_file = get_monitor_history_path(monitor_dir)
    if not history_file.is_file():
        raise FileNotFoundError(f"Training monitor history not found: {history_file}")

    monitor = TrainingMonitor(monitor_dir)
    if not monitor.plot_file.is_file():
        raise RuntimeError(f"Failed to redraw monitor plot: {monitor.plot_file}")
    return history_file, monitor.plot_file.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--run-session", dest="run_session")
    selector.add_argument("--monitor-dir", dest="monitor_dir")
    selector.add_argument("--checkpoint")
    parser.add_argument("--result-root", dest="result_root")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    monitor_dir = resolve_monitor_dir(args)
    history_file, plot_file = redraw_monitor(monitor_dir)
    print(f"monitor_dir={Path(monitor_dir).resolve()}")
    print(f"history_file={history_file}")
    print(f"plot_file={plot_file}")


if __name__ == "__main__":
    main()
