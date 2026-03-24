#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/mpl-large-drl-{os.environ.get('USER', 'user')}")

from parameter import RuntimeConfig, SMOKE_FOLDER_NAME, get_run_identity_from_checkpoint, resolve_resume_checkpoint
from runtime_utils import configure_matplotlib_cache


def _parse_bool01(raw_value: str | bool | None) -> bool:
    if isinstance(raw_value, bool):
        return raw_value
    value = str(raw_value).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected 0/1 or true/false, got: {raw_value}")


def _add_bool_switch(parser: argparse.ArgumentParser, flag: str, dest: str) -> None:
    parser.add_argument(flag, dest=dest, nargs="?", const="1", type=_parse_bool01, metavar="0|1", default=None)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume-from", dest="resume_from")
    parser.add_argument("--max-episodes", dest="max_episodes", type=int)
    parser.add_argument("--num-meta-agent", dest="num_meta_agent", type=int)
    parser.add_argument("--ray-num-cpus", dest="ray_num_cpus", type=int)
    parser.add_argument("--ray-worker-num-cpus", dest="ray_worker_num_cpus", type=int)
    parser.add_argument("--worker-num-threads", dest="worker_num_threads", type=int)
    parser.add_argument("--max-episode-step", dest="max_episode_step", type=int)
    parser.add_argument("--minimum-buffer-size", dest="minimum_buffer_size", type=int)
    parser.add_argument("--batch-size", dest="batch_size", type=int)
    parser.add_argument("--replay-size", dest="replay_size", type=int)
    parser.add_argument("--save-img-gap", dest="save_img_gap", type=int)
    parser.add_argument("--save-model-gap", dest="save_model_gap", type=int)
    parser.add_argument("--summary-window", dest="summary_window", type=int)
    parser.add_argument("--train-updates-per-iter", dest="train_updates_per_iter", type=int)
    parser.add_argument("--result-bucket-episodes", dest="result_bucket_episodes", type=int)
    parser.add_argument("--monitor-window", dest="monitor_window", type=int)
    parser.add_argument("--monitor-snapshot-interval", dest="monitor_snapshot_interval", type=int)
    parser.add_argument("--auto-eval-map-count", dest="auto_eval_map_count", type=int)
    parser.add_argument("--auto-eval-interval", dest="auto_eval_interval", type=int)
    parser.add_argument("--run-name", dest="run_name")
    parser.add_argument("--run-session", dest="run_session")
    parser.add_argument("--maps-dir", dest="maps_dir")
    parser.add_argument("--num-gpu", dest="num_gpu", type=int)
    parser.add_argument("--disable-training-monitor", action="store_true")
    parser.add_argument("--disable-auto-eval", action="store_true")
    _add_bool_switch(parser, "--load-model", "load_model")
    _add_bool_switch(parser, "--use-gpu", "use_gpu")
    _add_bool_switch(parser, "--use-gpu-global", "use_gpu_global")
    _add_bool_switch(parser, "--auto-eval-greedy", "auto_eval_greedy")
    args = parser.parse_args()

    if args.resume_from is not None:
        if args.smoke:
            parser.error("--resume-from cannot be used with --smoke")
        if args.run_session is not None:
            parser.error("--resume-from cannot be used with --run-session")
        resume_path, resume_session = resolve_resume_checkpoint(args.resume_from)
        resume_run_name, _ = get_run_identity_from_checkpoint(resume_path)
        args.resume_from = str(resume_path)
        args.resume_session = resume_session
        args.resume_run_name = resume_run_name
    else:
        args.resume_session = None
        args.resume_run_name = None

    return args


def build_runtime_config(args) -> RuntimeConfig:
    config = RuntimeConfig()
    if args.smoke:
        config = config.with_overrides(
            run_name=SMOKE_FOLDER_NAME,
            max_episodes=2,
            num_meta_agent=1,
            max_episode_step=10,
            minimum_buffer_size=8,
            batch_size=8,
            replay_size=64,
            save_img_gap=1,
            save_model_gap=1,
            summary_window=1,
            train_updates_per_iter=1,
            result_bucket_episodes=1,
            auto_eval_interval=1,
            auto_eval_map_count=1,
        )

    overrides = {}
    for field in (
        "max_episodes",
        "num_meta_agent",
        "ray_num_cpus",
        "ray_worker_num_cpus",
        "worker_num_threads",
        "max_episode_step",
        "minimum_buffer_size",
        "batch_size",
        "replay_size",
        "save_img_gap",
        "save_model_gap",
        "summary_window",
        "train_updates_per_iter",
        "result_bucket_episodes",
        "monitor_window",
        "monitor_snapshot_interval",
        "auto_eval_map_count",
        "auto_eval_interval",
        "run_name",
        "run_session",
        "maps_dir",
        "num_gpu",
    ):
        value = getattr(args, field)
        if value is not None:
            overrides[field] = value

    if args.disable_training_monitor:
        overrides["enable_training_monitor"] = False
    if args.disable_auto_eval:
        overrides["enable_auto_eval"] = False

    for field in ("load_model", "use_gpu", "use_gpu_global", "auto_eval_greedy"):
        value = getattr(args, field)
        if value is not None:
            overrides[field] = value

    if args.resume_from is not None:
        overrides["resume_from"] = args.resume_from
        overrides["run_name"] = args.resume_run_name
        overrides["run_session"] = args.resume_session

    if overrides:
        config = config.with_overrides(**overrides)
    return config


def main_cli():
    configure_matplotlib_cache("main")
    args = parse_args()
    runtime_config = build_runtime_config(args)
    from driver import main as driver_main

    result = driver_main(runtime_config)
    print(f"checkpoint_path={result['checkpoint_path'] or '<none>'}")
    print(f"model_dir={result['model_dir'] or '<none>'}")
    print(f"result_gif_dir={result['result_gif_dir']}")
    print(f"result_eval_dir={result['result_eval_dir']}")
    print(f"train_dir={result['train_dir']}")
    print(f"completed_episodes={result['completed_episodes']}")


if __name__ == "__main__":
    main_cli()
