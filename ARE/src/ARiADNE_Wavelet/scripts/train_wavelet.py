#!/usr/bin/env python3
"""
Train:
source /root/miniconda3/etc/profile.d/conda.sh && conda activate ros_conda && cd /root/ros_ws/ARE/src && python -m ARiADNE_Wavelet.scripts.train_wavelet

Smoke train:
source /root/miniconda3/etc/profile.d/conda.sh && conda activate ros_conda && cd /root/ros_ws/ARE/src && python -m ARiADNE_Wavelet.scripts.train_wavelet --smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ARiADNE_Wavelet.driver import main
from ARiADNE_Wavelet.parameter import (
    RuntimeConfig,
    SMOKE_FOLDER_NAME,
    get_run_identity_from_checkpoint,
    resolve_resume_checkpoint,
)


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
    parser.add_argument(
        flag,
        dest=dest,
        nargs="?",
        const="1",
        type=_parse_bool01,
        metavar="0|1",
        default=None,
    )


def build_runtime_config(args):
    config = RuntimeConfig()
    if args.smoke:
        config = config.with_overrides(
            max_episodes=2,
            num_meta_agent=1,
            max_episode_step=10,
            minimum_buffer_size=8,
            batch_size=8,
            replay_size=64,
            save_img_gap=1,
            summary_window=1,
            train_updates_per_iter=1,
            result_bucket_episodes=1,
            run_name=SMOKE_FOLDER_NAME,
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
        "gif_frame_rate",
        "summary_window",
        "train_updates_per_iter",
        "result_bucket_episodes",
        "monitor_window",
        "monitor_snapshot_interval",
        "run_name",
        "run_session",
        "history_len",
        "history_input_dim",
        "history_wavelet_levels",
        "history_embed_dim",
        "history_feature_set",
        "history_encoder_mode",
        "utility_target_type",
        "utility_target_horizon",
        "utility_target_gamma",
        "utility_loss_mode",
        "utility_loss_weight",
        "utility_patch_size",
        "utility_patch_sigma",
        "utility_aux_loss_type",
        "utility_aux_base_weight",
        "utility_aux_wavelet_weight",
        "utility_wavelet_levels",
        "utility_wavelet_rho",
    ):
        value = getattr(args, field)
        if value is not None:
            overrides[field] = value

    if args.disable_training_monitor:
        overrides["enable_training_monitor"] = False
    if args.enable_wavelet_history is not None:
        overrides["enable_wavelet_history"] = bool(args.enable_wavelet_history)
    if args.enable_wavelet_utility_loss is not None:
        overrides["enable_wavelet_utility_loss"] = bool(args.enable_wavelet_utility_loss)

    if args.resume_from is not None:
        overrides["resume_from"] = args.resume_from
        overrides["run_name"] = args.resume_run_name
        overrides["run_session"] = args.resume_session

    if overrides:
        config = config.with_overrides(**overrides)
    return config


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
    parser.add_argument("--gif-frame-rate", dest="gif_frame_rate", type=float)
    parser.add_argument("--summary-window", dest="summary_window", type=int)
    parser.add_argument("--train-updates-per-iter", dest="train_updates_per_iter", type=int)
    parser.add_argument("--result-bucket-episodes", dest="result_bucket_episodes", type=int)
    parser.add_argument("--monitor-window", dest="monitor_window", type=int)
    parser.add_argument("--monitor-snapshot-interval", dest="monitor_snapshot_interval", type=int)
    parser.add_argument("--run-name", dest="run_name")
    parser.add_argument("--run-session", dest="run_session")
    parser.add_argument("--disable-training-monitor", action="store_true")

    _add_bool_switch(parser, "--enable-wavelet-history", "enable_wavelet_history")
    _add_bool_switch(parser, "--enable-wavelet-utility-loss", "enable_wavelet_utility_loss")
    parser.add_argument("--history-len", dest="history_len", type=int)
    parser.add_argument("--history-input-dim", dest="history_input_dim", type=int)
    parser.add_argument("--history-wavelet-levels", dest="history_wavelet_levels", type=int)
    parser.add_argument("--history-embed-dim", dest="history_embed_dim", type=int)
    parser.add_argument("--history-feature-set", dest="history_feature_set")
    parser.add_argument(
        "--history-encoder-mode",
        dest="history_encoder_mode",
        choices=("mlp_only", "wavelet_shared", "wavelet_split"),
    )
    parser.add_argument(
        "--utility-target-type",
        dest="utility_target_type",
        choices=("td_bootstrap", "n_step_return"),
    )
    parser.add_argument("--utility-target-horizon", dest="utility_target_horizon", type=int)
    parser.add_argument("--utility-target-gamma", dest="utility_target_gamma", type=float)
    parser.add_argument("--utility-loss-mode", dest="utility_loss_mode", choices=("basic", "spatial2d"))
    parser.add_argument("--utility-loss-weight", dest="utility_loss_weight", type=float)
    parser.add_argument("--utility-patch-size", dest="utility_patch_size", type=int)
    parser.add_argument("--utility-patch-sigma", dest="utility_patch_sigma", type=float)
    parser.add_argument("--utility-aux-loss-type", dest="utility_aux_loss_type", choices=("smoothl1", "mse"))
    parser.add_argument("--utility-aux-base-weight", dest="utility_aux_base_weight", type=float)
    parser.add_argument("--utility-aux-wavelet-weight", dest="utility_aux_wavelet_weight", type=float)
    parser.add_argument("--utility-wavelet-levels", dest="utility_wavelet_levels", type=int)
    parser.add_argument("--utility-wavelet-rho", dest="utility_wavelet_rho", type=float)

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


def main_cli():
    args = parse_args()
    runtime_config = build_runtime_config(args)
    result = main(runtime_config)
    print(f"checkpoint_path={result['checkpoint_path'] or '<none>'}")
    print(f"model_dir={result['model_dir'] or '<none>'}")
    print(f"result_gif_dir={result['result_gif_dir']}")
    print(f"result_eval_dir={result['result_eval_dir']}")


if __name__ == "__main__":
    main_cli()
