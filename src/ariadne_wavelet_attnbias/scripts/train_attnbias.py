#!/usr/bin/env python3
"""
Train:
source /root/miniconda3/etc/profile.d/conda.sh && conda activate ros_conda && cd /root/ros_ws/ARE/src && python -m ariadne_wavelet_attnbias.scripts.train_attnbias

Baseline smoke train:
source /root/miniconda3/etc/profile.d/conda.sh && conda activate ros_conda && cd /root/ros_ws/ARE/src && python -m ariadne_wavelet_attnbias.scripts.train_attnbias --smoke

Full-feature smoke train:
source /root/miniconda3/etc/profile.d/conda.sh && conda activate ros_conda && cd /root/ros_ws/ARE/src && python -m ariadne_wavelet_attnbias.scripts.train_attnbias --smoke-all-features
"""

import argparse
import sys
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ariadne_wavelet_attnbias.driver import main
from ariadne_wavelet_attnbias.parameter import (
    RuntimeConfig,
    SMOKE_FOLDER_NAME,
    configure_attention_bias,
    configure_rl_options,
    get_run_identity_from_checkpoint,
    get_rl_options,
    resolve_resume_checkpoint,
)


def _parse_csv_ints(value):
    return tuple(int(item.strip()) for item in str(value).split(",") if item.strip())


def _parse_csv_strings(value):
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def _parse_mapping_args(values):
    if not values:
        return None
    parsed = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE, got: {item}")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if not key or not raw_value:
            raise ValueError(f"Expected KEY=VALUE, got: {item}")
        parsed[key] = raw_value
    return parsed


def _parse_pattern_mapping_args(values):
    mapping = _parse_mapping_args(values)
    if mapping is None:
        return None
    return {key: tuple(pattern.strip() for pattern in value.split(",") if pattern.strip()) for key, value in mapping.items()}


def _baseline_smoke_overrides():
    return {
        "max_episodes": 2,
        "num_meta_agent": 1,
        "max_episode_step": 10,
        "minimum_buffer_size": 8,
        "batch_size": 8,
        "replay_size": 64,
        "save_img_gap": 1,
        "summary_window": 1,
        "train_updates_per_iter": 1,
        "result_bucket_episodes": 1,
        "auto_eval_episodes": 1,
        "auto_eval_device": "cpu",
        "run_name": SMOKE_FOLDER_NAME,
    }


def _full_feature_smoke_overrides():
    overrides = _baseline_smoke_overrides()
    overrides["max_episodes"] = 3
    return overrides


def build_runtime_config(args):
    config = RuntimeConfig(rl_options=get_rl_options())
    if args.smoke_all_features:
        config = config.with_overrides(**_full_feature_smoke_overrides())
    elif args.smoke:
        config = config.with_overrides(**_baseline_smoke_overrides())

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
        "gif_frame_rate",
        "summary_window",
        "train_updates_per_iter",
        "result_bucket_episodes",
        "monitor_window",
        "monitor_snapshot_interval",
        "auto_eval_episodes",
        "auto_eval_device",
        "run_name",
        "run_session",
    ):
        value = getattr(args, field)
        if value is not None:
            overrides[field] = value
    if args.disable_training_monitor:
        overrides["enable_training_monitor"] = False
    if args.disable_auto_eval:
        overrides["enable_auto_eval"] = False
    if args.sampled_auto_eval:
        overrides["auto_eval_greedy"] = False
    if args.disable_fixed_eval_maps:
        overrides["use_fixed_eval_maps"] = False
    if args.eval_benchmark_map:
        overrides["eval_benchmark_maps"] = tuple(args.eval_benchmark_map)
    if args.resume_from is not None:
        overrides["resume_from"] = args.resume_from
        overrides["run_name"] = args.resume_run_name
        overrides["run_session"] = args.resume_session
    if overrides:
        config = config.with_overrides(**overrides)
    return config.with_overrides(rl_options=get_rl_options())


def configure_rl_from_args(args):
    if args.smoke_all_features:
        configure_rl_options(
            use_n_step_return=True,
            n_step=3,
            use_reward_decomposition=True,
            use_privileged_distill=True,
            distill_warmup_updates=1,
            use_curriculum=True,
            curriculum_milestones=(0, 1, 2),
        )

    configure_rl_options(
        use_n_step_return=True if args.use_n_step_return else None,
        n_step=args.n_step,
        use_reward_decomposition=True if args.use_reward_decomposition else None,
        r_info_w=args.r_info_w,
        r_dist_w=args.r_dist_w,
        r_safe_w=args.r_safe_w,
        r_terminal_bonus=args.r_terminal_bonus,
        use_privileged_distill=True if args.use_privileged_distill else None,
        distill_lambda=args.distill_lambda,
        distill_tau=args.distill_tau,
        distill_warmup_updates=args.distill_warmup_updates,
        use_curriculum=True if args.use_curriculum else None,
        curriculum_milestones=_parse_csv_ints(args.curriculum_milestones) if args.curriculum_milestones else None,
        curriculum_levels=_parse_csv_strings(args.curriculum_levels) if args.curriculum_levels else None,
        curriculum_mode=args.curriculum_mode,
        curriculum_dirs=_parse_mapping_args(args.curriculum_dir),
        curriculum_patterns=_parse_pattern_mapping_args(args.curriculum_pattern),
        use_curriculum_in_eval=True if args.use_curriculum_in_eval else None,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-all-features", action="store_true")
    parser.add_argument("--resume-from", dest="resume_from")
    parser.add_argument("--disable-attention-bias", action="store_true")
    parser.add_argument("--attn-bias-mode", choices=("diff", "open", "hybrid"))
    parser.add_argument("--attn-bias-beta", type=float)
    parser.add_argument("--use-n-step-return", action="store_true")
    parser.add_argument("--n-step", type=int)
    parser.add_argument("--use-reward-decomposition", action="store_true")
    parser.add_argument("--r-info-w", dest="r_info_w", type=float)
    parser.add_argument("--r-dist-w", dest="r_dist_w", type=float)
    parser.add_argument("--r-safe-w", dest="r_safe_w", type=float)
    parser.add_argument("--r-terminal-bonus", dest="r_terminal_bonus", type=float)
    parser.add_argument("--use-privileged-distill", action="store_true")
    parser.add_argument("--distill-lambda", dest="distill_lambda", type=float)
    parser.add_argument("--distill-tau", dest="distill_tau", type=float)
    parser.add_argument("--distill-warmup-updates", dest="distill_warmup_updates", type=int)
    parser.add_argument("--use-curriculum", action="store_true")
    parser.add_argument("--curriculum-mode", choices=("dir", "pattern"))
    parser.add_argument("--curriculum-milestones")
    parser.add_argument("--curriculum-levels")
    parser.add_argument("--curriculum-dir", action="append")
    parser.add_argument("--curriculum-pattern", action="append")
    parser.add_argument("--use-curriculum-in-eval", action="store_true")
    parser.add_argument("--disable-fixed-eval-maps", action="store_true")
    parser.add_argument("--eval-benchmark-map", action="append")
    parser.add_argument("--max-episodes", dest="max_episodes", type=int)
    parser.add_argument("--num-meta-agent", dest="num_meta_agent", type=int)
    parser.add_argument("--max-episode-step", dest="max_episode_step", type=int)
    parser.add_argument("--minimum-buffer-size", dest="minimum_buffer_size", type=int)
    parser.add_argument("--batch-size", dest="batch_size", type=int)
    parser.add_argument("--replay-size", dest="replay_size", type=int)
    parser.add_argument("--save-img-gap", dest="save_img_gap", type=int)
    parser.add_argument("--gif-frame-rate", dest="gif_frame_rate", type=float)
    parser.add_argument("--summary-window", dest="summary_window", type=int)
    parser.add_argument("--train-updates-per-iter", dest="train_updates_per_iter", type=int)
    parser.add_argument("--result-bucket-episodes", dest="result_bucket_episodes", type=int)
    parser.add_argument("--monitor-window", dest="monitor_window", type=int)
    parser.add_argument("--monitor-snapshot-interval", dest="monitor_snapshot_interval", type=int)
    parser.add_argument("--auto-eval-episodes", dest="auto_eval_episodes", type=int)
    parser.add_argument("--auto-eval-device", dest="auto_eval_device", choices=("cpu", "cuda"))
    parser.add_argument("--disable-training-monitor", action="store_true")
    parser.add_argument("--disable-auto-eval", action="store_true")
    parser.add_argument("--sampled-auto-eval", action="store_true")
    parser.add_argument("--run-name", dest="run_name")
    parser.add_argument("--run-session", dest="run_session")
    parser.add_argument("--ray-num-cpus", dest="ray_num_cpus", type=int)
    parser.add_argument("--ray-worker-num-cpus", dest="ray_worker_num_cpus", type=int)
    parser.add_argument("--worker-num-threads", dest="worker_num_threads", type=int)
    args = parser.parse_args()

    if args.resume_from is not None:
        if args.smoke or args.smoke_all_features:
            parser.error("--resume-from cannot be used with smoke presets")
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
    configure_attention_bias(
        use_attention_bias=not args.disable_attention_bias,
        attn_bias_mode=args.attn_bias_mode,
        attn_bias_beta=args.attn_bias_beta,
    )
    configure_rl_from_args(args)
    runtime_config = build_runtime_config(args)
    result = main(runtime_config)
    print(f"checkpoint_path={result['checkpoint_path'] or '<none>'}")
    print(f"model_dir={result['model_dir'] or '<none>'}")
    print(f"result_gif_dir={result['result_gif_dir']}")
    print(f"result_eval_dir={result['result_eval_dir']}")


if __name__ == "__main__":
    main_cli()
