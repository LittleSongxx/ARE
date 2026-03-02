#!/usr/bin/env python3
"""
Train:
source /root/miniconda3/etc/profile.d/conda.sh && conda activate ros_conda && cd /root/ros_ws/ARE/src && python -m ariadne_wavelet_attnbias.scripts.train_attnbias

Smoke train:
source /root/miniconda3/etc/profile.d/conda.sh && conda activate ros_conda && cd /root/ros_ws/ARE/src && python -m ariadne_wavelet_attnbias.scripts.train_attnbias --smoke
"""

import argparse
import sys
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ariadne_wavelet_attnbias.driver import main
from ariadne_wavelet_attnbias.parameter import RuntimeConfig, SMOKE_FOLDER_NAME, configure_attention_bias


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
            use_gpu=False,
            use_gpu_global=False,
            num_gpu=0,
            auto_eval_episodes=1,
            auto_eval_device="cpu",
            run_name=SMOKE_FOLDER_NAME,
        )

    overrides = {}
    for field in (
        "max_episodes",
        "num_meta_agent",
        "max_episode_step",
        "minimum_buffer_size",
        "batch_size",
        "replay_size",
        "save_img_gap",
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
    if overrides:
        config = config.with_overrides(**overrides)
    return config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--disable-attention-bias", action="store_true")
    parser.add_argument("--attn-bias-mode", choices=("diff", "open", "hybrid"))
    parser.add_argument("--attn-bias-beta", type=float)
    parser.add_argument("--max-episodes", dest="max_episodes", type=int)
    parser.add_argument("--num-meta-agent", dest="num_meta_agent", type=int)
    parser.add_argument("--max-episode-step", dest="max_episode_step", type=int)
    parser.add_argument("--minimum-buffer-size", dest="minimum_buffer_size", type=int)
    parser.add_argument("--batch-size", dest="batch_size", type=int)
    parser.add_argument("--replay-size", dest="replay_size", type=int)
    parser.add_argument("--save-img-gap", dest="save_img_gap", type=int)
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
    return parser.parse_args()


def main_cli():
    args = parse_args()
    configure_attention_bias(
        use_attention_bias=not args.disable_attention_bias,
        attn_bias_mode=args.attn_bias_mode,
        attn_bias_beta=args.attn_bias_beta,
    )
    runtime_config = build_runtime_config(args)
    result = main(runtime_config)
    print(f"checkpoint_path={result['checkpoint_path']}")
    print(f"model_dir={result['model_dir']}")
    print(f"train_dir={result['train_dir']}")
    print(f"gif_dir={result['gif_dir']}")


if __name__ == "__main__":
    main_cli()
