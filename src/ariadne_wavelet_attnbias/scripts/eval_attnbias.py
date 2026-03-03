#!/usr/bin/env python3
"""
Evaluate:
source /root/miniconda3/etc/profile.d/conda.sh && conda activate ros_conda && cd /root/ros_ws/ARE/src && python -m ariadne_wavelet_attnbias.scripts.eval_attnbias

Smoke eval:
source /root/miniconda3/etc/profile.d/conda.sh && conda activate ros_conda && cd /root/ros_ws/ARE/src && python -m ariadne_wavelet_attnbias.scripts.eval_attnbias --checkpoint /root/ros_ws/ARE/src/ariadne_wavelet_attnbias/model/<run_session>/checkpoint.pth --episodes 1 --device cpu
"""

import argparse
import sys
from pathlib import Path

import torch


SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ariadne_wavelet_attnbias.evaluation import evaluate_policy, save_evaluation_metrics_plot, summarize_eval_results
from ariadne_wavelet_attnbias.parameter import (
    MAX_EPISODE_STEP,
    RuntimeConfig,
    configure_attention_bias,
    configure_rl_options,
    ensure_result_dirs,
    get_result_eval_path,
    get_rl_options,
    get_latest_checkpoint_path,
    get_run_identity_from_checkpoint,
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


def configure_rl_from_args(args):
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
    parser.add_argument("--checkpoint", default=str(get_latest_checkpoint_path()))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--sampled", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--result-bucket-episodes", type=int, default=100)
    parser.add_argument("--max-episode-step", type=int, default=MAX_EPISODE_STEP)
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
    return parser.parse_args()


def main():
    args = parse_args()
    configure_attention_bias(
        use_attention_bias=not args.disable_attention_bias,
        attn_bias_mode=args.attn_bias_mode,
        attn_bias_beta=args.attn_bias_beta,
    )
    configure_rl_from_args(args)

    run_name, run_session = get_run_identity_from_checkpoint(args.checkpoint)
    runtime_overrides = {}
    if args.disable_fixed_eval_maps:
        runtime_overrides["use_fixed_eval_maps"] = False
    if args.eval_benchmark_map:
        runtime_overrides["eval_benchmark_maps"] = tuple(args.eval_benchmark_map)
    output_config = RuntimeConfig(run_name=run_name, run_session=run_session, rl_options=get_rl_options()).with_overrides(
        **runtime_overrides
    )
    ensure_result_dirs(output_config)

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    results = evaluate_policy(
        checkpoint["policy_model"],
        output_config=output_config,
        episodes=args.episodes,
        start_episode=args.start_episode,
        greedy=args.greedy or not args.sampled,
        device=args.device,
        result_bucket_episodes=args.result_bucket_episodes,
        max_episode_step=args.max_episode_step,
    )
    plot_path = save_evaluation_metrics_plot(results, get_result_eval_path(output_config))
    for result in results:
        print(
            f"episode={result['episode']} "
            f"explored_rate={result['explored_rate']:.4f} "
            f"travel_dist={result['travel_dist']:.2f} "
            f"success={int(result['success'])} "
            f"episode_return={result['episode_return']:.4f} "
            f"steps_taken={result['steps_taken']} "
            f"gif_path={result['gif_path']} "
            f"png_path={result['png_path']}"
        )
    summary = summarize_eval_results(results)
    print(
        "summary "
        f"episodes={summary['episodes']} "
        f"explored_rate={summary['explored_rate']:.4f} "
        f"travel_dist={summary['travel_dist']:.2f} "
        f"success_rate={summary['success_rate']:.4f} "
        f"episode_return={summary['episode_return']:.4f} "
        f"steps_taken={summary['steps_taken']:.2f}"
    )
    print(f"eval_plot={plot_path}")


if __name__ == "__main__":
    main()
