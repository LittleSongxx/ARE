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

from ariadne_wavelet_attnbias.evaluation import evaluate_policy, summarize_eval_results
from ariadne_wavelet_attnbias.parameter import (
    MAX_EPISODE_STEP,
    RuntimeConfig,
    configure_attention_bias,
    ensure_output_dirs,
    get_latest_checkpoint_path,
    get_run_identity_from_checkpoint,
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
    return parser.parse_args()


def main():
    args = parse_args()
    configure_attention_bias(
        use_attention_bias=not args.disable_attention_bias,
        attn_bias_mode=args.attn_bias_mode,
        attn_bias_beta=args.attn_bias_beta,
    )

    run_name, run_session = get_run_identity_from_checkpoint(args.checkpoint)
    output_config = RuntimeConfig(run_name=run_name, run_session=run_session)
    ensure_output_dirs(output_config)

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


if __name__ == "__main__":
    main()
