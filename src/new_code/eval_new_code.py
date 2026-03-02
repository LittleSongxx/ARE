#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import torch


SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from evaluation import evaluate_policy, summarize_eval_results
from parameter import MAX_EPISODE_STEP, checkpoint_final_path, checkpoint_interrupted_path, checkpoint_path, eval_path


def get_default_checkpoint():
    for candidate in (checkpoint_interrupted_path, checkpoint_final_path, checkpoint_path):
        if Path(candidate).exists():
            return candidate
    return checkpoint_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=get_default_checkpoint())
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--sampled", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-episode-step", type=int, default=MAX_EPISODE_STEP)
    parser.add_argument("--output-dir", default=eval_path)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    results = evaluate_policy(
        checkpoint["policy_model"],
        output_dir=args.output_dir,
        episodes=args.episodes,
        start_episode=args.start_episode,
        greedy=not args.sampled,
        device=args.device,
        max_episode_step=args.max_episode_step,
    )
    for result in results:
        print(
            f"episode={result['episode']} "
            f"explored_rate={result['explored_rate']:.4f} "
            f"travel_dist={result['travel_dist']:.2f} "
            f"success={int(result['success'])} "
            f"episode_return={result['episode_return']:.4f} "
            f"steps_taken={result['steps_taken']}"
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
