#!/usr/bin/env python3
"""
Train:
source /root/miniconda3/etc/profile.d/conda.sh && conda activate ros_conda && cd /root/ros_ws/ARE/src && python -m ARiADNE.scripts.train_wavelet

Smoke train:
source /root/miniconda3/etc/profile.d/conda.sh && conda activate ros_conda && cd /root/ros_ws/ARE/src && python -m ARiADNE.scripts.train_wavelet --smoke
"""

import argparse
import sys
from pathlib import Path

import torch


SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ARiADNE.driver import main
from ARiADNE.parameter import (
    RuntimeConfig,
    SMOKE_FOLDER_NAME,
    get_run_identity_from_checkpoint,
    resolve_resume_checkpoint,
    runtime_config_from_checkpoint,
)


def _parse_csv_ints(value):
    return tuple(int(item.strip()) for item in str(value).split(",") if item.strip())


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
            auto_eval_episodes=1,
            auto_eval_device="cpu",
            run_name=SMOKE_FOLDER_NAME,
        )
    if args.resume_from is not None:
        checkpoint = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        config = runtime_config_from_checkpoint(
            checkpoint,
            config.with_overrides(run_name=args.resume_run_name, run_session=args.resume_session),
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
        "wavelet_feature_mode",
        "wavelet_local_pool",
        "wavelet_local_pool_radius_cells",
        "wavelet_norm_method",
        "wavelet_clip_percentile",
        "wavelet_fixed_clip_value",
        "wavelet_eps",
        "wavelet_attn_bias_type",
        "wavelet_attn_bias_beta",
        "wavelet_attn_bias_sigma",
        "wavelet_skip_thresh",
        "wavelet_skip_utility_low",
        "wavelet_skip_max_age_steps",
        "wavelet_skip_near_robot_radius",
        "wavelet_node_keep_thresh",
        "wavelet_node_keep_prob_low",
        "wavelet_node_min_keep",
        "wavelet_dth_alpha",
        "wavelet_dth_max_mult",
    ):
        value = getattr(args, field)
        if value is not None:
            overrides[field] = value
    if args.wavelet_scales is not None:
        overrides["wavelet_scales"] = _parse_csv_ints(args.wavelet_scales)
    if args.wavelet_scales_auto_mults is not None:
        overrides["wavelet_scales_auto_mults"] = _parse_csv_ints(args.wavelet_scales_auto_mults)
    if args.disable_training_monitor:
        overrides["enable_training_monitor"] = False
    if args.disable_auto_eval:
        overrides["enable_auto_eval"] = False
    if args.sampled_auto_eval:
        overrides["auto_eval_greedy"] = False
    if args.use_wavelet_feature:
        overrides["use_wavelet_feature"] = True
    if args.disable_wavelet_feature:
        overrides["use_wavelet_feature"] = False
    if args.wavelet_scales_auto:
        overrides["wavelet_scales_auto"] = True
    if args.disable_wavelet_scales_auto:
        overrides["wavelet_scales_auto"] = False
    if args.use_wavelet_attn_bias:
        overrides["use_wavelet_attn_bias"] = True
    if args.disable_wavelet_attn_bias:
        overrides["use_wavelet_attn_bias"] = False
    if args.wavelet_attn_bias_masked_only:
        overrides["wavelet_attn_bias_apply_on_masked_edges_only"] = True
    if args.wavelet_attn_bias_all_pairs:
        overrides["wavelet_attn_bias_apply_on_masked_edges_only"] = False
    if args.wavelet_skip_utility_updates:
        overrides["wavelet_skip_utility_updates"] = True
    if args.disable_wavelet_skip_utility_updates:
        overrides["wavelet_skip_utility_updates"] = False
    if args.wavelet_guided_node_sampling:
        overrides["wavelet_guided_node_sampling"] = True
    if args.disable_wavelet_guided_node_sampling:
        overrides["wavelet_guided_node_sampling"] = False
    if args.wavelet_node_always_keep_current_and_neighbors:
        overrides["wavelet_node_always_keep_current_and_neighbors"] = True
    if args.disable_wavelet_node_always_keep_current_and_neighbors:
        overrides["wavelet_node_always_keep_current_and_neighbors"] = False
    if args.wavelet_adaptive_dth:
        overrides["wavelet_adaptive_dth"] = True
    if args.disable_wavelet_adaptive_dth:
        overrides["wavelet_adaptive_dth"] = False
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
    parser.add_argument("--use-wavelet-feature", action="store_true")
    parser.add_argument("--disable-wavelet-feature", action="store_true")
    parser.add_argument("--wavelet-feature-mode", choices=("scalar", "scales", "scales_orient"))
    parser.add_argument("--wavelet-local-pool", choices=("none", "mean", "max"))
    parser.add_argument("--wavelet-local-pool-radius-cells", dest="wavelet_local_pool_radius_cells", type=int)
    parser.add_argument("--wavelet-scales-auto", action="store_true")
    parser.add_argument("--disable-wavelet-scales-auto", action="store_true")
    parser.add_argument("--wavelet-scales")
    parser.add_argument("--wavelet-scales-auto-mults")
    parser.add_argument("--wavelet-norm-method", dest="wavelet_norm_method", choices=("minmax", "percentile", "log_percentile", "fixed_clip"))
    parser.add_argument("--wavelet-clip-percentile", dest="wavelet_clip_percentile", type=float)
    parser.add_argument("--wavelet-fixed-clip-value", dest="wavelet_fixed_clip_value", type=float)
    parser.add_argument("--wavelet-eps", dest="wavelet_eps", type=float)
    parser.add_argument("--use-wavelet-attn-bias", action="store_true")
    parser.add_argument("--disable-wavelet-attn-bias", action="store_true")
    parser.add_argument("--wavelet-attn-bias-type", dest="wavelet_attn_bias_type", choices=("sim_exp", "neg_l1", "neg_l2"))
    parser.add_argument("--wavelet-attn-bias-beta", dest="wavelet_attn_bias_beta", type=float)
    parser.add_argument("--wavelet-attn-bias-sigma", dest="wavelet_attn_bias_sigma", type=float)
    parser.add_argument("--wavelet-attn-bias-masked-only", action="store_true")
    parser.add_argument("--wavelet-attn-bias-all-pairs", action="store_true")
    parser.add_argument("--wavelet-skip-utility-updates", action="store_true")
    parser.add_argument("--disable-wavelet-skip-utility-updates", action="store_true")
    parser.add_argument("--wavelet-skip-thresh", dest="wavelet_skip_thresh", type=float)
    parser.add_argument("--wavelet-skip-utility-low", dest="wavelet_skip_utility_low", type=float)
    parser.add_argument("--wavelet-skip-max-age-steps", dest="wavelet_skip_max_age_steps", type=int)
    parser.add_argument("--wavelet-skip-near-robot-radius", dest="wavelet_skip_near_robot_radius", type=float)
    parser.add_argument("--wavelet-guided-node-sampling", action="store_true")
    parser.add_argument("--disable-wavelet-guided-node-sampling", action="store_true")
    parser.add_argument("--wavelet-node-keep-thresh", dest="wavelet_node_keep_thresh", type=float)
    parser.add_argument("--wavelet-node-keep-prob-low", dest="wavelet_node_keep_prob_low", type=float)
    parser.add_argument("--wavelet-node-min-keep", dest="wavelet_node_min_keep", type=int)
    parser.add_argument("--wavelet-node-always-keep-current-and-neighbors", action="store_true")
    parser.add_argument("--disable-wavelet-node-always-keep-current-and-neighbors", action="store_true")
    parser.add_argument("--wavelet-adaptive-dth", action="store_true")
    parser.add_argument("--disable-wavelet-adaptive-dth", action="store_true")
    parser.add_argument("--wavelet-dth-alpha", dest="wavelet_dth_alpha", type=float)
    parser.add_argument("--wavelet-dth-max-mult", dest="wavelet_dth_max_mult", type=float)
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
