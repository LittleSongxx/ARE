#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/mpl-hpbg-rl-{os.environ.get('USER', 'user')}")

from hpbg_rl.parameter import (
    RuntimeConfig,
    SMOKE_FOLDER_NAME,
    apply_runtime_config,
    get_run_identity_from_checkpoint,
    resolve_resume_checkpoint,
)
from hpbg_rl.runtime_utils import configure_matplotlib_cache


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
    parser.add_argument("--corridor-max-width", dest="corridor_max_width", type=float)
    parser.add_argument("--corridor-min-length", dest="corridor_min_length", type=float)
    parser.add_argument("--smoothness-turn-penalty", dest="smoothness_turn_penalty", type=float)
    parser.add_argument("--smoothness-lateral-penalty", dest="smoothness_lateral_penalty", type=float)
    parser.add_argument("--run-name", dest="run_name")
    parser.add_argument("--run-session", dest="run_session")
    parser.add_argument("--maps-dir", dest="maps_dir")
    parser.add_argument("--train-maps-dir", dest="train_maps_dir")
    parser.add_argument("--val-maps-dir", dest="val_maps_dir")
    parser.add_argument("--test-maps-dir", dest="test_maps_dir")
    parser.add_argument("--split-manifest-path", dest="split_manifest_path")
    parser.add_argument("--split-seed", dest="split_seed", type=int)
    parser.add_argument("--val-map-count", dest="val_map_count", type=int)
    parser.add_argument("--test-map-count", dest="test_map_count", type=int)
    parser.add_argument("--eval-budget-mode", dest="eval_budget_mode")
    parser.add_argument("--large-scale-min-free-area", dest="large_scale_min_free_area", type=int)
    parser.add_argument("--large-scale-min-side", dest="large_scale_min_side", type=int)
    parser.add_argument("--num-gpu", dest="num_gpu", type=int)
    parser.add_argument("--disable-training-monitor", action="store_true")
    parser.add_argument("--disable-auto-eval", action="store_true")
    _add_bool_switch(parser, "--load-model", "load_model")
    _add_bool_switch(parser, "--use-gpu", "use_gpu")
    _add_bool_switch(parser, "--use-gpu-global", "use_gpu_global")
    _add_bool_switch(parser, "--auto-eval-greedy", "auto_eval_greedy")
    _add_bool_switch(parser, "--run-final-test", "run_final_test")
    _add_bool_switch(parser, "--allow-train-split-eval", "allow_train_split_eval")
    _add_bool_switch(parser, "--enable-corridor-graph-compression", "enable_corridor_graph_compression")
    _add_bool_switch(parser, "--enable-corridor-edge-pruning", "enable_corridor_edge_pruning")
    _add_bool_switch(parser, "--enable-smoothness-reward", "enable_smoothness_reward")
    _add_bool_switch(parser, "--use-lf-attention-hf-residual", "use_lf_attention_hf_residual")
    _add_bool_switch(parser, "--use-privileged-wavelet-distillation", "use_privileged_wavelet_distillation")
    _add_bool_switch(parser, "--use-hpbg", "use_hpbg")
    _add_bool_switch(parser, "--use-belief-state", "use_belief_state")
    _add_bool_switch(parser, "--use-map-prediction", "use_map_prediction")
    _add_bool_switch(parser, "--use-hierarchical-graph", "use_hierarchical_graph")
    _add_bool_switch(parser, "--use-expert-reward", "use_expert_reward")
    _add_bool_switch(parser, "--use-belief-distillation", "use_belief_distillation")
    _add_bool_switch(parser, "--wavelet-lf-qk", "wavelet_lf_qk")
    parser.add_argument("--hpbg-risk-weight", dest="hpbg_risk_weight", type=float)
    parser.add_argument("--hpbg-belief-ema-alpha", dest="hpbg_belief_ema_alpha", type=float)
    parser.add_argument("--hpbg-cluster-resolution", dest="hpbg_cluster_resolution", type=float)
    parser.add_argument("--hpbg-cluster-edge-hops", dest="hpbg_cluster_edge_hops", type=int)
    parser.add_argument("--hpbg-expert-reward-weight", dest="hpbg_expert_reward_weight", type=float)
    parser.add_argument("--hpbg-expert-potential-weight", dest="hpbg_expert_potential_weight", type=float)
    parser.add_argument("--hpbg-oracle-gain-weight", dest="hpbg_oracle_gain_weight", type=float)
    parser.add_argument("--hpbg-belief-distill-weight", dest="hpbg_belief_distill_weight", type=float)
    parser.add_argument("--hpbg-belief-distill-warmup-updates", dest="hpbg_belief_distill_warmup_updates", type=int)
    parser.add_argument("--hpbg-belief-distill-ramp-updates", dest="hpbg_belief_distill_ramp_updates", type=int)
    parser.add_argument("--wavelet-scales", dest="wavelet_scales")
    parser.add_argument("--wavelet-fuse-dim", dest="wavelet_fuse_dim", type=int)
    parser.add_argument("--wavelet-distill-weight", dest="wavelet_distill_weight", type=float)
    parser.add_argument("--wavelet-distill-lf-weight", dest="wavelet_distill_lf_weight", type=float)
    parser.add_argument("--wavelet-distill-hf-weight", dest="wavelet_distill_hf_weight", type=float)
    parser.add_argument("--wavelet-distill-warmup-updates", dest="wavelet_distill_warmup_updates", type=int)
    parser.add_argument("--wavelet-distill-ramp-updates", dest="wavelet_distill_ramp_updates", type=int)
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
            summary_window=1,
            train_updates_per_iter=1,
            result_bucket_episodes=1,
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
        "corridor_max_width",
        "corridor_min_length",
        "smoothness_turn_penalty",
        "smoothness_lateral_penalty",
        "run_name",
        "run_session",
        "maps_dir",
        "train_maps_dir",
        "val_maps_dir",
        "test_maps_dir",
        "split_manifest_path",
        "split_seed",
        "val_map_count",
        "test_map_count",
        "eval_budget_mode",
        "large_scale_min_free_area",
        "large_scale_min_side",
        "num_gpu",
        "hpbg_risk_weight",
        "hpbg_belief_ema_alpha",
        "hpbg_cluster_resolution",
        "hpbg_cluster_edge_hops",
        "hpbg_expert_reward_weight",
        "hpbg_expert_potential_weight",
        "hpbg_oracle_gain_weight",
        "hpbg_belief_distill_weight",
        "hpbg_belief_distill_warmup_updates",
        "hpbg_belief_distill_ramp_updates",
        "wavelet_fuse_dim",
        "wavelet_distill_weight",
        "wavelet_distill_lf_weight",
        "wavelet_distill_hf_weight",
        "wavelet_distill_warmup_updates",
        "wavelet_distill_ramp_updates",
    ):
        value = getattr(args, field)
        if value is not None:
            overrides[field] = value

    if args.disable_training_monitor:
        overrides["enable_training_monitor"] = False
    if args.disable_auto_eval:
        overrides["enable_auto_eval"] = False
    if args.auto_eval_map_count is not None:
        if args.val_map_count is None:
            overrides["val_map_count"] = args.auto_eval_map_count
        if args.test_map_count is None:
            overrides["test_map_count"] = args.auto_eval_map_count

    for field in (
        "load_model",
        "use_gpu",
        "use_gpu_global",
        "auto_eval_greedy",
        "run_final_test",
        "allow_train_split_eval",
        "enable_corridor_graph_compression",
        "enable_corridor_edge_pruning",
        "enable_smoothness_reward",
        "use_lf_attention_hf_residual",
        "use_privileged_wavelet_distillation",
        "use_hpbg",
        "use_belief_state",
        "use_map_prediction",
        "use_hierarchical_graph",
        "use_expert_reward",
        "use_belief_distillation",
        "wavelet_lf_qk",
    ):
        value = getattr(args, field)
        if value is not None:
            overrides[field] = value

    if args.wavelet_scales:
        overrides["wavelet_scales"] = tuple(int(token.strip()) for token in args.wavelet_scales.split(",") if token.strip())

    if args.resume_from is not None:
        overrides["resume_from"] = args.resume_from
        overrides["run_name"] = args.resume_run_name
        overrides["run_session"] = args.resume_session

    if overrides:
        config = config.with_overrides(**overrides)
    return config


def _cli_log(event: str, **fields) -> None:
    import time

    payload = " ".join(f"{key}={value}" for key, value in fields.items())
    if payload:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {event} {payload}", flush=True)
    else:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {event}", flush=True)


def main_cli():
    _cli_log("train_cli_start", pid=os.getpid(), cwd=os.getcwd())
    configure_matplotlib_cache("main")
    args = parse_args()
    runtime_config = apply_runtime_config(build_runtime_config(args))
    _cli_log(
        "train_cli_runtime_config_ready",
        run_name=runtime_config.run_name,
        run_session=runtime_config.run_session,
        max_episodes=runtime_config.max_episodes,
        num_meta_agent=runtime_config.num_meta_agent,
        max_episode_step=runtime_config.max_episode_step,
        smoke=args.smoke,
        resume_from=args.resume_from or "<none>",
    )
    _cli_log("train_driver_import_start")
    from hpbg_rl.driver import main as driver_main

    _cli_log("train_driver_import_done")
    result = driver_main(runtime_config)
    _cli_log("train_driver_returned", completed_episodes=result["completed_episodes"])
    print(f"checkpoint_path={result['checkpoint_path'] or '<none>'}")
    print(f"model_dir={result['model_dir'] or '<none>'}")
    print(f"result_gif_dir={result['result_gif_dir']}")
    print(f"result_eval_dir={result['result_eval_dir']}")
    print(f"train_dir={result['train_dir']}")
    print(f"completed_episodes={result['completed_episodes']}")


if __name__ == "__main__":
    main_cli()
