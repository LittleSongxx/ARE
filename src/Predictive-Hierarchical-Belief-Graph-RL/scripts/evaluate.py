#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import fields
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/mpl-hpbg-rl-{os.environ.get('USER', 'user')}")

from hpbg_rl.evaluation import evaluate_policy, save_evaluation_summary, summarize_eval_results
from hpbg_rl.map_splits import materialize_split_manifest, save_split_manifest
from hpbg_rl.parameter import (
    RuntimeConfig,
    apply_runtime_config,
    build_run_session,
    ensure_output_dirs,
    get_result_test_eval_path,
    get_result_validation_path,
    get_protocol_path,
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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a HPBG-RL checkpoint on a fixed validation/test split.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--run-name")
    parser.add_argument("--run-session")
    parser.add_argument("--maps-dir")
    parser.add_argument("--train-maps-dir")
    parser.add_argument("--val-maps-dir")
    parser.add_argument("--test-maps-dir")
    parser.add_argument("--split-manifest-path")
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--val-map-count", type=int)
    parser.add_argument("--test-map-count", type=int)
    parser.add_argument("--max-episode-step", type=int)
    parser.add_argument("--auto-eval-interval", type=int)
    parser.add_argument("--greedy", nargs="?", const="1", type=_parse_bool01, metavar="0|1")
    parser.add_argument("--use-gpu", nargs="?", const="1", type=_parse_bool01, metavar="0|1")
    return parser.parse_args()


def _checkpoint_runtime_config(checkpoint: dict[str, object]) -> RuntimeConfig:
    payload = checkpoint.get("runtime_config")
    if not isinstance(payload, dict):
        return RuntimeConfig()
    field_names = {field.name for field in fields(RuntimeConfig)}
    overrides = {key: value for key, value in payload.items() if key in field_names}
    try:
        return RuntimeConfig().with_overrides(**overrides)
    except TypeError:
        return RuntimeConfig()


def build_runtime_config(args, checkpoint: dict[str, object] | None = None) -> RuntimeConfig:
    base_config = _checkpoint_runtime_config(checkpoint or {})
    run_name = args.run_name or base_config.run_name
    overrides = {
        "run_name": run_name,
        "run_session": args.run_session or base_config.run_session or build_run_session(run_name),
    }
    if args.use_gpu is not None:
        overrides["use_gpu"] = bool(args.use_gpu)
        overrides["use_gpu_global"] = bool(args.use_gpu)
    if args.greedy is not None:
        overrides["auto_eval_greedy"] = bool(args.greedy)
    if args.auto_eval_interval is not None:
        overrides["auto_eval_interval"] = max(int(args.auto_eval_interval), 1)
    for field in (
        "maps_dir",
        "train_maps_dir",
        "val_maps_dir",
        "test_maps_dir",
        "split_manifest_path",
        "split_seed",
        "val_map_count",
        "test_map_count",
        "max_episode_step",
    ):
        value = getattr(args, field)
        if value is not None:
            overrides[field] = value
    return base_config.with_overrides(**overrides)


def main():
    configure_matplotlib_cache("evaluate")
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    policy_state = checkpoint.get("policy_model")
    if policy_state is None:
        raise ValueError(f"Checkpoint missing policy_model: {checkpoint_path}")

    runtime_config = apply_runtime_config(build_runtime_config(args, checkpoint))
    ensure_output_dirs(runtime_config)

    manifest = materialize_split_manifest(runtime_config)
    protocol_dir = get_protocol_path(runtime_config)
    manifest_path = save_split_manifest(manifest, protocol_dir / "split_manifest.json")
    runtime_config = apply_runtime_config(runtime_config.with_overrides(split_manifest_path=str(manifest_path)))

    split = args.split
    map_count = runtime_config.test_map_count if split == "test" else runtime_config.val_map_count
    episode_number = int(checkpoint.get("episode", 0) or 0)
    results = evaluate_policy(
        policy_state,
        runtime_config,
        episode_number=episode_number,
        device="cuda:0" if runtime_config.use_gpu else "cpu",
        greedy=runtime_config.auto_eval_greedy,
        max_episode_step=runtime_config.max_episode_step,
        split=split,
        map_count=map_count,
    )
    summary = summarize_eval_results(results)
    protocol = {
        "split": split,
        "manifest_hash": manifest.content_hash(),
        "split_manifest_path": str(manifest_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_episode": episode_number,
        "step_budget": int(runtime_config.max_episode_step),
        "greedy": bool(runtime_config.auto_eval_greedy),
    }
    output_dir = get_result_test_eval_path(runtime_config) if split == "test" else get_result_validation_path(runtime_config)
    save_evaluation_summary(
        output_dir,
        episode_number=episode_number,
        results=results,
        summary=summary,
        bucket_size=runtime_config.auto_eval_interval,
        protocol=protocol,
    )
    report_path = Path(output_dir) / ("final_test_report.json" if split == "test" else "validation_eval_report.json")
    report_path.write_text(
        json.dumps(
            {"episode": episode_number, "protocol": protocol, "summary": summary, "results": results},
            indent=2,
            sort_keys=True,
        )
    )
    print(f"split={split}")
    print(f"evaluated_maps={summary['evaluated_maps']}")
    print(f"explored_rate={summary['explored_rate']:.6f}")
    print(f"success_rate={summary['success_rate']:.6f}")
    print(f"report_path={report_path}")


if __name__ == "__main__":
    main()