from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from ac_pbgrl.config import CONFIG_ROOT, PROJECT_ROOT, load_config
from ac_pbgrl.runtime.gpu import (
    GPULease,
    calibrate_micro_batch,
    query_gpus,
    recommended_micro_batch,
    select_gpus,
)


def _common_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="full")
    parser.add_argument("--system", default=None)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ac-pbgrl")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="inspect dependencies, storage, and GPU selection")
    _common_config(doctor)
    doctor.add_argument("--gpu-policy", choices=["idle-only", "prefer-idle", "shared-ok"], default=None)

    train = subparsers.add_parser("train", help="launch one training run")
    _common_config(train)
    train.add_argument("--gpus", default="auto", help="auto, cpu, or comma-separated physical indices")
    train.add_argument("--gpu-policy", choices=["idle-only", "prefer-idle", "shared-ok"], default=None)
    train.add_argument("--min-gpus", type=int, default=None)
    train.add_argument("--max-gpus", type=int, default=None)
    train.add_argument("--resume", default="auto")
    train.add_argument("--smoke", action="store_true")

    supervise = subparsers.add_parser("supervise", help="checkpoint-aware resource supervisor")
    _common_config(supervise)
    supervise.add_argument("--gpus", default="auto")
    supervise.add_argument("--gpu-policy", choices=["idle-only", "prefer-idle", "shared-ok"], default=None)
    supervise.add_argument("--min-gpus", type=int, default=None)
    supervise.add_argument("--max-gpus", type=int, default=None)
    supervise.add_argument("--no-wait", action="store_true")
    supervise.add_argument("--smoke", action="store_true")

    labels = subparsers.add_parser("labels", help="generate privileged future-gain shards")
    _common_config(labels)
    labels.add_argument("--split", default="train")
    labels.add_argument("--samples", type=int, default=1000)
    labels.add_argument("--output", default=None)
    labels.add_argument("--device", default="cpu")
    labels.add_argument("--smoke", action="store_true")

    calibrate = subparsers.add_parser("calibrate", help="fit held-out variance temperatures for KF")
    _common_config(calibrate)
    calibrate.add_argument("--checkpoint", required=True)
    calibrate.add_argument("--labels-root", default=None)
    calibrate.add_argument("--split", default="validation")
    calibrate.add_argument("--samples", type=int, default=2048)
    calibrate.add_argument("--batch-size", type=int, default=32)
    calibrate.add_argument("--output", default=None)
    calibrate.add_argument("--device", default="cpu")
    calibrate.add_argument("--smoke", action="store_true")

    train_gru = subparsers.add_parser("train-gru", help="train the recurrent temporal control on stable-ID sequences")
    _common_config(train_gru)
    train_gru.add_argument("--actor-checkpoint", required=True)
    train_gru.add_argument("--labels-root", default=None)
    train_gru.add_argument("--samples", type=int, default=20000)
    train_gru.add_argument("--epochs", type=int, default=20)
    train_gru.add_argument("--batch-size", type=int, default=64)
    train_gru.add_argument("--extraction-batch-size", type=int, default=16)
    train_gru.add_argument("--learning-rate", type=float, default=1.0e-3)
    train_gru.add_argument("--output", default=None)
    train_gru.add_argument("--device", default="cpu")
    train_gru.add_argument("--smoke", action="store_true")

    splits = subparsers.add_parser("splits", help="create leakage-safe D4 map splits")
    _common_config(splits)
    splits.add_argument("--output", default=None)

    evaluate = subparsers.add_parser("evaluate", help="evaluate one checkpoint")
    _common_config(evaluate)
    evaluate.add_argument("--checkpoint", default=None)
    evaluate.add_argument("--output", default=None)
    evaluate.add_argument("--split", default="iid_test")
    evaluate.add_argument("--map-limit", type=int, default=None)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--seeds", default=None, help="comma-separated evaluation seeds")
    evaluate.add_argument("--smoke", action="store_true")

    figures = subparsers.add_parser("figures", help="generate paper figures from run artifacts")
    figures.add_argument("--runs-root", default=None)
    figures.add_argument("--output", default=None)
    figures.add_argument(
        "--evidence-level",
        choices=(
            "multi_seed_formal",
            "multi_seed_credibility_pilot",
            "single_run_directional_screening",
        ),
        default="multi_seed_formal",
    )

    export = subparsers.add_parser("export", help="export a policy to ONNX")
    _common_config(export)
    export.add_argument("--checkpoint", default=None)
    export.add_argument("--output", default=None)
    export.add_argument("--format", choices=["onnx"], default="onnx")
    export.add_argument("--device", default="cpu")
    export.add_argument("--validate", action="store_true")
    export.add_argument("--smoke", action="store_true")

    ablate = subparsers.add_parser("ablate", help="run a configured ablation suite")
    ablate.add_argument("--suite", default="main")
    ablate.add_argument("--gpus", default="auto")
    ablate.add_argument("--gpu-policy", default="prefer-idle")
    ablate.add_argument("--system", default=None)
    ablate.add_argument("--smoke", action="store_true")

    paper = subparsers.add_parser("paper", help="teacher, labels, main runs, ablations, evaluation, figures")
    paper.add_argument("--gpus", default="auto")
    paper.add_argument("--gpu-policy", default="prefer-idle")
    paper.add_argument("--label-samples", type=int, default=100000)
    paper.add_argument("--pilot-only", action="store_true")
    paper.add_argument(
        "--single-run-screening",
        action="store_true",
        help="allow exactly one pilot seed for directional screening without statistical claims",
    )
    paper.add_argument("--pilot-seeds", type=int, default=3)
    paper.add_argument("--pilot-early-steps", type=int, default=200000)
    paper.add_argument("--pilot-steps", type=int, default=500000)
    paper.add_argument("--pilot-label-samples", type=int, default=20000)
    paper.add_argument("--pilot-map-limit", type=int, default=100)

    test = subparsers.add_parser("test", help="run the repository test suite")
    test.add_argument("pytest_args", nargs=argparse.REMAINDER)
    return parser


def command_doctor(args) -> int:
    config = load_config(args.config, system=args.system, overrides=args.set)
    scheduler = config.gpu_scheduler
    policy = args.gpu_policy or str(scheduler.policy)
    gpus = query_gpus()
    selected = select_gpus(
        gpus,
        policy=policy,
        min_gpus=int(scheduler.min_gpus),
        max_gpus=int(scheduler.max_gpus),
        min_free_memory_gib=float(scheduler.min_free_memory_gib),
        max_utilization_pct=int(scheduler.max_utilization_pct),
        max_temperature_c=int(scheduler.max_temperature_c),
        idle_used_memory_mib=int(scheduler.idle_used_memory_mib),
        idle_utilization_pct=int(scheduler.idle_utilization_pct),
    )
    dependency_pins = {
        "numpy": "1.24.3",
        "scipy": "1.10.1",
        "scikit-image": "0.21.0",
        "matplotlib": "3.7.5",
        "tensorboard": "2.14.0",
        "PyYAML": "6.0.3",
        "torch": "2.4.1",
        "ray": "2.10.0",
        "h5py": "3.11.0",
        "pandas": "2.0.3",
        "seaborn": "0.13.2",
        "scikit-learn": "1.3.2",
        "onnx": "1.16.2",
        "onnxruntime": "1.16.3",
        "pytest": "8.3.5",
        "pytest-cov": "5.0.0",
    }
    dependencies = {}
    for package, expected in dependency_pins.items():
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        dependencies[package] = {
            "installed": installed,
            "expected": expected,
            "compatible": bool(installed and (installed == expected or installed.startswith(expected + "+"))),
        }
    storage_path = Path(config.project.data_root).expanduser()
    existing_storage = storage_path
    while not existing_storage.exists() and existing_storage != existing_storage.parent:
        existing_storage = existing_storage.parent
    storage = shutil.disk_usage(existing_storage)
    payload = {
        "python": sys.version.split()[0],
        "project_root": str(PROJECT_ROOT),
        "data_root": str(config.project.data_root),
        "maps": len(list(Path(config.project.maps_dir).glob("*.png"))),
        "gpu_policy": policy,
        "gpus": [vars(item) for item in gpus],
        "selected_indices": [item.index for item in selected],
        "recommended_micro_batch": recommended_micro_batch(selected, float(scheduler.memory_reserve_gib)),
        "dependencies": dependencies,
        "storage": {
            "checked_path": str(existing_storage),
            "target_writable": os.access(str(existing_storage), os.W_OK),
            "free_gib": round(storage.free / 1024**3, 2),
        },
    }
    micro_batch = int(payload["recommended_micro_batch"] or 0)
    world_size = len(selected)
    if micro_batch and world_size:
        import math

        local_max = math.ceil(int(config.train.global_batch_size) / world_size)
        payload["gradient_accumulation_steps"] = math.ceil(local_max / micro_batch)
        payload["recommended_command"] = (
            f"./run.sh supervise --config {args.config} --min-gpus 1 "
            f"--max-gpus {int(scheduler.max_gpus)}"
        )
    else:
        payload["gradient_accumulation_steps"] = None
        payload["recommended_command"] = None
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _default_checkpoint(config) -> Path:
    run_name = str(config.project.run_name)
    if run_name == "auto":
        run_name = f"{config.project.experiment}/seed_{int(config.project.seed)}"
    return Path(config.project.data_root) / "runs" / run_name / "checkpoints" / "latest.pt"


def _resolve_checkpoint(config, value: str | None) -> Path:
    checkpoint = Path(value) if value else _default_checkpoint(config)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"checkpoint not found: {checkpoint}; train the configuration first or pass --checkpoint"
        )
    return checkpoint


def _train_command(args, indices: list[int] | None, micro_batch: int | None = None) -> int:
    environment = os.environ.copy()
    if indices is None:
        command = [sys.executable, "-m", "ac_pbgrl.learning.train", "--device", "cpu"]
    else:
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in indices)
        inventory = {gpu.index: gpu for gpu in query_gpus()}
        selected = [inventory[index] for index in indices if index in inventory]
        environment["ACPBGRL_SELECTED_GPU_INDICES"] = ",".join(str(index) for index in indices)
        environment["ACPBGRL_SELECTED_GPU_UUIDS"] = ",".join(gpu.uuid for gpu in selected)
        environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
        environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={len(indices)}",
            "-m",
            "ac_pbgrl.learning.train",
        ]
    command.extend(("--config", args.config))
    if args.system:
        command.extend(("--system", args.system))
    for override in args.set:
        command.extend(("--set", override))
    if getattr(args, "resume", None):
        command.extend(("--resume", args.resume))
    if micro_batch:
        command.extend(("--micro-batch", str(micro_batch)))
    if getattr(args, "smoke", False):
        command.append("--smoke")
    return subprocess.call(command, env=environment, cwd=str(PROJECT_ROOT))


def command_train(args) -> int:
    config = load_config(args.config, system=args.system, overrides=args.set)
    if args.gpus == "cpu":
        return _train_command(args, None)
    if args.gpus != "auto":
        indices = [int(value) for value in args.gpus.split(",") if value.strip()]
        inventory = {gpu.index: gpu for gpu in query_gpus()}
        missing = [index for index in indices if index not in inventory]
        if missing:
            print(f"Unknown GPU indices: {missing}", file=sys.stderr)
            return 2
        selected = [inventory[index] for index in indices]
        lease = GPULease(Path(config.project.data_root) / "gpu_locks", selected)
        if not lease.acquire():
            print("One or more selected GPUs are already leased by AC-PBGRL.", file=sys.stderr)
            return 75
        try:
            micro_batch, attempts = calibrate_micro_batch(
                selected,
                experiment=args.config,
                system=args.system,
                overrides=args.set,
                candidates=config.gpu_scheduler.micro_batch_candidates,
                start=int(config.train.micro_batch_size),
                reserve_gib=float(config.gpu_scheduler.memory_reserve_gib),
                smoke=args.smoke,
            )
            if micro_batch <= 0:
                print(json.dumps({"error": "GPU memory probe failed", "attempts": attempts}, indent=2), file=sys.stderr)
                return 75
            return _train_command(args, indices, micro_batch)
        finally:
            lease.release()
    scheduler = config.gpu_scheduler
    policy = args.gpu_policy or str(scheduler.policy)
    selected = select_gpus(
        query_gpus(),
        policy=policy,
        min_gpus=int(args.min_gpus or scheduler.min_gpus),
        max_gpus=int(args.max_gpus or scheduler.max_gpus),
        min_free_memory_gib=float(scheduler.min_free_memory_gib),
        max_utilization_pct=int(scheduler.max_utilization_pct),
        max_temperature_c=int(scheduler.max_temperature_c),
        idle_used_memory_mib=int(scheduler.idle_used_memory_mib),
        idle_utilization_pct=int(scheduler.idle_utilization_pct),
    )
    if not selected:
        print("No GPU currently satisfies the configured safety policy.", file=sys.stderr)
        return 75
    micro_batch = recommended_micro_batch(selected, float(scheduler.memory_reserve_gib))
    lease = GPULease(Path(config.project.data_root) / "gpu_locks", selected)
    if not lease.acquire():
        print("Selected GPUs were leased by another AC-PBGRL run; retrying is safe.", file=sys.stderr)
        return 75
    try:
        calibrated, attempts = calibrate_micro_batch(
            selected,
            experiment=args.config,
            system=args.system,
            overrides=args.set,
            candidates=config.gpu_scheduler.micro_batch_candidates,
            start=micro_batch,
            reserve_gib=float(scheduler.memory_reserve_gib),
            smoke=args.smoke,
        )
        if calibrated <= 0:
            print(json.dumps({"error": "GPU memory probe failed", "attempts": attempts}, indent=2), file=sys.stderr)
            return 75
        return _train_command(args, [item.index for item in selected], calibrated)
    finally:
        lease.release()


def command_supervise(args) -> int:
    from ac_pbgrl.runtime.supervisor import supervise_training

    if args.gpus != "auto":
        return command_train(args)
    config = load_config(args.config, system=args.system, overrides=args.set)
    return supervise_training(
        config,
        experiment=args.config,
        system=args.system,
        overrides=args.set,
        min_gpus=int(args.min_gpus or config.gpu_scheduler.min_gpus),
        max_gpus=int(args.max_gpus or config.gpu_scheduler.max_gpus),
        policy=args.gpu_policy or str(config.gpu_scheduler.policy),
        wait=not args.no_wait,
        smoke=args.smoke,
    )


def command_splits(args) -> int:
    from ac_pbgrl.data.map_splits import create_map_splits

    config = load_config(args.config, system=args.system, overrides=args.set)
    output = Path(args.output) if args.output else Path(config.project.data_root) / "map_splits.json"
    manifest = create_map_splits(config.project.maps_dir, output)
    print(json.dumps({"path": str(output), "counts": manifest["counts"], "hash": manifest["split_hash"]}, indent=2))
    return 0


def command_calibrate(args) -> int:
    from ac_pbgrl.learning.calibration import (
        default_calibration_path,
        fit_variance_calibration,
        save_calibration,
    )
    from ac_pbgrl.learning.train import apply_cli_overrides, apply_smoke_overrides

    config = load_config(args.config, system=args.system, overrides=args.set)
    if args.smoke:
        config = apply_smoke_overrides(config)
        config = apply_cli_overrides(config, args.set)
    label_root = (
        Path(args.labels_root)
        if args.labels_root
        else Path(config.project.data_root) / "labels"
    )
    report = fit_variance_calibration(
        config,
        args.checkpoint,
        label_root,
        split=args.split,
        samples=args.samples,
        batch_size=args.batch_size,
        device=args.device,
    )
    output = Path(args.output) if args.output else default_calibration_path(config)
    save_calibration(report, output)
    print(json.dumps({"output": str(output), **report}, indent=2, ensure_ascii=False))
    return 0


def command_train_gru(args) -> int:
    from ac_pbgrl.learning.gru_control import default_gru_checkpoint, train_gru_control
    from ac_pbgrl.learning.train import apply_cli_overrides, apply_smoke_overrides

    config = load_config(args.config, system=args.system, overrides=args.set)
    if args.smoke:
        config = apply_smoke_overrides(config)
        config = apply_cli_overrides(config, args.set)
    label_root = Path(args.labels_root) if args.labels_root else Path(config.project.data_root) / "labels"
    output = Path(args.output) if args.output else default_gru_checkpoint(config)
    report = train_gru_control(
        config,
        args.actor_checkpoint,
        label_root,
        output,
        samples=args.samples,
        extraction_batch_size=args.extraction_batch_size,
        sequence_batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=args.device,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True))
    return 0


def command_labels(args) -> int:
    from ac_pbgrl.data.labels import LabelShardWriter
    from ac_pbgrl.data.map_splits import create_map_splits, load_split_paths
    from ac_pbgrl.learning.future_gain import FutureGainLabeler
    from ac_pbgrl.learning.teacher import HeuristicTeacher
    from ac_pbgrl.learning.train import (
        apply_cli_overrides,
        apply_smoke_overrides,
        build_environment,
        teacher_checkpoint_path,
    )
    from ac_pbgrl.utils import sha256_file

    config = load_config(args.config, system=args.system, overrides=args.set)
    if args.smoke:
        config = apply_smoke_overrides(config)
        config = apply_cli_overrides(config, args.set)
    split_manifest = None
    if args.smoke:
        teacher = HeuristicTeacher()
        maps = [None]
    else:
        checkpoint = str(teacher_checkpoint_path(config))
        teacher = None
        split_manifest = Path(config.project.data_root) / "map_splits.json"
        if not split_manifest.is_file():
            create_map_splits(config.project.maps_dir, split_manifest)
        maps = load_split_paths(split_manifest, args.split)
    # Canonical shards retain the raw ARiADNE action graph. Methods that use the
    # hierarchy compact it only after loading, preserving identical targets and
    # candidate slots across every ablation.
    config.method.hierarchy = False
    labeler = (
        FutureGainLabeler(teacher, int(config.teacher.horizon), float(config.teacher.gamma))
        if teacher is not None
        else None
    )
    environment = build_environment(config) if args.smoke else None
    output = Path(args.output) if args.output else Path(config.project.data_root) / "labels"
    episode = 0
    provenance = {
        "teacher": "heuristic-smoke" if args.smoke else str(Path(checkpoint).resolve()),
        "teacher_sha256": "heuristic-smoke" if args.smoke else sha256_file(Path(checkpoint)),
        "horizon": int(config.teacher.horizon),
        "gamma": float(config.teacher.gamma),
        "label_definition": "discounted-baseline-frontier-minus-distance-v1",
        "terminal_bonus_included": False,
        "distance_weight": float(config.environment.distance_weight),
        "representation": "raw-action-preserving",
        "map_split_sha256": (
            "synthetic-smoke" if split_manifest is None else sha256_file(split_manifest)
        ),
    }
    pool = None
    try:
        if not args.smoke:
            from ac_pbgrl.learning.ray_labels import RayLabelPool

            pool = RayLabelPool(config, checkpoint)
        with LabelShardWriter(
            output,
            args.split,
            int(config.teacher.label_shard_size),
            provenance=provenance,
        ) as writer:
            written = writer.total
            episode = written
            while written < args.samples:
                if pool is not None:
                    states_per_task = int(config.teacher.label_states_per_task)
                    remaining = args.samples - written
                    tasks = min(pool.count, max(1, (remaining + states_per_task - 1) // states_per_task))
                    batches = pool.generate(
                        episode,
                        maps,
                        tasks=tasks,
                        states_per_task=states_per_task,
                    )
                    episode += tasks
                    for records in batches:
                        for state, labels, metadata in records:
                            if written >= args.samples:
                                break
                            writer.append(state, labels, metadata)
                            written += 1
                    continue
                map_path = maps[episode % len(maps)]
                try:
                    state, _ = environment.reset(episode=episode, map_path=map_path)
                except TypeError:
                    state, _ = environment.reset(seed=episode)
                for step in range(int(config.environment.max_episode_steps)):
                    labels = labeler.label(environment, state)
                    writer.append(
                        state,
                        labels,
                        {
                            "episode": episode,
                            "step": step,
                            "map": "synthetic" if map_path is None else Path(map_path).name,
                        },
                    )
                    written += 1
                    if written >= args.samples:
                        break
                    action = teacher.select(state)
                    result = environment.step(action)
                    state = result.state
                    if result.done:
                        break
                episode += 1
    finally:
        if pool is not None:
            pool.close()
    print(json.dumps({"output": str(output), "split": args.split, "samples": written}, indent=2))
    return 0


def command_evaluate(args) -> int:
    from ac_pbgrl.evaluation.evaluator import evaluate_checkpoint
    from ac_pbgrl.learning.train import apply_cli_overrides, apply_smoke_overrides

    config = load_config(args.config, system=args.system, overrides=args.set)
    if args.smoke:
        config = apply_smoke_overrides(config)
        config = apply_cli_overrides(config, args.set)
    checkpoint = _resolve_checkpoint(config, args.checkpoint)
    output = Path(args.output) if args.output else checkpoint.parent.parent / "evaluation"
    manifest = Path(config.project.data_root) / "map_splits.json"
    rows = evaluate_checkpoint(
        config,
        checkpoint,
        output,
        split_manifest=manifest if manifest.is_file() else None,
        split=args.split,
        map_limit=args.map_limit,
        device=args.device,
        seeds=None if not args.seeds else [int(value) for value in args.seeds.split(",")],
    )
    print(json.dumps({"output": str(output), "episodes": len(rows)}, indent=2))
    return 0


def command_figures(args) -> int:
    from ac_pbgrl.config import default_data_root
    from ac_pbgrl.evaluation.figures import generate_paper_figures

    runs_root = Path(args.runs_root) if args.runs_root else default_data_root() / "runs"
    output = Path(args.output) if args.output else default_data_root() / "paper_figures"
    generated = generate_paper_figures(
        runs_root,
        output,
        evidence_level=str(args.evidence_level),
    )
    print(json.dumps({"output": str(output), "files": [str(path) for path in generated]}, indent=2))
    return 0


def command_export(args) -> int:
    from ac_pbgrl.export import compare_onnx, export_onnx
    from ac_pbgrl.learning.train import apply_cli_overrides, apply_smoke_overrides

    config = load_config(args.config, system=args.system, overrides=args.set)
    if args.smoke:
        config = apply_smoke_overrides(config)
        config = apply_cli_overrides(config, args.set)
    checkpoint = _resolve_checkpoint(config, args.checkpoint)
    output = Path(args.output) if args.output else Path(config.project.data_root) / "exports" / f"{config.project.experiment}.onnx"
    export_onnx(config, checkpoint, output, device=args.device)
    payload = {"output": str(output)}
    if args.validate:
        payload["max_abs_error"] = compare_onnx(config, checkpoint, output, device=args.device)
    print(json.dumps(payload, indent=2))
    return 0


def _suite_entries(suite: dict, groups=("main", "ablations")) -> list[tuple[str, int]]:
    entries = []
    for group in groups:
        for config_name in suite[group]["configs"]:
            for seed in suite[group]["seeds"]:
                entries.append((config_name, int(seed)))
    return entries


def _minimum_pilot_environment_steps(config) -> int:
    """Return the first transition count with fully weighted auxiliaries."""

    update_ratio = float(config.train.gradient_updates_per_transition)
    if update_ratio <= 0:
        raise ValueError("gradient_updates_per_transition must be positive")
    scheduled_updates = int(config.loss.warmup_steps) + int(config.loss.ramp_steps)
    return int(math.ceil(scheduled_updates / update_ratio))


def _pilot_evidence_level(
    seed_count: int,
    *,
    single_run_screening: bool,
    available_seed_count: int,
) -> str:
    """Validate the requested pilot replication level and name its evidence tier."""

    seed_count = int(seed_count)
    if seed_count <= 0:
        raise ValueError("pilot seed count must be positive")
    if seed_count > int(available_seed_count):
        raise ValueError("pilot seed count exceeds the registered main-suite seeds")
    if bool(single_run_screening):
        if seed_count != 1:
            raise ValueError("single-run screening requires exactly one pilot seed")
        return "single_run_directional_screening"
    if seed_count < 3:
        raise ValueError(
            "a credibility pilot requires at least three independent seeds; "
            "use --single-run-screening with --pilot-seeds 1 for a direction-only check"
        )
    return "multi_seed_credibility_pilot"


def command_ablate(args) -> int:
    suite_path = CONFIG_ROOT / "suites" / f"{args.suite}.yaml"
    suite = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    entries = _suite_entries(suite, getattr(args, "groups", ("main", "ablations")))
    for config_name, seed in entries:
        command = [
            str(PROJECT_ROOT / "run.sh"),
            "supervise",
            "--config",
            config_name,
            "--gpus",
            args.gpus,
            "--gpu-policy",
            args.gpu_policy,
            "--set",
            f"project.seed={seed}",
            "--set",
            f"project.run_name={config_name}/seed_{seed}",
        ]
        if args.system:
            command.extend(("--system", args.system))
        if args.smoke:
            command.append("--smoke")
        result = subprocess.call(command, cwd=str(PROJECT_ROOT))
        if result:
            return result
    return 0


def command_paper(args) -> int:
    from ac_pbgrl.utils import atomic_write_json

    config = load_config("full", system="server_a40")
    data_root = Path(config.project.data_root)
    run_script = str(PROJECT_ROOT / "run.sh")
    suite = yaml.safe_load((CONFIG_ROOT / "suites" / "main.yaml").read_text(encoding="utf-8"))
    main_seeds = [int(value) for value in suite["main"]["seeds"]]
    ablation_seeds = [int(value) for value in suite["ablations"]["seeds"]]
    if bool(args.single_run_screening) and not bool(args.pilot_only):
        raise ValueError("--single-run-screening requires --pilot-only")
    if bool(args.pilot_only):
        pilot_seed_count = int(args.pilot_seeds)
        pilot_early_steps = int(args.pilot_early_steps)
        pilot_steps = int(args.pilot_steps)
        required_steps = _minimum_pilot_environment_steps(config)
        pilot_evidence_level = _pilot_evidence_level(
            pilot_seed_count,
            single_run_screening=bool(args.single_run_screening),
            available_seed_count=len(main_seeds),
        )
        if pilot_early_steps <= 0 or pilot_early_steps >= pilot_steps:
            raise ValueError("pilot_early_steps must be positive and smaller than pilot_steps")
        if pilot_steps < required_steps:
            raise ValueError(
                f"pilot_steps={pilot_steps} ends before auxiliary ramp completion at "
                f"{required_steps} environment transitions"
            )

    def invoke(arguments, *, environment=None) -> int:
        return subprocess.call([run_script, *arguments], cwd=str(PROJECT_ROOT), env=environment)

    def train_until(method: str, seed: int, environment_steps: int) -> int:
        return invoke(
            [
                "supervise",
                "--config",
                method,
                "--system",
                "server_a40",
                "--gpus",
                args.gpus,
                "--gpu-policy",
                args.gpu_policy,
                "--set",
                f"project.seed={seed}",
                "--set",
                f"project.run_name={method}/seed_{seed}",
                "--set",
                f"train.max_environment_steps={int(environment_steps)}",
            ]
        )

    def calibrate_seed(method: str, seed: int) -> int:
        checkpoint = data_root / "runs" / method / f"seed_{seed}" / "checkpoints" / "latest.pt"
        return invoke(
            [
                "calibrate",
                "--config",
                method,
                "--system",
                "server_a40",
                "--checkpoint",
                str(checkpoint),
                "--set",
                f"project.seed={seed}",
                "--samples",
                "2048",
            ]
        )

    def evaluate_entries(entries, output_root: Path, map_limit: int | None = None) -> int:
        evaluation_environment = os.environ.copy()
        evaluation_device = "cpu"
        lease = None
        if args.gpus != "cpu":
            scheduler = config.gpu_scheduler
            selected = select_gpus(
                query_gpus(),
                policy=args.gpu_policy,
                min_gpus=1,
                max_gpus=1,
                min_free_memory_gib=float(scheduler.min_free_memory_gib),
                max_utilization_pct=int(scheduler.max_utilization_pct),
                max_temperature_c=int(scheduler.max_temperature_c),
                idle_used_memory_mib=int(scheduler.idle_used_memory_mib),
                idle_utilization_pct=int(scheduler.idle_utilization_pct),
            )
            if not selected:
                return 75
            lease = GPULease(data_root / "gpu_locks", selected)
            if not lease.acquire():
                return 75
            evaluation_environment["CUDA_VISIBLE_DEVICES"] = str(selected[0].index)
            evaluation_environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            evaluation_device = "cuda:0"
        try:
            for method, seed in entries:
                run_root = data_root / "runs" / method / f"seed_{seed}"
                evaluation_root = output_root / method / f"seed_{seed}" / "evaluation"
                command = [
                    "evaluate",
                    "--config",
                    method,
                    "--system",
                    "server_a40",
                    "--checkpoint",
                    str(run_root / "checkpoints" / "latest.pt"),
                    "--output",
                    str(evaluation_root),
                    "--device",
                    evaluation_device,
                    "--seeds",
                    str(seed),
                    "--set",
                    f"project.seed={seed}",
                ]
                if map_limit is not None:
                    command.extend(("--map-limit", str(int(map_limit))))
                result = invoke(command, environment=evaluation_environment)
                if result:
                    return result
        finally:
            if lease is not None:
                lease.release()
        return 0

    teacher_path = data_root / "teachers" / "ariadne_pi" / f"step_{int(config.teacher.checkpoint_step)}.pt"
    if not teacher_path.is_file():
        teacher_run = "teacher_training/ariadne_pi"
        result = invoke(
            [
                "supervise",
                "--config",
                "ariadne_pi",
                "--system",
                "server_a40",
                "--gpus",
                args.gpus,
                "--gpu-policy",
                args.gpu_policy,
                "--set",
                f"project.run_name={teacher_run}",
                "--set",
                f"train.max_environment_steps={int(config.teacher.checkpoint_step)}",
            ]
        )
        if result:
            return result
        source = data_root / "runs" / teacher_run / "checkpoints" / "latest.pt"
        teacher_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, teacher_path)
    split_manifest = data_root / "map_splits.json"
    if not split_manifest.is_file():
        if invoke(["splits", "--config", "full", "--system", "server_a40"]):
            return 1
    label_samples = (
        min(int(args.label_samples), int(args.pilot_label_samples))
        if bool(args.pilot_only)
        else int(args.label_samples)
    )
    validation_samples = max(2048, label_samples // 10)
    for split, samples in (("train", label_samples), ("validation", validation_samples)):
        result = invoke(
            [
                "labels",
                "--config",
                "full",
                "--system",
                "server_a40",
                "--split",
                split,
                "--samples",
                str(samples),
            ]
        )
        if result:
            return result

    phase_steps = int(config.train.calibration_phase_steps)

    def temporal_prephase(method: str, seed: int, kind: str) -> int:
        run_name = f"{method}/seed_{seed}"
        checkpoint = data_root / "runs" / run_name / "checkpoints" / "latest.pt"
        artifact = (
            data_root / "calibration" / method / f"seed_{seed}.json"
            if kind == "kf"
            else data_root / "temporal" / "gru" / method / f"seed_{seed}.pt"
        )
        if artifact.is_file():
            return 0
        result = invoke(
            [
                "supervise",
                "--config",
                method,
                "--system",
                "server_a40",
                "--gpus",
                args.gpus,
                "--gpu-policy",
                args.gpu_policy,
                "--set",
                f"project.seed={seed}",
                "--set",
                f"project.run_name={run_name}",
                "--set",
                "method.temporal=none",
                "--set",
                f"train.max_environment_steps={phase_steps}",
            ]
        )
        if result:
            return result
        if kind == "kf":
            return invoke(
                [
                    "calibrate",
                    "--config",
                    method,
                    "--system",
                    "server_a40",
                    "--checkpoint",
                    str(checkpoint),
                    "--set",
                    f"project.seed={seed}",
                    "--samples",
                    "2048",
                ]
            )
        return invoke(
            [
                "train-gru",
                "--config",
                method,
                "--system",
                "server_a40",
                "--actor-checkpoint",
                str(checkpoint),
                "--set",
                f"project.seed={seed}",
                "--samples",
                str(min(int(args.label_samples), 20000)),
                "--device",
                "cpu",
            ]
        )

    if bool(args.pilot_only):
        pilot_seeds = main_seeds[:pilot_seed_count]
        pilot_entries = [
            (method, seed)
            for method in ("ariadne_pi", "full")
            for seed in pilot_seeds
        ]
        for stage_steps in (pilot_early_steps, pilot_steps):
            for seed in pilot_seeds:
                result = train_until("ariadne_pi", seed, stage_steps)
                if result:
                    return result
            for seed in pilot_seeds:
                result = temporal_prephase("full", seed, "kf")
                if result:
                    return result
                result = train_until("full", seed, stage_steps)
                if result:
                    return result
            if stage_steps == pilot_steps:
                for seed in pilot_seeds:
                    result = calibrate_seed("full", seed)
                    if result:
                        return result

            stage_root = data_root / "pilot" / f"step_{stage_steps}"
            evaluation_runs = stage_root / "runs"
            result = evaluate_entries(
                pilot_entries,
                evaluation_runs,
                map_limit=int(args.pilot_map_limit),
            )
            if result:
                return result
            figure_root = stage_root / "figures"
            result = invoke(
                [
                    "figures",
                    "--runs-root",
                    str(evaluation_runs),
                    "--output",
                    str(figure_root),
                    "--evidence-level",
                    pilot_evidence_level,
                ]
            )
            if result:
                return result
            if bool(args.single_run_screening):
                stage_kind = (
                    "single_run_early_diagnostic"
                    if stage_steps == pilot_early_steps
                    else "single_run_screening"
                )
            else:
                stage_kind = (
                    "early_diagnostic"
                    if stage_steps == pilot_early_steps
                    else "credibility_pilot"
                )
            atomic_write_json(
                stage_root / "manifest.json",
                {
                    "kind": stage_kind,
                    "evidence_level": pilot_evidence_level,
                    "statistical_claims_supported": False,
                    "methods": ["ariadne_pi", "full"],
                    "seeds": pilot_seeds,
                    "environment_steps_per_run": stage_steps,
                    "minimum_steps_for_full_auxiliary_weight": required_steps,
                    "auxiliary_weight_fully_ramped": stage_steps >= required_steps,
                    "train_label_samples": label_samples,
                    "validation_label_samples": validation_samples,
                    "evaluation_map_limit": int(args.pilot_map_limit),
                    "evaluation_runs_root": str(evaluation_runs),
                    "figure_root": str(figure_root),
                },
            )
        return 0

    # Complete every main-comparison model before spending compute on controls
    # that are only needed by the ablation suite.  The 30k full-method phase is
    # resumed by its 1M run, so it remains part of the same transition budget.
    for seed in main_seeds:
        result = temporal_prephase("full", seed, "kf")
        if result:
            return result
    result = command_ablate(
        argparse.Namespace(
            suite="main",
            gpus=args.gpus,
            gpu_policy=args.gpu_policy,
            system="server_a40",
            smoke=False,
            groups=("main",),
        )
    )
    if result:
        return result

    # Ablation-only temporal controls are deliberately deferred until all four
    # main methods and their seeds have reached the formal budget.
    for method, kind in (("potential_kf", "kf"), ("gru_control", "gru")):
        for seed in ablation_seeds:
            result = temporal_prephase(method, seed, kind)
            if result:
                return result
    result = command_ablate(
        argparse.Namespace(
            suite="main",
            gpus=args.gpus,
            gpu_policy=args.gpu_policy,
            system="server_a40",
            smoke=False,
            groups=("ablations",),
        )
    )
    if result:
        return result

    # Refit final KF temperatures on validation data, then evaluate every method
    # on one fixed physical GPU so planning latency remains comparable.
    for method, seeds in (("full", main_seeds), ("potential_kf", ablation_seeds)):
        for seed in seeds:
            result = calibrate_seed(method, seed)
            if result:
                return result

    entries = _suite_entries(suite)
    result = evaluate_entries(entries, data_root / "runs")
    if result:
        return result
    return invoke(
        [
            "figures",
            "--runs-root",
            str(data_root / "runs"),
            "--output",
            str(data_root / "paper_figures"),
        ]
    )


def command_test(args) -> int:
    command = [sys.executable, "-m", "pytest", "-q"] + list(args.pytest_args)
    return subprocess.call(command, cwd=str(PROJECT_ROOT))


COMMANDS = {
    "doctor": command_doctor,
    "train": command_train,
    "supervise": command_supervise,
    "labels": command_labels,
    "calibrate": command_calibrate,
    "train-gru": command_train_gru,
    "splits": command_splits,
    "evaluate": command_evaluate,
    "figures": command_figures,
    "export": command_export,
    "ablate": command_ablate,
    "paper": command_paper,
    "test": command_test,
}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return int(COMMANDS[args.command](args))


if __name__ == "__main__":
    raise SystemExit(main())
