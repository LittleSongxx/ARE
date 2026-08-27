from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from ac_pbgrl.config import PROJECT_ROOT, Config, deep_merge, load_config, parse_overrides, save_resolved_config
from ac_pbgrl.data.labels import LabelDataset
from ac_pbgrl.data.map_splits import create_map_splits, load_split_paths
from ac_pbgrl.envs.ariadne.adapter import AriadneExplorationEnv
from ac_pbgrl.envs.synthetic import SyntheticGraphExplorationEnv
from ac_pbgrl.runtime.gpu import gpu_inventory
from ac_pbgrl.runtime.manifest import build_run_manifest, save_run_manifest
from ac_pbgrl.utils import seed_everything

from .checkpoint import load_checkpoint, save_checkpoint
from .distributed import BatchSchedule, initialize_distributed
from .future_gain import FutureGainLabeler
from .ray_pool import RayRolloutPool
from .replay import PersistentReplayBuffer
from .rollout import EpisodeCollector
from .sac import DiscreteSACLearner
from .teacher import FrozenPolicyTeacher, FrozenQTeacher, HeuristicQTeacher, HeuristicTeacher


TEMPORARY_RESOURCE_EXIT = 75
_stop_requested = False


def _signal_handler(signum, frame):
    del signum, frame
    global _stop_requested
    _stop_requested = True


def _run_stop_requested(run_root: Path) -> bool:
    return bool(_stop_requested or (run_root / "graceful_stop.request").is_file())


def _coordinated_run_stop_requested(context, run_root: Path) -> bool:
    """Return rank zero's stop decision identically on every learner rank."""

    observed = _run_stop_requested(run_root) if context.is_primary else None
    return bool(context.broadcast_object(observed))


def apply_smoke_overrides(config: Config) -> Config:
    config = config.clone()
    config.environment.backend = "synthetic"
    config.environment.node_padding = 32
    config.environment.critic_node_padding = 32
    config.environment.candidate_padding = 8
    config.environment.max_episode_steps = 8
    config.model.embedding_dim = 32
    config.model.heads = 4
    config.model.encoder_layers = 2
    config.graph_context.local_budget = 24
    config.graph_context.region_budget = 8
    config.train.episodes = 2
    config.train.max_environment_steps = 0
    config.train.gradient_updates_per_transition = 0.125
    config.train.minimum_replay = 1
    config.train.replay_size = 128
    config.train.global_batch_size = 8
    config.train.micro_batch_size = 4
    config.train.checkpoint_seconds = 30
    config.train.ddp_sync_check_interval = 1
    config.train.rollout_backend = "sequential"
    config.loss.warmup_steps = 0
    config.loss.ramp_steps = 1
    config.filter.require_calibration = False
    config.filter.require_gru_checkpoint = False
    return config


def resolve_run_root(config: Config) -> Path:
    experiment = str(config.project.experiment)
    seed = int(config.project.seed)
    run_name = str(config.project.run_name)
    if run_name == "auto":
        run_name = f"{experiment}/seed_{seed}"
        config.project.run_name = run_name
    return Path(config.project.data_root) / "runs" / run_name


def build_environment(config: Config):
    if str(config.environment.backend) == "synthetic":
        return SyntheticGraphExplorationEnv(
            node_padding=int(config.environment.node_padding),
            candidate_padding=int(config.environment.candidate_padding),
            seed=int(config.project.seed),
        )
    return AriadneExplorationEnv(
        maps_dir=config.project.maps_dir,
        node_padding=int(config.environment.node_padding),
        critic_node_padding=int(config.environment.critic_node_padding),
        candidate_padding=int(config.environment.candidate_padding),
        max_episode_steps=int(config.environment.max_episode_steps),
        completion_threshold=float(config.environment.completion_threshold),
        terminal_reward=float(config.environment.terminal_reward),
        hierarchy=bool(config.method.hierarchy),
        local_budget=int(config.graph_context.local_budget),
        region_budget=int(config.graph_context.region_budget),
        region_size_m=float(config.graph_context.region_size_m),
        seed=int(config.project.seed),
    )


def build_labeler(config: Config, smoke: bool, device: torch.device):
    if not bool(config.method.potential):
        return None
    if smoke:
        teacher = HeuristicTeacher()
    else:
        checkpoint = str(config.teacher.checkpoint)
        if checkpoint == "auto":
            checkpoint = str(
                Path(config.project.data_root)
                / "teachers"
                / "ariadne_pi"
                / f"step_{int(config.teacher.checkpoint_step)}.pt"
            )
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(
                f"frozen teacher checkpoint is required for formal potential training: {checkpoint}"
            )
        teacher = FrozenPolicyTeacher.from_checkpoint(
            checkpoint,
            node_feature_dim=int(config.environment.node_feature_dim),
            edge_feature_dim=int(config.environment.edge_feature_dim),
            embedding_dim=int(config.model.embedding_dim),
            heads=int(config.model.heads),
            layers=int(config.model.encoder_layers),
            device=device,
        )
    return FutureGainLabeler(teacher, int(config.teacher.horizon), float(config.teacher.gamma))


def teacher_checkpoint_path(config: Config) -> Path:
    checkpoint = str(config.teacher.checkpoint)
    if checkpoint == "auto":
        return (
            Path(config.project.data_root)
            / "teachers"
            / "ariadne_pi"
            / f"step_{int(config.teacher.checkpoint_step)}.pt"
        )
    return Path(checkpoint)


def build_q_teacher(config: Config, smoke: bool, device: torch.device):
    if not bool(config.method.q_distillation):
        return None
    if smoke:
        return HeuristicQTeacher()
    checkpoint = teacher_checkpoint_path(config)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"frozen privileged Q teacher is required: {checkpoint}")
    return FrozenQTeacher.from_checkpoint(
        checkpoint,
        node_feature_dim=int(config.environment.critic_feature_dim),
        edge_feature_dim=int(config.environment.edge_feature_dim),
        embedding_dim=int(config.model.embedding_dim),
        heads=int(config.model.heads),
        layers=int(config.model.encoder_layers),
        device=device,
    )


def apply_cli_overrides(config: Config, overrides: list[str]) -> Config:
    """Reapply CLI overrides after smoke defaults so explicit values win."""

    return Config.convert(deep_merge(config.plain(), parse_overrides(overrides)))


def configure_cuda_memory(config: Config, device: torch.device) -> None:
    if device.type != "cuda":
        return
    reserve_gib = float(os.environ.get("ACPBGRL_MEMORY_RESERVE_GIB", config.gpu_scheduler.memory_reserve_gib))
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    reserve_bytes = reserve_gib * 1024**3
    allowed = max(0.1, min(0.95, (free_bytes - reserve_bytes) / total_bytes))
    torch.cuda.set_per_process_memory_fraction(float(allowed), device=device)


def append_metrics(path: Path, step: int, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = {}
    for key, value in metrics.items():
        numeric = float(value)
        values[key] = numeric if math.isfinite(numeric) else None
    record = {"time": time.time(), "step": int(step), **values}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def reconcile_metrics_to_checkpoint(path: Path, *, episodes: int, update_step: int) -> int:
    """Archive metric records produced after the checkpoint being resumed."""

    if not path.is_file():
        return 0
    retained: list[str] = []
    orphaned: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        record = json.loads(line)
        if "train/update_step" in record:
            keep = int(record["train/update_step"]) <= int(update_step)
        elif "episode/steps" in record:
            keep = int(record["step"]) <= int(episodes)
        else:
            keep = True
        (retained if keep else orphaned).append(line)
    if not orphaned:
        return 0
    archive = path.with_name(f"{path.stem}.orphaned_{time.time_ns()}{path.suffix}")
    archive.write_text("".join(orphaned), encoding="utf-8")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.writelines(retained)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return len(orphaned)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="AC-PBGRL DDP trainer")
    parser.add_argument("--config", default="full")
    parser.add_argument("--system", default=None)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--micro-batch", type=int, default=None)
    parser.add_argument("--resume", default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    global _stop_requested
    args = parse_args(argv)
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    config = load_config(args.config, system=args.system, overrides=args.set)
    if args.smoke:
        config = apply_smoke_overrides(config)
        config = apply_cli_overrides(config, args.set)
    if args.micro_batch is not None:
        config.train.micro_batch_size = int(args.micro_batch)
    context = initialize_distributed(args.device)
    configure_cuda_memory(config, context.device)
    seed_everything(int(config.project.seed) + context.rank)
    run_root = resolve_run_root(config)
    checkpoint_root = run_root / "checkpoints"
    replay_root = Path(config.project.data_root) / "replay" / str(config.project.run_name)
    metrics_path = run_root / "metrics" / "train.jsonl"
    if context.is_primary:
        run_root.mkdir(parents=True, exist_ok=True)
        save_resolved_config(config, run_root / "config_resolved.yaml")
        inventory = gpu_inventory()
        selected_indices = {
            int(value)
            for value in os.environ.get("ACPBGRL_SELECTED_GPU_INDICES", "").split(",")
            if value
        }
        manifest = build_run_manifest(
            config,
            PROJECT_ROOT,
            selected_gpus=[item for item in inventory if int(item["index"]) in selected_indices],
            micro_batch=int(config.train.micro_batch_size),
        )
        save_run_manifest(run_root / "run_manifest.json", manifest)
    context.barrier()

    learner = DiscreteSACLearner(config, context)
    resume_path = checkpoint_root / "latest.pt" if args.resume == "auto" else Path(args.resume)
    checkpoint_payload = None
    if args.resume != "none" and resume_path.is_file():
        checkpoint_payload = load_checkpoint(resume_path, learner, map_location=context.device)
    context.barrier()

    train_maps = None
    if str(config.environment.backend) != "synthetic":
        split_manifest = Path(config.project.data_root) / "map_splits.json"
        if context.is_primary and not split_manifest.is_file():
            create_map_splits(config.project.maps_dir, split_manifest)
        context.barrier()
        train_maps = load_split_paths(split_manifest, "train")

    label_dataset = None
    label_rng = None
    if bool(config.method.potential) and not args.smoke:
        label_root = Path(str(config.teacher.get("labels_root", "auto")))
        if str(label_root) == "auto":
            label_root = Path(config.project.data_root) / "labels"
        manifest = label_root / "manifest_train.json"
        if not manifest.is_file():
            raise FileNotFoundError(
                f"offline future-gain labels are required for potential training: {manifest}; "
                "run './run.sh labels --split train' first"
            )
        label_dataset = LabelDataset(label_root, "train")
        label_rng = np.random.default_rng(
            int(config.project.seed)
            + 7919 * context.rank
            + 104729 * int(learner.state.update_step)
        )

    # Rank zero alone mutates simulation state and the persistent replay. Formal
    # runs fan episodes out to Ray CPU actors; all learner ranks read the same
    # atomically committed replay fields.
    rollout_pool = None
    collector = None
    if context.is_primary:
        replay = PersistentReplayBuffer(
            replay_root,
            int(config.train.replay_size),
            int(config.project.seed) + 104729 * int(learner.state.update_step),
        )
        if checkpoint_payload is not None:
            replay_manifest = checkpoint_payload.get("replay")
            if not replay_manifest:
                raise ValueError("resume checkpoint does not contain a replay manifest")
            checkpoint_total = int(replay_manifest.get("total_added", replay_manifest["size"]))
            if checkpoint_total != int(learner.state.environment_steps):
                raise ValueError(
                    "checkpoint learner environment_steps differs from its replay manifest"
                )
            orphaned_transitions = replay.restore_checkpoint_manifest(replay_manifest)
            orphaned_metrics = reconcile_metrics_to_checkpoint(
                metrics_path,
                episodes=int(learner.state.episodes),
                update_step=int(learner.state.update_step),
            )
            if orphaned_transitions or orphaned_metrics:
                recovery_path = run_root / "recovery_events.jsonl"
                with recovery_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "time": time.time(),
                                "checkpoint": str(resume_path),
                                "orphaned_transitions": orphaned_transitions,
                                "orphaned_metric_records": orphaned_metrics,
                                "restored_environment_steps": checkpoint_total,
                                "restored_update_step": int(learner.state.update_step),
                                "restored_episodes": int(learner.state.episodes),
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
        rollout_backend = str(config.train.get("rollout_backend", "sequential"))
        if rollout_backend == "ray" and not args.smoke:
            rollout_pool = RayRolloutPool(config, context.world_size)
        else:
            environment = build_environment(config)
            labeler = (
                build_labeler(config, True, context.device)
                if args.smoke and bool(config.method.potential)
                else None
            )
            q_teacher = build_q_teacher(config, args.smoke, context.device)
            collector = EpisodeCollector(
                environment,
                learner.actor_raw,
                config,
                device=context.device,
                labeler=labeler,
                q_teacher=q_teacher,
            )

        def collect_episodes(episode_start: int, count: int, step_budget: int | None = None) -> int:
            if rollout_pool is not None:
                results = rollout_pool.collect(
                    learner.actor_raw,
                    episode_start=episode_start,
                    count=count,
                    maps=train_maps,
                    step_budget=step_budget,
                )
                for offset, (transitions, metrics) in enumerate(results):
                    replay.add_many(transitions)
                    metrics["train/environment_steps"] = float(replay.total_added)
                    append_metrics(metrics_path, episode_start + offset + 1, metrics)
                return len(results)
            assert collector is not None
            remaining_budget = None if step_budget is None else int(step_budget)
            for offset in range(count):
                episode = episode_start + offset
                map_path = None if train_maps is None else train_maps[episode % len(train_maps)]
                transitions, metrics = collector.collect(
                    episode=episode,
                    map_path=map_path,
                    max_steps=remaining_budget,
                )
                replay.add_many(transitions)
                metrics["train/environment_steps"] = float(replay.total_added)
                append_metrics(metrics_path, episode + 1, metrics)
                if remaining_budget is not None:
                    remaining_budget = max(0, remaining_budget - len(transitions))
                    if remaining_budget <= 0:
                        return offset + 1
            return count

        warmup_episode = int(learner.state.episodes)
        while len(replay) < int(config.train.minimum_replay) and not _stop_requested:
            remaining = int(config.train.minimum_replay) - len(replay)
            estimate = max(1, int(config.environment.max_episode_steps))
            desired = max(1, int(np.ceil(remaining / estimate)))
            parallelism = rollout_pool.count if rollout_pool is not None else 1
            collected = collect_episodes(
                warmup_episode,
                min(desired, parallelism, max(1, remaining)),
                step_budget=remaining,
            )
            warmup_episode += collected
            learner.state.episodes = warmup_episode
            learner.state.environment_steps = replay.total_added
    context.barrier()
    synchronized_state = context.broadcast_object(vars(learner.state) if context.is_primary else None)
    learner.state = learner.state.__class__(**synchronized_state)
    if not context.is_primary:
        replay = PersistentReplayBuffer(
            replay_root,
            int(config.train.replay_size),
            int(config.project.seed)
            + context.rank
            + 104729 * int(learner.state.update_step),
        )

    schedule = BatchSchedule(
        int(config.train.global_batch_size),
        context.world_size,
        context.rank,
        int(config.train.micro_batch_size),
    )
    last_checkpoint = time.monotonic()
    last_checkpoint_episode = int(learner.state.episodes)
    episode_cursor = int(learner.state.episodes)
    completed = False
    try:
        while episode_cursor < int(config.train.episodes):
            if context.is_primary and _run_stop_requested(run_root):
                collection_count = 0
                transitions_collected = 0
            elif context.is_primary:
                remaining_episodes = int(config.train.episodes) - episode_cursor
                parallelism = rollout_pool.count if rollout_pool is not None else 1
                max_environment_steps = int(config.train.get("max_environment_steps", 0))
                step_budget = None
                if max_environment_steps > 0:
                    remaining_steps = max_environment_steps - int(learner.state.environment_steps)
                    if remaining_steps <= 0:
                        collection_count = 0
                    else:
                        step_budget = remaining_steps
                        budget_count = max(
                            1,
                            int(np.ceil(remaining_steps / max(1, int(config.environment.max_episode_steps)))),
                        )
                        collection_count = min(
                            remaining_episodes,
                            parallelism,
                            budget_count,
                            remaining_steps,
                        )
                else:
                    collection_count = min(remaining_episodes, parallelism)
                if collection_count:
                    before_collection = replay.total_added
                    collection_count = collect_episodes(
                        episode_cursor,
                        collection_count,
                        step_budget=step_budget,
                    )
                    transitions_collected = replay.total_added - before_collection
                    learner.state.episodes = episode_cursor + collection_count
                    learner.state.environment_steps = replay.total_added
                else:
                    transitions_collected = 0
            else:
                collection_count = None
                transitions_collected = None
            collection_count = int(context.broadcast_object(collection_count))
            transitions_collected = int(context.broadcast_object(transitions_collected))
            if collection_count <= 0:
                break
            episode_cursor += collection_count
            context.barrier()
            replay.refresh()
            synchronized_state = context.broadcast_object(vars(learner.state) if context.is_primary else None)
            learner.state = learner.state.__class__(**synchronized_state)
            aggregate: dict[str, float] = {}
            if context.is_primary:
                learner.state.update_credit += transitions_collected * float(
                    config.train.gradient_updates_per_transition
                )
                update_count = int(math.floor(learner.state.update_credit + 1.0e-12))
                learner.state.update_credit -= update_count
                update_payload = (update_count, learner.state.update_credit)
            else:
                update_payload = None
            update_count, update_credit = context.broadcast_object(update_payload)
            learner.state.update_credit = float(update_credit)
            for _ in range(update_count):
                chunks = [replay.sample(size, device="cpu") for size in schedule.chunk_sizes if size > 0]
                potential_chunks = None
                # Hierarchical label compaction performs CPU graph searches.
                # During the auxiliary-loss warmup both scheduled weights are
                # exactly zero, so constructing these batches cannot affect
                # gradients and only starves the GPUs.
                if label_dataset is not None and learner.requires_offline_potential_batch():
                    potential_chunks = [
                        label_dataset.sample(
                            size,
                            label_rng,
                            hierarchy=bool(config.method.hierarchy),
                            local_budget=int(config.graph_context.local_budget),
                            region_budget=int(config.graph_context.region_budget),
                            region_size_m=float(config.graph_context.region_size_m),
                        )
                        for size in schedule.chunk_sizes
                        if size > 0
                    ]
                update_metrics = learner.update_chunks(
                    chunks,
                    schedule,
                    potential_chunks=potential_chunks,
                    potential_schedule=schedule if potential_chunks is not None else None,
                )
                for key, value in update_metrics.items():
                    aggregate[key] = aggregate.get(key, 0.0) + value / update_count
            aggregate["train/update_step"] = float(learner.state.update_step)
            aggregate["train/environment_steps"] = float(learner.state.environment_steps)
            aggregate["train/updates_per_transition"] = float(
                config.train.gradient_updates_per_transition
            )
            if context.is_primary:
                append_metrics(metrics_path, learner.state.update_step, aggregate)
                now = time.monotonic()
                run_stop_requested = _run_stop_requested(run_root)
                due = (
                    now - last_checkpoint >= float(config.train.checkpoint_seconds)
                    or episode_cursor - last_checkpoint_episode >= int(config.train.checkpoint_episodes)
                    or run_stop_requested
                )
                if due:
                    save_checkpoint(
                        checkpoint_root / f"episode_{episode_cursor:06d}.pt",
                        learner,
                        config=config.plain(),
                        replay_manifest=replay.manifest(),
                        run_manifest={"world_size": context.world_size, "micro_batch": schedule.micro_batch_size},
                    )
                    last_checkpoint = now
                    last_checkpoint_episode = episode_cursor
            else:
                run_stop_requested = None
            stop = context.broadcast_object(run_stop_requested)
            max_environment_steps = int(config.train.get("max_environment_steps", 0))
            reached_budget = max_environment_steps > 0 and learner.state.environment_steps >= max_environment_steps
            reached_budget = context.broadcast_object(reached_budget if context.is_primary else None)
            if stop or reached_budget:
                break
        max_environment_steps = int(config.train.get("max_environment_steps", 0))
        completed = episode_cursor >= int(config.train.episodes) or (
            max_environment_steps > 0 and learner.state.environment_steps >= max_environment_steps
        )
        # Only rank zero owns the request-file lifecycle.  Broadcasting its
        # observation prevents a teardown race where one rank returns success
        # while another reads the file after the supervisor has archived it
        # and returns the temporary-resource code to torch elastic.
        gracefully_stopped = _coordinated_run_stop_requested(context, run_root)
        if context.is_primary:
            save_checkpoint(
                checkpoint_root / ("final.pt" if completed else "interrupted.pt"),
                learner,
                config=config.plain(),
                replay_manifest=replay.manifest(),
                run_manifest={"world_size": context.world_size, "micro_batch": schedule.micro_batch_size},
            )
            if not completed:
                (run_root / "restart.request").write_text(
                    "training stopped at an update boundary and is safe to resume\n",
                    encoding="utf-8",
                )
        context.barrier()
        # A coordinated request is a successful checkpointed shutdown.  The
        # supervisor uses restart.request to distinguish it from completion;
        # returning zero keeps torch elastic from classifying healthy ranks as
        # failures and killing a peer during teardown.
        return 0 if completed or gracefully_stopped else TEMPORARY_RESOURCE_EXIT
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            if context.is_primary:
                (run_root / "oom.request").write_text(str(exc), encoding="utf-8")
            return TEMPORARY_RESOURCE_EXIT
        raise
    finally:
        if context.is_primary and rollout_pool is not None:
            rollout_pool.close()
        context.close()


if __name__ == "__main__":
    raise SystemExit(main())
