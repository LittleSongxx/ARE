from __future__ import annotations

import copy
import os
import signal
import time

import numpy as np
import ray
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    PACKAGE_ROOT = Path(__file__).resolve().parent
    if str(PACKAGE_ROOT.parent) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT.parent))

    from ARiADNE.evaluation import evaluate_policy, summarize_eval_results
    from ARiADNE.model import PolicyNet, QNet
    from ARiADNE.parameter import (
        EMBEDDING_DIM,
        GAMMA,
        K_SIZE,
        LR,
        SRC_ROOT,
        RuntimeConfig,
        build_run_session,
        checkpoint_model_config,
        ensure_checkpoint_compatible,
        ensure_output_dirs,
        get_critic_node_input_dim,
        get_checkpoint_path,
        get_checkpoint_final_path,
        get_checkpoint_interrupted_path,
        get_gifs_path,
        get_latest_checkpoint_path,
        get_model_path,
        get_monitor_path,
        get_node_input_dim,
        get_result_eval_path,
        get_train_path,
        is_smoke_run,
    )
    from ARiADNE.replay_buffer import ReplayBuffer, episode_buffer_to_transitions
    from ARiADNE.runtime_utils import resolve_ray_num_cpus, resolve_ray_worker_num_cpus, resolve_worker_num_threads
    from ARiADNE.training_monitor import TrainingMonitor
    from ARiADNE.runner import Runner
    from ARiADNE.utils import ensure_bucket_dir, get_bucket_name, get_numba_runtime_status
else:
    from .evaluation import evaluate_policy, summarize_eval_results
    from .model import PolicyNet, QNet
    from .parameter import (
        EMBEDDING_DIM,
        GAMMA,
        K_SIZE,
        LR,
        SRC_ROOT,
        RuntimeConfig,
        build_run_session,
        checkpoint_model_config,
        ensure_checkpoint_compatible,
        ensure_output_dirs,
        get_critic_node_input_dim,
        get_checkpoint_path,
        get_checkpoint_final_path,
        get_checkpoint_interrupted_path,
        get_gifs_path,
        get_latest_checkpoint_path,
        get_model_path,
        get_monitor_path,
        get_node_input_dim,
        get_result_eval_path,
        get_train_path,
        is_smoke_run,
    )
    from .replay_buffer import ReplayBuffer, episode_buffer_to_transitions
    from .runtime_utils import resolve_ray_num_cpus, resolve_ray_worker_num_cpus, resolve_worker_num_threads
    from .training_monitor import TrainingMonitor
    from .runner import Runner
    from .utils import ensure_bucket_dir, get_bucket_name, get_numba_runtime_status


# ── interrupt handling ──────────────────────────────────────────────────────────
_interrupted = False
_TENSORBOARD_TAGS = {
    "value": "Losses/Value",
    "policy_loss": "Losses/Policy Loss",
    "alpha_loss": "Losses/Alpha Loss",
    "q_value_loss": "Losses/Q Value Loss",
    "entropy": "Losses/Entropy",
    "policy_grad_norm": "Losses/Policy Grad Norm",
    "q_value_grad_norm": "Losses/Q Value Grad Norm",
    "log_alpha": "Losses/Log Alpha",
    "reward": "Perf/Reward",
    "reward_info": "Perf/Reward Info",
    "reward_dist": "Perf/Reward Dist",
    "reward_safe": "Perf/Reward Safe",
    "reward_terminal": "Perf/Reward Terminal",
    "travel_dist": "Perf/Travel Distance",
    "explored_rate": "Perf/Explored Rate",
    "success_rate": "Perf/Success Rate",
    "episode_return": "Perf/Episode Return",
    "episode_steps": "Perf/Episode Steps",
    "curriculum_level_index": "Curriculum/Level Index",
    "buffer_size": "Replay/buffer_size",
    "replay_ratio": "Replay/replay_ratio",
    "per_beta": "Replay/per_beta",
    "is_weight_mean": "Replay/is_weight_mean",
    "n_step_mean": "Replay/n_step_mean",
    "n_valid_mean": "SAC/n_valid_mean",
    "entropy_target_mean": "SAC/entropy_target_mean",
    "td_error_mean": "PER/td_error_mean",
}


def _signal_handler(signum, frame):
    """Raise KeyboardInterrupt immediately so ray.wait() is not awaited."""
    global _interrupted
    _interrupted = True
    print(
        f"\nInterrupt received (signal {signum}). "
        "Saving last completed checkpoint and exiting immediately..."
    )
    raise KeyboardInterrupt


def _build_checkpoint_dict(
    policy_net, q1, q2, log_alpha,
    policy_opt, q1_opt, q2_opt, alpha_opt, episode,
    runtime_config: RuntimeConfig,
    learner_update_step: int = 0,
    policy_update_step: int = 0,
    target_q_update_counter: int = 1,
):
    """Snapshot the current training state with independent tensor copies.

    All tensors are cloned to CPU so the cached dict is safe to keep across
    in-place optimizer updates.
    """
    def _cpu_sd(module):
        return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}

    return {
        "policy_model": _cpu_sd(policy_net),
        "q_net1_model": _cpu_sd(q1),
        "q_net2_model": _cpu_sd(q2),
        "log_alpha": log_alpha.detach().cpu().clone(),
        "policy_optimizer": copy.deepcopy(policy_opt.state_dict()),
        "q_net1_optimizer": copy.deepcopy(q1_opt.state_dict()),
        "q_net2_optimizer": copy.deepcopy(q2_opt.state_dict()),
        "log_alpha_optimizer": copy.deepcopy(alpha_opt.state_dict()),
        "episode": episode,
        "learner_update_step": int(learner_update_step),
        "policy_update_step": int(policy_update_step),
        "target_q_update_counter": int(target_q_update_counter),
        "config_snapshot": checkpoint_model_config(runtime_config),
        "node_input_dim": get_node_input_dim(runtime_config),
        "critic_node_input_dim": get_critic_node_input_dim(runtime_config),
    }


def _get_worker_gpu_share(runtime_config: RuntimeConfig, worker_uses_cuda: bool) -> float:
    return 0.0


def _state_dict_to_device(state_dict, device):
    return {key: value.detach().to(device) for key, value in state_dict.items()}


def _visible_gpu_count_from_env() -> int | None:
    raw_value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw_value is None:
        return None
    entries = [entry.strip() for entry in raw_value.split(",") if entry.strip()]
    if not entries:
        return None
    if any(entry == "-1" for entry in entries):
        return 0
    return len(entries)


def _cuda_runtime_status() -> tuple[bool, str | None]:
    if not torch.cuda.is_available():
        return False, "No CUDA GPUs are available"
    try:
        probe = torch.zeros(1, device="cuda")
        probe = probe + 1
        probe.item()
        torch.cuda.synchronize()
    except (AssertionError, RuntimeError) as exc:
        return False, str(exc)
    return True, None


def _resolve_learner_device(runtime_config: RuntimeConfig) -> torch.device:
    if not runtime_config.use_gpu_global:
        return torch.device("cpu")
    cuda_ok, reason = _cuda_runtime_status()
    if cuda_ok:
        return torch.device("cuda")
    print(f"CUDA is unavailable for the learner; falling back to CPU. Reason: {reason}")
    return torch.device("cpu")


def _resolve_learner_gpu_count(runtime_config: RuntimeConfig, device: torch.device) -> int:
    if device.type != "cuda":
        return 0

    env_visible_gpu_count = _visible_gpu_count_from_env()
    try:
        detected_gpu_count = int(torch.cuda.device_count())
    except (AssertionError, RuntimeError):
        detected_gpu_count = 0

    visible_gpu_count = detected_gpu_count
    if visible_gpu_count <= 0 and env_visible_gpu_count is not None:
        visible_gpu_count = env_visible_gpu_count
    if visible_gpu_count <= 0:
        visible_gpu_count = 1

    requested_gpu_count = int(runtime_config.num_gpu)
    if requested_gpu_count <= 0:
        return visible_gpu_count
    return max(1, min(requested_gpu_count, visible_gpu_count))


def _wrap_data_parallel(module: nn.Module, device_ids: list[int]) -> nn.Module:
    if len(device_ids) <= 1:
        return module
    return nn.DataParallel(module, device_ids=device_ids, output_device=device_ids[0])


def _resolve_worker_runtime(
    runtime_config: RuntimeConfig,
    available_gpus: float | None = None,
) -> tuple[RuntimeConfig, float]:
    if runtime_config.use_gpu:
        print("worker_gpu_policy Rollout workers are pinned to CPU; learner keeps the visible GPUs.")
    return runtime_config.with_overrides(use_gpu=False, num_gpu=0), 0.0


def _ensure_pythonpath_entry(path: str) -> str:
    existing = os.environ.get("PYTHONPATH", "")
    entries = [entry for entry in existing.split(os.pathsep) if entry]
    if path not in entries:
        entries.insert(0, path)
    pythonpath = os.pathsep.join(entries)
    os.environ["PYTHONPATH"] = pythonpath
    return pythonpath


def _format_metric(value) -> str:
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isnan(value):
        return "nan"
    return f"{value:.4f}"


def _nanmean_or_nan(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or np.all(np.isnan(array)):
        return float("nan")
    return float(np.nanmean(array))


def _resolve_per_beta(runtime_config: RuntimeConfig, learner_update_step: int) -> float:
    if not runtime_config.enable_per:
        return float("nan")
    progress = min(max(float(learner_update_step), 0.0) / float(runtime_config.per_beta_steps), 1.0)
    return float(runtime_config.per_beta0 + progress * (1.0 - runtime_config.per_beta0))


def _resolve_updates_to_run(runtime_config: RuntimeConfig, pending_update_budget: float) -> int:
    if runtime_config.replay_ratio <= 0:
        return int(runtime_config.train_updates_per_iter)
    return min(int(max(pending_update_budget, 0.0)), int(runtime_config.train_updates_per_iter))


def _actor_update_due(learner_update_step: int, policy_delay: int) -> bool:
    return ((int(learner_update_step) + 1) % max(int(policy_delay), 1)) == 0


def _resume_counters_from_checkpoint(checkpoint: dict[str, object]) -> tuple[int, int, int]:
    return (
        int(checkpoint.get("learner_update_step", 0)),
        int(checkpoint.get("policy_update_step", 0)),
        int(checkpoint.get("target_q_update_counter", 1)),
    )


def _soft_update(target_module: nn.Module, source_module: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_param, source_param in zip(target_module.parameters(), source_module.parameters()):
            target_param.mul_(1.0 - tau).add_(source_param, alpha=tau)


def _maybe_sync_target_networks(
    runtime_config: RuntimeConfig,
    target_q_update_counter: int,
    global_target_q_net1: nn.Module,
    global_target_q_net2: nn.Module,
    global_q_net1: nn.Module,
    global_q_net2: nn.Module,
) -> int:
    if runtime_config.enable_soft_target_update and runtime_config.tau > 0:
        _soft_update(global_target_q_net1, global_q_net1, runtime_config.tau)
        _soft_update(global_target_q_net2, global_q_net2, runtime_config.tau)
        return target_q_update_counter

    target_q_update_counter += 1
    if target_q_update_counter > 64:
        target_q_update_counter = 1
        global_target_q_net1.load_state_dict(global_q_net1.state_dict())
        global_target_q_net2.load_state_dict(global_q_net2.state_dict())
        global_target_q_net1.eval()
        global_target_q_net2.eval()
    return target_q_update_counter


def _resolve_entropy_target_tensor(
    edge_padding_mask: torch.Tensor,
    runtime_config: RuntimeConfig,
    default_entropy_target: float,
) -> tuple[torch.Tensor, float, float]:
    if not runtime_config.enable_adaptive_entropy_target:
        return (
            torch.full(
                (edge_padding_mask.size(0),),
                float(default_entropy_target),
                dtype=torch.float32,
                device=edge_padding_mask.device,
            ),
            float("nan"),
            float("nan"),
        )

    n_valid = (edge_padding_mask == 0).sum(dim=-1).float().squeeze(1).clamp_min(1.0)
    entropy_target_tensor = runtime_config.entropy_target_scale * torch.log(n_valid)
    return entropy_target_tensor, float(n_valid.mean().item()), float(entropy_target_tensor.mean().item())


def _move_batch_to_device(batch: dict[str, torch.Tensor | None], device: torch.device) -> dict[str, torch.Tensor | None]:
    return {
        field_name: value.to(device) if isinstance(value, torch.Tensor) else None
        for field_name, value in batch.items()
    }


def _build_actor_observation(batch: dict[str, torch.Tensor | None]) -> list[torch.Tensor | None]:
    return [
        batch["node_inputs"],
        batch["node_padding_mask"],
        batch["edge_mask"],
        batch["current_index"],
        batch["current_edge"],
        batch["edge_padding_mask"],
        batch["actor_attn_bias"],
    ]


def _build_next_actor_observation(batch: dict[str, torch.Tensor | None]) -> list[torch.Tensor | None]:
    return [
        batch["next_node_inputs"],
        batch["next_node_padding_mask"],
        batch["next_edge_mask"],
        batch["next_current_index"],
        batch["next_current_edge"],
        batch["next_edge_padding_mask"],
        batch["actor_next_attn_bias"],
    ]


def _build_critic_observation(batch: dict[str, torch.Tensor | None]) -> list[torch.Tensor | None]:
    return [
        batch["critic_node_inputs"],
        batch["critic_node_padding_mask"],
        batch["critic_edge_mask"],
        batch["critic_current_index"],
        batch["critic_current_edge"],
        batch["critic_edge_padding_mask"],
        batch["critic_attn_bias"],
    ]


def _build_next_critic_observation(batch: dict[str, torch.Tensor | None]) -> list[torch.Tensor | None]:
    return [
        batch["critic_next_node_inputs"],
        batch["critic_next_node_padding_mask"],
        batch["critic_next_edge_mask"],
        batch["critic_next_current_index"],
        batch["critic_next_current_edge"],
        batch["critic_next_edge_padding_mask"],
        batch["critic_next_attn_bias"],
    ]


def _print_train_progress(curr_episode, max_episodes, train_metrics, buffer_size, elapsed_sec):
    total_episodes = max(max_episodes, 1)
    progress = 100.0 * curr_episode / total_episodes
    print(
        "train_progress "
        f"episode={curr_episode}/{max_episodes} "
        f"progress={progress:.1f}% "
        f"buffer={buffer_size} "
        f"elapsed_sec={elapsed_sec:.1f} "
        f"reward={_format_metric(train_metrics['reward'])} "
        f"explored_rate={_format_metric(train_metrics['explored_rate'])} "
        f"success_rate={_format_metric(train_metrics['success_rate'])} "
        f"episode_return={_format_metric(train_metrics['episode_return'])} "
        f"episode_steps={_format_metric(train_metrics['episode_steps'])} "
        f"policy_loss={_format_metric(train_metrics['policy_loss'])} "
        f"q_value_loss={_format_metric(train_metrics['q_value_loss'])}"
    )


def _save_checkpoint(
    global_policy_net,
    global_q_net1,
    global_q_net2,
    log_alpha,
    global_policy_optimizer,
    global_q_net1_optimizer,
    global_q_net2_optimizer,
    log_alpha_optimizer,
    curr_episode,
    learner_update_step,
    policy_update_step,
    target_q_update_counter,
    bucket_size,
    current_checkpoint_path,
    current_model_path,
    runtime_config: RuntimeConfig,
):
    if current_checkpoint_path is None:
        return

    checkpoint = _build_checkpoint_dict(
        global_policy_net,
        global_q_net1,
        global_q_net2,
        log_alpha,
        global_policy_optimizer,
        global_q_net1_optimizer,
        global_q_net2_optimizer,
        log_alpha_optimizer,
        curr_episode,
        runtime_config,
        learner_update_step=learner_update_step,
        policy_update_step=policy_update_step,
        target_q_update_counter=target_q_update_counter,
    )
    torch.save(checkpoint, current_checkpoint_path)
    if current_model_path is not None:
        bucket_dir = ensure_bucket_dir(current_model_path, curr_episode, bucket_size)
        torch.save(checkpoint, bucket_dir / "checkpoint.pth")


def _write_to_tensor_board(writer, tensorboard_data, curr_episode):
    aggregated: dict[str, float] = {}
    keys = {key for record in tensorboard_data for key in record}
    for key in keys:
        aggregated[key] = _nanmean_or_nan(record.get(key, float("nan")) for record in tensorboard_data)

    for key, tag in _TENSORBOARD_TAGS.items():
        if key not in aggregated or np.isnan(aggregated[key]):
            continue
        writer.add_scalar(tag=tag, scalar_value=aggregated[key], global_step=curr_episode)
    return aggregated


def _append_perf_metrics(perf_metrics: dict[str, list[float]], metrics: dict[str, object]) -> None:
    for name, value in metrics.items():
        perf_metrics.setdefault(name, []).append(float(value))


def _write_eval_to_tensor_board(writer, eval_summary, curr_episode):
    writer.add_scalar(tag="Eval/Explored Rate", scalar_value=eval_summary["explored_rate"], global_step=curr_episode)
    writer.add_scalar(tag="Eval/Travel Distance", scalar_value=eval_summary["travel_dist"], global_step=curr_episode)
    writer.add_scalar(tag="Eval/Success Rate", scalar_value=eval_summary["success_rate"], global_step=curr_episode)
    writer.add_scalar(tag="Eval/Episode Return", scalar_value=eval_summary["episode_return"], global_step=curr_episode)
    writer.add_scalar(tag="Eval/Steps Taken", scalar_value=eval_summary["steps_taken"], global_step=curr_episode)


def _maybe_rotate_writer(writer, current_writer_bucket, current_train_path, episode, bucket_size):
    writer_bucket = get_bucket_name(episode, bucket_size)
    if writer_bucket != current_writer_bucket:
        writer.close()
        writer = SummaryWriter(ensure_bucket_dir(current_train_path, episode, bucket_size))
        current_writer_bucket = writer_bucket
    return writer, current_writer_bucket


def main(runtime_config: RuntimeConfig | None = None) -> dict:
    global _interrupted
    _interrupted = False

    runtime_config = runtime_config or RuntimeConfig()
    if runtime_config.run_session is None:
        runtime_config = runtime_config.with_overrides(run_session=build_run_session(runtime_config.run_name))
    ensure_output_dirs(runtime_config)

    current_checkpoint_path = None if is_smoke_run(runtime_config) else get_checkpoint_path(runtime_config)
    current_model_path = None if is_smoke_run(runtime_config) else get_model_path(runtime_config)
    current_eval_path = get_result_eval_path(runtime_config)
    current_train_path = get_train_path(runtime_config)
    current_gifs_path = get_gifs_path(runtime_config)
    train_start_time = time.time()

    device = _resolve_learner_device(runtime_config)

    current_writer_bucket = get_bucket_name(1, runtime_config.result_bucket_episodes)
    writer = SummaryWriter(ensure_bucket_dir(current_train_path, 1, runtime_config.result_bucket_episodes))

    training_monitor = None
    if runtime_config.enable_training_monitor:
        training_monitor = TrainingMonitor(
            get_monitor_path(runtime_config),
            window_size=runtime_config.monitor_window,
            snapshot_interval=runtime_config.monitor_snapshot_interval,
        )

    pythonpath = _ensure_pythonpath_entry(str(SRC_ROOT))
    requested_ray_num_cpus = resolve_ray_num_cpus(runtime_config)
    worker_num_cpus = resolve_ray_worker_num_cpus(runtime_config)
    worker_num_threads = resolve_worker_num_threads(runtime_config, worker_num_cpus)
    learner_gpu_count = _resolve_learner_gpu_count(runtime_config, device)
    learner_device_ids = list(range(learner_gpu_count))
    worker_gpu_share = 0.0

    ray_init_kwargs = {
        "ignore_reinit_error": True,
        "runtime_env": {
            "env_vars": {
                "PYTHONPATH": pythonpath,
                "ARIADNE_RAY_WORKER_NUM_CPUS": str(worker_num_cpus),
                "ARIADNE_WORKER_NUM_THREADS": str(worker_num_threads),
            }
        },
    }
    if requested_ray_num_cpus is not None:
        ray_init_kwargs["num_cpus"] = requested_ray_num_cpus

    ray.init(**ray_init_kwargs)
    cluster_resources = ray.cluster_resources()
    cluster_cpus = int(cluster_resources.get("CPU", 0))
    cluster_gpus = float(cluster_resources.get("GPU", 0.0))
    worker_runtime_config, worker_gpu_share = _resolve_worker_runtime(runtime_config, cluster_gpus)
    local_device = torch.device("cpu")

    max_parallel_workers = 0
    if cluster_cpus > 0:
        max_parallel_workers = max(1, cluster_cpus // worker_num_cpus)
    print(
        "ray_rollout_config "
        f"requested_cluster_cpus={requested_ray_num_cpus if requested_ray_num_cpus is not None else 'auto'} "
        f"cluster_cpus={cluster_cpus} "
        f"cluster_gpus={cluster_gpus} "
        f"meta_agents={runtime_config.num_meta_agent} "
        f"worker_num_cpus={worker_num_cpus} "
        f"worker_num_threads={worker_num_threads} "
        f"worker_num_gpus={worker_gpu_share} "
        f"max_parallel_workers={max_parallel_workers} "
        f"pythonpath_root={SRC_ROOT} "
        f"learner_device={device.type} "
        f"learner_num_gpus={learner_gpu_count} "
        f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES') or '<all-visible>'} "
        f"worker_device={local_device.type}"
    )
    if cluster_cpus <= 1 and runtime_config.num_meta_agent > 1:
        print(
            "ray_cpu_warning "
            "Ray currently sees only one CPU. "
            "Set ARIADNE_RAY_NUM_CPUS or --ray-num-cpus if the server actually has more available cores."
        )
    if cluster_cpus > 0 and runtime_config.num_meta_agent * worker_num_cpus > cluster_cpus:
        print(
            "ray_worker_capacity_warning "
            f"requested_worker_cpus={runtime_config.num_meta_agent * worker_num_cpus} "
            f"cluster_cpus={cluster_cpus} "
            f"max_parallel_workers={max_parallel_workers}. "
            "Some Ray workers will queue until CPU resources free up."
        )
    if worker_num_threads > worker_num_cpus:
        print(
            "ray_worker_thread_warning "
            f"worker_num_threads={worker_num_threads} exceeds worker_num_cpus={worker_num_cpus}. "
            "Each actor may oversubscribe its reserved CPU resources."
        )
    print(
        "training_start "
        f"run_session={runtime_config.run_session} "
        f"max_episodes={runtime_config.max_episodes} "
        f"summary_window={runtime_config.summary_window} "
        f"minimum_buffer_size={runtime_config.minimum_buffer_size} "
        f"batch_size={runtime_config.batch_size} "
        f"train_updates_per_iter={runtime_config.train_updates_per_iter}"
    )
    if runtime_config.enable_curriculum:
        print(
            "curriculum_config "
            f"source={runtime_config.curriculum_source} "
            f"levels={runtime_config.curriculum_levels} "
            f"milestones={runtime_config.curriculum_milestones} "
            f"mix_window={runtime_config.curriculum_mix_window} "
            f"use_in_eval={int(runtime_config.use_curriculum_in_eval)}"
        )
    numba_status = get_numba_runtime_status()
    print(
        "numba_runtime "
        f"installed={int(bool(numba_status['installed']))} "
        f"version={numba_status['version'] or '<none>'} "
        f"sensor_accel={int(bool(numba_status['sensor_enabled']))} "
        f"collision_accel={int(bool(numba_status['collision_enabled']))}"
    )

    # ── install signal handlers ─────────────────────────────────────────────
    old_sigint = signal.signal(signal.SIGINT, _signal_handler)
    old_sigterm = signal.signal(signal.SIGTERM, _signal_handler)

    actor_node_dim = get_node_input_dim(runtime_config)
    critic_node_dim = get_critic_node_input_dim(runtime_config)
    global_policy_net = PolicyNet(actor_node_dim, EMBEDDING_DIM).to(device)
    global_q_net1 = QNet(critic_node_dim, EMBEDDING_DIM).to(device)
    global_q_net2 = QNet(critic_node_dim, EMBEDDING_DIM).to(device)
    global_target_q_net1 = QNet(critic_node_dim, EMBEDDING_DIM).to(device)
    global_target_q_net2 = QNet(critic_node_dim, EMBEDDING_DIM).to(device)
    log_alpha = torch.FloatTensor([-2]).to(device)
    log_alpha.requires_grad = True

    global_policy_optimizer = optim.Adam(global_policy_net.parameters(), lr=LR)
    global_q_net1_optimizer = optim.Adam(global_q_net1.parameters(), lr=LR)
    global_q_net2_optimizer = optim.Adam(global_q_net2.parameters(), lr=LR)
    log_alpha_optimizer = optim.Adam([log_alpha], lr=1e-4)
    entropy_target = runtime_config.entropy_target_scale * (-np.log(1 / K_SIZE))

    curr_episode = 0
    next_episode = 0
    learner_update_step = 0
    policy_update_step = 0
    target_q_update_counter = 1

    checkpoint_source = None
    if runtime_config.resume_from is not None:
        checkpoint_source = runtime_config.resume_from
    elif runtime_config.load_model:
        latest = get_latest_checkpoint_path(runtime_config.run_name)
        if latest.exists():
            checkpoint_source = latest
        elif current_checkpoint_path is not None and current_checkpoint_path.exists():
            checkpoint_source = current_checkpoint_path

    if checkpoint_source is not None:
        checkpoint = torch.load(checkpoint_source, map_location=device, weights_only=False)
        ensure_checkpoint_compatible(checkpoint, runtime_config)
        global_policy_net.load_state_dict(checkpoint["policy_model"])
        global_q_net1.load_state_dict(checkpoint["q_net1_model"])
        global_q_net2.load_state_dict(checkpoint["q_net2_model"])
        log_alpha = checkpoint["log_alpha"].to(device)
        log_alpha.requires_grad_(True)
        log_alpha_optimizer = optim.Adam([log_alpha], lr=1e-4)

        global_policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        global_q_net1_optimizer.load_state_dict(checkpoint["q_net1_optimizer"])
        global_q_net2_optimizer.load_state_dict(checkpoint["q_net2_optimizer"])
        log_alpha_optimizer.load_state_dict(checkpoint["log_alpha_optimizer"])
        curr_episode = checkpoint["episode"]
        next_episode = curr_episode
        learner_update_step, policy_update_step, target_q_update_counter = _resume_counters_from_checkpoint(checkpoint)
        print(
            f"loading_model checkpoint={checkpoint_source} resume_episode={curr_episode} "
            f"learner_update_step={learner_update_step} "
            f"policy_update_step={policy_update_step}"
        )
        print(
            "resume_replay_reset "
            f"minimum_buffer_size={runtime_config.minimum_buffer_size} "
            "replay_restored=0"
        )

    global_target_q_net1.load_state_dict(global_q_net1.state_dict())
    global_target_q_net2.load_state_dict(global_q_net2.state_dict())
    global_target_q_net1.eval()
    global_target_q_net2.eval()

    runner_options = {
        "num_cpus": worker_num_cpus,
        "num_gpus": worker_gpu_share,
    }
    RemoteRunner = ray.remote(Runner)
    meta_agents = [
        RemoteRunner.options(**runner_options).remote(i, worker_runtime_config)
        for i in range(runtime_config.num_meta_agent)
    ]

    policy_wrapper = _wrap_data_parallel(global_policy_net, learner_device_ids) if device.type == "cuda" else global_policy_net
    q1_wrapper = _wrap_data_parallel(global_q_net1, learner_device_ids) if device.type == "cuda" else global_q_net1
    q2_wrapper = _wrap_data_parallel(global_q_net2, learner_device_ids) if device.type == "cuda" else global_q_net2
    target_q1_wrapper = _wrap_data_parallel(global_target_q_net1, learner_device_ids) if device.type == "cuda" else global_target_q_net1
    target_q2_wrapper = _wrap_data_parallel(global_target_q_net2, learner_device_ids) if device.type == "cuda" else global_target_q_net2

    weights_set = [_state_dict_to_device(global_policy_net.state_dict(), local_device)]
    job_list = []
    for meta_agent in meta_agents:
        if next_episode >= runtime_config.max_episodes:
            break
        next_episode += 1
        job_list.append(meta_agent.job.remote(weights_set, next_episode))

    training_data: list[dict[str, float]] = []
    perf_metrics: dict[str, list[float]] = {}
    replay_buffer = ReplayBuffer(
        runtime_config.replay_size,
        prioritized=runtime_config.enable_per,
        alpha=runtime_config.per_alpha,
    )
    pending_update_budget = 0.0
    save_gap = max(int(runtime_config.save_img_gap), 1)
    auto_eval_episodes = set()
    _last_good_checkpoint = None

    try:
        while job_list:
            done_id, job_list = ray.wait(job_list)
            done_jobs = ray.get(done_id)
            completed_episodes = []

            for job in done_jobs:
                job_results, metrics, info = job
                completed_episodes.append(info["episode_number"])
                for transition in episode_buffer_to_transitions(job_results):
                    replay_buffer.push(transition)
                _append_perf_metrics(perf_metrics, metrics)

            new_transitions = sum(len(job[0][0]) for job in done_jobs)
            buffer_size = replay_buffer.size

            curr_episode = max(job[2]["episode_number"] for job in done_jobs)
            latest_metrics = done_jobs[-1][1]
            print(
                "episode_complete "
                f"episode={curr_episode}/{runtime_config.max_episodes} "
                f"meta_agent={done_jobs[-1][2]['id']} "
                f"buffer={buffer_size} "
                f"travel_dist={_format_metric(latest_metrics['travel_dist'])} "
                f"explored_rate={_format_metric(latest_metrics['explored_rate'])} "
                f"success_rate={_format_metric(latest_metrics['success_rate'])} "
                f"episode_return={_format_metric(latest_metrics['episode_return'])} "
                f"episode_steps={_format_metric(latest_metrics['episode_steps'])}"
            )

            if next_episode < runtime_config.max_episodes:
                next_episode += 1
                job_list.append(meta_agents[info["id"]].job.remote(weights_set, next_episode))

            if runtime_config.replay_ratio > 0:
                pending_update_budget += new_transitions * runtime_config.replay_ratio

            if buffer_size >= runtime_config.minimum_buffer_size:
                updates_to_run = _resolve_updates_to_run(runtime_config, pending_update_budget)

                sample_batch_size = min(runtime_config.batch_size, buffer_size)
                for _ in range(updates_to_run):
                    if sample_batch_size <= 0:
                        break

                    per_beta = _resolve_per_beta(runtime_config, learner_update_step)
                    replay_sample = replay_buffer.sample(
                        sample_batch_size,
                        beta=per_beta if runtime_config.enable_per else None,
                    )
                    batch = _move_batch_to_device(replay_sample.batch, device)
                    is_weights = replay_sample.is_weights.to(device).unsqueeze(-1)

                    observation = _build_actor_observation(batch)
                    next_observation = _build_next_actor_observation(batch)
                    critic_observation = _build_critic_observation(batch)
                    critic_next_observation = _build_next_critic_observation(batch)

                    action = batch["action"]
                    reward = batch["reward"]
                    done = batch["done"].float()
                    gamma_pow = batch["gamma_pow"]
                    n_step_actual = batch["n_step_actual"].float()

                    with torch.no_grad():
                        next_logp = policy_wrapper(*next_observation)
                        next_q_values1 = target_q1_wrapper(*critic_next_observation)
                        next_q_values2 = target_q2_wrapper(*critic_next_observation)
                        next_q_values = torch.min(next_q_values1, next_q_values2)
                        value_prime = torch.sum(
                            next_logp.exp().unsqueeze(2)
                            * (next_q_values - log_alpha.exp() * next_logp.unsqueeze(2)),
                            dim=1,
                        ).unsqueeze(1)
                        target_q = reward + gamma_pow * (1.0 - done) * value_prime

                    q_values1 = q1_wrapper(*critic_observation)
                    q1 = torch.gather(q_values1, 1, action)
                    q1_error = torch.square(q1 - target_q.detach())
                    q1_loss = (q1_error * is_weights).mean() if runtime_config.enable_per else q1_error.mean()
                    global_q_net1_optimizer.zero_grad()
                    q1_loss.backward()
                    q1_grad_norm = torch.nn.utils.clip_grad_norm_(
                        global_q_net1.parameters(),
                        max_norm=20000,
                        norm_type=2,
                    )
                    global_q_net1_optimizer.step()

                    q_values2 = q2_wrapper(*critic_observation)
                    q2 = torch.gather(q_values2, 1, action)
                    q2_error = torch.square(q2 - target_q.detach())
                    q2_loss = (q2_error * is_weights).mean() if runtime_config.enable_per else q2_error.mean()
                    global_q_net2_optimizer.zero_grad()
                    q2_loss.backward()
                    q2_grad_norm = torch.nn.utils.clip_grad_norm_(
                        global_q_net2.parameters(),
                        max_norm=20000,
                        norm_type=2,
                    )
                    global_q_net2_optimizer.step()

                    td_error_mean = float("nan")
                    if runtime_config.enable_per:
                        with torch.no_grad():
                            updated_q1 = torch.gather(q1_wrapper(*critic_observation), 1, action)
                            updated_q2 = torch.gather(q2_wrapper(*critic_observation), 1, action)
                            td_error = 0.5 * (
                                (updated_q1 - target_q).abs() + (updated_q2 - target_q).abs()
                            )
                        td_error_cpu = td_error.squeeze(-1).squeeze(-1).detach().cpu().numpy() + runtime_config.per_eps
                        replay_buffer.update_priorities(replay_sample.indices, td_error_cpu)
                        td_error_mean = float(td_error.mean().item())

                    target_q_update_counter = _maybe_sync_target_networks(
                        runtime_config,
                        target_q_update_counter,
                        global_target_q_net1,
                        global_target_q_net2,
                        global_q_net1,
                        global_q_net2,
                    )

                    actor_step = _actor_update_due(learner_update_step, runtime_config.policy_delay)
                    policy_loss_value = float("nan")
                    alpha_loss_value = float("nan")
                    entropy_value = float("nan")
                    policy_grad_norm_value = float("nan")
                    n_valid_mean = float("nan")
                    entropy_target_mean = float("nan")
                    if actor_step:
                        q_values1 = q1_wrapper(*critic_observation)
                        q_values2 = q2_wrapper(*critic_observation)
                        q_values = torch.min(q_values1, q_values2)
                        logp = policy_wrapper(*observation)
                        policy_loss = torch.sum(
                            logp.exp().unsqueeze(2)
                            * (log_alpha.exp().detach() * logp.unsqueeze(2) - q_values.detach()),
                            dim=1,
                        ).mean()

                        global_policy_optimizer.zero_grad()
                        policy_loss.backward()
                        policy_grad_norm = torch.nn.utils.clip_grad_norm_(
                            global_policy_net.parameters(),
                            max_norm=100,
                            norm_type=2,
                        )
                        global_policy_optimizer.step()
                        policy_update_step += 1

                        entropy = (logp * logp.exp()).sum(dim=-1)
                        entropy_target_tensor, n_valid_mean, entropy_target_mean = _resolve_entropy_target_tensor(
                            batch["edge_padding_mask"],
                            runtime_config,
                            entropy_target,
                        )

                        alpha_loss = -(log_alpha * (entropy.detach() + entropy_target_tensor)).mean()
                        log_alpha_optimizer.zero_grad()
                        alpha_loss.backward()
                        log_alpha_optimizer.step()

                        policy_loss_value = float(policy_loss.item())
                        alpha_loss_value = float(alpha_loss.item())
                        entropy_value = float(entropy.mean().item())
                        policy_grad_norm_value = float(policy_grad_norm.item())
                    elif runtime_config.enable_adaptive_entropy_target:
                        _, n_valid_mean, entropy_target_mean = _resolve_entropy_target_tensor(
                            batch["edge_padding_mask"],
                            runtime_config,
                            entropy_target,
                        )

                    learner_update_step += 1
                    if runtime_config.replay_ratio > 0:
                        pending_update_budget = max(0.0, pending_update_budget - 1.0)

                    perf_data = {
                        name: _nanmean_or_nan(values)
                        for name, values in perf_metrics.items()
                        if values
                    }
                    training_record = {
                        "reward": float(reward.mean().item()),
                        "value": float(value_prime.mean().item()),
                        "policy_loss": policy_loss_value,
                        "q_value_loss": float(0.5 * (q1_loss.item() + q2_loss.item())),
                        "entropy": entropy_value,
                        "policy_grad_norm": policy_grad_norm_value,
                        "q_value_grad_norm": float(max(float(q1_grad_norm.item()), float(q2_grad_norm.item()))),
                        "log_alpha": float(log_alpha.item()),
                        "alpha_loss": alpha_loss_value,
                        "buffer_size": float(buffer_size),
                        "replay_ratio": float(runtime_config.replay_ratio),
                        "per_beta": per_beta,
                        "is_weight_mean": float(is_weights.mean().item()),
                        "n_step_mean": float(n_step_actual.mean().item()),
                        "n_valid_mean": n_valid_mean,
                        "entropy_target_mean": entropy_target_mean,
                        "td_error_mean": td_error_mean,
                        **perf_data,
                    }
                    training_data.append(training_record)

            if len(training_data) >= runtime_config.summary_window:
                writer, current_writer_bucket = _maybe_rotate_writer(
                    writer,
                    current_writer_bucket,
                    current_train_path,
                    curr_episode,
                    runtime_config.result_bucket_episodes,
                )
                train_metrics = _write_to_tensor_board(writer, training_data, curr_episode)
                if training_monitor is not None:
                    training_monitor.update_train(curr_episode, train_metrics)
                    training_monitor.update_system(
                        curr_episode,
                        {
                            "buffer_size": replay_buffer.size,
                            "completed_episodes": curr_episode,
                            "learner_updates": learner_update_step,
                        },
                    )
                _print_train_progress(
                    curr_episode,
                    runtime_config.max_episodes,
                    train_metrics,
                    replay_buffer.size,
                    time.time() - train_start_time,
                )
                training_data = []
                perf_metrics = {}

            weights_set = [_state_dict_to_device(global_policy_net.state_dict(), local_device)]

            # ── cache checkpoint (last completed iteration) ─────────────────
            _last_good_checkpoint = _build_checkpoint_dict(
                global_policy_net, global_q_net1, global_q_net2, log_alpha,
                global_policy_optimizer, global_q_net1_optimizer,
                global_q_net2_optimizer, log_alpha_optimizer, curr_episode,
                runtime_config,
                learner_update_step=learner_update_step,
                policy_update_step=policy_update_step,
                target_q_update_counter=target_q_update_counter,
            )

            if not runtime_config.enable_auto_eval:
                continue

            for episode_number in sorted(completed_episodes):
                if episode_number % save_gap != 0 or episode_number in auto_eval_episodes:
                    continue
                eval_results = evaluate_policy(
                    global_policy_net.state_dict(),
                    output_config=runtime_config,
                    episodes=runtime_config.auto_eval_episodes,
                    start_episode=episode_number,
                    greedy=runtime_config.auto_eval_greedy,
                    device=runtime_config.auto_eval_device,
                    result_bucket_episodes=runtime_config.result_bucket_episodes,
                    max_episode_step=runtime_config.max_episode_step,
                )
                eval_summary = summarize_eval_results(eval_results)
                writer, current_writer_bucket = _maybe_rotate_writer(
                    writer,
                    current_writer_bucket,
                    current_train_path,
                    episode_number,
                    runtime_config.result_bucket_episodes,
                )
                _write_eval_to_tensor_board(writer, eval_summary, episode_number)
                if training_monitor is not None:
                    training_monitor.update_eval(episode_number, eval_summary)
                auto_eval_episodes.add(episode_number)
                print(
                    f"auto_eval episode={episode_number} "
                    f"episodes={eval_summary['episodes']} "
                    f"explored_rate={eval_summary['explored_rate']:.4f} "
                    f"travel_dist={eval_summary['travel_dist']:.2f} "
                    f"success_rate={eval_summary['success_rate']:.4f} "
                    f"episode_return={eval_summary['episode_return']:.4f} "
                    f"steps_taken={eval_summary['steps_taken']:.2f}"
                )

        # ── normal completion — save final checkpoint ───────────────────────
        _save_checkpoint(
            global_policy_net, global_q_net1, global_q_net2, log_alpha,
            global_policy_optimizer, global_q_net1_optimizer,
            global_q_net2_optimizer, log_alpha_optimizer,
            curr_episode, learner_update_step, policy_update_step, target_q_update_counter,
            runtime_config.result_bucket_episodes,
            current_checkpoint_path, current_model_path, runtime_config,
        )
        if current_checkpoint_path is not None:
            final_path = get_checkpoint_final_path(runtime_config)
            torch.save(
                torch.load(current_checkpoint_path, map_location="cpu", weights_only=False),
                final_path,
            )
            print(
                f"final_model_saved checkpoint={current_checkpoint_path} "
                f"final={final_path}"
            )

    except KeyboardInterrupt:
        # ── interrupted — save last *completed* iteration's snapshot ─────
        if _last_good_checkpoint is not None and current_checkpoint_path is not None:
            interrupt_path = get_checkpoint_interrupted_path(runtime_config)
            torch.save(_last_good_checkpoint, interrupt_path)
            if current_model_path is not None:
                bucket_dir = ensure_bucket_dir(
                    current_model_path,
                    _last_good_checkpoint["episode"],
                    runtime_config.result_bucket_episodes,
                )
                torch.save(_last_good_checkpoint, bucket_dir / "checkpoint.pth")
            print(
                f"interrupt_checkpoint_saved path={interrupt_path} "
                f"episode={_last_good_checkpoint['episode']}"
            )
        else:
            print("interrupt — no completed training iteration to save")

    finally:
        writer.close()
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)
        for actor in meta_agents:
            ray.kill(actor)
        ray.shutdown()

    return {
        "checkpoint_path": str(current_checkpoint_path) if current_checkpoint_path is not None else None,
        "model_dir": str(current_model_path) if current_model_path is not None else None,
        "result_eval_dir": str(current_eval_path),
        "result_gif_dir": str(current_gifs_path),
        "train_dir": str(current_train_path),
        "gif_dir": str(current_gifs_path),
        "episode": curr_episode,
    }


if __name__ == "__main__":
    main()
