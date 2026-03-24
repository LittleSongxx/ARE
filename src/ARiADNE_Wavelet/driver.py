from __future__ import annotations

import copy
import os
import random
import signal
import shutil
import time

import ray
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from .model import PolicyNet, QNet, compute_wavelet_utility_aux_loss
from .parameter import (
    EMBEDDING_DIM,
    GAMMA,
    K_SIZE,
    LR,
    NODE_INPUT_DIM,
    SRC_ROOT,
    RuntimeConfig,
    build_run_session,
    ensure_output_dirs,
    get_checkpoint_final_path,
    get_checkpoint_interrupted_path,
    get_checkpoint_path,
    get_gifs_path,
    get_latest_checkpoint_path,
    get_model_path,
    get_monitor_path,
    get_result_eval_path,
    get_train_path,
    is_smoke_run,
    resolve_resume_checkpoint,
)
from .runtime_utils import configure_worker_process_threads, resolve_ray_num_cpus, resolve_ray_worker_num_cpus, resolve_worker_num_threads
from .runner import Runner
from .training_monitor import TrainingMonitor
from .utils import ensure_bucket_dir, get_bucket_name


_interrupted = False
_interrupt_raised = False
_saving_interrupt_checkpoint = False


def _signal_handler(signum, frame):
    del frame
    global _interrupted, _interrupt_raised, _saving_interrupt_checkpoint
    _interrupted = True
    if _saving_interrupt_checkpoint:
        return
    if _interrupt_raised:
        print(f"\nInterrupt received again (signal {signum}). Waiting for checkpoint save...")
        return
    _interrupt_raised = True
    print(f"\nInterrupt received (signal {signum}). Saving last completed checkpoint and exiting...")
    raise KeyboardInterrupt


def _build_checkpoint_dict(
    policy_net,
    q1,
    q2,
    log_alpha,
    policy_opt,
    q1_opt,
    q2_opt,
    alpha_opt,
    episode,
):
    def _cpu_sd(module):
        return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}

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
    }


def _state_dict_to_device(state_dict, device):
    return {key: value.detach().to(device) for key, value in state_dict.items()}


def _visible_gpu_count_from_env() -> int | None:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    entries = [entry.strip() for entry in raw.split(",") if entry.strip()]
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
    ok, reason = _cuda_runtime_status()
    if ok:
        return torch.device("cuda")
    print(f"[Run] learner GPU requested but CUDA is unavailable; fallback to CPU ({reason})")
    return torch.device("cpu")


def _resolve_learner_gpu_count(runtime_config: RuntimeConfig, device: torch.device) -> int:
    if device.type != "cuda":
        return 0

    env_visible = _visible_gpu_count_from_env()
    try:
        detected = int(torch.cuda.device_count())
    except (AssertionError, RuntimeError):
        detected = 0

    visible = detected
    if visible <= 0 and env_visible is not None:
        visible = env_visible
    if visible <= 0:
        visible = 1

    requested = int(runtime_config.num_gpu)
    if requested <= 0:
        return visible
    return max(1, min(requested, visible))


def _wrap_data_parallel(module: nn.Module, device_ids: list[int]) -> nn.Module:
    if len(device_ids) <= 1:
        return module
    return nn.DataParallel(module, device_ids=device_ids, output_device=device_ids[0])


def _resolve_worker_runtime(runtime_config: RuntimeConfig) -> tuple[RuntimeConfig, float]:
    if runtime_config.use_gpu:
        print("[Run] worker GPU request is ignored; collectors are forced to CPU-only mode")
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


def _print_train_progress(curr_episode, max_episodes, train_metrics, buffer_size, elapsed_sec):
    total_episodes = max(max_episodes, 1)
    progress = 100.0 * curr_episode / total_episodes
    print(
        "train_progress "
        f"episode={curr_episode}/{max_episodes} "
        f"progress={progress:.1f}% "
        f"buffer={buffer_size} "
        f"elapsed_sec={elapsed_sec:.1f} "
        f"reward={_format_metric(train_metrics.get('reward'))} "
        f"explored_rate={_format_metric(train_metrics.get('explored_rate'))} "
        f"success_rate={_format_metric(train_metrics.get('success_rate'))} "
        f"episode_return={_format_metric(train_metrics.get('episode_return'))} "
        f"episode_steps={_format_metric(train_metrics.get('episode_steps'))} "
        f"policy_loss={_format_metric(train_metrics.get('policy_loss'))} "
        f"q_value_loss={_format_metric(train_metrics.get('q_value_loss'))} "
        f"utility_aux_loss={_format_metric(train_metrics.get('utility_aux_loss'))}"
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
    bucket_size,
    current_checkpoint_path,
    current_model_path,
):
    if current_checkpoint_path is None:
        return

    checkpoint = {
        "policy_model": global_policy_net.state_dict(),
        "q_net1_model": global_q_net1.state_dict(),
        "q_net2_model": global_q_net2.state_dict(),
        "log_alpha": log_alpha.detach().cpu(),
        "policy_optimizer": global_policy_optimizer.state_dict(),
        "q_net1_optimizer": global_q_net1_optimizer.state_dict(),
        "q_net2_optimizer": global_q_net2_optimizer.state_dict(),
        "log_alpha_optimizer": log_alpha_optimizer.state_dict(),
        "episode": curr_episode,
    }
    torch.save(checkpoint, current_checkpoint_path)
    if current_model_path is not None:
        bucket_dir = ensure_bucket_dir(current_model_path, curr_episode, bucket_size)
        torch.save(checkpoint, bucket_dir / "checkpoint.pth")


def _resolve_runtime_session(runtime_config: RuntimeConfig) -> RuntimeConfig:
    if runtime_config.run_session is not None:
        return runtime_config

    if runtime_config.resume_from:
        try:
            _, resume_session = resolve_resume_checkpoint(runtime_config.resume_from)
        except ValueError:
            return runtime_config.with_overrides(run_session=build_run_session(runtime_config.run_name))
        return runtime_config.with_overrides(run_session=resume_session)

    if runtime_config.load_model:
        latest = get_latest_checkpoint_path(runtime_config.run_name)
        if latest.exists():
            try:
                _, resume_session = resolve_resume_checkpoint(latest)
            except ValueError:
                pass
            else:
                print(f"[Run] load_model detected; continuing in run_session={resume_session}")
                return runtime_config.with_overrides(run_session=resume_session)

    return runtime_config.with_overrides(run_session=build_run_session(runtime_config.run_name))


def _save_interrupt_checkpoint(
    runtime_config: RuntimeConfig,
    current_checkpoint_path,
    current_model_path,
    policy_net,
    q1,
    q2,
    log_alpha,
    policy_opt,
    q1_opt,
    q2_opt,
    alpha_opt,
    curr_episode: int,
) -> bool:
    if current_checkpoint_path is None:
        return False

    interrupt_path = get_checkpoint_interrupted_path(runtime_config)
    checkpoint_payload = _build_checkpoint_dict(
        policy_net,
        q1,
        q2,
        log_alpha,
        policy_opt,
        q1_opt,
        q2_opt,
        alpha_opt,
        curr_episode,
    )
    torch.save(checkpoint_payload, interrupt_path)
    torch.save(checkpoint_payload, current_checkpoint_path)

    if current_model_path is not None:
        bucket_dir = ensure_bucket_dir(
            current_model_path,
            curr_episode,
            runtime_config.result_bucket_episodes,
        )
        torch.save(checkpoint_payload, bucket_dir / "checkpoint.pth")

    print(f"interrupt_checkpoint_saved path={interrupt_path} episode={curr_episode}")
    return True


def _write_to_tensor_board(writer, tensorboard_data, curr_episode):
    tensorboard_data = np.array(tensorboard_data)
    tensorboard_data = list(np.nanmean(tensorboard_data, axis=0))
    (
        reward,
        value,
        policy_loss,
        q_value_loss,
        entropy,
        policy_grad_norm,
        q_value_grad_norm,
        log_alpha,
        alpha_loss,
        travel_dist,
        success_rate,
        explored_rate,
        episode_return,
        episode_steps,
        utility_aux_loss,
        utility_aux_base,
        utility_aux_wavelet,
    ) = tensorboard_data

    writer.add_scalar(tag="Losses/Value", scalar_value=value, global_step=curr_episode)
    writer.add_scalar(tag="Losses/Policy Loss", scalar_value=policy_loss, global_step=curr_episode)
    writer.add_scalar(tag="Losses/Alpha Loss", scalar_value=alpha_loss, global_step=curr_episode)
    writer.add_scalar(tag="Losses/Q Value Loss", scalar_value=q_value_loss, global_step=curr_episode)
    writer.add_scalar(tag="Losses/Entropy", scalar_value=entropy, global_step=curr_episode)
    writer.add_scalar(tag="Losses/Policy Grad Norm", scalar_value=policy_grad_norm, global_step=curr_episode)
    writer.add_scalar(tag="Losses/Q Value Grad Norm", scalar_value=q_value_grad_norm, global_step=curr_episode)
    writer.add_scalar(tag="Losses/Log Alpha", scalar_value=log_alpha, global_step=curr_episode)
    writer.add_scalar(tag="Losses/Utility Aux", scalar_value=utility_aux_loss, global_step=curr_episode)
    writer.add_scalar(tag="Losses/Utility Aux Base", scalar_value=utility_aux_base, global_step=curr_episode)
    writer.add_scalar(tag="Losses/Utility Aux Wavelet", scalar_value=utility_aux_wavelet, global_step=curr_episode)

    writer.add_scalar(tag="Perf/Reward", scalar_value=reward, global_step=curr_episode)
    writer.add_scalar(tag="Perf/Travel Distance", scalar_value=travel_dist, global_step=curr_episode)
    writer.add_scalar(tag="Perf/Explored Rate", scalar_value=explored_rate, global_step=curr_episode)
    writer.add_scalar(tag="Perf/Success Rate", scalar_value=success_rate, global_step=curr_episode)
    writer.add_scalar(tag="Perf/Episode Return", scalar_value=episode_return, global_step=curr_episode)
    writer.add_scalar(tag="Perf/Episode Steps", scalar_value=episode_steps, global_step=curr_episode)

    return {
        "reward": reward,
        "value": value,
        "policy_loss": policy_loss,
        "q_value_loss": q_value_loss,
        "entropy": entropy,
        "policy_grad_norm": policy_grad_norm,
        "q_value_grad_norm": q_value_grad_norm,
        "log_alpha": log_alpha,
        "alpha_loss": alpha_loss,
        "travel_dist": travel_dist,
        "success_rate": success_rate,
        "explored_rate": explored_rate,
        "episode_return": episode_return,
        "episode_steps": episode_steps,
        "utility_aux_loss": utility_aux_loss,
        "utility_aux_base": utility_aux_base,
        "utility_aux_wavelet": utility_aux_wavelet,
    }


def main(runtime_config: RuntimeConfig | None = None) -> dict:
    global _interrupted, _interrupt_raised, _saving_interrupt_checkpoint
    _interrupted = False
    _interrupt_raised = False
    _saving_interrupt_checkpoint = False

    runtime_config = runtime_config or RuntimeConfig()
    runtime_config = _resolve_runtime_session(runtime_config)
    ensure_output_dirs(runtime_config)

    current_checkpoint_path = None if is_smoke_run(runtime_config) else get_checkpoint_path(runtime_config)
    current_model_path = None if is_smoke_run(runtime_config) else get_model_path(runtime_config)
    current_eval_path = get_result_eval_path(runtime_config)
    current_train_path = get_train_path(runtime_config)
    current_gifs_path = get_gifs_path(runtime_config)
    train_start_time = time.time()

    device = _resolve_learner_device(runtime_config)

    writer = SummaryWriter(current_train_path)

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

    configure_worker_process_threads(worker_num_threads)

    learner_gpu_count = _resolve_learner_gpu_count(runtime_config, device)
    learner_device_ids = list(range(learner_gpu_count))
    worker_runtime_config, worker_gpu_share = _resolve_worker_runtime(runtime_config)
    local_device = torch.device("cpu")

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

    print(f"[Run] result_root={runtime_config.run_session}")
    print(f"[Run] model_path={current_model_path}")
    print(f"[Run] train_path={current_train_path}")
    print(f"[Run] gifs_path={current_gifs_path}")
    print(f"[Run] ray_num_cpus={requested_ray_num_cpus if requested_ray_num_cpus is not None else 'auto'}")
    print(f"[Run] ray_worker_num_cpus={worker_num_cpus}")
    print(f"[Run] worker_num_threads={worker_num_threads}")
    print(
        f"[Run] role_split: collector_device=cpu, learner_device={device.type}, "
        f"learner_num_gpus={learner_gpu_count}, visible_cuda_devices={os.environ.get('CUDA_VISIBLE_DEVICES') or '<all-visible>'}"
    )
    print(
        f"[Run] ray_cluster_resources={cluster_resources} "
        f"worker_gpu_request={worker_gpu_share} cluster_gpus={cluster_gpus} cluster_cpus={cluster_cpus}"
    )

    old_sigint = signal.signal(signal.SIGINT, _signal_handler)
    old_sigterm = signal.signal(signal.SIGTERM, _signal_handler)

    global_policy_net = PolicyNet(
        NODE_INPUT_DIM,
        EMBEDDING_DIM,
        enable_wavelet_history=runtime_config.enable_wavelet_history,
        history_input_dim=runtime_config.history_input_dim,
        history_embed_dim=runtime_config.history_embed_dim,
        history_wavelet_levels=runtime_config.history_wavelet_levels,
        history_encoder_mode=runtime_config.history_encoder_mode,
    ).to(device)
    global_q_net1 = QNet(
        NODE_INPUT_DIM,
        EMBEDDING_DIM,
        enable_wavelet_history=runtime_config.enable_wavelet_history,
        history_input_dim=runtime_config.history_input_dim,
        history_embed_dim=runtime_config.history_embed_dim,
        history_wavelet_levels=runtime_config.history_wavelet_levels,
        history_encoder_mode=runtime_config.history_encoder_mode,
        enable_wavelet_utility_loss=runtime_config.enable_wavelet_utility_loss,
    ).to(device)
    global_q_net2 = QNet(
        NODE_INPUT_DIM,
        EMBEDDING_DIM,
        enable_wavelet_history=runtime_config.enable_wavelet_history,
        history_input_dim=runtime_config.history_input_dim,
        history_embed_dim=runtime_config.history_embed_dim,
        history_wavelet_levels=runtime_config.history_wavelet_levels,
        history_encoder_mode=runtime_config.history_encoder_mode,
        enable_wavelet_utility_loss=runtime_config.enable_wavelet_utility_loss,
    ).to(device)
    global_target_q_net1 = QNet(
        NODE_INPUT_DIM,
        EMBEDDING_DIM,
        enable_wavelet_history=runtime_config.enable_wavelet_history,
        history_input_dim=runtime_config.history_input_dim,
        history_embed_dim=runtime_config.history_embed_dim,
        history_wavelet_levels=runtime_config.history_wavelet_levels,
        history_encoder_mode=runtime_config.history_encoder_mode,
        enable_wavelet_utility_loss=False,
    ).to(device)
    global_target_q_net2 = QNet(
        NODE_INPUT_DIM,
        EMBEDDING_DIM,
        enable_wavelet_history=runtime_config.enable_wavelet_history,
        history_input_dim=runtime_config.history_input_dim,
        history_embed_dim=runtime_config.history_embed_dim,
        history_wavelet_levels=runtime_config.history_wavelet_levels,
        history_encoder_mode=runtime_config.history_encoder_mode,
        enable_wavelet_utility_loss=False,
    ).to(device)

    log_alpha = torch.FloatTensor([-2]).to(device)
    log_alpha.requires_grad = True

    global_policy_optimizer = optim.Adam(global_policy_net.parameters(), lr=LR)
    global_q_net1_optimizer = optim.Adam(global_q_net1.parameters(), lr=LR)
    global_q_net2_optimizer = optim.Adam(global_q_net2.parameters(), lr=LR)
    log_alpha_optimizer = optim.Adam([log_alpha], lr=1e-4)
    entropy_target = 0.05 * (-np.log(1 / K_SIZE))

    curr_episode = 0
    next_episode = 0
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
        print(f"loading_model checkpoint={checkpoint_source} resume_episode={curr_episode}")

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

    metric_name = ["travel_dist", "success_rate", "explored_rate", "episode_return", "episode_steps"]
    training_data = []
    perf_metrics = {name: [] for name in metric_name}
    experience_buffer = [[] for _ in range(runtime_config.replay_channels)]

    print(
        "training_start "
        f"max_episodes={runtime_config.max_episodes} "
        f"summary_window={runtime_config.summary_window} "
        f"minimum_buffer_size={runtime_config.minimum_buffer_size} "
        f"batch_size={runtime_config.batch_size} "
        f"train_updates_per_iter={runtime_config.train_updates_per_iter} "
        f"enable_wavelet_history={runtime_config.enable_wavelet_history} "
        f"enable_wavelet_utility_loss={runtime_config.enable_wavelet_utility_loss} "
        f"history_encoder_mode={runtime_config.history_encoder_mode} "
        f"utility_target_type={runtime_config.utility_target_type} "
        f"utility_loss_mode={runtime_config.utility_loss_mode}"
    )

    try:
        while job_list:
            done_id, job_list = ray.wait(job_list)
            done_jobs = ray.get(done_id)

            completed_episodes = []
            for job in done_jobs:
                job_results, metrics, info = job
                completed_episodes.append(info["episode_number"])
                for i in range(len(experience_buffer)):
                    experience_buffer[i] += job_results[i]
                for name in metric_name:
                    perf_metrics[name].append(metrics[name])

            curr_episode = max(job[2]["episode_number"] for job in done_jobs)
            latest_metrics = done_jobs[-1][1]
            print(
                "episode_complete "
                f"episode={curr_episode}/{runtime_config.max_episodes} "
                f"meta_agent={done_jobs[-1][2]['id']} "
                f"buffer={len(experience_buffer[0])} "
                f"travel_dist={_format_metric(latest_metrics['travel_dist'])} "
                f"explored_rate={_format_metric(latest_metrics['explored_rate'])} "
                f"success_rate={_format_metric(latest_metrics['success_rate'])} "
                f"episode_return={_format_metric(latest_metrics['episode_return'])} "
                f"episode_steps={_format_metric(latest_metrics['episode_steps'])}"
            )

            if next_episode < runtime_config.max_episodes:
                next_episode += 1
                job_list.append(meta_agents[info["id"]].job.remote(weights_set, next_episode))

            if len(experience_buffer[0]) >= runtime_config.minimum_buffer_size:
                if len(experience_buffer[0]) >= runtime_config.replay_size:
                    for i in range(len(experience_buffer)):
                        experience_buffer[i] = experience_buffer[i][-runtime_config.replay_size :]

                indices = list(range(len(experience_buffer[0])))
                sample_batch_size = min(runtime_config.batch_size, len(indices))

                for _ in range(runtime_config.train_updates_per_iter):
                    if sample_batch_size == 0:
                        break
                    sample_indices = random.sample(indices, sample_batch_size)
                    rollouts = []
                    for i in range(len(experience_buffer)):
                        rollouts.append([experience_buffer[i][index] for index in sample_indices])

                    node_inputs = torch.stack(rollouts[0]).to(device)
                    node_padding_mask = torch.stack(rollouts[1]).to(device)
                    edge_mask = torch.stack(rollouts[2]).to(device)
                    current_index = torch.stack(rollouts[3]).to(device)
                    current_edge = torch.stack(rollouts[4]).to(device)
                    edge_padding_mask = torch.stack(rollouts[5]).to(device)
                    action = torch.stack(rollouts[6]).to(device)
                    reward = torch.stack(rollouts[7]).to(device)
                    done = torch.stack(rollouts[8]).float().to(device)

                    next_node_inputs = torch.stack(rollouts[9]).to(device)
                    next_node_padding_mask = torch.stack(rollouts[10]).to(device)
                    next_edge_mask = torch.stack(rollouts[11]).to(device)
                    next_current_index = torch.stack(rollouts[12]).to(device)
                    next_current_edge = torch.stack(rollouts[13]).to(device)
                    next_edge_padding_mask = torch.stack(rollouts[14]).to(device)

                    history_inputs = torch.stack(rollouts[15]).to(device)
                    next_history_inputs = torch.stack(rollouts[16]).to(device)
                    utility_target_hint = torch.stack(rollouts[17]).to(device).squeeze(1)

                    observation = [
                        node_inputs,
                        node_padding_mask,
                        edge_mask,
                        current_index,
                        current_edge,
                        edge_padding_mask,
                        history_inputs,
                    ]
                    next_observation = [
                        next_node_inputs,
                        next_node_padding_mask,
                        next_edge_mask,
                        next_current_index,
                        next_current_edge,
                        next_edge_padding_mask,
                        next_history_inputs,
                    ]

                    with torch.no_grad():
                        q_values1 = q1_wrapper(*observation)
                        q_values2 = q2_wrapper(*observation)
                        q_values = torch.min(q_values1, q_values2)

                    logp = policy_wrapper(*observation)
                    policy_loss = torch.sum(
                        logp.exp().unsqueeze(2) * (log_alpha.exp().detach() * logp.unsqueeze(2) - q_values.detach()),
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

                    with torch.no_grad():
                        next_logp = policy_wrapper(*next_observation)
                        next_q_values1 = target_q1_wrapper(*next_observation)
                        next_q_values2 = target_q2_wrapper(*next_observation)
                        next_q_values = torch.min(next_q_values1, next_q_values2)
                        value_prime = torch.sum(
                            next_logp.unsqueeze(2).exp() * (next_q_values - log_alpha.exp() * next_logp.unsqueeze(2)),
                            dim=1,
                        ).unsqueeze(1)
                        target_q = reward + GAMMA * (1 - done) * value_prime
                        utility_td_target = torch.min(target_q1_wrapper(*observation), target_q2_wrapper(*observation))

                    mse_loss = nn.MSELoss()
                    candidate_valid_mask = ~edge_padding_mask.squeeze(1).bool()
                    candidate_indices = current_edge.squeeze(-1).long()
                    candidate_coords = torch.gather(
                        node_inputs[..., :2],
                        1,
                        candidate_indices.unsqueeze(-1).expand(-1, -1, 2),
                    )
                    utility_supervision_mask = candidate_valid_mask

                    utility_aux_loss_1 = torch.tensor(0.0, device=device)
                    utility_aux_base_1 = torch.tensor(0.0, device=device)
                    utility_aux_wavelet_1 = torch.tensor(0.0, device=device)

                    if runtime_config.enable_wavelet_utility_loss:
                        q_values1, utility_pred1 = q1_wrapper(*observation, return_aux=True)
                    else:
                        q_values1 = q1_wrapper(*observation)

                    q1 = torch.gather(q_values1, 1, action)
                    q1_loss = mse_loss(q1, target_q.detach()).mean()

                    if runtime_config.enable_wavelet_utility_loss:
                        utility_target = utility_td_target.detach().clone()
                        if runtime_config.utility_target_type == "n_step_return":
                            valid_hint = torch.isfinite(utility_target_hint.squeeze(-1))
                            utility_target = torch.where(valid_hint.unsqueeze(-1), utility_target_hint, utility_target)
                        utility_aux_loss_1, utility_aux_base_1, utility_aux_wavelet_1 = compute_wavelet_utility_aux_loss(
                            utility_pred1,
                            utility_target,
                            candidate_valid_mask,
                            candidate_coords=candidate_coords,
                            supervision_mask=utility_supervision_mask,
                            loss_mode=runtime_config.utility_loss_mode,
                            loss_weight=runtime_config.utility_loss_weight,
                            loss_type=runtime_config.utility_aux_loss_type,
                            base_weight=runtime_config.utility_aux_base_weight,
                            wavelet_weight=runtime_config.utility_aux_wavelet_weight,
                            patch_size=runtime_config.utility_patch_size,
                            patch_sigma=runtime_config.utility_patch_sigma,
                            wavelet_levels=runtime_config.utility_wavelet_levels,
                            wavelet_rho=runtime_config.utility_wavelet_rho,
                        )
                        q1_total_loss = q1_loss + utility_aux_loss_1
                    else:
                        q1_total_loss = q1_loss

                    global_q_net1_optimizer.zero_grad()
                    q1_total_loss.backward()
                    q1_grad_norm = torch.nn.utils.clip_grad_norm_(
                        global_q_net1.parameters(),
                        max_norm=20000,
                        norm_type=2,
                    )
                    global_q_net1_optimizer.step()

                    utility_aux_loss_2 = torch.tensor(0.0, device=device)
                    utility_aux_base_2 = torch.tensor(0.0, device=device)
                    utility_aux_wavelet_2 = torch.tensor(0.0, device=device)

                    if runtime_config.enable_wavelet_utility_loss:
                        q_values2, utility_pred2 = q2_wrapper(*observation, return_aux=True)
                    else:
                        q_values2 = q2_wrapper(*observation)

                    q2 = torch.gather(q_values2, 1, action)
                    q2_loss = mse_loss(q2, target_q.detach()).mean()

                    if runtime_config.enable_wavelet_utility_loss:
                        utility_target = utility_td_target.detach().clone()
                        if runtime_config.utility_target_type == "n_step_return":
                            valid_hint = torch.isfinite(utility_target_hint.squeeze(-1))
                            utility_target = torch.where(valid_hint.unsqueeze(-1), utility_target_hint, utility_target)
                        utility_aux_loss_2, utility_aux_base_2, utility_aux_wavelet_2 = compute_wavelet_utility_aux_loss(
                            utility_pred2,
                            utility_target,
                            candidate_valid_mask,
                            candidate_coords=candidate_coords,
                            supervision_mask=utility_supervision_mask,
                            loss_mode=runtime_config.utility_loss_mode,
                            loss_weight=runtime_config.utility_loss_weight,
                            loss_type=runtime_config.utility_aux_loss_type,
                            base_weight=runtime_config.utility_aux_base_weight,
                            wavelet_weight=runtime_config.utility_aux_wavelet_weight,
                            patch_size=runtime_config.utility_patch_size,
                            patch_sigma=runtime_config.utility_patch_sigma,
                            wavelet_levels=runtime_config.utility_wavelet_levels,
                            wavelet_rho=runtime_config.utility_wavelet_rho,
                        )
                        q2_total_loss = q2_loss + utility_aux_loss_2
                    else:
                        q2_total_loss = q2_loss

                    global_q_net2_optimizer.zero_grad()
                    q2_total_loss.backward()
                    q2_grad_norm = torch.nn.utils.clip_grad_norm_(
                        global_q_net2.parameters(),
                        max_norm=20000,
                        norm_type=2,
                    )
                    global_q_net2_optimizer.step()

                    entropy = (logp * logp.exp()).sum(dim=-1)
                    alpha_loss = -(log_alpha * (entropy.detach() + entropy_target)).mean()

                    log_alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    log_alpha_optimizer.step()

                    target_q_update_counter += 1
                    perf_data = [np.nanmean(perf_metrics[name]) for name in metric_name]

                    utility_aux_loss = 0.5 * (utility_aux_loss_1.item() + utility_aux_loss_2.item())
                    utility_aux_base = 0.5 * (utility_aux_base_1.item() + utility_aux_base_2.item())
                    utility_aux_wavelet = 0.5 * (
                        utility_aux_wavelet_1.item() + utility_aux_wavelet_2.item()
                    )

                    training_data.append(
                        [
                            reward.mean().item(),
                            value_prime.mean().item(),
                            policy_loss.item(),
                            0.5 * (q1_loss.item() + q2_loss.item()),
                            entropy.mean().item(),
                            policy_grad_norm.item(),
                            0.5 * (q1_grad_norm.item() + q2_grad_norm.item()),
                            log_alpha.item(),
                            alpha_loss.item(),
                            *perf_data,
                            utility_aux_loss,
                            utility_aux_base,
                            utility_aux_wavelet,
                        ]
                    )

            if len(training_data) >= runtime_config.summary_window:
                train_metrics = _write_to_tensor_board(writer, training_data, curr_episode)
                if training_monitor is not None:
                    training_monitor.update_train(curr_episode, train_metrics)
                    training_monitor.update_system(
                        curr_episode,
                        {
                            "buffer_size": len(experience_buffer[0]),
                            "completed_episodes": curr_episode,
                        },
                    )
                _print_train_progress(
                    curr_episode,
                    runtime_config.max_episodes,
                    train_metrics,
                    len(experience_buffer[0]),
                    time.time() - train_start_time,
                )
                training_data = []
                perf_metrics = {name: [] for name in metric_name}

            weights_set = [_state_dict_to_device(global_policy_net.state_dict(), local_device)]

            if target_q_update_counter > 64:
                target_q_update_counter = 1
                global_target_q_net1.load_state_dict(global_q_net1.state_dict())
                global_target_q_net2.load_state_dict(global_q_net2.state_dict())
                global_target_q_net1.eval()
                global_target_q_net2.eval()

            if current_checkpoint_path is not None and curr_episode % max(int(runtime_config.save_model_gap), 1) == 0:
                _save_checkpoint(
                    global_policy_net,
                    global_q_net1,
                    global_q_net2,
                    log_alpha,
                    global_policy_optimizer,
                    global_q_net1_optimizer,
                    global_q_net2_optimizer,
                    log_alpha_optimizer,
                    curr_episode,
                    runtime_config.result_bucket_episodes,
                    current_checkpoint_path,
                    current_model_path,
                )
                print(
                    f"checkpoint_saved episode={curr_episode} path={current_checkpoint_path} "
                    f"bucket={get_bucket_name(curr_episode, runtime_config.result_bucket_episodes)}"
                )

        _save_checkpoint(
            global_policy_net,
            global_q_net1,
            global_q_net2,
            log_alpha,
            global_policy_optimizer,
            global_q_net1_optimizer,
            global_q_net2_optimizer,
            log_alpha_optimizer,
            curr_episode,
            runtime_config.result_bucket_episodes,
            current_checkpoint_path,
            current_model_path,
        )
        if current_checkpoint_path is not None:
            final_path = get_checkpoint_final_path(runtime_config)
            torch.save(torch.load(current_checkpoint_path, map_location="cpu", weights_only=False), final_path)
            print(f"final_model_saved checkpoint={current_checkpoint_path} final={final_path} episode={curr_episode}")

    except KeyboardInterrupt:
        saved = False
        try:
            _saving_interrupt_checkpoint = True
            saved = _save_interrupt_checkpoint(
                runtime_config,
                current_checkpoint_path,
                current_model_path,
                global_policy_net,
                global_q_net1,
                global_q_net2,
                log_alpha,
                global_policy_optimizer,
                global_q_net1_optimizer,
                global_q_net2_optimizer,
                log_alpha_optimizer,
                curr_episode,
            )
        except Exception as exc:  # pragma: no cover - best effort fallback on forced interrupts
            if current_checkpoint_path is not None and current_checkpoint_path.exists():
                try:
                    interrupt_path = get_checkpoint_interrupted_path(runtime_config)
                    shutil.copy2(current_checkpoint_path, interrupt_path)
                    print(
                        f"interrupt_checkpoint_saved path={interrupt_path} "
                        f"episode={curr_episode} source=fallback_current_checkpoint error={exc}"
                    )
                    saved = True
                except Exception as copy_exc:
                    print(f"interrupt_checkpoint_save_failed error={exc} fallback_error={copy_exc}")
            else:
                print(f"interrupt_checkpoint_save_failed error={exc}")
        finally:
            _saving_interrupt_checkpoint = False

        if not saved:
            print("interrupt no checkpoint available to save")

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
