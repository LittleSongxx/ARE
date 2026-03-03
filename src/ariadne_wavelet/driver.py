from __future__ import annotations

import random

import numpy as np
import ray
import torch
import torch.nn as nn
import torch.optim as optim

from .evaluation import evaluate_policy, summarize_eval_results
from .model import PolicyNet, QNet
from .parameter import (
    EMBEDDING_DIM,
    GAMMA,
    K_SIZE,
    LR,
    NODE_INPUT_DIM,
    RuntimeConfig,
    build_run_session,
    ensure_output_dirs,
    get_checkpoint_path,
    get_gifs_path,
    get_model_path,
    get_monitor_path,
    get_result_eval_path,
    is_smoke_run,
)
from .training_monitor import TrainingMonitor
from .runner import Runner


def _state_dict_to_device(state_dict, device):
    return {key: value.detach().to(device) for key, value in state_dict.items()}


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


def _resolve_worker_runtime(runtime_config: RuntimeConfig) -> tuple[RuntimeConfig, float]:
    if not runtime_config.use_gpu or runtime_config.num_gpu <= 0:
        return runtime_config.with_overrides(use_gpu=False, num_gpu=0), 0.0

    cuda_ok, reason = _cuda_runtime_status()
    if not cuda_ok:
        print(f"CUDA is unavailable for workers; remote workers will run on CPU. Reason: {reason}")
        return runtime_config.with_overrides(use_gpu=False, num_gpu=0), 0.0

    available_gpus = float(ray.cluster_resources().get("GPU", 0.0))
    if available_gpus <= 0:
        print("Ray reports no GPU resources for workers; remote workers will run on CPU.")
        return runtime_config.with_overrides(use_gpu=False, num_gpu=0), 0.0

    requested_gpus = min(float(runtime_config.num_gpu), available_gpus)
    return runtime_config, requested_gpus / max(runtime_config.num_meta_agent, 1)


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
    current_checkpoint_path,
):
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


def _aggregate_train_metrics(tensorboard_data):
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
    ) = tensorboard_data

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
    }


def main(runtime_config: RuntimeConfig | None = None) -> dict:
    runtime_config = runtime_config or RuntimeConfig()
    if runtime_config.run_session is None:
        runtime_config = runtime_config.with_overrides(run_session=build_run_session(runtime_config.run_name))
    ensure_output_dirs(runtime_config)
    current_checkpoint_path = None if is_smoke_run(runtime_config) else get_checkpoint_path(runtime_config)
    current_model_path = None if is_smoke_run(runtime_config) else get_model_path(runtime_config)
    current_eval_path = get_result_eval_path(runtime_config)
    current_gifs_path = get_gifs_path(runtime_config)

    device = _resolve_learner_device(runtime_config)

    training_monitor = None
    if runtime_config.enable_training_monitor:
        training_monitor = TrainingMonitor(
            get_monitor_path(runtime_config),
            window_size=runtime_config.monitor_window,
            snapshot_interval=runtime_config.monitor_snapshot_interval,
        )
    ray.init(ignore_reinit_error=True)
    worker_runtime_config, worker_num_gpus = _resolve_worker_runtime(runtime_config)

    global_policy_net = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM).to(device)
    global_q_net1 = QNet(NODE_INPUT_DIM + 1, EMBEDDING_DIM).to(device)
    global_q_net2 = QNet(NODE_INPUT_DIM + 1, EMBEDDING_DIM).to(device)
    global_target_q_net1 = QNet(NODE_INPUT_DIM + 1, EMBEDDING_DIM).to(device)
    global_target_q_net2 = QNet(NODE_INPUT_DIM + 1, EMBEDDING_DIM).to(device)
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
    elif runtime_config.load_model and current_checkpoint_path is not None and current_checkpoint_path.exists():
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

    global_target_q_net1.load_state_dict(global_q_net1.state_dict())
    global_target_q_net2.load_state_dict(global_q_net2.state_dict())
    global_target_q_net1.eval()
    global_target_q_net2.eval()

    runner_options = {"num_cpus": 1, "num_gpus": worker_num_gpus}
    RemoteRunner = ray.remote(Runner)
    meta_agents = [
        RemoteRunner.options(**runner_options).remote(i, worker_runtime_config)
        for i in range(runtime_config.num_meta_agent)
    ]

    policy_wrapper = nn.DataParallel(global_policy_net) if device.type == "cuda" and torch.cuda.device_count() > 1 else global_policy_net
    q1_wrapper = nn.DataParallel(global_q_net1) if device.type == "cuda" and torch.cuda.device_count() > 1 else global_q_net1
    q2_wrapper = nn.DataParallel(global_q_net2) if device.type == "cuda" and torch.cuda.device_count() > 1 else global_q_net2
    target_q1_wrapper = (
        nn.DataParallel(global_target_q_net1) if device.type == "cuda" and torch.cuda.device_count() > 1 else global_target_q_net1
    )
    target_q2_wrapper = (
        nn.DataParallel(global_target_q_net2) if device.type == "cuda" and torch.cuda.device_count() > 1 else global_target_q_net2
    )

    weights_set = [_state_dict_to_device(global_policy_net.state_dict(), torch.device("cpu"))]
    job_list = []
    for meta_agent in meta_agents:
        if next_episode >= runtime_config.max_episodes:
            break
        next_episode += 1
        job_list.append(meta_agent.job.remote(weights_set, next_episode))

    metric_name = ["travel_dist", "success_rate", "explored_rate", "episode_return", "episode_steps"]
    training_data = []
    perf_metrics = {name: [] for name in metric_name}
    experience_buffer = [[] for _ in range(27)]
    save_gap = max(int(runtime_config.save_img_gap), 1)
    auto_eval_episodes = set()

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

            buffer_size = len(experience_buffer[0])
            for i in range(len(experience_buffer)):
                assert len(experience_buffer[i]) == buffer_size

            curr_episode = max(job[2]["episode_number"] for job in done_jobs)
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
                    done = torch.stack(rollouts[8]).to(device)
                    next_node_inputs = torch.stack(rollouts[9]).to(device)
                    next_node_padding_mask = torch.stack(rollouts[10]).to(device)
                    next_edge_mask = torch.stack(rollouts[11]).to(device)
                    next_current_index = torch.stack(rollouts[12]).to(device)
                    next_current_edge = torch.stack(rollouts[13]).to(device)
                    next_edge_padding_mask = torch.stack(rollouts[14]).to(device)

                    critic_node_inputs = torch.stack(rollouts[15]).to(device)
                    critic_node_padding_mask = torch.stack(rollouts[16]).to(device)
                    critic_edge_mask = torch.stack(rollouts[17]).to(device)
                    critic_current_index = torch.stack(rollouts[18]).to(device)
                    critic_current_edge = torch.stack(rollouts[19]).to(device)
                    critic_edge_padding_mask = torch.stack(rollouts[20]).to(device)
                    critic_next_node_inputs = torch.stack(rollouts[21]).to(device)
                    critic_next_node_padding_mask = torch.stack(rollouts[22]).to(device)
                    critic_next_edge_mask = torch.stack(rollouts[23]).to(device)
                    critic_next_current_index = torch.stack(rollouts[24]).to(device)
                    critic_next_current_edge = torch.stack(rollouts[25]).to(device)
                    critic_next_edge_padding_mask = torch.stack(rollouts[26]).to(device)

                    observation = [
                        node_inputs,
                        node_padding_mask,
                        edge_mask,
                        current_index,
                        current_edge,
                        edge_padding_mask,
                    ]
                    next_observation = [
                        next_node_inputs,
                        next_node_padding_mask,
                        next_edge_mask,
                        next_current_index,
                        next_current_edge,
                        next_edge_padding_mask,
                    ]

                    critic_observation = [
                        critic_node_inputs,
                        critic_node_padding_mask,
                        critic_edge_mask,
                        critic_current_index,
                        critic_current_edge,
                        critic_edge_padding_mask,
                    ]
                    critic_next_observation = [
                        critic_next_node_inputs,
                        critic_next_node_padding_mask,
                        critic_next_edge_mask,
                        critic_next_current_index,
                        critic_next_current_edge,
                        critic_next_edge_padding_mask,
                    ]

                    with torch.no_grad():
                        q_values1 = q1_wrapper(*critic_observation)
                        q_values2 = q2_wrapper(*critic_observation)
                        q_values = torch.min(q_values1, q_values2)

                    logp = policy_wrapper(*observation)
                    policy_loss = torch.sum(
                        logp.exp().unsqueeze(2) * (log_alpha.exp().detach() * logp.unsqueeze(2) - q_values.detach()),
                        dim=1,
                    ).mean()

                    global_policy_optimizer.zero_grad()
                    policy_loss.backward()
                    policy_grad_norm = torch.nn.utils.clip_grad_norm_(global_policy_net.parameters(), max_norm=100, norm_type=2)
                    global_policy_optimizer.step()

                    with torch.no_grad():
                        next_logp = policy_wrapper(*next_observation)
                        next_q_values1 = target_q1_wrapper(*critic_next_observation)
                        next_q_values2 = target_q2_wrapper(*critic_next_observation)
                        next_q_values = torch.min(next_q_values1, next_q_values2)
                        value_prime = torch.sum(
                            next_logp.unsqueeze(2).exp()
                            * (next_q_values - log_alpha.exp() * next_logp.unsqueeze(2)),
                            dim=1,
                        ).unsqueeze(1)
                        target_q = reward + GAMMA * (1 - done) * value_prime

                    mse_loss = nn.MSELoss()
                    q_values1 = q1_wrapper(*critic_observation)
                    q1 = torch.gather(q_values1, 1, action)
                    q1_loss = mse_loss(q1, target_q.detach()).mean()

                    global_q_net1_optimizer.zero_grad()
                    q1_loss.backward()
                    q_grad_norm = torch.nn.utils.clip_grad_norm_(global_q_net1.parameters(), max_norm=20000, norm_type=2)
                    global_q_net1_optimizer.step()

                    q_values2 = q2_wrapper(*critic_observation)
                    q2 = torch.gather(q_values2, 1, action)
                    q2_loss = mse_loss(q2, target_q.detach()).mean()

                    global_q_net2_optimizer.zero_grad()
                    q2_loss.backward()
                    q_grad_norm = torch.nn.utils.clip_grad_norm_(global_q_net2.parameters(), max_norm=20000, norm_type=2)
                    global_q_net2_optimizer.step()

                    entropy = (logp * logp.exp()).sum(dim=-1)
                    alpha_loss = -(log_alpha * (entropy.detach() + entropy_target)).mean()

                    log_alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    log_alpha_optimizer.step()

                    target_q_update_counter += 1
                    perf_data = [np.nanmean(perf_metrics[name]) for name in metric_name]
                    training_data.append(
                        [
                            reward.mean().item(),
                            value_prime.mean().item(),
                            policy_loss.item(),
                            q1_loss.item(),
                            entropy.mean().item(),
                            policy_grad_norm.item(),
                            q_grad_norm.item(),
                            log_alpha.item(),
                            alpha_loss.item(),
                            *perf_data,
                        ]
                    )

            if len(training_data) >= runtime_config.summary_window:
                if training_monitor is not None:
                    train_metrics = _aggregate_train_metrics(training_data)
                    training_monitor.update_train(curr_episode, train_metrics)
                    training_monitor.update_system(
                        curr_episode,
                        {
                            "buffer_size": len(experience_buffer[0]),
                            "completed_episodes": curr_episode,
                        },
                    )
                training_data = []
                perf_metrics = {name: [] for name in metric_name}

            weights_set = [_state_dict_to_device(global_policy_net.state_dict(), torch.device("cpu"))]

            if target_q_update_counter > 64:
                target_q_update_counter = 1
                global_target_q_net1.load_state_dict(global_q_net1.state_dict())
                global_target_q_net2.load_state_dict(global_q_net2.state_dict())
                global_target_q_net1.eval()
                global_target_q_net2.eval()

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

        if current_checkpoint_path is not None:
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
                current_checkpoint_path,
            )
    finally:
        for actor in meta_agents:
            ray.kill(actor)
        ray.shutdown()

    return {
        "checkpoint_path": str(current_checkpoint_path) if current_checkpoint_path is not None else None,
        "model_dir": str(current_model_path) if current_model_path is not None else None,
        "result_eval_dir": str(current_eval_path),
        "result_gif_dir": str(current_gifs_path),
        "train_dir": str(current_eval_path),
        "gif_dir": str(current_gifs_path),
        "episode": curr_episode,
    }


if __name__ == "__main__":
    main()
