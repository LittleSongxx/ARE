from __future__ import annotations

import random

import numpy as np
import ray
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

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
    get_train_path,
)
from .training_monitor import TrainingMonitor
from .runner import Runner
from .utils import ensure_bucket_dir, get_bucket_name


def _state_dict_to_device(state_dict, device):
    return {key: value.detach().to(device) for key, value in state_dict.items()}


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
    bucket_dir = ensure_bucket_dir(current_model_path, curr_episode, bucket_size)
    torch.save(checkpoint, bucket_dir / "checkpoint.pth")


def _aggregate_metric_records(metric_records):
    aggregated = {}
    for record in metric_records:
        for key, value in record.items():
            aggregated.setdefault(key, []).append(float(value))
    return {key: float(np.nanmean(values)) for key, values in aggregated.items() if values}


def _write_to_tensor_board(writer, tensorboard_data, curr_episode):
    metrics = _aggregate_metric_records(tensorboard_data)

    writer.add_scalar(tag="Losses/Value", scalar_value=metrics["value"], global_step=curr_episode)
    writer.add_scalar(tag="Losses/Policy Loss", scalar_value=metrics["policy_loss"], global_step=curr_episode)
    writer.add_scalar(tag="Losses/Alpha Loss", scalar_value=metrics["alpha_loss"], global_step=curr_episode)
    writer.add_scalar(tag="Losses/Q Value Loss", scalar_value=metrics["q_value_loss"], global_step=curr_episode)
    writer.add_scalar(tag="Losses/Entropy", scalar_value=metrics["entropy"], global_step=curr_episode)
    writer.add_scalar(tag="Losses/Policy Grad Norm", scalar_value=metrics["policy_grad_norm"], global_step=curr_episode)
    writer.add_scalar(tag="Losses/Q Value Grad Norm", scalar_value=metrics["q_value_grad_norm"], global_step=curr_episode)
    writer.add_scalar(tag="Losses/Log Alpha", scalar_value=metrics["log_alpha"], global_step=curr_episode)
    writer.add_scalar(tag="Perf/Reward", scalar_value=metrics["reward"], global_step=curr_episode)
    writer.add_scalar(tag="Perf/Travel Distance", scalar_value=metrics["travel_dist"], global_step=curr_episode)
    writer.add_scalar(tag="Perf/Explored Rate", scalar_value=metrics["explored_rate"], global_step=curr_episode)
    writer.add_scalar(tag="Perf/Success Rate", scalar_value=metrics["success_rate"], global_step=curr_episode)
    writer.add_scalar(tag="Perf/Episode Return", scalar_value=metrics["episode_return"], global_step=curr_episode)
    writer.add_scalar(tag="Perf/Episode Steps", scalar_value=metrics["episode_steps"], global_step=curr_episode)

    optional_tags = {
        "reward_info": "Perf/Reward Info",
        "reward_dist": "Perf/Reward Dist",
        "reward_safe": "Perf/Reward Safe",
        "reward_terminal": "Perf/Reward Terminal",
        "distill_loss": "Losses/Distill",
        "distill_lambda": "Losses/Distill Lambda",
        "curriculum_level_index": "Perf/Curriculum Level Index",
    }
    for key, tag in optional_tags.items():
        if key in metrics:
            writer.add_scalar(tag=tag, scalar_value=metrics[key], global_step=curr_episode)

    return metrics


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


def compute_distill_teacher_probs(q_values, edge_padding_mask, tau):
    teacher_logits = q_values.detach().squeeze(-1)
    invalid_mask = edge_padding_mask.squeeze(1).bool()
    tau = max(float(tau), 1e-6)
    teacher_logits = teacher_logits.masked_fill(invalid_mask, -1e8)
    teacher_probs = torch.softmax(teacher_logits / tau, dim=-1)
    teacher_probs = teacher_probs.masked_fill(invalid_mask, 0.0)
    teacher_probs = teacher_probs / teacher_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return teacher_probs


def compute_distill_loss(logp, teacher_probs):
    return -(teacher_probs * logp).sum(dim=-1).mean()


def compute_distill_lambda(update_count, max_lambda, warmup_updates):
    if warmup_updates <= 0:
        return float(max_lambda)
    return float(max_lambda) * min(max(int(update_count), 0) / float(warmup_updates), 1.0)


def get_policy_optimizer_update_count(optimizer):
    return int(optimizer.param_groups[0].get("distill_update_count", 0))


def set_policy_optimizer_update_count(optimizer, update_count):
    for param_group in optimizer.param_groups:
        param_group["distill_update_count"] = int(update_count)


def main(runtime_config: RuntimeConfig | None = None) -> dict:
    runtime_config = runtime_config or RuntimeConfig()
    if runtime_config.run_session is None:
        runtime_config = runtime_config.with_overrides(run_session=build_run_session(runtime_config.run_name))
    ensure_output_dirs(runtime_config)
    current_checkpoint_path = get_checkpoint_path(runtime_config)
    current_model_path = get_model_path(runtime_config)
    current_train_path = get_train_path(runtime_config)
    current_gifs_path = get_gifs_path(runtime_config)

    device = torch.device("cuda") if runtime_config.use_gpu_global else torch.device("cpu")
    local_device = torch.device("cuda") if runtime_config.use_gpu else torch.device("cpu")

    current_writer_bucket = get_bucket_name(1, runtime_config.result_bucket_episodes)
    writer = SummaryWriter(ensure_bucket_dir(current_train_path, 1, runtime_config.result_bucket_episodes))
    training_monitor = None
    if runtime_config.enable_training_monitor:
        training_monitor = TrainingMonitor(
            get_monitor_path(runtime_config),
            window_size=runtime_config.monitor_window,
            snapshot_interval=runtime_config.monitor_snapshot_interval,
        )
    ray.init(ignore_reinit_error=True)

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
    set_policy_optimizer_update_count(global_policy_optimizer, 0)
    entropy_target = 0.05 * (-np.log(1 / K_SIZE))

    curr_episode = 0
    next_episode = 0
    target_q_update_counter = 1

    if runtime_config.load_model and current_checkpoint_path.exists():
        checkpoint = torch.load(current_checkpoint_path, map_location=device, weights_only=False)
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
        set_policy_optimizer_update_count(
            global_policy_optimizer,
            get_policy_optimizer_update_count(global_policy_optimizer),
        )
        curr_episode = checkpoint["episode"]
        next_episode = curr_episode

    global_target_q_net1.load_state_dict(global_q_net1.state_dict())
    global_target_q_net2.load_state_dict(global_q_net2.state_dict())
    global_target_q_net1.eval()
    global_target_q_net2.eval()

    runner_options = {"num_cpus": 1, "num_gpus": runtime_config.num_gpu / max(runtime_config.num_meta_agent, 1)}
    RemoteRunner = ray.remote(Runner)
    meta_agents = [RemoteRunner.options(**runner_options).remote(i, runtime_config) for i in range(runtime_config.num_meta_agent)]

    policy_wrapper = nn.DataParallel(global_policy_net) if device.type == "cuda" and torch.cuda.device_count() > 1 else global_policy_net
    q1_wrapper = nn.DataParallel(global_q_net1) if device.type == "cuda" and torch.cuda.device_count() > 1 else global_q_net1
    q2_wrapper = nn.DataParallel(global_q_net2) if device.type == "cuda" and torch.cuda.device_count() > 1 else global_q_net2
    target_q1_wrapper = (
        nn.DataParallel(global_target_q_net1) if device.type == "cuda" and torch.cuda.device_count() > 1 else global_target_q_net1
    )
    target_q2_wrapper = (
        nn.DataParallel(global_target_q_net2) if device.type == "cuda" and torch.cuda.device_count() > 1 else global_target_q_net2
    )

    weights_set = [_state_dict_to_device(global_policy_net.state_dict(), local_device)]
    job_list = []
    for meta_agent in meta_agents:
        if next_episode >= runtime_config.max_episodes:
            break
        next_episode += 1
        job_list.append(meta_agent.job.remote(weights_set, next_episode))

    training_data = []
    perf_metrics = {}
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
                for name, value in metrics.items():
                    perf_metrics.setdefault(name, []).append(value)

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
                    distill_loss = torch.zeros((), device=device)
                    distill_lambda = 0.0
                    policy_loss_total = policy_loss
                    if runtime_config.rl_options.use_privileged_distill:
                        teacher_probs = compute_distill_teacher_probs(
                            q_values,
                            edge_padding_mask,
                            runtime_config.rl_options.distill_tau,
                        )
                        distill_loss = compute_distill_loss(logp, teacher_probs)
                        current_update_count = get_policy_optimizer_update_count(global_policy_optimizer) + 1
                        distill_lambda = compute_distill_lambda(
                            current_update_count,
                            runtime_config.rl_options.distill_lambda,
                            runtime_config.rl_options.distill_warmup_updates,
                        )
                        policy_loss_total = policy_loss + distill_lambda * distill_loss

                    global_policy_optimizer.zero_grad()
                    policy_loss_total.backward()
                    policy_grad_norm = torch.nn.utils.clip_grad_norm_(global_policy_net.parameters(), max_norm=100, norm_type=2)
                    global_policy_optimizer.step()
                    if runtime_config.rl_options.use_privileged_distill:
                        set_policy_optimizer_update_count(global_policy_optimizer, current_update_count)

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
                        gamma_multiplier = GAMMA
                        if runtime_config.rl_options.use_n_step_return:
                            gamma_multiplier = GAMMA ** runtime_config.rl_options.n_step
                        target_q = reward + gamma_multiplier * (1 - done) * value_prime

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
                    perf_data = {
                        name: float(np.nanmean(values))
                        for name, values in perf_metrics.items()
                        if values
                    }
                    training_record = {
                        "reward": reward.mean().item(),
                        "value": value_prime.mean().item(),
                        "policy_loss": policy_loss_total.item(),
                        "q_value_loss": q1_loss.item(),
                        "entropy": entropy.mean().item(),
                        "policy_grad_norm": policy_grad_norm.item(),
                        "q_value_grad_norm": q_grad_norm.item(),
                        "log_alpha": log_alpha.item(),
                        "alpha_loss": alpha_loss.item(),
                    }
                    if runtime_config.rl_options.use_privileged_distill:
                        training_record["distill_loss"] = distill_loss.item()
                        training_record["distill_lambda"] = float(distill_lambda)
                    training_record.update(perf_data)
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
                            "buffer_size": len(experience_buffer[0]),
                            "completed_episodes": curr_episode,
                        },
                    )
                training_data = []
                perf_metrics = {}

            weights_set = [_state_dict_to_device(global_policy_net.state_dict(), local_device)]

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
    finally:
        writer.close()
        for actor in meta_agents:
            ray.kill(actor)
        ray.shutdown()

    return {
        "checkpoint_path": str(current_checkpoint_path),
        "model_dir": str(current_model_path),
        "train_dir": str(current_train_path),
        "gif_dir": str(current_gifs_path),
        "episode": curr_episode,
    }


if __name__ == "__main__":
    main()
