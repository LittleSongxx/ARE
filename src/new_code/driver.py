import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import random

import numpy as np
import ray
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from evaluation import evaluate_policy, summarize_eval_results
from model import PolicyNet, QNet
from parameter import *
from runner import RLRunner
from training_monitor import TrainingMonitor


def _state_dict_to_device(state_dict, device):
    return {key: value.detach().to(device) for key, value in state_dict.items()}


def _build_checkpoint(global_policy_net, global_q_net1, global_q_net2, log_alpha, global_policy_optimizer, global_q_net1_optimizer, global_q_net2_optimizer, log_alpha_optimizer, episode):
    return {
        "policy_model": global_policy_net.state_dict(),
        "q_net1_model": global_q_net1.state_dict(),
        "q_net2_model": global_q_net2.state_dict(),
        "log_alpha": log_alpha.detach().cpu(),
        "policy_optimizer": global_policy_optimizer.state_dict(),
        "q_net1_optimizer": global_q_net1_optimizer.state_dict(),
        "q_net2_optimizer": global_q_net2_optimizer.state_dict(),
        "log_alpha_optimizer": log_alpha_optimizer.state_dict(),
        "episode": episode,
    }


def _save_checkpoint(path, checkpoint):
    torch.save(checkpoint, path)


def _load_checkpoint(device):
    for candidate in (checkpoint_interrupted_path, checkpoint_final_path, checkpoint_path):
        if os.path.exists(candidate):
            checkpoint = torch.load(candidate, map_location=device, weights_only=False)
            return candidate, checkpoint
    return None, None


def _write_train_to_tensorboard(writer, tensorboard_data, curr_episode):
    tensorboard_data = np.array(tensorboard_data)
    tensorboard_data = list(np.nanmean(tensorboard_data, axis=0))
    reward, value, policy_loss, q_value_loss, entropy, policy_grad_norm, q_value_grad_norm, log_alpha, alpha_loss, travel_dist, success_rate, explored_rate, episode_return, episode_steps = tensorboard_data

    writer.add_scalar(tag="Losses/Value", scalar_value=value, global_step=curr_episode)
    writer.add_scalar(tag="Losses/Policy Loss", scalar_value=policy_loss, global_step=curr_episode)
    writer.add_scalar(tag="Losses/Alpha Loss", scalar_value=alpha_loss, global_step=curr_episode)
    writer.add_scalar(tag="Losses/Q Value Loss", scalar_value=q_value_loss, global_step=curr_episode)
    writer.add_scalar(tag="Losses/Entropy", scalar_value=entropy, global_step=curr_episode)
    writer.add_scalar(tag="Losses/Policy Grad Norm", scalar_value=policy_grad_norm, global_step=curr_episode)
    writer.add_scalar(tag="Losses/Q Value Grad Norm", scalar_value=q_value_grad_norm, global_step=curr_episode)
    writer.add_scalar(tag="Losses/Log Alpha", scalar_value=log_alpha, global_step=curr_episode)
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
    }


def _write_eval_to_tensorboard(writer, eval_summary, curr_episode):
    writer.add_scalar(tag="Eval/Explored Rate", scalar_value=eval_summary["explored_rate"], global_step=curr_episode)
    writer.add_scalar(tag="Eval/Travel Distance", scalar_value=eval_summary["travel_dist"], global_step=curr_episode)
    writer.add_scalar(tag="Eval/Success Rate", scalar_value=eval_summary["success_rate"], global_step=curr_episode)
    writer.add_scalar(tag="Eval/Episode Return", scalar_value=eval_summary["episode_return"], global_step=curr_episode)
    writer.add_scalar(tag="Eval/Steps Taken", scalar_value=eval_summary["steps_taken"], global_step=curr_episode)


def main():
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(gifs_path, exist_ok=True)
    os.makedirs(eval_path, exist_ok=True)

    if USE_GPU and NUM_GPU > 0:
        ray.init(ignore_reinit_error=True, num_gpus=NUM_GPU)
    else:
        ray.init(ignore_reinit_error=True)
    print(f"Welcome to RL autonomous exploration! worker_gpus={NUM_GPU} learner_device={'cuda' if USE_GPU_GLOBAL else 'cpu'}")

    writer = SummaryWriter(train_path)
    monitor = TrainingMonitor(save_dir=monitor_path, window_size=MONITOR_WINDOW, snapshot_interval=MONITOR_SNAPSHOT_INTERVAL)

    device = torch.device("cuda") if USE_GPU_GLOBAL and torch.cuda.is_available() else torch.device("cpu")
    local_device = torch.device("cuda") if USE_GPU and torch.cuda.is_available() else torch.device("cpu")

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

    if LOAD_MODEL:
        checkpoint_file, checkpoint = _load_checkpoint(device)
        if checkpoint is None:
            print("[warning] no checkpoint found, start from scratch")
        else:
            print(f"Loading Model from: {checkpoint_file}")
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

    meta_agents = [RLRunner.remote(i) for i in range(NUM_META_AGENT)]

    policy_wrapper = nn.DataParallel(global_policy_net) if device.type == "cuda" and torch.cuda.device_count() > 1 else global_policy_net
    q1_wrapper = nn.DataParallel(global_q_net1) if device.type == "cuda" and torch.cuda.device_count() > 1 else global_q_net1
    q2_wrapper = nn.DataParallel(global_q_net2) if device.type == "cuda" and torch.cuda.device_count() > 1 else global_q_net2
    target_q1_wrapper = nn.DataParallel(global_target_q_net1) if device.type == "cuda" and torch.cuda.device_count() > 1 else global_target_q_net1
    target_q2_wrapper = nn.DataParallel(global_target_q_net2) if device.type == "cuda" and torch.cuda.device_count() > 1 else global_target_q_net2

    weights_set = [_state_dict_to_device(global_policy_net.state_dict(), local_device)]
    job_list = []
    for meta_agent in meta_agents:
        if not INFINITE_TRAINING and next_episode >= MAX_EPISODES:
            break
        next_episode += 1
        job_list.append(meta_agent.job.remote(weights_set, next_episode))

    metric_names = ["travel_dist", "success_rate", "explored_rate", "episode_return", "episode_steps"]
    perf_metrics = {name: [] for name in metric_names}
    training_data = []
    experience_buffer = [[] for _ in range(27)]
    auto_eval_triggers = set()
    save_gap = max(int(SAVE_IMG_GAP), 1)

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
                for name in metric_names:
                    perf_metrics[name].append(metrics[name])

            buffer_size = len(experience_buffer[0])
            for i in range(len(experience_buffer)):
                assert len(experience_buffer[i]) == buffer_size

            curr_episode = max(job[2]["episode_number"] for job in done_jobs)
            if INFINITE_TRAINING or next_episode < MAX_EPISODES:
                next_episode += 1
                job_list.append(meta_agents[info["id"]].job.remote(weights_set, next_episode))

            if len(experience_buffer[0]) >= MINIMUM_BUFFER_SIZE:
                if len(experience_buffer[0]) >= REPLAY_SIZE:
                    for i in range(len(experience_buffer)):
                        experience_buffer[i] = experience_buffer[i][-REPLAY_SIZE:]

                indices = list(range(len(experience_buffer[0])))
                sample_batch_size = min(BATCH_SIZE, len(indices))

                for _ in range(TRAIN_UPDATES_PER_ITER):
                    if sample_batch_size == 0:
                        break
                    sample_indices = random.sample(indices, sample_batch_size)
                    rollouts = [[experience_buffer[i][index] for index in sample_indices] for i in range(len(experience_buffer))]

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

                    observation = [node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask]
                    next_observation = [next_node_inputs, next_node_padding_mask, next_edge_mask, next_current_index, next_current_edge, next_edge_padding_mask]
                    critic_observation = [critic_node_inputs, critic_node_padding_mask, critic_edge_mask, critic_current_index, critic_current_edge, critic_edge_padding_mask]
                    critic_next_observation = [critic_next_node_inputs, critic_next_node_padding_mask, critic_next_edge_mask, critic_next_current_index, critic_next_current_edge, critic_next_edge_padding_mask]

                    with torch.no_grad():
                        q_values1 = q1_wrapper(*critic_observation)
                        q_values2 = q2_wrapper(*critic_observation)
                        q_values = torch.min(q_values1, q_values2)

                    logp = policy_wrapper(*observation)
                    policy_loss = torch.sum(logp.exp().unsqueeze(2) * (log_alpha.exp().detach() * logp.unsqueeze(2) - q_values.detach()), dim=1).mean()

                    global_policy_optimizer.zero_grad()
                    policy_loss.backward()
                    policy_grad_norm = torch.nn.utils.clip_grad_norm_(global_policy_net.parameters(), max_norm=100, norm_type=2)
                    global_policy_optimizer.step()

                    with torch.no_grad():
                        next_logp = policy_wrapper(*next_observation)
                        next_q_values1 = target_q1_wrapper(*critic_next_observation)
                        next_q_values2 = target_q2_wrapper(*critic_next_observation)
                        next_q_values = torch.min(next_q_values1, next_q_values2)
                        value_prime = torch.sum(next_logp.unsqueeze(2).exp() * (next_q_values - log_alpha.exp() * next_logp.unsqueeze(2)), dim=1).unsqueeze(1)
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
                    perf_data = [np.nanmean(perf_metrics[name]) for name in metric_names]
                    training_data.append([
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
                    ])

            if len(training_data) >= SUMMARY_WINDOW:
                train_metrics = _write_train_to_tensorboard(writer, training_data, curr_episode)
                monitor.update_train(curr_episode, train_metrics)
                monitor.update_system(curr_episode, {"buffer_size": len(experience_buffer[0]), "completed_episodes": curr_episode})
                training_data = []
                perf_metrics = {name: [] for name in metric_names}

            weights_set = [_state_dict_to_device(global_policy_net.state_dict(), local_device)]

            if target_q_update_counter > 64:
                target_q_update_counter = 1
                global_target_q_net1.load_state_dict(global_q_net1.state_dict())
                global_target_q_net2.load_state_dict(global_q_net2.state_dict())
                global_target_q_net1.eval()
                global_target_q_net2.eval()

            if AUTO_EVAL:
                for episode_number in sorted(completed_episodes):
                    if episode_number % save_gap != 0 or episode_number in auto_eval_triggers:
                        continue
                    eval_results = evaluate_policy(
                        global_policy_net.state_dict(),
                        episodes=AUTO_EVAL_EPISODES,
                        start_episode=episode_number,
                        greedy=AUTO_EVAL_GREEDY,
                        device=AUTO_EVAL_DEVICE,
                        max_episode_step=MAX_EPISODE_STEP,
                    )
                    eval_summary = summarize_eval_results(eval_results)
                    _write_eval_to_tensorboard(writer, eval_summary, episode_number)
                    monitor.update_eval(episode_number, eval_summary)
                    auto_eval_triggers.add(episode_number)
                    print(
                        f"auto_eval episode={episode_number} "
                        f"episodes={eval_summary['episodes']} "
                        f"explored_rate={eval_summary['explored_rate']:.4f} "
                        f"travel_dist={eval_summary['travel_dist']:.2f} "
                        f"success_rate={eval_summary['success_rate']:.4f} "
                        f"episode_return={eval_summary['episode_return']:.4f} "
                        f"steps_taken={eval_summary['steps_taken']:.2f}"
                    )

        final_checkpoint = _build_checkpoint(
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
        _save_checkpoint(checkpoint_path, final_checkpoint)
        _save_checkpoint(checkpoint_final_path, final_checkpoint)
        print(f"Final model saved to {checkpoint_final_path}")
    except KeyboardInterrupt:
        interrupted_checkpoint = _build_checkpoint(
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
        _save_checkpoint(checkpoint_interrupted_path, interrupted_checkpoint)
        print(f"Interrupted model saved to {checkpoint_interrupted_path}")
    finally:
        writer.close()
        for actor in meta_agents:
            ray.kill(actor)
        ray.shutdown()


def write_to_tensor_board(writer, tensorboard_data, curr_episode):
    return _write_train_to_tensorboard(writer, tensorboard_data, curr_episode)


if __name__ == "__main__":
    main()
