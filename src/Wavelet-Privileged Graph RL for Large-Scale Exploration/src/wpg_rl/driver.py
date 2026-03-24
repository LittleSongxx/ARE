from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import copy
import os
import random
import shutil
import signal
import time

import numpy as np
import ray
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from .evaluation import evaluate_policy, get_evaluated_episodes, save_evaluation_summary, summarize_eval_results
from .model import PolicyNet, QNet
from .parameter import (
    CRITIC_NODE_INPUT_DIM,
    EMBEDDING_DIM,
    GAMMA,
    K_SIZE,
    LR,
    NODE_INPUT_DIM,
    POLICY_GRAD_CLIP,
    Q_GRAD_CLIP,
    SRC_ROOT,
    RuntimeConfig,
    TARGET_Q_UPDATE_INTERVAL,
    apply_runtime_config,
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
from .runner import Runner
from .runtime_utils import configure_worker_process_threads, resolve_ray_num_cpus, resolve_ray_worker_num_cpus, resolve_worker_num_threads
from .training_monitor import TrainingMonitor
from .utils import ensure_bucket_dir, get_bucket_name
from .wavelet_losses import WaveletDistillationLoss
from .wavelet_graph import build_overlap_valid_mask, decompose_graph_hidden, gather_by_index


_interrupted = False
_interrupt_raised = False
_saving_interrupt_checkpoint = False


ROLLOUT_BATCH_KEYS = (
    "node_inputs",
    "node_padding_mask",
    "edge_mask",
    "current_index",
    "current_edge",
    "edge_padding_mask",
    "action",
    "reward",
    "done",
    "next_node_inputs",
    "next_node_padding_mask",
    "next_edge_mask",
    "next_current_index",
    "next_current_edge",
    "next_edge_padding_mask",
    "critic_node_inputs",
    "critic_node_padding_mask",
    "critic_edge_mask",
    "critic_current_index",
    "critic_current_edge",
    "critic_edge_padding_mask",
    "critic_next_node_inputs",
    "critic_next_node_padding_mask",
    "critic_next_edge_mask",
    "critic_next_current_index",
    "critic_next_current_edge",
    "critic_next_edge_padding_mask",
    "actor_to_critic_index",
)


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


@dataclass
class LearnerState:
    policy_net: PolicyNet
    q_net1: QNet
    q_net2: QNet
    target_q_net1: QNet
    target_q_net2: QNet
    policy_wrapper: nn.Module
    q_net1_wrapper: nn.Module
    q_net2_wrapper: nn.Module
    target_q_net1_wrapper: nn.Module
    target_q_net2_wrapper: nn.Module
    policy_optimizer: optim.Optimizer
    q_net1_optimizer: optim.Optimizer
    q_net2_optimizer: optim.Optimizer
    log_alpha: torch.Tensor
    log_alpha_optimizer: optim.Optimizer
    entropy_target: float
    wavelet_distillation: WaveletDistillationLoss
    device: torch.device
    update_step: int = 0
    target_q_update_counter: int = 1


def _policy_kwargs(runtime_config: RuntimeConfig) -> dict:
    return {
        "use_lf_attention_hf_residual": runtime_config.use_lf_attention_hf_residual,
        "use_privileged_wavelet_distillation": runtime_config.use_privileged_wavelet_distillation,
        "wavelet_scales": runtime_config.wavelet_scales,
        "wavelet_fuse_dim": runtime_config.wavelet_fuse_dim,
        "wavelet_lf_qk": runtime_config.wavelet_lf_qk,
    }


def _critic_kwargs(runtime_config: RuntimeConfig) -> dict:
    return {
        "use_lf_attention_hf_residual": runtime_config.use_lf_attention_hf_residual,
        "wavelet_scales": runtime_config.wavelet_scales,
        "wavelet_fuse_dim": runtime_config.wavelet_fuse_dim,
        "wavelet_lf_qk": runtime_config.wavelet_lf_qk,
    }


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
        return torch.device("cuda:0")
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
        f"wavelet_loss={_format_metric(train_metrics.get('wavelet_loss'))} "
        f"wavelet_lf_loss={_format_metric(train_metrics.get('wavelet_lf_loss'))} "
        f"wavelet_hf_loss={_format_metric(train_metrics.get('wavelet_hf_loss'))} "
        f"wavelet_lambda_eff={_format_metric(train_metrics.get('wavelet_lambda_eff'))}"
    )


def _state_dict_to_cpu(module: nn.Module) -> dict:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _checkpoint_dict(learner_state: LearnerState, completed_episodes: int) -> dict:
    return {
        "policy_model": _state_dict_to_cpu(learner_state.policy_net),
        "q_net1_model": _state_dict_to_cpu(learner_state.q_net1),
        "q_net2_model": _state_dict_to_cpu(learner_state.q_net2),
        "log_alpha": learner_state.log_alpha.detach().cpu().clone(),
        "policy_optimizer": copy.deepcopy(learner_state.policy_optimizer.state_dict()),
        "q_net1_optimizer": copy.deepcopy(learner_state.q_net1_optimizer.state_dict()),
        "q_net2_optimizer": copy.deepcopy(learner_state.q_net2_optimizer.state_dict()),
        "log_alpha_optimizer": copy.deepcopy(learner_state.log_alpha_optimizer.state_dict()),
        "update_step": learner_state.update_step,
        "target_q_update_counter": learner_state.target_q_update_counter,
        "episode": completed_episodes,
    }


def _save_checkpoint(
    learner_state: LearnerState,
    runtime_config: RuntimeConfig,
    completed_episodes: int,
    checkpoint_path: Path,
    bucket_model_dir: Path,
):
    payload = _checkpoint_dict(learner_state, completed_episodes)
    torch.save(payload, checkpoint_path)
    bucket_dir = ensure_bucket_dir(bucket_model_dir, completed_episodes, runtime_config.result_bucket_episodes)
    torch.save(payload, bucket_dir / "checkpoint.pth")


def _save_interrupt_checkpoint(
    learner_state: LearnerState,
    runtime_config: RuntimeConfig,
    completed_episodes: int,
    checkpoint_path: Path,
    interrupt_path: Path,
    bucket_model_dir: Path,
):
    payload = _checkpoint_dict(learner_state, completed_episodes)
    torch.save(payload, interrupt_path)
    torch.save(payload, checkpoint_path)
    bucket_dir = ensure_bucket_dir(bucket_model_dir, completed_episodes, runtime_config.result_bucket_episodes)
    torch.save(payload, bucket_dir / "checkpoint.pth")
    print(f"interrupt_checkpoint_saved path={interrupt_path} episode={completed_episodes}")


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


def create_learner_state(runtime_config: RuntimeConfig, device: torch.device) -> LearnerState:
    policy_net = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM, **_policy_kwargs(runtime_config)).to(device)
    q_net1 = QNet(CRITIC_NODE_INPUT_DIM, EMBEDDING_DIM, **_critic_kwargs(runtime_config)).to(device)
    q_net2 = QNet(CRITIC_NODE_INPUT_DIM, EMBEDDING_DIM, **_critic_kwargs(runtime_config)).to(device)
    target_q_net1 = QNet(CRITIC_NODE_INPUT_DIM, EMBEDDING_DIM, **_critic_kwargs(runtime_config)).to(device)
    target_q_net2 = QNet(CRITIC_NODE_INPUT_DIM, EMBEDDING_DIM, **_critic_kwargs(runtime_config)).to(device)
    target_q_net1.load_state_dict(q_net1.state_dict())
    target_q_net2.load_state_dict(q_net2.state_dict())
    target_q_net1.eval()
    target_q_net2.eval()

    log_alpha = torch.FloatTensor([-2.0]).to(device)
    log_alpha.requires_grad = True

    return LearnerState(
        policy_net=policy_net,
        q_net1=q_net1,
        q_net2=q_net2,
        target_q_net1=target_q_net1,
        target_q_net2=target_q_net2,
        policy_wrapper=policy_net,
        q_net1_wrapper=q_net1,
        q_net2_wrapper=q_net2,
        target_q_net1_wrapper=target_q_net1,
        target_q_net2_wrapper=target_q_net2,
        policy_optimizer=optim.Adam(policy_net.parameters(), lr=LR),
        q_net1_optimizer=optim.Adam(q_net1.parameters(), lr=LR),
        q_net2_optimizer=optim.Adam(q_net2.parameters(), lr=LR),
        log_alpha=log_alpha,
        log_alpha_optimizer=optim.Adam([log_alpha], lr=1e-4),
        entropy_target=0.05 * (-np.log(1 / K_SIZE)),
        wavelet_distillation=WaveletDistillationLoss(
            runtime_config.wavelet_distill_lf_weight,
            runtime_config.wavelet_distill_hf_weight,
            runtime_config.wavelet_distill_weight,
            runtime_config.wavelet_distill_warmup_updates,
            runtime_config.wavelet_distill_ramp_updates,
        ),
        device=device,
    )


def load_checkpoint(learner_state: LearnerState, checkpoint_path: Path) -> int:
    checkpoint = torch.load(checkpoint_path, map_location=learner_state.device, weights_only=False)
    learner_state.policy_net.load_state_dict(checkpoint["policy_model"])
    learner_state.q_net1.load_state_dict(checkpoint["q_net1_model"])
    learner_state.q_net2.load_state_dict(checkpoint["q_net2_model"])
    learner_state.target_q_net1.load_state_dict(checkpoint["q_net1_model"])
    learner_state.target_q_net2.load_state_dict(checkpoint["q_net2_model"])

    learner_state.log_alpha = checkpoint["log_alpha"].to(learner_state.device)
    learner_state.log_alpha.requires_grad_(True)
    learner_state.log_alpha_optimizer = optim.Adam([learner_state.log_alpha], lr=1e-4)

    learner_state.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
    learner_state.q_net1_optimizer.load_state_dict(checkpoint["q_net1_optimizer"])
    learner_state.q_net2_optimizer.load_state_dict(checkpoint["q_net2_optimizer"])
    learner_state.log_alpha_optimizer.load_state_dict(checkpoint["log_alpha_optimizer"])
    learner_state.update_step = int(checkpoint.get("update_step", 0))
    learner_state.target_q_update_counter = int(checkpoint.get("target_q_update_counter", 1))
    return int(checkpoint.get("episode", 0))


def sample_rollout_batch(experience_buffer, batch_size: int, replace: bool = False):
    buffer_size = len(experience_buffer[0])
    if buffer_size == 0:
        raise ValueError("experience_buffer is empty")
    if replace or buffer_size < batch_size:
        sample_indices = [random.randrange(buffer_size) for _ in range(batch_size)]
    else:
        sample_indices = random.sample(range(buffer_size), batch_size)

    rollouts = []
    for i in range(len(experience_buffer)):
        rollouts.append([experience_buffer[i][index] for index in sample_indices])
    return rollouts


def stack_batch_tensors(rollouts, device: torch.device) -> dict[str, torch.Tensor]:
    return {key: torch.stack(values).to(device) for key, values in zip(ROLLOUT_BATCH_KEYS, rollouts)}


def _default_wavelet_metrics(device: torch.device) -> dict:
    zero = torch.tensor(0.0, device=device)
    return {
        "loss_lf": zero,
        "loss_hf": zero,
        "loss": zero,
        "weighted_loss": zero,
        "lambda_eff": 0.0,
    }


def train_step(batch: dict[str, torch.Tensor], learner_state: LearnerState, runtime_config: RuntimeConfig) -> dict[str, float]:
    observation = [
        batch["node_inputs"],
        batch["node_padding_mask"],
        batch["edge_mask"],
        batch["current_index"],
        batch["current_edge"],
        batch["edge_padding_mask"],
    ]
    next_observation = [
        batch["next_node_inputs"],
        batch["next_node_padding_mask"],
        batch["next_edge_mask"],
        batch["next_current_index"],
        batch["next_current_edge"],
        batch["next_edge_padding_mask"],
    ]
    critic_observation = [
        batch["critic_node_inputs"],
        batch["critic_node_padding_mask"],
        batch["critic_edge_mask"],
        batch["critic_current_index"],
        batch["critic_current_edge"],
        batch["critic_edge_padding_mask"],
    ]
    critic_next_observation = [
        batch["critic_next_node_inputs"],
        batch["critic_next_node_padding_mask"],
        batch["critic_next_edge_mask"],
        batch["critic_next_current_index"],
        batch["critic_next_current_edge"],
        batch["critic_next_edge_padding_mask"],
    ]

    with torch.no_grad():
        q_values1 = learner_state.q_net1_wrapper(*critic_observation)
        q_values2 = learner_state.q_net2_wrapper(*critic_observation)
        q_values = torch.min(q_values1, q_values2)

    if runtime_config.use_privileged_wavelet_distillation:
        logp, actor_hidden = learner_state.policy_wrapper(*observation, return_hidden=True)
    else:
        logp = learner_state.policy_wrapper(*observation)
        actor_hidden = None

    policy_loss_original = torch.sum(
        logp.exp().unsqueeze(2) * (learner_state.log_alpha.exp().detach() * logp.unsqueeze(2) - q_values.detach()),
        dim=1,
    ).mean()

    wavelet_metrics = _default_wavelet_metrics(learner_state.device)
    if runtime_config.use_privileged_wavelet_distillation:
        actor_lf, actor_hf, *_ = decompose_graph_hidden(
            actor_hidden,
            batch["edge_mask"],
            batch["node_padding_mask"],
            scales=runtime_config.wavelet_scales,
        )
        overlap_valid_mask = build_overlap_valid_mask(batch["actor_to_critic_index"], batch["node_padding_mask"])

        with torch.no_grad():
            _, critic_hidden = learner_state.target_q_net1_wrapper(*critic_observation, return_hidden=True)
            critic_lf_full, critic_hf_full, *_ = decompose_graph_hidden(
                critic_hidden,
                batch["critic_edge_mask"],
                batch["critic_node_padding_mask"],
                scales=runtime_config.wavelet_scales,
            )
            critic_lf = gather_by_index(critic_lf_full, batch["actor_to_critic_index"])
            critic_hf = gather_by_index(critic_hf_full, batch["actor_to_critic_index"])
            critic_valid_full = (~batch["critic_node_padding_mask"].squeeze(1).bool()).to(dtype=actor_lf.dtype)
            critic_valid_overlap = gather_by_index(
                critic_valid_full.unsqueeze(-1),
                batch["actor_to_critic_index"],
            ).squeeze(-1) > 0
            overlap_valid_mask = overlap_valid_mask & critic_valid_overlap

        wavelet_metrics = learner_state.wavelet_distillation(
            actor_lf,
            actor_hf,
            critic_lf,
            critic_hf,
            overlap_valid_mask,
            learner_state.update_step,
        )

    policy_loss = policy_loss_original + wavelet_metrics["weighted_loss"]
    learner_state.policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_grad_norm = torch.nn.utils.clip_grad_norm_(
        learner_state.policy_net.parameters(),
        max_norm=POLICY_GRAD_CLIP,
        norm_type=2,
    )
    learner_state.policy_optimizer.step()

    with torch.no_grad():
        next_logp = learner_state.policy_wrapper(*next_observation)
        next_q_values1 = learner_state.target_q_net1_wrapper(*critic_next_observation)
        next_q_values2 = learner_state.target_q_net2_wrapper(*critic_next_observation)
        next_q_values = torch.min(next_q_values1, next_q_values2)
        value_prime = torch.sum(
            next_logp.unsqueeze(2).exp()
            * (next_q_values - learner_state.log_alpha.exp() * next_logp.unsqueeze(2)),
            dim=1,
        ).unsqueeze(1)
        target_q = batch["reward"] + GAMMA * (1 - batch["done"]) * value_prime

    mse_loss = nn.MSELoss()

    q_values1 = learner_state.q_net1_wrapper(*critic_observation)
    q1 = torch.gather(q_values1, 1, batch["action"])
    q1_loss = mse_loss(q1, target_q.detach()).mean()
    learner_state.q_net1_optimizer.zero_grad()
    q1_loss.backward()
    q1_grad_norm = torch.nn.utils.clip_grad_norm_(
        learner_state.q_net1.parameters(),
        max_norm=Q_GRAD_CLIP,
        norm_type=2,
    )
    learner_state.q_net1_optimizer.step()

    q_values2 = learner_state.q_net2_wrapper(*critic_observation)
    q2 = torch.gather(q_values2, 1, batch["action"])
    q2_loss = mse_loss(q2, target_q.detach()).mean()
    learner_state.q_net2_optimizer.zero_grad()
    q2_loss.backward()
    q2_grad_norm = torch.nn.utils.clip_grad_norm_(
        learner_state.q_net2.parameters(),
        max_norm=Q_GRAD_CLIP,
        norm_type=2,
    )
    learner_state.q_net2_optimizer.step()

    entropy = (logp * logp.exp()).sum(dim=-1)
    alpha_loss = -(learner_state.log_alpha * (entropy.detach() + learner_state.entropy_target)).mean()
    learner_state.log_alpha_optimizer.zero_grad()
    alpha_loss.backward()
    learner_state.log_alpha_optimizer.step()

    learner_state.target_q_update_counter += 1
    if learner_state.target_q_update_counter > TARGET_Q_UPDATE_INTERVAL:
        learner_state.target_q_update_counter = 1
        learner_state.target_q_net1.load_state_dict(learner_state.q_net1.state_dict())
        learner_state.target_q_net2.load_state_dict(learner_state.q_net2.state_dict())
        learner_state.target_q_net1.eval()
        learner_state.target_q_net2.eval()

    learner_state.update_step += 1
    return {
        "reward": batch["reward"].mean().item(),
        "value": value_prime.mean().item(),
        "policy_loss": policy_loss.item(),
        "policy_loss_original": policy_loss_original.item(),
        "q_value_loss": 0.5 * (q1_loss.item() + q2_loss.item()),
        "entropy": entropy.mean().item(),
        "policy_grad_norm": float(policy_grad_norm),
        "q_value_grad_norm": float(max(q1_grad_norm, q2_grad_norm)),
        "log_alpha": learner_state.log_alpha.item(),
        "alpha_loss": alpha_loss.item(),
        "wavelet_lf_loss": float(wavelet_metrics["loss_lf"]),
        "wavelet_hf_loss": float(wavelet_metrics["loss_hf"]),
        "wavelet_loss": float(wavelet_metrics["loss"]),
        "wavelet_weighted_loss": float(wavelet_metrics["weighted_loss"]),
        "wavelet_lambda_eff": float(wavelet_metrics["lambda_eff"]),
    }


def write_to_tensor_board(writer: SummaryWriter, tensorboard_data: list[dict], curr_episode: int) -> dict[str, float]:
    metrics = {}
    for key in tensorboard_data[0].keys():
        metrics[key] = float(np.nanmean([item[key] for item in tensorboard_data]))

    writer.add_scalar("Losses/Value", metrics["value"], curr_episode)
    writer.add_scalar("Losses/Policy Loss", metrics["policy_loss"], curr_episode)
    writer.add_scalar("Losses/Policy Loss Original", metrics["policy_loss_original"], curr_episode)
    writer.add_scalar("Losses/Alpha Loss", metrics["alpha_loss"], curr_episode)
    writer.add_scalar("Losses/Q Value Loss", metrics["q_value_loss"], curr_episode)
    writer.add_scalar("Losses/Entropy", metrics["entropy"], curr_episode)
    writer.add_scalar("Losses/Policy Grad Norm", metrics["policy_grad_norm"], curr_episode)
    writer.add_scalar("Losses/Q Value Grad Norm", metrics["q_value_grad_norm"], curr_episode)
    writer.add_scalar("Losses/Log Alpha", metrics["log_alpha"], curr_episode)
    writer.add_scalar("Losses/Wavelet LF", metrics["wavelet_lf_loss"], curr_episode)
    writer.add_scalar("Losses/Wavelet HF", metrics["wavelet_hf_loss"], curr_episode)
    writer.add_scalar("Losses/Wavelet Total", metrics["wavelet_loss"], curr_episode)
    writer.add_scalar("Losses/Wavelet Weighted", metrics["wavelet_weighted_loss"], curr_episode)
    return metrics


def _write_eval_to_tensor_board(writer: SummaryWriter, eval_summary: dict[str, object], curr_episode: int) -> None:
    writer.add_scalar("Eval/Explored Rate", float(eval_summary["explored_rate"]), curr_episode)
    writer.add_scalar("Eval/Travel Distance", float(eval_summary["travel_dist"]), curr_episode)
    writer.add_scalar("Eval/Success Rate", float(eval_summary["success_rate"]), curr_episode)
    writer.add_scalar("Eval/Episode Return", float(eval_summary["episode_return"]), curr_episode)
    writer.add_scalar("Eval/Steps Taken", float(eval_summary["steps_taken"]), curr_episode)


def main(runtime_config: RuntimeConfig | None = None) -> dict:
    global _interrupted, _interrupt_raised, _saving_interrupt_checkpoint
    _interrupted = False
    _interrupt_raised = False
    _saving_interrupt_checkpoint = False

    runtime_config = runtime_config or RuntimeConfig()
    runtime_config = _resolve_runtime_session(runtime_config)
    runtime_config = apply_runtime_config(runtime_config)
    ensure_output_dirs(runtime_config)

    current_checkpoint_path = None if is_smoke_run(runtime_config) else get_checkpoint_path(runtime_config)
    current_checkpoint_final_path = None if is_smoke_run(runtime_config) else get_checkpoint_final_path(runtime_config)
    current_checkpoint_interrupted_path = None if is_smoke_run(runtime_config) else get_checkpoint_interrupted_path(runtime_config)
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
    evaluated_eval_episodes = get_evaluated_episodes(current_eval_path)

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
                "WPG_RL_RAY_WORKER_NUM_CPUS": str(worker_num_cpus),
                "WPG_RL_WORKER_NUM_THREADS": str(worker_num_threads),
                "MPLCONFIGDIR": os.environ.get("MPLCONFIGDIR", ""),
            }
        },
    }
    if requested_ray_num_cpus is not None:
        ray_init_kwargs["num_cpus"] = requested_ray_num_cpus

    learner_state = create_learner_state(runtime_config, device)
    meta_agents = []
    ray_started = False

    checkpoint_source = None
    if runtime_config.resume_from is not None:
        checkpoint_source = Path(runtime_config.resume_from)
    elif runtime_config.load_model:
        latest = get_latest_checkpoint_path(runtime_config.run_name)
        if latest.exists():
            checkpoint_source = latest
        elif current_checkpoint_path is not None and current_checkpoint_path.exists():
            checkpoint_source = current_checkpoint_path

    curr_episode = 0
    next_episode = 0
    if checkpoint_source is not None:
        curr_episode = load_checkpoint(learner_state, checkpoint_source)
        next_episode = curr_episode
        print(f"loading_model checkpoint={checkpoint_source} resume_episode={curr_episode}")

    learner_state.policy_wrapper = _wrap_data_parallel(learner_state.policy_net, learner_device_ids)
    learner_state.q_net1_wrapper = _wrap_data_parallel(learner_state.q_net1, learner_device_ids)
    learner_state.q_net2_wrapper = _wrap_data_parallel(learner_state.q_net2, learner_device_ids)
    learner_state.target_q_net1_wrapper = _wrap_data_parallel(learner_state.target_q_net1, learner_device_ids)
    learner_state.target_q_net2_wrapper = _wrap_data_parallel(learner_state.target_q_net2, learner_device_ids)

    ray.init(**ray_init_kwargs)
    ray_started = True
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

    RemoteRunner = ray.remote(Runner)
    runner_options = {"num_cpus": worker_num_cpus, "num_gpus": worker_gpu_share}
    meta_agents = [
        RemoteRunner.options(**runner_options).remote(i, worker_runtime_config)
        for i in range(runtime_config.num_meta_agent)
    ]

    weights_set = [_state_dict_to_cpu(learner_state.policy_net)]
    job_list = []
    for meta_agent in meta_agents:
        if next_episode >= runtime_config.max_episodes:
            break
        next_episode += 1
        job_list.append(meta_agent.job.remote(weights_set, next_episode))

    metric_names = ["travel_dist", "success_rate", "explored_rate", "episode_return", "episode_steps"]
    perf_metrics = {name: [] for name in metric_names}
    experience_buffer = [[] for _ in range(len(ROLLOUT_BATCH_KEYS))]
    training_data = []
    latest_train_metrics = {}
    print(
        "training_start "
        f"max_episodes={runtime_config.max_episodes} "
        f"summary_window={runtime_config.summary_window} "
        f"minimum_buffer_size={runtime_config.minimum_buffer_size} "
        f"batch_size={runtime_config.batch_size} "
        f"train_updates_per_iter={runtime_config.train_updates_per_iter} "
        f"use_lf_attention_hf_residual={runtime_config.use_lf_attention_hf_residual} "
        f"use_privileged_wavelet_distillation={runtime_config.use_privileged_wavelet_distillation} "
        f"enable_corridor_graph_compression={runtime_config.enable_corridor_graph_compression} "
        f"enable_corridor_edge_pruning={runtime_config.enable_corridor_edge_pruning} "
        f"enable_smoothness_reward={runtime_config.enable_smoothness_reward} "
        f"wavelet_scales={','.join(str(scale) for scale in runtime_config.wavelet_scales)} "
        f"wavelet_fuse_dim={runtime_config.wavelet_fuse_dim}"
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
                for name in metric_names:
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
                    rollouts = sample_rollout_batch(experience_buffer, sample_batch_size, replace=False)
                    batch = stack_batch_tensors(rollouts, learner_state.device)
                    latest_train_metrics = train_step(batch, learner_state, runtime_config)
                    if perf_metrics["travel_dist"]:
                        latest_train_metrics.update(
                            {
                                "travel_dist": float(np.nanmean(perf_metrics["travel_dist"])),
                                "success_rate": float(np.nanmean(perf_metrics["success_rate"])),
                                "explored_rate": float(np.nanmean(perf_metrics["explored_rate"])),
                                "episode_return": float(np.nanmean(perf_metrics["episode_return"])),
                                "episode_steps": float(np.nanmean(perf_metrics["episode_steps"])),
                            }
                        )
                    training_data.append(latest_train_metrics)

            if len(training_data) >= runtime_config.summary_window:
                latest_train_metrics = write_to_tensor_board(writer, training_data, curr_episode)
                if training_monitor is not None:
                    training_monitor.update_train(curr_episode, latest_train_metrics)
                    training_monitor.update_system(
                        curr_episode,
                        {
                            "buffer_size": len(experience_buffer[0]),
                            "completed_episodes": curr_episode,
                        },
                    )
                for name in metric_names:
                    if perf_metrics[name]:
                        writer.add_scalar(
                            f"Perf/{name.replace('_', ' ').title()}",
                            float(np.nanmean(perf_metrics[name])),
                            curr_episode,
                        )
                        latest_train_metrics[name] = float(np.nanmean(perf_metrics[name]))
                _print_train_progress(
                    curr_episode,
                    runtime_config.max_episodes,
                    latest_train_metrics,
                    len(experience_buffer[0]),
                    time.time() - train_start_time,
                )
                training_data = []
                perf_metrics = {name: [] for name in metric_names}

            weights_set = [_state_dict_to_cpu(learner_state.policy_net)]

            if current_checkpoint_path is not None and curr_episode % max(int(runtime_config.save_model_gap), 1) == 0:
                _save_checkpoint(
                    learner_state,
                    runtime_config,
                    curr_episode,
                    current_checkpoint_path,
                    current_model_path,
                )
                print(
                    "checkpoint_saved "
                    f"episode={curr_episode} "
                    f"path={current_checkpoint_path} "
                    f"bucket={get_bucket_name(curr_episode, runtime_config.result_bucket_episodes)}"
                )

            if runtime_config.enable_auto_eval and runtime_config.auto_eval_map_count > 0:
                for episode_number in sorted(completed_episodes):
                    if episode_number % runtime_config.auto_eval_interval != 0:
                        continue
                    if episode_number in evaluated_eval_episodes:
                        continue

                    eval_results = evaluate_policy(
                        _state_dict_to_cpu(learner_state.policy_net),
                        runtime_config,
                        episode_number=episode_number,
                        device=learner_state.device,
                        greedy=runtime_config.auto_eval_greedy,
                        max_episode_step=runtime_config.max_episode_step,
                    )
                    eval_summary = summarize_eval_results(eval_results)
                    save_evaluation_summary(
                        current_eval_path,
                        episode_number,
                        eval_results,
                        eval_summary,
                        runtime_config.auto_eval_interval,
                    )
                    _write_eval_to_tensor_board(writer, eval_summary, episode_number)
                    if training_monitor is not None:
                        training_monitor.update_eval(episode_number, eval_summary)
                    evaluated_eval_episodes.add(episode_number)
                    print(
                        "auto_eval "
                        f"episode={episode_number} "
                        f"maps={int(eval_summary['evaluated_maps'])} "
                        f"explored_rate={float(eval_summary['explored_rate']):.4f} "
                        f"travel_dist={float(eval_summary['travel_dist']):.2f} "
                        f"success_rate={float(eval_summary['success_rate']):.4f} "
                        f"episode_return={float(eval_summary['episode_return']):.4f} "
                        f"steps_taken={float(eval_summary['steps_taken']):.2f}"
                    )

        if current_checkpoint_path is not None and current_model_path is not None:
            final_episode = max(curr_episode, 1)
            _save_checkpoint(
                learner_state,
                runtime_config,
                final_episode,
                current_checkpoint_path,
                current_model_path,
            )
            torch.save(torch.load(current_checkpoint_path, map_location="cpu", weights_only=False), current_checkpoint_final_path)
            print(f"final_model_saved checkpoint={current_checkpoint_path} final={current_checkpoint_final_path} episode={final_episode}")
    except KeyboardInterrupt:
        saved = False
        try:
            if current_checkpoint_path is not None and current_checkpoint_interrupted_path is not None and current_model_path is not None:
                _saving_interrupt_checkpoint = True
                _save_interrupt_checkpoint(
                    learner_state,
                    runtime_config,
                    max(curr_episode, 1),
                    current_checkpoint_path,
                    current_checkpoint_interrupted_path,
                    current_model_path,
                )
                saved = True
        except Exception as exc:  # pragma: no cover - best effort fallback on forced interrupts
            if current_checkpoint_path is not None and current_checkpoint_interrupted_path is not None and current_checkpoint_path.exists():
                try:
                    shutil.copy2(current_checkpoint_path, current_checkpoint_interrupted_path)
                    print(
                        f"interrupt_checkpoint_saved path={current_checkpoint_interrupted_path} "
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
        if ray_started:
            ray.shutdown()

    return {
        "checkpoint_path": str(current_checkpoint_path) if current_checkpoint_path is not None else None,
        "checkpoint_final_path": (
            str(current_checkpoint_final_path) if current_checkpoint_final_path is not None else None
        ),
        "checkpoint_interrupted_path": (
            str(current_checkpoint_interrupted_path) if current_checkpoint_interrupted_path is not None else None
        ),
        "model_dir": str(current_model_path) if current_model_path is not None else None,
        "result_eval_dir": str(current_eval_path),
        "result_gif_dir": str(current_gifs_path),
        "train_dir": str(current_train_path),
        "gif_dir": str(current_gifs_path),
        "gifs_dir": str(current_gifs_path),
        "eval_dir": str(current_eval_path),
        "episode": curr_episode,
        "completed_episodes": curr_episode,
    }
