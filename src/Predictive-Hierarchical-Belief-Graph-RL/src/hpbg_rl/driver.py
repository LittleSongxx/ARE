from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import copy
import json
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

from .belief_losses import BeliefDistillationLoss
from .evaluation import evaluate_policy, get_evaluated_episodes, save_evaluation_summary, summarize_eval_results
from .map_splits import materialize_split_manifest, save_split_manifest
from .model import PolicyNet, QNet
from .parameter import (
    CRITIC_NODE_INPUT_DIM,
    EMBEDDING_DIM,
    GAMMA,
    HPBG_CRITIC_PRIVILEGED_DIM,
    K_SIZE,
    LR,
    NODE_INPUT_DIM,
    POLICY_GRAD_CLIP,
    Q_GRAD_CLIP,
    SENSOR_RANGE,
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
    get_protocol_path,
    get_result_eval_path,
    get_result_test_eval_path,
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
RAY_WAIT_HEARTBEAT_SEC = 10.0


def _log_value(value) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, default=str)
    if value is None:
        return "<none>"
    return str(value)


def _log_event(event: str, **fields) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    payload = " ".join(f"{key}={_log_value(value)}" for key, value in fields.items())
    if payload:
        print(f"[{timestamp}] {event} {payload}", flush=True)
    else:
        print(f"[{timestamp}] {event}", flush=True)


def _wait_for_refs_with_heartbeat(refs, event: str, heartbeat_sec: float = RAY_WAIT_HEARTBEAT_SEC) -> list:
    pending = list(refs)
    total = len(pending)
    results = []
    start_time = time.time()
    while pending:
        ready_refs, pending = ray.wait(pending, num_returns=1, timeout=heartbeat_sec)
        elapsed_sec = time.time() - start_time
        if not ready_refs:
            _log_event(
                f"{event}_waiting",
                ready=len(results),
                pending=len(pending),
                total=total,
                elapsed_sec=elapsed_sec,
            )
            continue
        results.extend(ray.get(ready_refs))
        _log_event(
            f"{event}_ready",
            ready=len(results),
            pending=len(pending),
            total=total,
            elapsed_sec=elapsed_sec,
        )
    return results

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
    belief_distillation: BeliefDistillationLoss
    device: torch.device
    update_step: int = 0
    target_q_update_counter: int = 1


def _policy_kwargs(runtime_config: RuntimeConfig) -> dict:
    return {
        "use_lf_attention_hf_residual": runtime_config.use_lf_attention_hf_residual,
        "use_privileged_wavelet_distillation": runtime_config.use_hpbg
        and runtime_config.use_privileged_wavelet_distillation,
        "use_hierarchical_context": runtime_config.use_hpbg and runtime_config.use_hierarchical_graph,
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
        f"raw_return={_format_metric(train_metrics.get('episode_raw_return'))} "
        f"expert_delta={_format_metric(train_metrics.get('expert_reward_delta'))} "
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


def _checkpoint_dict(learner_state: LearnerState, completed_episodes: int, runtime_config: RuntimeConfig | None = None) -> dict:
    payload = {
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
    if runtime_config is not None:
        payload["runtime_config"] = asdict(runtime_config)
    return payload


def _save_checkpoint(
    learner_state: LearnerState,
    runtime_config: RuntimeConfig,
    completed_episodes: int,
    checkpoint_path: Path,
    bucket_model_dir: Path,
):
    payload = _checkpoint_dict(learner_state, completed_episodes, runtime_config)
    torch.save(payload, checkpoint_path)
    bucket_dir = ensure_bucket_dir(bucket_model_dir, completed_episodes, runtime_config.result_bucket_episodes)
    torch.save(payload, bucket_dir / "checkpoint.pth")


def _save_named_checkpoint(
    learner_state: LearnerState,
    runtime_config: RuntimeConfig,
    completed_episodes: int,
    checkpoint_path: Path,
    extra: dict[str, object] | None = None,
) -> None:
    payload = _checkpoint_dict(learner_state, completed_episodes, runtime_config)
    if extra:
        payload.update(extra)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)


def _validation_score(eval_summary: dict[str, object]) -> float:
    explored = float(eval_summary.get("explored_rate") or 0.0)
    success = float(eval_summary.get("success_rate") or 0.0)
    efficiency = float(eval_summary.get("normalized_exploration_efficiency") or 0.0)
    auc = float(eval_summary.get("exploration_auc") or 0.0)
    return explored + 0.5 * success + 0.25 * auc + 0.05 * efficiency


def _evaluation_protocol(runtime_config: RuntimeConfig, split: str, manifest_hash: str | None = None) -> dict[str, object]:
    return {
        "split": split,
        "manifest_hash": manifest_hash,
        "step_budget": int(runtime_config.max_episode_step),
        "budget_mode": runtime_config.eval_budget_mode,
        "greedy": bool(runtime_config.auto_eval_greedy),
        "sensor_range": float(SENSOR_RANGE),
        "split_manifest_path": runtime_config.split_manifest_path,
    }


def _clean_protocol(protocol: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in protocol.items() if value is not None}


def _materialize_run_protocol(runtime_config: RuntimeConfig) -> tuple[RuntimeConfig, str, Path]:
    protocol_dir = get_protocol_path(runtime_config)
    protocol_dir.mkdir(parents=True, exist_ok=True)
    manifest = materialize_split_manifest(runtime_config)
    manifest_path = save_split_manifest(manifest, protocol_dir / "split_manifest.json")
    manifest_hash = manifest.content_hash()
    split_counts = {split: len(manifest.split_entries(split)) for split in ("train", "val", "test")}
    _log_event(
        "split_manifest_materialized",
        manifest_path=manifest_path,
        manifest_hash=manifest_hash,
        train_maps=split_counts["train"],
        val_maps=split_counts["val"],
        test_maps=split_counts["test"],
        generated_from=manifest.generated_from,
    )

    runtime_config = runtime_config.with_overrides(split_manifest_path=str(manifest_path))
    runtime_config = apply_runtime_config(runtime_config)
    protocol_payload = {
        "manifest_path": str(manifest_path),
        "manifest_hash": manifest_hash,
        "split_counts": split_counts,
        "generated_from": manifest.generated_from,
        "step_budget": int(runtime_config.max_episode_step),
        "budget_mode": runtime_config.eval_budget_mode,
        "validation_map_count": runtime_config.val_map_count,
        "test_map_count": runtime_config.test_map_count,
        "sensor_range": float(SENSOR_RANGE),
        "runtime_config": asdict(runtime_config),
    }
    (protocol_dir / "run_protocol.json").write_text(json.dumps(protocol_payload, indent=2, sort_keys=True))
    return runtime_config, manifest_hash, manifest_path


def _save_final_test_report(
    output_dir: Path,
    episode_number: int,
    results: list[dict[str, object]],
    summary: dict[str, object],
    protocol: dict[str, object],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "final_test_report.json"
    report_path.write_text(
        json.dumps(
            {
                "episode": int(episode_number),
                "protocol": protocol,
                "summary": summary,
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return report_path


def _checkpoint_policy_state(checkpoint_path: Path) -> dict | None:
    if not checkpoint_path.is_file():
        return None
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return checkpoint.get("policy_model")


def _save_interrupt_checkpoint(
    learner_state: LearnerState,
    runtime_config: RuntimeConfig,
    completed_episodes: int,
    checkpoint_path: Path,
    interrupt_path: Path,
    bucket_model_dir: Path,
):
    payload = _checkpoint_dict(learner_state, completed_episodes, runtime_config)
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
        belief_distillation=BeliefDistillationLoss(
            runtime_config.hpbg_belief_distill_weight,
            runtime_config.hpbg_belief_distill_warmup_updates,
            runtime_config.hpbg_belief_distill_ramp_updates,
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


def _default_belief_metrics(device: torch.device) -> dict:
    zero = torch.tensor(0.0, device=device)
    return {
        "loss": zero,
        "weighted_loss": zero,
        "lambda_eff": 0.0,
        "explored_loss": zero,
        "oracle_loss": zero,
        "potential_loss": zero,
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

    use_actor_hidden = runtime_config.use_hpbg and (
        runtime_config.use_privileged_wavelet_distillation or runtime_config.use_belief_distillation
    )
    if use_actor_hidden:
        logp, actor_hidden = learner_state.policy_wrapper(*observation, return_hidden=True)
    else:
        logp = learner_state.policy_wrapper(*observation)
        actor_hidden = None

    policy_loss_original = torch.sum(
        logp.exp().unsqueeze(2) * (learner_state.log_alpha.exp().detach() * logp.unsqueeze(2) - q_values.detach()),
        dim=1,
    ).mean()

    wavelet_metrics = _default_wavelet_metrics(learner_state.device)
    belief_metrics = _default_belief_metrics(learner_state.device)
    overlap_valid_mask = None
    if use_actor_hidden:
        overlap_valid_mask = build_overlap_valid_mask(batch["actor_to_critic_index"], batch["node_padding_mask"])
        critic_valid_full = (~batch["critic_node_padding_mask"].squeeze(1).bool()).to(dtype=actor_hidden.dtype)
        critic_valid_overlap = gather_by_index(
            critic_valid_full.unsqueeze(-1),
            batch["actor_to_critic_index"],
        ).squeeze(-1) > 0
        overlap_valid_mask = overlap_valid_mask & critic_valid_overlap

    if runtime_config.use_hpbg and runtime_config.use_privileged_wavelet_distillation:
        actor_lf, actor_hf, *_ = decompose_graph_hidden(
            actor_hidden,
            batch["edge_mask"],
            batch["node_padding_mask"],
            scales=runtime_config.wavelet_scales,
        )

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

        wavelet_metrics = learner_state.wavelet_distillation(
            actor_lf,
            actor_hf,
            critic_lf,
            critic_hf,
            overlap_valid_mask,
            learner_state.update_step,
        )

    if runtime_config.use_hpbg and runtime_config.use_belief_distillation:
        privileged_target_full = batch["critic_node_inputs"][..., -HPBG_CRITIC_PRIVILEGED_DIM:].detach()
        privileged_target = gather_by_index(privileged_target_full, batch["actor_to_critic_index"])
        belief_prediction = learner_state.policy_net.predict_belief_targets(actor_hidden)
        belief_metrics = learner_state.belief_distillation(
            belief_prediction,
            privileged_target,
            overlap_valid_mask,
            learner_state.update_step,
        )

    policy_loss = policy_loss_original + wavelet_metrics["weighted_loss"] + belief_metrics["weighted_loss"]
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
    q1_td_error = q1 - target_q.detach()
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
    q2_td_error = q2 - target_q.detach()
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
    td_error = 0.5 * (q1_td_error.detach() + q2_td_error.detach())
    policy_q_grad_ratio = float(policy_grad_norm) / max(float(max(q1_grad_norm, q2_grad_norm)), 1e-12)
    return {
        "reward": batch["reward"].mean().item(),
        "value": value_prime.mean().item(),
        "policy_loss": policy_loss.item(),
        "policy_loss_original": policy_loss_original.item(),
        "q_value_loss": 0.5 * (q1_loss.item() + q2_loss.item()),
        "td_error_mean": float(td_error.mean().item()),
        "td_error_std": float(td_error.std(unbiased=False).item()),
        "entropy": entropy.mean().item(),
        "policy_grad_norm": float(policy_grad_norm),
        "q_value_grad_norm": float(max(q1_grad_norm, q2_grad_norm)),
        "policy_q_grad_ratio": policy_q_grad_ratio,
        "update_step": float(learner_state.update_step),
        "log_alpha": learner_state.log_alpha.item(),
        "alpha_loss": alpha_loss.item(),
        "wavelet_lf_loss": float(wavelet_metrics["loss_lf"]),
        "wavelet_hf_loss": float(wavelet_metrics["loss_hf"]),
        "wavelet_loss": float(wavelet_metrics["loss"]),
        "wavelet_weighted_loss": float(wavelet_metrics["weighted_loss"]),
        "wavelet_lambda_eff": float(wavelet_metrics["lambda_eff"]),
        "belief_loss": float(belief_metrics["loss"]),
        "belief_weighted_loss": float(belief_metrics["weighted_loss"]),
        "belief_lambda_eff": float(belief_metrics["lambda_eff"]),
        "belief_explored_loss": float(belief_metrics["explored_loss"]),
        "belief_oracle_loss": float(belief_metrics["oracle_loss"]),
        "belief_potential_loss": float(belief_metrics["potential_loss"]),
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
    writer.add_scalar("Losses/TD Error Mean", metrics["td_error_mean"], curr_episode)
    writer.add_scalar("Losses/TD Error Std", metrics["td_error_std"], curr_episode)
    writer.add_scalar("Losses/Entropy", metrics["entropy"], curr_episode)
    writer.add_scalar("Losses/Policy Grad Norm", metrics["policy_grad_norm"], curr_episode)
    writer.add_scalar("Losses/Q Value Grad Norm", metrics["q_value_grad_norm"], curr_episode)
    writer.add_scalar("Losses/Policy Q Grad Ratio", metrics["policy_q_grad_ratio"], curr_episode)
    writer.add_scalar("Losses/Log Alpha", metrics["log_alpha"], curr_episode)
    writer.add_scalar("Losses/Wavelet LF", metrics["wavelet_lf_loss"], curr_episode)
    writer.add_scalar("Losses/Wavelet HF", metrics["wavelet_hf_loss"], curr_episode)
    writer.add_scalar("Losses/Wavelet Total", metrics["wavelet_loss"], curr_episode)
    writer.add_scalar("Losses/Wavelet Weighted", metrics["wavelet_weighted_loss"], curr_episode)
    writer.add_scalar("Losses/Belief Distill", metrics["belief_loss"], curr_episode)
    writer.add_scalar("Losses/Belief Distill Weighted", metrics["belief_weighted_loss"], curr_episode)
    writer.add_scalar("Losses/Belief Distill Explored", metrics["belief_explored_loss"], curr_episode)
    writer.add_scalar("Losses/Belief Distill Oracle", metrics["belief_oracle_loss"], curr_episode)
    writer.add_scalar("Losses/Belief Distill Potential", metrics["belief_potential_loss"], curr_episode)
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
    _log_event("startup_begin", pid=os.getpid(), cwd=os.getcwd())
    runtime_config = _resolve_runtime_session(runtime_config)
    runtime_config = apply_runtime_config(runtime_config)
    _log_event(
        "runtime_config_resolved",
        run_name=runtime_config.run_name,
        run_session=runtime_config.run_session,
        max_episodes=runtime_config.max_episodes,
        max_episode_step=runtime_config.max_episode_step,
        num_meta_agent=runtime_config.num_meta_agent,
        minimum_buffer_size=runtime_config.minimum_buffer_size,
        batch_size=runtime_config.batch_size,
        replay_size=runtime_config.replay_size,
        train_updates_per_iter=runtime_config.train_updates_per_iter,
        use_hpbg=runtime_config.use_hpbg,
        use_belief_state=runtime_config.use_belief_state,
        use_map_prediction=runtime_config.use_map_prediction,
        use_hierarchical_graph=runtime_config.use_hierarchical_graph,
        use_expert_reward=runtime_config.use_expert_reward,
        use_belief_distillation=runtime_config.use_belief_distillation,
        enable_auto_eval=runtime_config.enable_auto_eval,
        auto_eval_interval=runtime_config.auto_eval_interval,
        val_map_count=runtime_config.val_map_count,
        test_map_count=runtime_config.test_map_count,
        run_final_test=runtime_config.run_final_test,
    )
    _log_event("output_dirs_prepare_start")
    ensure_output_dirs(runtime_config)
    _log_event("output_dirs_prepare_done", result_session_root=runtime_config.run_session)
    _log_event(
        "split_manifest_prepare_start",
        maps_dir=runtime_config.maps_dir,
        train_maps_dir=runtime_config.train_maps_dir,
        val_maps_dir=runtime_config.val_maps_dir,
        test_maps_dir=runtime_config.test_maps_dir,
        split_manifest_path=runtime_config.split_manifest_path,
        split_seed=runtime_config.split_seed,
    )
    runtime_config, split_manifest_hash, split_manifest_path = _materialize_run_protocol(runtime_config)
    _log_event("run_protocol_written", manifest_path=split_manifest_path, manifest_hash=split_manifest_hash)
    ensure_output_dirs(runtime_config)

    current_checkpoint_path = None if is_smoke_run(runtime_config) else get_checkpoint_path(runtime_config)
    current_checkpoint_final_path = None if is_smoke_run(runtime_config) else get_checkpoint_final_path(runtime_config)
    current_checkpoint_interrupted_path = None if is_smoke_run(runtime_config) else get_checkpoint_interrupted_path(runtime_config)
    current_model_path = None if is_smoke_run(runtime_config) else get_model_path(runtime_config)
    current_eval_path = get_result_eval_path(runtime_config)
    current_test_eval_path = get_result_test_eval_path(runtime_config)
    current_train_path = get_train_path(runtime_config)
    current_gifs_path = get_gifs_path(runtime_config)
    train_start_time = time.time()

    _log_event("learner_device_resolve_start", use_gpu_global=runtime_config.use_gpu_global)
    device = _resolve_learner_device(runtime_config)
    _log_event("learner_device_resolved", learner_device=device.type, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES") or "<all-visible>")

    _log_event("tensorboard_writer_start", train_path=current_train_path)
    writer = SummaryWriter(current_train_path)
    _log_event("tensorboard_writer_ready", train_path=current_train_path)
    training_monitor = None
    if runtime_config.enable_training_monitor:
        _log_event("training_monitor_start", monitor_path=get_monitor_path(runtime_config))
        training_monitor = TrainingMonitor(
            get_monitor_path(runtime_config),
            window_size=runtime_config.monitor_window,
            snapshot_interval=runtime_config.monitor_snapshot_interval,
        )
        _log_event(
            "training_monitor_ready",
            monitor_path=get_monitor_path(runtime_config),
            monitor_window=runtime_config.monitor_window,
            snapshot_interval=runtime_config.monitor_snapshot_interval,
        )
    evaluated_eval_episodes = get_evaluated_episodes(current_eval_path)
    _log_event("existing_validation_evals_loaded", eval_path=current_eval_path, evaluated_episodes=len(evaluated_eval_episodes))

    pythonpath = _ensure_pythonpath_entry(str(SRC_ROOT))
    requested_ray_num_cpus = resolve_ray_num_cpus(runtime_config)
    worker_num_cpus = resolve_ray_worker_num_cpus(runtime_config)
    worker_num_threads = resolve_worker_num_threads(runtime_config, worker_num_cpus)
    configure_worker_process_threads(worker_num_threads)

    learner_gpu_count = _resolve_learner_gpu_count(runtime_config, device)
    learner_device_ids = list(range(learner_gpu_count))
    worker_runtime_config, worker_gpu_share = _resolve_worker_runtime(runtime_config)
    _log_event(
        "resource_config_resolved",
        ray_num_cpus=requested_ray_num_cpus if requested_ray_num_cpus is not None else "auto",
        worker_num_cpus=worker_num_cpus,
        worker_num_threads=worker_num_threads,
        learner_device=device.type,
        learner_gpu_count=learner_gpu_count,
        learner_device_ids=learner_device_ids,
        worker_gpu_request=worker_gpu_share,
        pythonpath=pythonpath,
    )
    ray_init_kwargs = {
        "ignore_reinit_error": True,
        "runtime_env": {
            "env_vars": {
                "PYTHONPATH": pythonpath,
                "HPBG_RL_RAY_WORKER_NUM_CPUS": str(worker_num_cpus),
                "HPBG_RL_WORKER_NUM_THREADS": str(worker_num_threads),
                "MPLCONFIGDIR": os.environ.get("MPLCONFIGDIR", ""),
            }
        },
    }
    if requested_ray_num_cpus is not None:
        ray_init_kwargs["num_cpus"] = requested_ray_num_cpus

    _log_event("learner_create_start", node_input_dim=NODE_INPUT_DIM, critic_node_input_dim=CRITIC_NODE_INPUT_DIM, embedding_dim=EMBEDDING_DIM)
    learner_state = create_learner_state(runtime_config, device)
    _log_event("learner_create_done", learner_device=learner_state.device.type)
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
        _log_event("checkpoint_load_start", checkpoint=checkpoint_source)
        curr_episode = load_checkpoint(learner_state, checkpoint_source)
        next_episode = curr_episode
        _log_event("checkpoint_load_done", checkpoint=checkpoint_source, resume_episode=curr_episode)

    _log_event("learner_wrapper_start", learner_device_ids=learner_device_ids)
    learner_state.policy_wrapper = _wrap_data_parallel(learner_state.policy_net, learner_device_ids)
    learner_state.q_net1_wrapper = _wrap_data_parallel(learner_state.q_net1, learner_device_ids)
    learner_state.q_net2_wrapper = _wrap_data_parallel(learner_state.q_net2, learner_device_ids)
    learner_state.target_q_net1_wrapper = _wrap_data_parallel(learner_state.target_q_net1, learner_device_ids)
    learner_state.target_q_net2_wrapper = _wrap_data_parallel(learner_state.target_q_net2, learner_device_ids)
    _log_event("learner_wrapper_done", data_parallel=len(learner_device_ids) > 1)

    _log_event("ray_init_start", ray_init_kwargs={key: value for key, value in ray_init_kwargs.items() if key != "runtime_env"})
    ray.init(**ray_init_kwargs)
    ray_started = True
    cluster_resources = ray.cluster_resources()
    cluster_cpus = int(cluster_resources.get("CPU", 0))
    cluster_gpus = float(cluster_resources.get("GPU", 0.0))
    _log_event("ray_init_done", cluster_resources=cluster_resources, cluster_cpus=cluster_cpus, cluster_gpus=cluster_gpus)
    _log_event(
        "artifact_paths",
        result_root=runtime_config.run_session,
        protocol_manifest=split_manifest_path,
        model_path=current_model_path,
        train_path=current_train_path,
        gifs_path=current_gifs_path,
    )
    _log_event(
        "role_split",
        collector_device="cpu",
        learner_device=device.type,
        learner_num_gpus=learner_gpu_count,
        visible_cuda_devices=os.environ.get("CUDA_VISIBLE_DEVICES") or "<all-visible>",
        worker_gpu_request=worker_gpu_share,
    )

    old_sigint = signal.signal(signal.SIGINT, _signal_handler)
    old_sigterm = signal.signal(signal.SIGTERM, _signal_handler)

    RemoteRunner = ray.remote(Runner)
    runner_options = {"num_cpus": worker_num_cpus, "num_gpus": worker_gpu_share}
    _log_event(
        "ray_actor_create_start",
        workers=runtime_config.num_meta_agent,
        runner_options=runner_options,
        max_episode_step=runtime_config.max_episode_step,
    )
    meta_agents = [
        RemoteRunner.options(**runner_options).remote(i, worker_runtime_config)
        for i in range(runtime_config.num_meta_agent)
    ]
    _log_event("ray_actor_create_submitted", workers=len(meta_agents))
    actor_ready_refs = [meta_agent.healthcheck.remote() for meta_agent in meta_agents]
    actor_health = _wait_for_refs_with_heartbeat(actor_ready_refs, "ray_actor_ready")
    _log_event("ray_actor_create_done", workers=len(meta_agents), actor_health=actor_health)

    weights_set = [_state_dict_to_cpu(learner_state.policy_net)]
    job_list = []
    _log_event("initial_episode_submit_start", start_episode=next_episode + 1, worker_slots=len(meta_agents))
    for meta_agent in meta_agents:
        if next_episode >= runtime_config.max_episodes:
            break
        next_episode += 1
        job_list.append(meta_agent.job.remote(weights_set, next_episode))
    _log_event("initial_episode_submit_done", submitted=len(job_list), next_episode=next_episode)

    metric_names = [
        "travel_dist",
        "success_rate",
        "explored_rate",
        "episode_return",
        "episode_raw_return",
        "expert_reward_delta",
        "episode_steps",
    ]
    perf_metrics = {name: [] for name in metric_names}
    experience_buffer = [[] for _ in range(len(ROLLOUT_BATCH_KEYS))]
    training_data = []
    latest_train_metrics = {}
    best_validation_score = float("-inf")
    best_validation_episode = None
    best_validation_summary = None
    final_test_summary = None
    final_test_report_path = None
    _log_event(
        "training_start",
        max_episodes=runtime_config.max_episodes,
        summary_window=runtime_config.summary_window,
        minimum_buffer_size=runtime_config.minimum_buffer_size,
        batch_size=runtime_config.batch_size,
        replay_size=runtime_config.replay_size,
        train_updates_per_iter=runtime_config.train_updates_per_iter,
        active_jobs=len(job_list),
        use_lf_attention_hf_residual=runtime_config.use_lf_attention_hf_residual,
        use_privileged_wavelet_distillation=runtime_config.use_privileged_wavelet_distillation,
        enable_corridor_graph_compression=runtime_config.enable_corridor_graph_compression,
        enable_corridor_edge_pruning=runtime_config.enable_corridor_edge_pruning,
        enable_smoothness_reward=runtime_config.enable_smoothness_reward,
        wavelet_scales=runtime_config.wavelet_scales,
        wavelet_fuse_dim=runtime_config.wavelet_fuse_dim,
    )

    try:
        last_wait_log = 0.0
        training_updates_started = False
        while job_list:
            done_id, job_list = ray.wait(job_list, timeout=RAY_WAIT_HEARTBEAT_SEC)
            if not done_id:
                elapsed_sec = time.time() - train_start_time
                now = time.time()
                if now - last_wait_log >= RAY_WAIT_HEARTBEAT_SEC:
                    _log_event(
                        "episode_waiting",
                        completed_episodes=curr_episode,
                        submitted_until=next_episode,
                        pending_jobs=len(job_list),
                        replay_size=len(experience_buffer[0]),
                        training_updates=learner_state.update_step,
                        elapsed_sec=elapsed_sec,
                    )
                    last_wait_log = now
                continue
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
            _log_event(
                "episode_complete",
                episode=f"{curr_episode}/{runtime_config.max_episodes}",
                meta_agent=done_jobs[-1][2]["id"],
                completed_batch=len(done_jobs),
                pending_jobs=len(job_list),
                buffer=len(experience_buffer[0]),
                travel_dist=_format_metric(latest_metrics["travel_dist"]),
                explored_rate=_format_metric(latest_metrics["explored_rate"]),
                success_rate=_format_metric(latest_metrics["success_rate"]),
                episode_return=_format_metric(latest_metrics["episode_return"]),
                episode_raw_return=_format_metric(latest_metrics["episode_raw_return"]),
                expert_reward_delta=_format_metric(latest_metrics["expert_reward_delta"]),
                episode_steps=_format_metric(latest_metrics["episode_steps"]),
            )

            if next_episode < runtime_config.max_episodes:
                next_episode += 1
                job_list.append(meta_agents[info["id"]].job.remote(weights_set, next_episode))
                _log_event(
                    "episode_submit",
                    episode=next_episode,
                    meta_agent=info["id"],
                    pending_jobs=len(job_list),
                )

            if len(experience_buffer[0]) >= runtime_config.minimum_buffer_size:
                if not training_updates_started:
                    _log_event(
                        "learner_update_start",
                        episode=curr_episode,
                        replay_size=len(experience_buffer[0]),
                        minimum_buffer_size=runtime_config.minimum_buffer_size,
                        batch_size=runtime_config.batch_size,
                        train_updates_per_iter=runtime_config.train_updates_per_iter,
                    )
                    training_updates_started = True
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
                                "episode_raw_return": float(np.nanmean(perf_metrics["episode_raw_return"])),
                                "expert_reward_delta": float(np.nanmean(perf_metrics["expert_reward_delta"])),
                                "episode_steps": float(np.nanmean(perf_metrics["episode_steps"])),
                            }
                        )
                    training_data.append(latest_train_metrics)
            elif not training_updates_started:
                _log_event(
                    "learner_warmup_waiting",
                    episode=curr_episode,
                    replay_size=len(experience_buffer[0]),
                    minimum_buffer_size=runtime_config.minimum_buffer_size,
                    remaining=max(runtime_config.minimum_buffer_size - len(experience_buffer[0]), 0),
                )

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
                if "episode_raw_return" in latest_train_metrics:
                    writer.add_scalar("Perf/Episode Raw Return", latest_train_metrics["episode_raw_return"], curr_episode)
                if "expert_reward_delta" in latest_train_metrics:
                    writer.add_scalar("Perf/Expert Reward Delta", latest_train_metrics["expert_reward_delta"], curr_episode)
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

            if runtime_config.enable_auto_eval and runtime_config.val_map_count and runtime_config.val_map_count > 0:
                for episode_number in sorted(completed_episodes):
                    if episode_number % runtime_config.auto_eval_interval != 0:
                        continue
                    if episode_number in evaluated_eval_episodes:
                        continue

                    validation_results = evaluate_policy(
                        _state_dict_to_cpu(learner_state.policy_net),
                        runtime_config,
                        episode_number=episode_number,
                        device=learner_state.device,
                        greedy=runtime_config.auto_eval_greedy,
                        max_episode_step=runtime_config.max_episode_step,
                        split="val",
                        map_count=runtime_config.val_map_count,
                    )
                    validation_summary = summarize_eval_results(validation_results)
                    validation_score = _validation_score(validation_summary)
                    validation_summary["validation_score"] = validation_score
                    if latest_train_metrics:
                        train_explored = float(latest_train_metrics.get("explored_rate", 0.0))
                        validation_summary["generalization_gap_explored_rate"] = train_explored - float(
                            validation_summary.get("explored_rate", 0.0)
                        )
                    validation_protocol = _clean_protocol(
                        _evaluation_protocol(runtime_config, "val", split_manifest_hash)
                        | {"checkpoint_episode": int(episode_number), "validation_score": validation_score}
                    )
                    save_evaluation_summary(
                        current_eval_path,
                        episode_number,
                        validation_results,
                        validation_summary,
                        runtime_config.auto_eval_interval,
                        protocol=validation_protocol,
                    )
                    writer.add_scalar("Validation/Composite Score", validation_score, episode_number)
                    _write_eval_to_tensor_board(writer, validation_summary, episode_number)
                    if training_monitor is not None:
                        training_monitor.update_eval(episode_number, validation_summary)
                    if current_model_path is not None and validation_score > best_validation_score:
                        best_validation_score = validation_score
                        best_validation_episode = episode_number
                        best_validation_summary = dict(validation_summary)
                        _save_named_checkpoint(
                            learner_state,
                            runtime_config,
                            episode_number,
                            current_model_path / "best_val_checkpoint.pth",
                            extra={
                                "validation_score": validation_score,
                                "validation_summary": best_validation_summary,
                                "split_manifest_hash": split_manifest_hash,
                                "split_manifest_path": str(split_manifest_path),
                            },
                        )
                        print(
                            "best_validation_checkpoint_saved "
                            f"episode={episode_number} "
                            f"score={validation_score:.6f} "
                            f"path={current_model_path / 'best_val_checkpoint.pth'}"
                        )
                    evaluated_eval_episodes.add(episode_number)
                    print(
                        "validation_eval "
                        f"episode={episode_number} "
                        f"maps={int(validation_summary['evaluated_maps'])} "
                        f"score={validation_score:.6f} "
                        f"explored_rate={float(validation_summary['explored_rate']):.4f} "
                        f"travel_dist={float(validation_summary['travel_dist']):.2f} "
                        f"success_rate={float(validation_summary['success_rate']):.4f} "
                        f"episode_return={float(validation_summary['episode_return']):.4f} "
                        f"steps_taken={float(validation_summary['steps_taken']):.2f}"
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

            if runtime_config.run_final_test and runtime_config.test_map_count and runtime_config.test_map_count > 0:
                best_checkpoint_path = current_model_path / "best_val_checkpoint.pth"
                policy_state = _checkpoint_policy_state(best_checkpoint_path)
                checkpoint_for_test = best_checkpoint_path
                if policy_state is None:
                    policy_state = _state_dict_to_cpu(learner_state.policy_net)
                    checkpoint_for_test = current_checkpoint_final_path

                final_test_results = evaluate_policy(
                    policy_state,
                    runtime_config,
                    episode_number=final_episode,
                    device=learner_state.device,
                    greedy=runtime_config.auto_eval_greedy,
                    max_episode_step=runtime_config.max_episode_step,
                    split="test",
                    map_count=runtime_config.test_map_count,
                )
                final_test_summary = summarize_eval_results(final_test_results)
                final_test_protocol = _clean_protocol(
                    _evaluation_protocol(runtime_config, "test", split_manifest_hash)
                    | {
                        "checkpoint_episode": int(final_episode),
                        "checkpoint_path": str(checkpoint_for_test),
                        "selection": "best_validation" if checkpoint_for_test == best_checkpoint_path else "final_checkpoint",
                        "best_validation_episode": best_validation_episode,
                        "best_validation_score": best_validation_score if np.isfinite(best_validation_score) else None,
                    }
                )
                save_evaluation_summary(
                    current_test_eval_path,
                    final_episode,
                    final_test_results,
                    final_test_summary,
                    runtime_config.auto_eval_interval,
                    protocol=final_test_protocol,
                )
                final_test_report_path = _save_final_test_report(
                    current_test_eval_path,
                    final_episode,
                    final_test_results,
                    final_test_summary,
                    final_test_protocol,
                )
                print(
                    "final_test_eval "
                    f"episode={final_episode} "
                    f"maps={int(final_test_summary['evaluated_maps'])} "
                    f"explored_rate={float(final_test_summary['explored_rate']):.4f} "
                    f"travel_dist={float(final_test_summary['travel_dist']):.2f} "
                    f"success_rate={float(final_test_summary['success_rate']):.4f} "
                    f"report={final_test_report_path}"
                )
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
        "result_test_eval_dir": str(current_test_eval_path),
        "result_gif_dir": str(current_gifs_path),
        "train_dir": str(current_train_path),
        "gif_dir": str(current_gifs_path),
        "gifs_dir": str(current_gifs_path),
        "eval_dir": str(current_eval_path),
        "test_eval_dir": str(current_test_eval_path),
        "split_manifest_path": str(split_manifest_path),
        "split_manifest_hash": split_manifest_hash,
        "best_validation_episode": best_validation_episode,
        "best_validation_score": best_validation_score if np.isfinite(best_validation_score) else None,
        "best_validation_summary": best_validation_summary,
        "final_test_summary": final_test_summary,
        "final_test_report_path": str(final_test_report_path) if final_test_report_path is not None else None,
        "episode": curr_episode,
        "completed_episodes": curr_episode,
    }
