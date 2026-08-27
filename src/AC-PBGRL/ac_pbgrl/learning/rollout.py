from __future__ import annotations

from dataclasses import replace
import time
from typing import Callable, Optional

import numpy as np
import torch

from ac_pbgrl.config import Config
from ac_pbgrl.models.temporal import (
    AdaptivePotentialKF,
    EMAPotentialMemory,
    GRUPotentialMemory,
    NoPotentialMemory,
)
from ac_pbgrl.state import ExplorationState, TransitionBatch

from .future_gain import FutureGainLabeler
from .teacher import QTeacher


def build_temporal_memory(config: Config):
    mode = str(config.method.temporal)
    if mode == "kf":
        from .calibration import load_calibrator, resolve_calibration_path

        calibration_path = resolve_calibration_path(config)
        if calibration_path.is_file():
            calibrator = load_calibrator(calibration_path)
        elif bool(config.filter.get("require_calibration", True)):
            raise FileNotFoundError(
                f"KF requires held-out variance calibration: {calibration_path}; "
                "run './run.sh calibrate ...' before enabling method.temporal=kf"
            )
        else:
            from ac_pbgrl.models.temporal import VarianceCalibrator

            calibrator = VarianceCalibrator(float(config.filter.get("calibration_temperature", 1.0)))
        return AdaptivePotentialKF(
            p0=float(config.filter.p0),
            q_stable=float(config.filter.q_stable),
            q_event=float(config.filter.q_event),
            r_min=float(config.filter.r_min),
            r_max=float(config.filter.r_max),
            nis_threshold=float(config.filter.nis_threshold),
            ttl_steps=int(config.filter.ttl_steps),
            calibrator=calibrator,
        )
    if mode == "ema":
        return EMAPotentialMemory(float(config.filter.ema_alpha), int(config.filter.ttl_steps))
    if mode == "gru":
        from .gru_control import resolve_gru_checkpoint

        checkpoint = resolve_gru_checkpoint(config)
        if not checkpoint.is_file() and bool(config.filter.get("require_gru_checkpoint", True)):
            raise FileNotFoundError(
                f"GRU control requires an offline-trained temporal checkpoint: {checkpoint}; "
                "run './run.sh train-gru ...' first"
            )
        return GRUPotentialMemory(
            hidden_dim=int(config.filter.get("gru_hidden_dim", 16)),
            ttl_steps=int(config.filter.ttl_steps),
            seed=int(config.project.seed),
            checkpoint=str(checkpoint) if checkpoint.is_file() else None,
        )
    return NoPotentialMemory()


class EpisodeCollector:
    def __init__(
        self,
        environment,
        actor,
        config: Config,
        *,
        device: str | torch.device = "cpu",
        labeler: FutureGainLabeler | None = None,
        q_teacher: QTeacher | None = None,
        label_budget: int | None = None,
        label_interval: int = 1,
        greedy: bool = False,
    ) -> None:
        self.environment = environment
        self.actor = actor
        self.config = config
        self.device = torch.device(device)
        self.labeler = labeler
        self.q_teacher = q_teacher
        self.label_budget = None if label_budget is None else int(label_budget)
        self.label_interval = max(1, int(label_interval))
        self.greedy = bool(greedy)
        self.rng = torch.Generator(device=self.device.type)
        self.rng.manual_seed(int(config.project.seed))
        self.temporal = build_temporal_memory(config)
        self.region_variance_temperature = 1.0
        self.action_variance_temperature = 1.0
        if bool(config.method.potential):
            from .calibration import load_variance_temperatures, resolve_calibration_path

            calibration_path = resolve_calibration_path(config)
            if calibration_path.is_file():
                self.region_variance_temperature, self.action_variance_temperature = (
                    load_variance_temperatures(calibration_path)
                )
        self.planning_times_ms: list[float] = []

    def _decorate(self, state: ExplorationState, step: int):
        device_state = state.to(self.device)
        started = time.perf_counter()
        with torch.no_grad():
            raw = self.actor(device_state)
        if raw.action_mean is None:
            self.planning_times_ms.append((time.perf_counter() - started) * 1000.0)
            return state, raw
        present_ids = state.stable_ids[state.node_mask].cpu().numpy()
        if hasattr(self.temporal, "retire_missing"):
            self.temporal.retire_missing(present_ids)
        candidate_ids = torch.gather(state.stable_ids, 1, state.candidate_indices)[0]
        mask = state.candidate_mask[0]
        valid_ids = candidate_ids[mask].cpu().numpy()
        region_mean = raw.region_mean[0, mask].float().cpu().numpy()
        region_variance = raw.region_log_variance[0, mask].float().exp().cpu().numpy()
        temporal_region_variance = (
            region_variance
            if isinstance(self.temporal, AdaptivePotentialKF)
            else region_variance * self.region_variance_temperature
        )
        events = (
            np.zeros(len(valid_ids), dtype=np.int16)
            if state.candidate_events is None
            else state.candidate_events[0, mask].cpu().numpy()
        )
        posterior_region_mean, posterior_region_variance = self.temporal.update_many(
            valid_ids, region_mean, temporal_region_variance, events, step=step
        )
        posterior_mean = raw.action_mean[0].float().cpu().clone()
        posterior_variance = (
            raw.action_log_variance[0].float().exp().cpu().clone()
            * self.action_variance_temperature
        )
        raw_region_mean = raw.region_mean[0, mask].float().cpu()
        calibrated_region_variance = (
            raw.region_log_variance[0, mask].float().exp().cpu()
            * self.region_variance_temperature
        )
        residual_variance = (posterior_variance[mask] - calibrated_region_variance).clamp_min(1.0e-6)
        posterior_mean[mask] = (
            raw.action_mean[0, mask].float().cpu()
            - raw_region_mean
            + torch.from_numpy(posterior_region_mean)
        )
        posterior_variance[mask] = torch.from_numpy(posterior_region_variance) + residual_variance
        decorated = replace(
            state,
            posterior_mean=posterior_mean.unsqueeze(0),
            posterior_variance=posterior_variance.unsqueeze(0),
        )
        with torch.no_grad():
            rescored = self.actor(decorated.to(self.device))
        self.planning_times_ms.append((time.perf_counter() - started) * 1000.0)
        return decorated, rescored

    def collect(
        self,
        *,
        episode: int,
        map_path=None,
        transition_callback: Callable[[TransitionBatch], None] | None = None,
        max_steps: int | None = None,
    ) -> tuple[list[TransitionBatch], dict[str, float]]:
        self.temporal.reset()
        self.planning_times_ms = []
        self.rng.manual_seed(int(self.config.project.seed) + int(episode))
        try:
            state, critic_state = self.environment.reset(episode=episode, map_path=map_path)
        except TypeError:
            state, critic_state = self.environment.reset(seed=int(self.config.project.seed) + episode)
        state, output = self._decorate(state, step=0)
        transitions: list[TransitionBatch] = []
        episode_return = 0.0
        travel_distance = 0.0
        direction_switches = 0
        repeated_edges = 0
        previous_delta: np.ndarray | None = None
        traversed: set[tuple[int, int]] = set()
        backtracks = 0
        node_path: list[int] = []
        coverage_95_distance = float("nan")
        cumulative_distance = 0.0
        node_counts: list[int] = []
        trajectory: list[list[float]] = []
        last_info = {"explored_rate": 0.0, "success": False}
        executed_steps = 0
        maximum_steps = int(self.config.environment.max_episode_steps)
        if max_steps is not None:
            maximum_steps = min(maximum_steps, max(1, int(max_steps)))
        labels_generated = 0
        for step in range(maximum_steps):
            if not bool(state.candidate_mask.any()):
                break
            should_label = (
                self.labeler is not None
                and step % self.label_interval == 0
                and (self.label_budget is None or labels_generated < self.label_budget)
            )
            labels = self.labeler.label(self.environment, state) if should_label else None
            labels_generated += int(labels is not None)
            teacher_q = self.q_teacher.values(critic_state) if self.q_teacher is not None else None
            if self.greedy:
                action = int(torch.argmax(output.logits, dim=-1)[0])
            else:
                action = int(torch.multinomial(output.probabilities[0], 1, generator=self.rng))
            current_id = int(state.stable_ids[0, state.current_index[0]])
            candidate_node = int(state.candidate_indices[0, action])
            target_id = int(state.stable_ids[0, candidate_node])
            edge = (current_id, target_id)
            repeated_edges += int(edge in traversed)
            traversed.add(edge)
            node_path.append(current_id)
            backtracks += int(len(node_path) >= 2 and target_id == node_path[-2])
            current_xy = state.node_xy[0, state.current_index[0]].cpu().numpy()
            target_xy = state.node_xy[0, candidate_node].cpu().numpy()
            delta = target_xy - current_xy
            if previous_delta is not None:
                denominator = max(float(np.linalg.norm(previous_delta) * np.linalg.norm(delta)), 1.0e-8)
                cosine = float(np.dot(previous_delta, delta) / denominator)
                direction_switches += int(cosine < 0.0)
            previous_delta = delta
            trajectory.append(current_xy.astype(float).tolist())
            node_counts.append(int(state.node_mask.sum()))
            result = self.environment.step(action)
            executed_steps += 1
            if hasattr(self.temporal, "retire"):
                self.temporal.retire(target_id)
            next_state, next_output = self._decorate(result.state, step=step + 1)
            transition = TransitionBatch(
                state=state,
                action=torch.tensor([action], dtype=torch.long),
                reward=torch.tensor([result.reward], dtype=torch.float32),
                done=torch.tensor([float(result.done)], dtype=torch.float32),
                next_state=next_state,
                critic_state=critic_state,
                critic_next_state=result.critic_state,
                future_gain=None if labels is None else torch.from_numpy(labels.values).unsqueeze(0),
                future_gain_mask=None if labels is None else torch.from_numpy(labels.mask).unsqueeze(0),
                teacher_q=teacher_q,
            )
            if transition_callback is not None:
                transition_callback(transition)
            else:
                transitions.append(transition)
            episode_return += float(result.reward)
            cumulative_distance += float(result.info.get("distance", 0.0))
            travel_distance = float(result.info.get("travel_distance", cumulative_distance))
            if not np.isfinite(coverage_95_distance) and float(result.info.get("explored_rate", 0.0)) >= 0.95:
                coverage_95_distance = travel_distance
            last_info = result.info
            state, output, critic_state = next_state, next_output, result.critic_state
            if result.done:
                break
        temporal_metrics = self.temporal.metrics() if hasattr(self.temporal, "metrics") else {}
        if state is not None:
            final_xy = state.node_xy[0, state.current_index[0]].cpu().numpy().astype(float).tolist()
            trajectory.append(final_xy)
        self.last_trajectory = trajectory
        metrics = {
            "episode/return": episode_return,
            "episode/steps": float(executed_steps),
            "episode/travel_distance": travel_distance,
            "episode/completion_distance": travel_distance if bool(last_info.get("success", False)) else float("nan"),
            "episode/coverage_95_distance": coverage_95_distance,
            "episode/explored_rate": float(last_info.get("explored_rate", 0.0)),
            "episode/success": float(bool(last_info.get("success", False))),
            "episode/direction_switches": float(direction_switches),
            "episode/repeated_edges": float(repeated_edges),
            "episode/backtracks": float(backtracks),
            "episode/makespan_steps": float(executed_steps),
            "graph/nodes_mean": float(np.mean(node_counts)) if node_counts else 0.0,
            "graph/nodes_max": float(max(node_counts)) if node_counts else 0.0,
            "system/planning_latency_mean_ms": float(np.mean(self.planning_times_ms)) if self.planning_times_ms else 0.0,
            "system/planning_latency_p95_ms": float(np.percentile(self.planning_times_ms, 95)) if self.planning_times_ms else 0.0,
            **temporal_metrics,
        }
        return transitions, metrics
