from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ac_pbgrl.state import ExplorationState

from .teacher import TeacherPolicy


@dataclass
class FutureGainLabels:
    values: np.ndarray
    mask: np.ndarray
    rollout_lengths: np.ndarray
    terminal: np.ndarray


class FutureGainLabeler:
    """Finite-horizon, action-conditioned rollout target generator."""

    def __init__(self, teacher: TeacherPolicy, horizon: int = 6, gamma: float = 0.95) -> None:
        if horizon < 1:
            raise ValueError("horizon must be positive")
        self.teacher = teacher
        self.horizon = int(horizon)
        self.gamma = float(gamma)

    def label(self, environment, state: ExplorationState) -> FutureGainLabels:
        state.validate()
        if state.batch_size != 1:
            raise ValueError("rollout label generation expects one state at a time")
        candidate_count = state.candidate_count
        values = np.full(candidate_count, np.nan, dtype=np.float32)
        mask = state.candidate_mask[0].detach().cpu().numpy().astype(np.bool_, copy=True)
        lengths = np.zeros(candidate_count, dtype=np.int16)
        terminal = np.zeros(candidate_count, dtype=np.bool_)
        for action_slot in np.flatnonzero(mask):
            branch = environment.clone()
            total = 0.0
            discount = 1.0
            result = branch.step(int(action_slot))
            total += discount * float(result.info.get("label_reward", result.reward))
            lengths[action_slot] = 1
            terminal[action_slot] = result.done
            rollout_state = result.state
            for depth in range(1, self.horizon):
                if result.done or not bool(rollout_state.candidate_mask.any()):
                    break
                discount *= self.gamma
                next_action = self.teacher.select(rollout_state)
                result = branch.step(next_action)
                total += discount * float(result.info.get("label_reward", result.reward))
                rollout_state = result.state
                lengths[action_slot] = depth + 1
                terminal[action_slot] = result.done
            values[action_slot] = total
        return FutureGainLabels(values, mask, lengths, terminal)
