from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass
class StepResult:
    reward: float
    cost: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class Observation:
    node_features: np.ndarray
    critic_node_features: np.ndarray
    edge_mask: np.ndarray
    action_mask: np.ndarray
    current_index: int
    neighbor_indices: np.ndarray
    node_positions: np.ndarray
    fallback_action_slot: int = 0
    expert_action_slot: int | None = None

    def safe_action_slots(self) -> np.ndarray:
        return np.flatnonzero(~self.action_mask)

    @property
    def has_safe_action(self) -> bool:
        return bool(np.any(~self.action_mask))


@dataclass
class EpisodeMetrics:
    explored_rate: float = 0.0
    travel_dist: float = 0.0
    episode_return: float = 0.0
    episode_cost: float = 0.0
    collision_count: int = 0
    near_miss_count: int = 0
    risk_integral: float = 0.0
    budget_violation: int = 0
    return_success: int = 0
    shield_interventions: int = 0
    unsafe_action_proposals: int = 0
    episode_steps: int = 0
    success_rate: float = 0.0

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)
