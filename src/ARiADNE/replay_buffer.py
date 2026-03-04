from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


ACTOR_OBS_FIELDS = (
    "node_inputs",
    "node_padding_mask",
    "edge_mask",
    "current_index",
    "current_edge",
    "edge_padding_mask",
)
ACTOR_ATTN_BIAS_FIELD = "actor_attn_bias"
NEXT_ACTOR_OBS_FIELDS = (
    "next_node_inputs",
    "next_node_padding_mask",
    "next_edge_mask",
    "next_current_index",
    "next_current_edge",
    "next_edge_padding_mask",
)
ACTOR_NEXT_ATTN_BIAS_FIELD = "actor_next_attn_bias"
CRITIC_OBS_FIELDS = (
    "critic_node_inputs",
    "critic_node_padding_mask",
    "critic_edge_mask",
    "critic_current_index",
    "critic_current_edge",
    "critic_edge_padding_mask",
)
CRITIC_ATTN_BIAS_FIELD = "critic_attn_bias"
NEXT_CRITIC_OBS_FIELDS = (
    "critic_next_node_inputs",
    "critic_next_node_padding_mask",
    "critic_next_edge_mask",
    "critic_next_current_index",
    "critic_next_current_edge",
    "critic_next_edge_padding_mask",
)
CRITIC_NEXT_ATTN_BIAS_FIELD = "critic_next_attn_bias"
TRANSITION_FIELDS = (
    *ACTOR_OBS_FIELDS,
    "action",
    "reward",
    "done",
    *NEXT_ACTOR_OBS_FIELDS,
    *CRITIC_OBS_FIELDS,
    *NEXT_CRITIC_OBS_FIELDS,
    ACTOR_ATTN_BIAS_FIELD,
    ACTOR_NEXT_ATTN_BIAS_FIELD,
    CRITIC_ATTN_BIAS_FIELD,
    CRITIC_NEXT_ATTN_BIAS_FIELD,
    "gamma_pow",
    "n_step_actual",
)
TRANSITION_FIELD_INDEX = {field_name: field_index for field_index, field_name in enumerate(TRANSITION_FIELDS)}
OPTIONAL_BATCH_CONCAT_FIELDS = {
    ACTOR_ATTN_BIAS_FIELD,
    ACTOR_NEXT_ATTN_BIAS_FIELD,
    CRITIC_ATTN_BIAS_FIELD,
    CRITIC_NEXT_ATTN_BIAS_FIELD,
}

EPISODE_BUFFER_SLOT_COUNT = len(TRANSITION_FIELDS)
NEXT_OBSERVATION_SLOTS = tuple(
    TRANSITION_FIELD_INDEX[field_name]
    for field_name in NEXT_ACTOR_OBS_FIELDS
)
CRITIC_NEXT_OBSERVATION_SLOTS = tuple(
    TRANSITION_FIELD_INDEX[field_name]
    for field_name in NEXT_CRITIC_OBS_FIELDS
)
GAMMA_POW_SLOT = EPISODE_BUFFER_SLOT_COUNT - 2
N_STEP_ACTUAL_SLOT = EPISODE_BUFFER_SLOT_COUNT - 1


@dataclass(frozen=True)
class ReplaySample:
    batch: dict[str, torch.Tensor | None]
    indices: np.ndarray
    is_weights: torch.Tensor


def empty_episode_buffer() -> list[list[torch.Tensor | None]]:
    return [[] for _ in range(EPISODE_BUFFER_SLOT_COUNT)]


def get_episode_buffer_size(episode_buffer: list[list[torch.Tensor | None]]) -> int:
    if len(episode_buffer) != EPISODE_BUFFER_SLOT_COUNT:
        raise ValueError(
            f"episode_buffer must have {EPISODE_BUFFER_SLOT_COUNT} slots, got {len(episode_buffer)}"
        )
    if not episode_buffer:
        return 0
    buffer_size = len(episode_buffer[0])
    for slot_index, slot in enumerate(episode_buffer):
        if len(slot) != buffer_size:
            raise ValueError(
                f"episode_buffer slot {slot_index} has len={len(slot)}; expected {buffer_size}"
            )
    return buffer_size


def episode_buffer_to_transitions(
    episode_buffer: list[list[torch.Tensor | None]],
) -> list[dict[str, torch.Tensor | None]]:
    buffer_size = get_episode_buffer_size(episode_buffer)
    return [
        {
            field_name: episode_buffer[field_index][transition_index]
            for field_index, field_name in enumerate(TRANSITION_FIELDS)
        }
        for transition_index in range(buffer_size)
    ]


def stack_transition_field(field_name: str, values: list[torch.Tensor | None]) -> torch.Tensor | None:
    if not values or values[0] is None:
        return None
    tensors = [value for value in values if value is not None]
    if len(tensors) != len(values):
        raise ValueError(f"Field {field_name} mixes None and tensor values")
    if field_name in OPTIONAL_BATCH_CONCAT_FIELDS:
        return torch.cat(tensors, dim=0)
    return torch.stack(tensors)


class ReplayBuffer:
    def __init__(self, capacity: int, prioritized: bool = False, alpha: float = 0.6):
        self.capacity = max(int(capacity), 1)
        self.prioritized = bool(prioritized)
        self.alpha = max(float(alpha), 0.0)
        self._storage = {field_name: [None] * self.capacity for field_name in TRANSITION_FIELDS}
        self._priorities = np.zeros(self.capacity, dtype=np.float32)
        self._size = 0
        self._next_index = 0
        self._max_priority = 1.0

    @property
    def size(self) -> int:
        return self._size

    def clear(self) -> None:
        self._storage = {field_name: [None] * self.capacity for field_name in TRANSITION_FIELDS}
        self._priorities.fill(0.0)
        self._size = 0
        self._next_index = 0
        self._max_priority = 1.0

    def push(self, transition: dict[str, torch.Tensor | None]) -> None:
        missing = [field_name for field_name in TRANSITION_FIELDS if field_name not in transition]
        if missing:
            raise KeyError(f"transition is missing fields: {missing}")

        for field_name in TRANSITION_FIELDS:
            self._storage[field_name][self._next_index] = transition[field_name]

        self._priorities[self._next_index] = self._max_priority if self.prioritized else 1.0
        self._next_index = (self._next_index + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def _sample_indices(self, batch_size: int) -> np.ndarray:
        batch_size = min(max(int(batch_size), 1), self._size)
        if not self.prioritized:
            return np.random.choice(self._size, size=batch_size, replace=False)

        priorities = np.asarray(self._priorities[: self._size], dtype=np.float64)
        scaled = np.power(np.maximum(priorities, 1e-12), self.alpha)
        total = scaled.sum()
        if total <= 0.0 or not np.isfinite(total):
            return np.random.choice(self._size, size=batch_size, replace=False)
        probabilities = scaled / total
        return np.random.choice(self._size, size=batch_size, replace=False, p=probabilities)

    def sample(self, batch_size: int, beta: float | None = None) -> ReplaySample:
        if self._size <= 0:
            raise ValueError("Cannot sample from an empty replay buffer")

        indices = self._sample_indices(batch_size)
        batch = {
            field_name: stack_transition_field(
                field_name,
                [self._storage[field_name][int(index)] for index in indices],
            )
            for field_name in TRANSITION_FIELDS
        }

        if not self.prioritized:
            is_weights = torch.ones((len(indices), 1), dtype=torch.float32)
            return ReplaySample(batch=batch, indices=indices.astype(np.int64), is_weights=is_weights)

        beta_value = min(max(float(beta if beta is not None else 1.0), 0.0), 1.0)
        priorities = np.asarray(self._priorities[: self._size], dtype=np.float64)
        scaled = np.power(np.maximum(priorities, 1e-12), self.alpha)
        probabilities = scaled / max(scaled.sum(), 1e-12)
        sample_probabilities = probabilities[indices]
        weights = np.power(self._size * np.maximum(sample_probabilities, 1e-12), -beta_value)
        weights /= max(weights.max(), 1.0)
        is_weights = torch.as_tensor(weights, dtype=torch.float32).unsqueeze(1)
        return ReplaySample(batch=batch, indices=indices.astype(np.int64), is_weights=is_weights)

    def update_priorities(self, indices: np.ndarray | list[int], td_errors: np.ndarray | list[float]) -> None:
        td_errors_array = np.asarray(td_errors, dtype=np.float32).reshape(-1)
        indices_array = np.asarray(indices, dtype=np.int64).reshape(-1)
        if td_errors_array.shape != indices_array.shape:
            raise ValueError(
                f"indices shape {indices_array.shape} does not match td_errors shape {td_errors_array.shape}"
            )

        priorities = np.maximum(np.abs(td_errors_array), 1e-12)
        for index, priority in zip(indices_array, priorities):
            self._priorities[int(index)] = float(priority)
        if priorities.size > 0:
            self._max_priority = max(self._max_priority, float(priorities.max()))
