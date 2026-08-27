from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Dict, Optional

import torch

from ac_pbgrl.events import GraphEvent


@dataclass
class ExplorationState:
    """Padded belief-graph batch.

    Masks use True for valid entries. Candidate slots retain their external action
    identity even when the graph has been compacted.
    """

    node_features: torch.Tensor
    node_xy: torch.Tensor
    node_mask: torch.Tensor
    adjacency: torch.Tensor
    stable_ids: torch.Tensor
    current_index: torch.Tensor
    candidate_indices: torch.Tensor
    candidate_mask: torch.Tensor
    edge_features: torch.Tensor
    candidate_events: Optional[torch.Tensor] = None
    posterior_mean: Optional[torch.Tensor] = None
    posterior_variance: Optional[torch.Tensor] = None
    metadata: Optional[Dict[str, Any]] = None

    def validate(self) -> "ExplorationState":
        if torch.jit.is_tracing() or torch.onnx.is_in_onnx_export():
            return self
        if self.node_features.ndim != 3:
            raise ValueError("node_features must have shape [B,N,F]")
        batch, nodes, _ = self.node_features.shape
        if self.node_xy.shape != (batch, nodes, 2):
            raise ValueError("node_xy must have shape [B,N,2]")
        if self.node_mask.shape != (batch, nodes):
            raise ValueError("node_mask must have shape [B,N]")
        if self.adjacency.shape != (batch, nodes, nodes):
            raise ValueError("adjacency must have shape [B,N,N]")
        if self.stable_ids.shape != (batch, nodes):
            raise ValueError("stable_ids must have shape [B,N]")
        if self.current_index.shape != (batch,):
            raise ValueError("current_index must have shape [B]")
        if self.candidate_indices.ndim != 2 or self.candidate_indices.shape[0] != batch:
            raise ValueError("candidate_indices must have shape [B,K]")
        if self.candidate_mask.shape != self.candidate_indices.shape:
            raise ValueError("candidate_mask shape mismatch")
        if self.edge_features.shape[:2] != self.candidate_indices.shape:
            raise ValueError("edge_features must have shape [B,K,E]")
        if torch.any(self.current_index < 0) or torch.any(self.current_index >= nodes):
            raise ValueError("current_index out of range")
        valid_candidates = self.candidate_indices[self.candidate_mask]
        if valid_candidates.numel() and (torch.any(valid_candidates < 0) or torch.any(valid_candidates >= nodes)):
            raise ValueError("candidate index out of range")
        return self

    @property
    def batch_size(self) -> int:
        return int(self.node_features.shape[0])

    @property
    def candidate_count(self) -> int:
        return int(self.candidate_indices.shape[1])

    def to(self, device: torch.device | str, non_blocking: bool = False) -> "ExplorationState":
        values: Dict[str, Any] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, torch.Tensor):
                value = value.to(device, non_blocking=non_blocking)
            values[field.name] = value
        return replace(self, **values)

    def detach(self) -> "ExplorationState":
        values: Dict[str, Any] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            values[field.name] = value.detach() if isinstance(value, torch.Tensor) else value
        return replace(self, **values)

    def index(self, indices: torch.Tensor) -> "ExplorationState":
        values: Dict[str, Any] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == self.batch_size:
                value = value.index_select(0, indices)
            values[field.name] = value
        return replace(self, **values)


@dataclass
class PrivilegedState:
    graph: ExplorationState
    ground_truth_map: Optional[torch.Tensor] = None
    map_id: Optional[list[str]] = None

    def to(self, device: torch.device | str) -> "PrivilegedState":
        gt = self.ground_truth_map
        return PrivilegedState(
            graph=self.graph.to(device),
            ground_truth_map=None if gt is None else gt.to(device),
            map_id=self.map_id,
        )


@dataclass
class PolicyOutput:
    logits: torch.Tensor
    log_probs: torch.Tensor
    probabilities: torch.Tensor
    base_logits: torch.Tensor
    action_mean: Optional[torch.Tensor] = None
    action_log_variance: Optional[torch.Tensor] = None
    region_mean: Optional[torch.Tensor] = None
    region_log_variance: Optional[torch.Tensor] = None
    posterior_mean: Optional[torch.Tensor] = None
    posterior_variance: Optional[torch.Tensor] = None
    candidate_embeddings: Optional[torch.Tensor] = None


@dataclass
class TransitionBatch:
    state: ExplorationState
    action: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    next_state: ExplorationState
    critic_state: Optional[ExplorationState] = None
    critic_next_state: Optional[ExplorationState] = None
    future_gain: Optional[torch.Tensor] = None
    future_gain_mask: Optional[torch.Tensor] = None
    teacher_q: Optional[torch.Tensor] = None

    def to(self, device: torch.device | str) -> "TransitionBatch":
        return TransitionBatch(
            state=self.state.to(device),
            action=self.action.to(device),
            reward=self.reward.to(device),
            done=self.done.to(device),
            next_state=self.next_state.to(device),
            critic_state=None if self.critic_state is None else self.critic_state.to(device),
            critic_next_state=None if self.critic_next_state is None else self.critic_next_state.to(device),
            future_gain=None if self.future_gain is None else self.future_gain.to(device),
            future_gain_mask=None if self.future_gain_mask is None else self.future_gain_mask.to(device),
            teacher_q=None if self.teacher_q is None else self.teacher_q.to(device),
        )


@dataclass
class PotentialSupervisionBatch:
    """Held-out/offline privileged rollout targets used by the Actor only."""

    state: ExplorationState
    future_gain: torch.Tensor
    future_gain_mask: torch.Tensor

    def to(self, device: torch.device | str) -> "PotentialSupervisionBatch":
        return PotentialSupervisionBatch(
            state=self.state.to(device),
            future_gain=self.future_gain.to(device),
            future_gain_mask=self.future_gain_mask.to(device),
        )
