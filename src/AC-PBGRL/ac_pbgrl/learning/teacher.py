from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from ac_pbgrl.models.policy import ACPolicyNetwork, PrivilegedQNetwork
from ac_pbgrl.state import ExplorationState


class TeacherPolicy(Protocol):
    def select(self, state: ExplorationState) -> int: ...


class QTeacher(Protocol):
    def values(self, state: ExplorationState) -> torch.Tensor: ...


class HeuristicTeacher:
    """Deterministic smoke-only teacher using observed utility then path cost."""

    def select(self, state: ExplorationState) -> int:
        valid = torch.nonzero(state.candidate_mask[0], as_tuple=False).flatten()
        if not valid.numel():
            raise RuntimeError("state has no valid candidate")
        candidate_nodes = state.candidate_indices[0, valid]
        utility = state.node_features[0, candidate_nodes, 2]
        distance = state.edge_features[0, valid, 2]
        score = utility - 0.1 * distance
        return int(valid[torch.argmax(score)])


class HeuristicQTeacher:
    """Smoke-only Q target with the same action-slot contract as the critic."""

    def values(self, state: ExplorationState) -> torch.Tensor:
        candidate_nodes = state.candidate_indices
        utility = torch.gather(state.node_features[..., 2], 1, candidate_nodes)
        distance = state.edge_features[..., 2]
        return (utility - 0.1 * distance).masked_fill(~state.candidate_mask, float("nan"))


class FrozenPolicyTeacher:
    def __init__(self, policy: ACPolicyNetwork, device: str | torch.device = "cpu") -> None:
        self.policy = policy.to(device).eval()
        self.device = torch.device(device)
        for parameter in self.policy.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def select(self, state: ExplorationState) -> int:
        output = self.policy(state.to(self.device))
        return int(torch.argmax(output.logits, dim=-1)[0])

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        node_feature_dim: int = 4,
        edge_feature_dim: int = 4,
        embedding_dim: int = 128,
        heads: int = 8,
        layers: int = 6,
        device: str | torch.device = "cpu",
    ) -> "FrozenPolicyTeacher":
        model = ACPolicyNetwork(
            node_feature_dim,
            edge_feature_dim,
            embedding_dim,
            heads,
            layers,
            use_potential=False,
            use_diffusion=False,
        )
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        container = payload.get("learner", payload)
        state = container.get("actor", container.get("policy_model")) if isinstance(container, dict) else None
        if state is None:
            raise KeyError("checkpoint does not contain actor/policy_model weights")
        model.load_state_dict(state, strict=True)
        return cls(model, device=device)


class FrozenQTeacher:
    def __init__(
        self,
        q1: PrivilegedQNetwork,
        q2: PrivilegedQNetwork,
        device: str | torch.device = "cpu",
    ) -> None:
        self.q1 = q1.to(device).eval()
        self.q2 = q2.to(device).eval()
        self.device = torch.device(device)
        for parameter in list(self.q1.parameters()) + list(self.q2.parameters()):
            parameter.requires_grad_(False)

    @torch.no_grad()
    def values(self, state: ExplorationState) -> torch.Tensor:
        state = state.to(self.device)
        values = torch.minimum(self.q1(state), self.q2(state))
        return values.masked_fill(~state.candidate_mask, float("nan")).cpu()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        node_feature_dim: int = 5,
        edge_feature_dim: int = 4,
        embedding_dim: int = 128,
        heads: int = 8,
        layers: int = 6,
        device: str | torch.device = "cpu",
    ) -> "FrozenQTeacher":
        def network() -> PrivilegedQNetwork:
            return PrivilegedQNetwork(
                node_feature_dim,
                edge_feature_dim,
                embedding_dim,
                heads,
                layers,
                use_diffusion=False,
            )

        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        container = payload.get("learner", payload)
        if not isinstance(container, dict) or "q1" not in container or "q2" not in container:
            raise KeyError("checkpoint does not contain q1/q2 weights for Q distillation")
        q1, q2 = network(), network()
        q1.load_state_dict(container["q1"], strict=True)
        q2.load_state_dict(container["q2"], strict=True)
        return cls(q1, q2, device=device)


def select_action(policy: TeacherPolicy, state: ExplorationState, rng: np.random.Generator | None = None) -> int:
    del rng
    return policy.select(state)
