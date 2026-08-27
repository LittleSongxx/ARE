from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch

from ac_pbgrl.state import ExplorationState
from ac_pbgrl.utils import stable_coordinate_id

from .protocol import StepResult


class SyntheticGraphExplorationEnv:
    """Fast deterministic graph environment used for smoke and algorithm tests."""

    def __init__(self, node_padding: int = 32, candidate_padding: int = 8, seed: int = 0) -> None:
        self.node_padding = int(node_padding)
        self.candidate_padding = int(candidate_padding)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.current = 0
        self.travel_distance = 0.0
        self.visited: set[int] = set()
        self.node_count = min(24, self.node_padding)
        side = int(np.ceil(np.sqrt(self.node_count)))
        self.xy = np.asarray([(i % side, i // side) for i in range(self.node_count)], dtype=np.float32) * 4.0
        self.adjacency = np.zeros((self.node_count, self.node_count), dtype=np.bool_)
        for i in range(self.node_count):
            for j in range(self.node_count):
                if np.abs(self.xy[i] - self.xy[j]).sum() <= 4.01:
                    self.adjacency[i, j] = True
        self.hidden_gain = self.rng.uniform(0.2, 2.0, size=self.node_count).astype(np.float32)

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.seed = int(seed)
            self.rng = np.random.default_rng(seed)
            self.hidden_gain = self.rng.uniform(0.2, 2.0, size=self.node_count).astype(np.float32)
        self.steps = 0
        self.current = 0
        self.travel_distance = 0.0
        self.visited = {0}
        return self._state(privileged=False), self._state(privileged=True)

    def clone(self):
        return copy.deepcopy(self)

    def _state(self, privileged: bool) -> ExplorationState:
        feature_dim = 5 if privileged else 4
        features = np.zeros((self.node_padding, feature_dim), dtype=np.float32)
        relative = (self.xy - self.xy[self.current]) / 32.0
        utility = self.hidden_gain.copy()
        if not privileged:
            # Belief aliases distant nodes on purpose; GT critic sees exact gain.
            utility = np.where(np.linalg.norm(relative, axis=1) < 0.4, utility, 0.5)
        features[: self.node_count, :2] = relative
        features[: self.node_count, 2] = utility
        features[: self.node_count, 3] = np.asarray([i in self.visited for i in range(self.node_count)])
        if privileged:
            features[: self.node_count, 4] = self.hidden_gain
        node_xy = np.zeros((self.node_padding, 2), dtype=np.float32)
        node_xy[: self.node_count] = self.xy
        node_mask = np.zeros(self.node_padding, dtype=np.bool_)
        node_mask[: self.node_count] = True
        adjacency = np.zeros((self.node_padding, self.node_padding), dtype=np.bool_)
        adjacency[: self.node_count, : self.node_count] = self.adjacency
        stable_ids = np.zeros(self.node_padding, dtype=np.int64)
        stable_ids[: self.node_count] = [stable_coordinate_id(*xy) for xy in self.xy]
        neighbors = np.flatnonzero(self.adjacency[self.current] & (np.arange(self.node_count) != self.current))
        candidate_indices = np.zeros(self.candidate_padding, dtype=np.int64)
        candidate_mask = np.zeros(self.candidate_padding, dtype=np.bool_)
        count = min(len(neighbors), self.candidate_padding)
        candidate_indices[:count] = neighbors[:count]
        candidate_mask[:count] = True
        edge_features = np.zeros((self.candidate_padding, 4), dtype=np.float32)
        if count:
            delta = self.xy[neighbors[:count]] - self.xy[self.current]
            distance = np.linalg.norm(delta, axis=1)
            edge_features[:count] = np.column_stack((delta / 32.0, distance / 32.0, distance / 32.0))
        state = ExplorationState(
            node_features=torch.from_numpy(features).unsqueeze(0),
            node_xy=torch.from_numpy(node_xy).unsqueeze(0),
            node_mask=torch.from_numpy(node_mask).unsqueeze(0),
            adjacency=torch.from_numpy(adjacency).unsqueeze(0),
            stable_ids=torch.from_numpy(stable_ids).unsqueeze(0),
            current_index=torch.tensor([self.current], dtype=torch.long),
            candidate_indices=torch.from_numpy(candidate_indices).unsqueeze(0),
            candidate_mask=torch.from_numpy(candidate_mask).unsqueeze(0),
            edge_features=torch.from_numpy(edge_features).unsqueeze(0),
            candidate_events=torch.zeros((1, self.candidate_padding), dtype=torch.int16),
            metadata={"map_id": f"synthetic-{self.seed}", "step": self.steps},
        )
        return state.validate()

    def step(self, action_slot: int) -> StepResult:
        actor_state = self._state(privileged=False)
        if not bool(actor_state.candidate_mask[0, action_slot]):
            raise ValueError(f"invalid action slot {action_slot}")
        next_node = int(actor_state.candidate_indices[0, action_slot])
        distance = float(np.linalg.norm(self.xy[next_node] - self.xy[self.current]))
        newly_visited = next_node not in self.visited
        gain = float(self.hidden_gain[next_node]) if newly_visited else 0.0
        self.current = next_node
        self.visited.add(next_node)
        self.steps += 1
        self.travel_distance += distance
        reward = gain - 0.05 * distance
        done = len(self.visited) >= self.node_count or self.steps >= 32
        return StepResult(
            state=self._state(privileged=False),
            critic_state=self._state(privileged=True),
            reward=reward,
            done=done,
            info={
                "frontier_gain": gain,
                "distance": distance,
                "label_reward": reward,
                "explored_rate": len(self.visited) / self.node_count,
                "travel_distance": self.travel_distance,
                "map_id": f"synthetic-{self.seed}",
            },
        )
