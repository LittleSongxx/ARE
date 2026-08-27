from __future__ import annotations

import copy
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from ac_pbgrl.models.context import action_preserving_context
from ac_pbgrl.events import GraphEvent
from ac_pbgrl.state import ExplorationState
from ac_pbgrl.utils import stable_coordinate_id

from ..protocol import StepResult
from .env import Env
from .ground_truth_node_manager import GroundTruthNodeManager
from .node_manager import NodeManager
from .parameter import FRONTIER_CELL_SIZE, SENSOR_RANGE, UPDATING_MAP_SIZE
from .utils import MapInfo, get_frontier_in_map


UTILITY_NORMALIZER = SENSOR_RANGE * 3.14 // FRONTIER_CELL_SIZE


class AriadneExplorationEnv:
    """AC-PBGRL state adapter around the vendored ARiADNE map simulator."""

    def __init__(
        self,
        *,
        maps_dir: str | Path,
        node_padding: int = 360,
        critic_node_padding: int = 512,
        candidate_padding: int = 25,
        max_episode_steps: int = 128,
        completion_threshold: float = 0.95,
        terminal_reward: float = 20.0,
        hierarchy: bool = False,
        local_budget: int = 192,
        region_budget: int = 32,
        region_size_m: float = 16.0,
        seed: int = 0,
    ) -> None:
        self.maps_dir = Path(maps_dir)
        self.node_padding = int(node_padding)
        self.critic_node_padding = int(critic_node_padding)
        self.candidate_padding = int(candidate_padding)
        self.max_episode_steps = int(max_episode_steps)
        self.completion_threshold = float(completion_threshold)
        self.terminal_reward = float(terminal_reward)
        self.hierarchy = bool(hierarchy)
        self.local_budget = int(local_budget)
        self.region_budget = int(region_budget)
        self.region_size_m = float(region_size_m)
        self.seed = int(seed)
        self.episode = 0
        self.step_index = 0
        self.simulator: Env | None = None
        self.node_manager: NodeManager | None = None
        self.gt_manager: GroundTruthNodeManager | None = None
        self._actor_coords: np.ndarray | None = None
        self._previous_utility: dict[int, float] = {}
        self._previous_neighbors: dict[int, set[int]] = {}
        self._last_frontier_count = 0
        self._last_state: ExplorationState | None = None
        self._last_critic: ExplorationState | None = None

    def clone(self):
        return copy.deepcopy(self)

    def _map_files(self) -> list[Path]:
        files = sorted(path for path in self.maps_dir.iterdir() if path.is_file() and path.suffix.lower() == ".png")
        if not files:
            raise FileNotFoundError(f"no PNG maps found in {self.maps_dir}")
        return files

    def reset(self, episode: int = 0, map_path: str | Path | None = None):
        self.episode = int(episode)
        self.step_index = 0
        if map_path is None:
            map_path = self._map_files()[self.episode % len(self._map_files())]
        self.simulator = Env(self.episode, plot=False, forced_map_path=map_path)
        self.node_manager = NodeManager(plot=False)
        self._update_belief_graph()
        self.gt_manager = GroundTruthNodeManager(
            self.node_manager, self.simulator.ground_truth_info, device="cpu", plot=False
        )
        self._previous_utility = {}
        self._previous_neighbors = {}
        self._last_frontier_count = len(self.simulator.global_frontiers)
        self._last_state = self._build_actor_state()
        self._last_critic = self._build_critic_state()
        return self._last_state, self._last_critic

    def _require_ready(self):
        if self.simulator is None or self.node_manager is None:
            raise RuntimeError("environment must be reset before use")

    def _update_belief_graph(self) -> None:
        self._require_simulator()
        simulator = self.simulator
        assert simulator is not None
        updating = self._updating_map(simulator.robot_location)
        frontier = get_frontier_in_map(updating)
        self.node_manager.update_graph(simulator.robot_location, frontier, updating, simulator.belief_info)
        simulator.global_frontiers = get_frontier_in_map(simulator.belief_info)

    def _require_simulator(self) -> None:
        if self.simulator is None:
            raise RuntimeError("simulator is not initialized")

    def _updating_map(self, location: np.ndarray) -> MapInfo:
        simulator = self.simulator
        assert simulator is not None
        half = UPDATING_MAP_SIZE / 2
        origin_x = max(simulator.belief_info.map_origin_x, location[0] - half)
        origin_y = max(simulator.belief_info.map_origin_y, location[1] - half)
        top_x = min(
            simulator.belief_info.map_origin_x + simulator.cell_size * (simulator.robot_belief.shape[1] - 1),
            location[0] + half,
        )
        top_y = min(
            simulator.belief_info.map_origin_y + simulator.cell_size * (simulator.robot_belief.shape[0] - 1),
            location[1] + half,
        )
        x0 = max(0, int(round((origin_x - simulator.belief_info.map_origin_x) / simulator.cell_size)))
        y0 = max(0, int(round((origin_y - simulator.belief_info.map_origin_y) / simulator.cell_size)))
        x1 = min(simulator.robot_belief.shape[1] - 1, int(round((top_x - simulator.belief_info.map_origin_x) / simulator.cell_size)))
        y1 = min(simulator.robot_belief.shape[0] - 1, int(round((top_y - simulator.belief_info.map_origin_y) / simulator.cell_size)))
        return MapInfo(simulator.robot_belief[y0 : y1 + 1, x0 : x1 + 1], origin_x, origin_y, simulator.cell_size)

    def _belief_arrays(self):
        self._require_ready()
        assert self.node_manager is not None and self.simulator is not None
        nodes = [item.data for item in self.node_manager.nodes_dict.__iter__()]
        coords = np.asarray([node.coords for node in nodes], dtype=np.float32).reshape(-1, 2)
        utility = np.asarray([node.utility for node in nodes], dtype=np.float32)
        visited = np.asarray([node.visited for node in nodes], dtype=np.float32)
        lookup = {complex(float(x), float(y)): index for index, (x, y) in enumerate(coords)}
        adjacency = np.zeros((len(nodes), len(nodes)), dtype=np.bool_)
        for index, node in enumerate(nodes):
            for neighbor in node.neighbor_set:
                target = lookup.get(complex(float(neighbor[0]), float(neighbor[1])))
                if target is not None:
                    adjacency[index, target] = True
        current = lookup[complex(float(self.simulator.robot_location[0]), float(self.simulator.robot_location[1]))]
        candidates = np.flatnonzero(adjacency[current])
        candidates = np.asarray([item for item in candidates if item != current], dtype=np.int64)
        return coords, utility, visited, adjacency, current, candidates

    def _events(self, coords, utility, visited, adjacency, candidates) -> np.ndarray:
        events = np.zeros(self.candidate_padding, dtype=np.int16)
        for slot, candidate in enumerate(candidates[: self.candidate_padding]):
            stable_id = stable_coordinate_id(*coords[candidate])
            previous_utility = self._previous_utility.get(stable_id)
            if previous_utility is not None and previous_utility != float(utility[candidate]):
                events[slot] |= int(GraphEvent.FRONTIER_CHANGED)
            neighbor_ids = {
                stable_coordinate_id(*coords[target]) for target in np.flatnonzero(adjacency[candidate])
            }
            previous_neighbors = self._previous_neighbors.get(stable_id)
            if previous_neighbors is not None and previous_neighbors != neighbor_ids:
                events[slot] |= int(GraphEvent.EDGE_INVALIDATED)
            if bool(visited[candidate]):
                events[slot] |= int(GraphEvent.VISITED)
        return events

    def _build_graph_state(
        self,
        coords: np.ndarray,
        features: np.ndarray,
        adjacency: np.ndarray,
        current: int,
        candidates: np.ndarray,
        max_nodes: int,
        events: np.ndarray | None = None,
        metadata: dict | None = None,
    ) -> ExplorationState:
        node_count = len(coords)
        adjacency = np.asarray(adjacency, dtype=np.bool_).copy()
        np.fill_diagonal(adjacency, True)
        raw_nodes = node_count if self.hierarchy and features.shape[1] == 4 else max_nodes
        if node_count > raw_nodes:
            raise RuntimeError(
                f"graph has {node_count} nodes but budget is {raw_nodes}; enable hierarchy or raise padding"
            )
        node_features = np.zeros((raw_nodes, features.shape[1]), dtype=np.float32)
        node_features[:node_count] = features
        node_xy = np.zeros((raw_nodes, 2), dtype=np.float32)
        node_xy[:node_count] = coords
        node_mask = np.zeros(raw_nodes, dtype=np.bool_)
        node_mask[:node_count] = True
        padded_adjacency = np.zeros((raw_nodes, raw_nodes), dtype=np.bool_)
        padded_adjacency[:node_count, :node_count] = adjacency
        stable_ids = np.zeros(raw_nodes, dtype=np.int64)
        stable_ids[:node_count] = [stable_coordinate_id(*xy) for xy in coords]
        candidate_indices = np.zeros(self.candidate_padding, dtype=np.int64)
        candidate_mask = np.zeros(self.candidate_padding, dtype=np.bool_)
        valid_candidates = candidates[: self.candidate_padding]
        candidate_indices[: len(valid_candidates)] = valid_candidates
        candidate_mask[: len(valid_candidates)] = True
        edge_features = np.zeros((self.candidate_padding, 4), dtype=np.float32)
        if len(valid_candidates):
            delta = coords[valid_candidates] - coords[current]
            distance = np.linalg.norm(delta, axis=1)
            edge_features[: len(valid_candidates)] = np.column_stack(
                (delta / UPDATING_MAP_SIZE, distance / UPDATING_MAP_SIZE, distance / UPDATING_MAP_SIZE)
            )
        if events is None:
            events = np.zeros(self.candidate_padding, dtype=np.int16)
        state = ExplorationState(
            node_features=torch.from_numpy(node_features).unsqueeze(0),
            node_xy=torch.from_numpy(node_xy).unsqueeze(0),
            node_mask=torch.from_numpy(node_mask).unsqueeze(0),
            adjacency=torch.from_numpy(padded_adjacency).unsqueeze(0),
            stable_ids=torch.from_numpy(stable_ids).unsqueeze(0),
            current_index=torch.tensor([current]),
            candidate_indices=torch.from_numpy(candidate_indices).unsqueeze(0),
            candidate_mask=torch.from_numpy(candidate_mask).unsqueeze(0),
            edge_features=torch.from_numpy(edge_features).unsqueeze(0),
            candidate_events=torch.from_numpy(events).unsqueeze(0),
            metadata=metadata,
        ).validate()
        if self.hierarchy and features.shape[1] == 4:
            state = action_preserving_context(
                state,
                local_budget=self.local_budget,
                region_budget=self.region_budget,
                region_size_m=self.region_size_m,
            )
        return state

    def _build_actor_state(self) -> ExplorationState:
        assert self.simulator is not None
        coords, utility, visited, adjacency, current, candidates = self._belief_arrays()
        relative = (coords - coords[current]) / UPDATING_MAP_SIZE / 2.0
        features = np.column_stack((relative, utility / UTILITY_NORMALIZER, visited)).astype(np.float32)
        events = self._events(coords, utility, visited, adjacency, candidates)
        state = self._build_graph_state(
            coords,
            features,
            adjacency,
            current,
            candidates,
            self.node_padding,
            events,
            metadata={"map_id": self.simulator.map_path.name, "step": self.step_index},
        )
        self._actor_coords = coords
        for index, coord in enumerate(coords):
            key = stable_coordinate_id(*coord)
            self._previous_utility[key] = float(utility[index])
            self._previous_neighbors[key] = {
                stable_coordinate_id(*coords[target]) for target in np.flatnonzero(adjacency[index])
            }
        return state

    def _build_critic_state(self) -> ExplorationState:
        assert self.gt_manager is not None and self.simulator is not None and self.node_manager is not None
        self.gt_manager.update_graph()
        explored_coords = [item.data.coords for item in self.node_manager.nodes_dict.__iter__()]
        hidden_coords = [item.data.coords for item in self.gt_manager.nodes_dict.__iter__() if item.data.explored == 0]
        coords = np.asarray(explored_coords + hidden_coords, dtype=np.float32).reshape(-1, 2)
        lookup = {complex(float(x), float(y)): index for index, (x, y) in enumerate(coords)}
        utility, explored, visited = [], [], []
        adjacency = np.zeros((len(coords), len(coords)), dtype=np.bool_)
        for index, coord in enumerate(coords):
            node = self.gt_manager.nodes_dict.find((float(coord[0]), float(coord[1]))).data
            utility.append(node.utility)
            explored.append(node.explored)
            visited.append(node.visited)
            for neighbor in node.neighbor_set:
                target = lookup.get(complex(float(neighbor[0]), float(neighbor[1])))
                if target is not None:
                    adjacency[index, target] = True
        current = lookup[complex(float(self.simulator.robot_location[0]), float(self.simulator.robot_location[1]))]
        assert self._last_state is not None or self._actor_coords is not None
        belief_coords, _, _, _, _, belief_candidates = self._belief_arrays()
        candidate_coords = belief_coords[belief_candidates]
        candidates = np.asarray(
            [lookup[complex(float(x), float(y))] for x, y in candidate_coords], dtype=np.int64
        )
        relative = (coords - coords[current]) / UPDATING_MAP_SIZE / 2.0
        features = np.column_stack(
            (
                relative,
                np.asarray(utility) / UTILITY_NORMALIZER,
                np.asarray(explored, dtype=np.float32),
                np.asarray(visited, dtype=np.float32),
            )
        ).astype(np.float32)
        return self._build_graph_state(
            coords,
            features,
            adjacency,
            current,
            candidates,
            self.critic_node_padding,
            metadata={"map_id": self.simulator.map_path.name, "privileged": True, "step": self.step_index},
        )

    def step(self, action_slot: int) -> StepResult:
        self._require_ready()
        assert self._last_state is not None and self._actor_coords is not None and self.simulator is not None
        if action_slot < 0 or action_slot >= self.candidate_padding or not bool(
            self._last_state.candidate_mask[0, action_slot]
        ):
            raise ValueError(f"invalid action slot {action_slot}")
        compact_index = int(self._last_state.candidate_indices[0, action_slot])
        if self.hierarchy:
            compaction = self._last_state.metadata["compaction"][0]
            original_index = compaction["candidate_old_indices"][
                list(np.flatnonzero(self._last_state.candidate_mask[0].cpu().numpy())).index(action_slot)
            ]
        else:
            original_index = compact_index
        next_waypoint = self._actor_coords[original_index]
        previous_frontiers = len(self.simulator.global_frontiers)
        previous_distance = self.simulator.travel_dist
        reward = float(self.simulator.step(next_waypoint))
        self.step_index += 1
        self._update_belief_graph()
        actor_state = self._build_actor_state()
        critic_state = self._build_critic_state()
        frontier_count = len(self.simulator.global_frontiers)
        frontier_gain = max(0, previous_frontiers - frontier_count)
        distance = float(self.simulator.travel_dist - previous_distance)
        utility_sum = float(sum(item.data.utility for item in self.node_manager.nodes_dict.__iter__()))
        done = (
            utility_sum <= 0
            or self.simulator.explored_rate >= self.completion_threshold
            or self.step_index >= self.max_episode_steps
        )
        success = utility_sum <= 0 or self.simulator.explored_rate >= self.completion_threshold
        label_reward = reward
        if success:
            reward += self.terminal_reward
        self._last_state, self._last_critic = actor_state, critic_state
        return StepResult(
            state=actor_state,
            critic_state=critic_state,
            reward=reward,
            done=done,
            info={
                "frontier_gain": float(frontier_gain),
                "distance": distance,
                "label_reward": float(label_reward),
                "travel_distance": float(self.simulator.travel_dist),
                "explored_rate": float(self.simulator.explored_rate),
                "success": bool(success),
                "map_id": self.simulator.map_path.name,
                "step": self.step_index,
            },
        )
