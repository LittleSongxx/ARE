from __future__ import annotations

from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch

from .node_manager import NodeManager
from .parameter import (
    FRONTIER_CELL_SIZE,
    HISTORY_FEATURE_SET,
    HISTORY_INPUT_DIM,
    HISTORY_LEN,
    K_SIZE,
    NODE_PADDING_SIZE,
    SENSOR_RANGE,
    UPDATING_MAP_SIZE,
)
from .utils import MapInfo, get_cell_position_from_coords, get_frontier_in_map


class HistoryFeatureBuilder:
    def __init__(self, feature_set: tuple[str, ...] | list[str] | str):
        if isinstance(feature_set, str):
            parsed = [feature.strip() for feature in feature_set.split(",") if feature.strip()]
        else:
            parsed = [str(feature).strip() for feature in feature_set if str(feature).strip()]
        self.feature_set = tuple(parsed) if parsed else HISTORY_FEATURE_SET
        self.utility_norm_denom = max(float(SENSOR_RANGE * 3.14 // FRONTIER_CELL_SIZE), 1.0)

    def build(
        self,
        utility: np.ndarray,
        guidepost: np.ndarray,
        frontier_count: int,
        frontier_count_delta: int,
        feedback: dict[str, float],
    ) -> np.ndarray:
        utility = np.asarray(utility, dtype=np.float32).reshape(-1)
        guidepost = np.asarray(guidepost, dtype=np.float32).reshape(-1)

        n_nodes = float(max(utility.size, 1))
        utility_norm = utility / self.utility_norm_denom

        feature_values = {
            "utility_sum": float(np.sum(utility_norm)),
            "utility_mean": float(np.mean(utility_norm)) if utility_norm.size > 0 else 0.0,
            "frontier_ratio": float(np.sum(utility > 0) / n_nodes),
            "visited_ratio": float(np.mean(guidepost)) if guidepost.size > 0 else 0.0,
            "frontier_count": float(frontier_count / n_nodes),
            "frontier_count_delta": float(frontier_count_delta / n_nodes),
            "reward_proxy": float(feedback.get("reward_proxy", 0.0)),
            "explored_delta": float(feedback.get("explored_delta", 0.0)),
            "travel_delta": float(feedback.get("travel_delta", 0.0)),
            "action_repeat": float(feedback.get("action_repeat", 0.0)),
            "oscillation": float(feedback.get("oscillation", 0.0)),
            "utility_to_cost": float(feedback.get("utility_to_cost", 0.0)),
        }

        values = [feature_values.get(name, 0.0) for name in self.feature_set]
        return np.asarray(values, dtype=np.float32)


class Agent:
    def __init__(
        self,
        policy_net,
        device="cpu",
        plot=False,
        history_len=HISTORY_LEN,
        history_input_dim=HISTORY_INPUT_DIM,
        history_feature_set: tuple[str, ...] | list[str] | str = HISTORY_FEATURE_SET,
    ):
        self.device = device
        self.policy_net = policy_net
        self.plot = plot

        self.location = None
        self.map_info = None

        self.cell_size = None
        self.node_resolution = None
        self.updating_map_size = UPDATING_MAP_SIZE

        self.updating_map_info = None
        self.frontier = set()

        self.node_manager = NodeManager(plot=self.plot)

        self.node_coords = None
        self.utility = None
        self.guidepost = None
        self.current_index = None
        self.adjacent_matrix = None
        self.neighbor_indices = None

        self.history_len = max(int(history_len), 1)
        self.history_input_dim = max(int(history_input_dim), 1)
        self.history_buffer = deque(maxlen=self.history_len)
        self.history_feature_builder = HistoryFeatureBuilder(history_feature_set)

        self._previous_frontier_count = None
        self._pending_feedback = self._zero_feedback()
        self._last_selected_node = None
        self._prev_selected_node = None

    @staticmethod
    def _zero_feedback() -> dict[str, float]:
        return {
            "reward_proxy": 0.0,
            "explored_delta": 0.0,
            "travel_delta": 0.0,
            "action_repeat": 0.0,
            "oscillation": 0.0,
            "utility_to_cost": 0.0,
        }

    def reset_history(self) -> None:
        self.history_buffer.clear()
        self._previous_frontier_count = None
        self._pending_feedback = self._zero_feedback()
        self._last_selected_node = None
        self._prev_selected_node = None

    def record_transition_feedback(
        self,
        reward_proxy: float,
        explored_delta: float,
        travel_delta: float,
        selected_node_index: int | None = None,
        selected_node_utility: float | None = None,
    ) -> None:
        action_repeat = 0.0
        oscillation = 0.0

        if selected_node_index is not None:
            selected_node_index = int(selected_node_index)
            action_repeat = 1.0 if self._last_selected_node == selected_node_index else 0.0
            oscillation = (
                1.0
                if self._prev_selected_node is not None
                and self._prev_selected_node == selected_node_index
                and self._last_selected_node != selected_node_index
                else 0.0
            )
            self._prev_selected_node = self._last_selected_node
            self._last_selected_node = selected_node_index

        selected_utility = float(selected_node_utility or 0.0)
        travel_abs = max(abs(float(travel_delta)), 1e-6)
        utility_to_cost = max(min(selected_utility / travel_abs, 10.0), -10.0)

        self._pending_feedback = {
            "reward_proxy": float(reward_proxy),
            "explored_delta": float(explored_delta),
            "travel_delta": float(travel_delta),
            "action_repeat": float(action_repeat),
            "oscillation": float(oscillation),
            "utility_to_cost": float(utility_to_cost),
        }

    def update_map(self, map_info: MapInfo) -> None:
        self.map_info = map_info

    def update_updating_map(self, location) -> None:
        self.updating_map_info = self.get_updating_map(location)

    def update_location(self, location) -> None:
        self.location = location
        node = self.node_manager.nodes_dict.find(location.tolist())
        if self.node_manager.nodes_dict.__len__() != 0:
            node.data.set_visited()

    def update_frontiers(self) -> None:
        self.frontier = get_frontier_in_map(self.updating_map_info)

    def get_updating_map(self, location) -> MapInfo:
        updating_map_origin_x = location[0] - self.updating_map_size / 2
        updating_map_origin_y = location[1] - self.updating_map_size / 2

        updating_map_top_x = updating_map_origin_x + self.updating_map_size
        updating_map_top_y = updating_map_origin_y + self.updating_map_size

        min_x = self.map_info.map_origin_x
        min_y = self.map_info.map_origin_y
        max_x = self.map_info.map_origin_x + self.map_info.cell_size * (self.map_info.map.shape[1] - 1)
        max_y = self.map_info.map_origin_y + self.map_info.cell_size * (self.map_info.map.shape[0] - 1)

        updating_map_origin_x = max(updating_map_origin_x, min_x)
        updating_map_origin_y = max(updating_map_origin_y, min_y)
        updating_map_top_x = min(updating_map_top_x, max_x)
        updating_map_top_y = min(updating_map_top_y, max_y)

        updating_map_origin_x = (updating_map_origin_x // self.map_info.cell_size + 1) * self.map_info.cell_size
        updating_map_origin_y = (updating_map_origin_y // self.map_info.cell_size + 1) * self.map_info.cell_size
        updating_map_top_x = (updating_map_top_x // self.map_info.cell_size) * self.map_info.cell_size
        updating_map_top_y = (updating_map_top_y // self.map_info.cell_size) * self.map_info.cell_size

        updating_map_origin_x = np.round(updating_map_origin_x, 1)
        updating_map_origin_y = np.round(updating_map_origin_y, 1)
        updating_map_top_x = np.round(updating_map_top_x, 1)
        updating_map_top_y = np.round(updating_map_top_y, 1)

        updating_map_origin = np.array([updating_map_origin_x, updating_map_origin_y])
        updating_map_origin_in_global_map = get_cell_position_from_coords(updating_map_origin, self.map_info)

        updating_map_top = np.array([updating_map_top_x, updating_map_top_y])
        updating_map_top_in_global_map = get_cell_position_from_coords(updating_map_top, self.map_info)

        updating_map = self.map_info.map[
            updating_map_origin_in_global_map[1] : updating_map_top_in_global_map[1] + 1,
            updating_map_origin_in_global_map[0] : updating_map_top_in_global_map[0] + 1,
        ]

        return MapInfo(updating_map, updating_map_origin_x, updating_map_origin_y, self.map_info.cell_size)

    def update_planning_state(self, global_map_info, location) -> None:
        self.update_map(global_map_info)
        self.update_location(location)
        self.update_updating_map(self.location)
        self.update_frontiers()
        self.node_manager.update_graph(
            self.location,
            self.frontier,
            self.updating_map_info,
            self.map_info,
        )
        (
            self.node_coords,
            self.utility,
            self.guidepost,
            self.adjacent_matrix,
            self.current_index,
            self.neighbor_indices,
        ) = self.update_observation()
        self._update_history_buffer()

    def update_observation(self):
        all_node_coords = []
        for node in self.node_manager.nodes_dict.__iter__():
            all_node_coords.append(node.data.coords)
        all_node_coords = np.array(all_node_coords).reshape(-1, 2)

        utility = []
        guidepost = []

        n_nodes = all_node_coords.shape[0]
        adjacent_matrix = np.ones((n_nodes, n_nodes)).astype(int)
        node_coords_to_check = all_node_coords[:, 0] + all_node_coords[:, 1] * 1j
        for i, coords in enumerate(all_node_coords):
            node = self.node_manager.nodes_dict.find((coords[0], coords[1])).data
            utility.append(node.utility)
            guidepost.append(node.visited)
            for neighbor in node.neighbor_set:
                index = np.argwhere(node_coords_to_check == neighbor[0] + neighbor[1] * 1j)
                index = index[0][0]
                adjacent_matrix[i, index] = 0

        utility = np.array(utility)
        guidepost = np.array(guidepost)

        current_index = np.argwhere(node_coords_to_check == self.location[0] + self.location[1] * 1j)[0][0]
        neighbor_indices = np.argwhere(adjacent_matrix[current_index] == 0).reshape(-1)
        return all_node_coords, utility, guidepost, adjacent_matrix, current_index, neighbor_indices

    def _update_history_buffer(self) -> None:
        if self.utility is None or self.guidepost is None:
            return

        frontier_count = int(len(self.frontier))
        previous = frontier_count if self._previous_frontier_count is None else int(self._previous_frontier_count)
        frontier_count_delta = frontier_count - previous
        self._previous_frontier_count = frontier_count

        history_vec = self.history_feature_builder.build(
            utility=np.asarray(self.utility, dtype=np.float32),
            guidepost=np.asarray(self.guidepost, dtype=np.float32),
            frontier_count=frontier_count,
            frontier_count_delta=frontier_count_delta,
            feedback=self._pending_feedback,
        )

        if self.history_input_dim < history_vec.shape[0]:
            history_vec = history_vec[: self.history_input_dim]
        elif self.history_input_dim > history_vec.shape[0]:
            history_vec = np.pad(history_vec, (0, self.history_input_dim - history_vec.shape[0]))

        self.history_buffer.append(history_vec)
        self._pending_feedback = self._zero_feedback()

    def _get_history_inputs_tensor(self) -> torch.Tensor:
        history = np.zeros((self.history_len, self.history_input_dim), dtype=np.float32)
        if len(self.history_buffer) > 0:
            items = np.array(list(self.history_buffer), dtype=np.float32)
            history[-len(items) :] = items
        return torch.as_tensor(history, dtype=torch.float32, device=self.device).unsqueeze(0)

    def get_observation(self):
        node_coords = self.node_coords
        node_utility = self.utility.reshape(-1, 1)
        node_guidepost = self.guidepost.reshape(-1, 1)
        current_index = self.current_index
        edge_mask = self.adjacent_matrix
        current_edge = self.neighbor_indices
        n_node = node_coords.shape[0]

        current_node_coords = node_coords[self.current_index]
        node_coords = np.concatenate(
            (
                node_coords[:, 0].reshape(-1, 1) - current_node_coords[0],
                node_coords[:, 1].reshape(-1, 1) - current_node_coords[1],
            ),
            axis=-1,
        ) / UPDATING_MAP_SIZE
        node_utility = node_utility / (SENSOR_RANGE * 3.14 // FRONTIER_CELL_SIZE)
        node_inputs = np.concatenate((node_coords, node_utility, node_guidepost), axis=1)
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)

        assert node_coords.shape[0] < NODE_PADDING_SIZE, (node_coords.shape[0], NODE_PADDING_SIZE)
        padding = torch.nn.ZeroPad2d((0, 0, 0, NODE_PADDING_SIZE - n_node))
        node_inputs = padding(node_inputs)

        node_padding_mask = torch.zeros((1, 1, n_node), dtype=torch.int16).to(self.device)
        node_padding = torch.ones((1, 1, NODE_PADDING_SIZE - n_node), dtype=torch.int16).to(self.device)
        node_padding_mask = torch.cat((node_padding_mask, node_padding), dim=-1)

        current_index = torch.tensor([current_index], dtype=torch.long, device=self.device).reshape(1, 1, 1)

        edge_mask = torch.tensor(edge_mask, dtype=torch.int16, device=self.device).unsqueeze(0)
        padding = torch.nn.ConstantPad2d((0, NODE_PADDING_SIZE - n_node, 0, NODE_PADDING_SIZE - n_node), 1)
        edge_mask = padding(edge_mask)

        current_in_edge = np.argwhere(current_edge == self.current_index)
        current_in_edge = int(current_in_edge[0][0]) if current_in_edge.size > 0 else 0

        current_edge = torch.tensor(current_edge, dtype=torch.long, device=self.device).unsqueeze(0)
        if current_edge.size(-1) > K_SIZE:
            current_edge = current_edge[:, :K_SIZE]
        k_size = current_edge.size()[-1]
        padding = torch.nn.ConstantPad1d((0, K_SIZE - k_size), 0)
        current_edge = padding(current_edge).unsqueeze(-1)

        edge_padding_mask = torch.zeros((1, 1, k_size), dtype=torch.int16).to(self.device)
        if k_size > 0:
            current_in_edge = min(current_in_edge, k_size - 1)
            edge_padding_mask[0, 0, current_in_edge] = 1
        padding = torch.nn.ConstantPad1d((0, K_SIZE - k_size), 1)
        edge_padding_mask = padding(edge_padding_mask)

        history_inputs = self._get_history_inputs_tensor()

        return [
            node_inputs,
            node_padding_mask,
            edge_mask,
            current_index,
            current_edge,
            edge_padding_mask,
            history_inputs,
        ]

    def select_next_waypoint(self, observation, greedy=False):
        _, _, _, _, current_edge, _, _ = observation
        with torch.no_grad():
            logp = self.policy_net(*observation)

        if greedy:
            action_index = torch.argmax(logp, dim=1).long()
        else:
            action_index = torch.multinomial(logp.exp(), 1).long().squeeze(1)

        next_node_index = int(current_edge[0, action_index.item(), 0].item())
        next_position = self.node_coords[next_node_index]

        return next_position, action_index, next_node_index

    def plot_env(self):
        plt.switch_backend("agg")

        plt.figure(figsize=(18, 5))
        plt.subplot(1, 3, 2)
        nodes = get_cell_position_from_coords(self.node_coords, self.map_info)
        if len(self.frontier) > 0:
            frontiers = get_cell_position_from_coords(np.array(list(self.frontier)), self.map_info).reshape(-1, 2)
            plt.scatter(frontiers[:, 0], frontiers[:, 1], c="r", s=2)
        robot = get_cell_position_from_coords(self.location, self.map_info)
        plt.imshow(self.map_info.map, cmap="gray")
        plt.axis("off")
        plt.scatter(nodes[:, 0], nodes[:, 1], c=self.utility, zorder=2)
        for node, utility in zip(nodes, self.utility):
            plt.text(node[0], node[1], str(utility), zorder=3)
        plt.plot(robot[0], robot[1], "mo", markersize=16, zorder=5)
        for coords in self.node_coords:
            node = self.node_manager.nodes_dict.find(coords.tolist()).data
            for neighbor_coords in node.neighbor_set:
                end = (np.array(neighbor_coords) - coords) / 2 + coords
                plt.plot(
                    (np.array([coords[0], end[0]]) - self.map_info.map_origin_x) / self.map_info.cell_size,
                    (np.array([coords[1], end[1]]) - self.map_info.map_origin_y) / self.map_info.cell_size,
                    "tan",
                    zorder=1,
                )

        plt.subplot(1, 3, 3)
        plt.imshow(self.map_info.map, cmap="gray")
        plt.axis("off")
        plt.scatter(nodes[:, 0], nodes[:, 1], c=self.guidepost, zorder=2)
        plt.plot(robot[0], robot[1], "mo", markersize=16, zorder=5)
