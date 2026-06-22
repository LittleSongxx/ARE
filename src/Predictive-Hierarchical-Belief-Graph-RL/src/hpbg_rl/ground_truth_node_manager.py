from __future__ import annotations

import numpy as np
import torch

from . import quads
from .corridor_refinement import build_refined_adjacency_matrix
from .belief_state import PredictionResult, zero_belief_features
from .expert_reward import compute_oracle_node_targets
from .hierarchical_graph import build_hierarchical_graph
from .map_prediction import HeuristicMapPredictor
from .parameter import (
    CELL_SIZE,
    CRITIC_NODE_INPUT_DIM,
    FREE,
    FRONTIER_CELL_SIZE,
    K_SIZE,
    NODE_PADDING_SIZE,
    NODE_RESOLUTION,
    RuntimeConfig,
    SENSOR_RANGE,
    UPDATING_MAP_SIZE,
)
from .utils import check_collision, get_cell_position_from_coords, get_frontier_in_map


def coords_to_lookup_key(coords, digits: int = 3):
    coords = np.asarray(coords, dtype=np.float64).reshape(-1)
    return tuple(np.round(coords[:2], digits).tolist())


def build_actor_to_critic_index(
    actor_node_coords,
    critic_node_coords,
    padded_size: int | None = None,
    invalid_index: int = -1,
    digits: int = 3,
):
    actor_node_coords = np.asarray(actor_node_coords, dtype=np.float64).reshape(-1, 2)
    critic_node_coords = np.asarray(critic_node_coords, dtype=np.float64).reshape(-1, 2)

    output_size = int(padded_size) if padded_size is not None else actor_node_coords.shape[0]
    index_map = np.full((output_size,), int(invalid_index), dtype=np.int64)
    critic_lookup = {
        coords_to_lookup_key(coords, digits=digits): index for index, coords in enumerate(critic_node_coords)
    }

    for actor_index, coords in enumerate(actor_node_coords):
        index_map[actor_index] = critic_lookup.get(coords_to_lookup_key(coords, digits=digits), invalid_index)
    return index_map


class GroundTruthNodeManager:
    def __init__(
        self,
        node_manager,
        ground_truth_map_info,
        device="cpu",
        plot=False,
        runtime_config: RuntimeConfig | None = None,
    ):
        self.runtime_config = runtime_config or RuntimeConfig()
        self.nodes_dict = quads.QuadTree((0, 0), 1000, 1000)
        self.node_manager = node_manager
        self.ground_truth_map_info = ground_truth_map_info
        self.ground_truth_node_coords = None
        self.ground_truth_node_utility = None
        self.explored_sign = None
        self.device = device
        self.plot = plot
        self.enable_corridor_graph_compression = self.runtime_config.enable_corridor_graph_compression
        self.enable_corridor_edge_pruning = self.runtime_config.enable_corridor_edge_pruning
        self.corridor_max_width = self.runtime_config.corridor_max_width
        self.corridor_min_length = self.runtime_config.corridor_min_length
        self.map_predictor = HeuristicMapPredictor(
            sensor_range=SENSOR_RANGE,
            risk_weight=self.runtime_config.hpbg_risk_weight,
        )

        self.initialize_graph()

    def _observed_utility_prediction(self, utility: np.ndarray, n_nodes: int) -> PredictionResult:
        if n_nodes == 0:
            empty = np.zeros(0, dtype=np.float32)
            return PredictionResult(empty, empty, empty)
        normalizer = max(float(SENSOR_RANGE * 3.14 // FRONTIER_CELL_SIZE), 1.0)
        observed_utility = np.clip(np.maximum(np.asarray(utility, dtype=np.float32).reshape(n_nodes), 0.0) / normalizer, 0.0, 1.0)
        uncertainty = np.zeros(n_nodes, dtype=np.float32)
        return PredictionResult(observed_utility, uncertainty, observed_utility.copy())

    def _build_critic_online_features(
        self,
        node_coords: np.ndarray,
        utility: np.ndarray,
        guidepost: np.ndarray,
        adjacent_matrix: np.ndarray,
        belief_map_info,
    ) -> np.ndarray:
        n_nodes = int(np.asarray(node_coords).reshape(-1, 2).shape[0])
        if n_nodes == 0 or not self.runtime_config.use_hpbg:
            return zero_belief_features(n_nodes)

        if self.runtime_config.use_map_prediction and belief_map_info is not None:
            frontier = get_frontier_in_map(belief_map_info)
            prediction = self.map_predictor.predict(
                node_coords,
                utility,
                guidepost,
                belief_map_info,
                frontier=frontier,
                utility_normalizer=max(float(SENSOR_RANGE * 3.14 // FRONTIER_CELL_SIZE), 1.0),
            )
        else:
            prediction = self._observed_utility_prediction(utility, n_nodes)

        cluster_prior = np.zeros(n_nodes, dtype=np.float32)
        if self.runtime_config.use_hierarchical_graph:
            hierarchical_graph = build_hierarchical_graph(
                node_coords,
                adjacent_matrix,
                node_scores=prediction.risk_aware_utility,
                cluster_resolution=self.runtime_config.hpbg_cluster_resolution,
                cluster_edge_hops=self.runtime_config.hpbg_cluster_edge_hops,
            )
            cluster_prior = hierarchical_graph.node_cluster_prior

        risk_aware = np.clip(prediction.risk_aware_utility + 0.15 * cluster_prior, -1.0, 1.0)
        features = np.stack(
            (
                prediction.predicted_utility,
                prediction.uncertainty,
                risk_aware,
                np.clip(cluster_prior, 0.0, 1.0),
            ),
            axis=-1,
        ).astype(np.float32)
        return np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

    def _build_privileged_features(
        self,
        node_coords: np.ndarray,
        explored_sign: np.ndarray,
        belief_map_info,
    ) -> np.ndarray:
        n_nodes = int(np.asarray(node_coords).reshape(-1, 2).shape[0])
        if n_nodes == 0:
            return np.zeros((n_nodes, 3), dtype=np.float32)
        explored = np.asarray(explored_sign, dtype=np.float32).reshape(n_nodes)
        if not self.runtime_config.use_hpbg:
            return np.stack(
                (
                    explored,
                    np.zeros(n_nodes, dtype=np.float32),
                    np.zeros(n_nodes, dtype=np.float32),
                ),
                axis=-1,
            ).astype(np.float32)
        oracle_targets = compute_oracle_node_targets(
            node_coords,
            self.ground_truth_map_info,
            belief_map_info,
            sensor_range=SENSOR_RANGE,
        )
        features = np.stack(
            (
                np.asarray(explored_sign, dtype=np.float32).reshape(n_nodes),
                oracle_targets.oracle_utility,
                oracle_targets.expert_potential,
            ),
            axis=-1,
        ).astype(np.float32)
        return np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)

    def get_ground_truth_observation(self, robot_location, belief_map_info=None):
        self.update_graph()

        all_node_coords = []
        for node in self.node_manager.nodes_dict.__iter__():
            all_node_coords.append(node.data.coords)
        for node in self.nodes_dict.__iter__():
            if node.data.explored == 0:
                all_node_coords.append(node.data.coords)
        all_node_coords = np.array(all_node_coords).reshape(-1, 2)
        utility = []
        explored_sign = []
        guidepost = []

        n_nodes = all_node_coords.shape[0]
        adjacent_matrix = np.ones((n_nodes, n_nodes)).astype(int)
        node_coords_to_check = all_node_coords[:, 0] + all_node_coords[:, 1] * 1j
        for i, coords in enumerate(all_node_coords):
            node = self.nodes_dict.find((coords[0], coords[1])).data
            utility.append(node.utility)
            explored_sign.append(node.explored)
            guidepost.append(node.visited)
            for neighbor in node.neighbor_set:
                index = np.argwhere(node_coords_to_check == neighbor[0] + neighbor[1] * 1j)[0][0]
                adjacent_matrix[i, index] = 0

        adjacent_matrix = build_refined_adjacency_matrix(
            all_node_coords,
            adjacent_matrix,
            self.ground_truth_map_info,
            enable_edge_pruning=self.enable_corridor_edge_pruning,
            enable_graph_compression=self.enable_corridor_graph_compression,
            corridor_max_width=self.corridor_max_width,
            corridor_min_length=self.corridor_min_length,
        )

        utility = np.array(utility)
        explored_sign = np.array(explored_sign)
        guidepost = np.array(guidepost)

        current_index = np.argwhere(node_coords_to_check == robot_location[0] + robot_location[1] * 1j)[0][0]

        neighbor_indices = np.argwhere(adjacent_matrix[current_index] == 0).reshape(-1)

        self.ground_truth_node_coords = all_node_coords
        self.ground_truth_node_utility = utility
        self.explored_sign = explored_sign

        absolute_node_coords = all_node_coords
        node_coords = all_node_coords
        node_utility = utility.reshape(-1, 1)
        node_guidepost = guidepost.reshape(-1, 1)
        edge_mask = adjacent_matrix
        current_edge = neighbor_indices
        n_node = node_coords.shape[0]

        online_features = self._build_critic_online_features(
            absolute_node_coords,
            utility,
            guidepost,
            adjacent_matrix,
            belief_map_info,
        )
        privileged_features = self._build_privileged_features(
            absolute_node_coords,
            explored_sign,
            belief_map_info,
        )

        current_node_coords = node_coords[current_index]
        node_coords = np.concatenate(
            (
                node_coords[:, 0].reshape(-1, 1) - current_node_coords[0],
                node_coords[:, 1].reshape(-1, 1) - current_node_coords[1],
            ),
            axis=-1,
        ) / UPDATING_MAP_SIZE / 2
        node_utility = node_utility / (SENSOR_RANGE * 3.14 // FRONTIER_CELL_SIZE)
        node_inputs = np.concatenate((node_coords, node_utility, node_guidepost, online_features, privileged_features), axis=1)
        if node_inputs.shape[1] != CRITIC_NODE_INPUT_DIM:
            raise RuntimeError(f"critic node input dim mismatch: {node_inputs.shape[1]} != {CRITIC_NODE_INPUT_DIM}")
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)

        assert node_coords.shape[0] < NODE_PADDING_SIZE, (node_coords.shape[0], NODE_PADDING_SIZE)
        padding = torch.nn.ZeroPad2d((0, 0, 0, NODE_PADDING_SIZE - n_node))
        node_inputs = padding(node_inputs)

        node_padding_mask = torch.zeros((1, 1, n_node), dtype=torch.int16).to(self.device)
        node_padding = torch.ones((1, 1, NODE_PADDING_SIZE - n_node), dtype=torch.int16).to(self.device)
        node_padding_mask = torch.cat((node_padding_mask, node_padding), dim=-1)

        edge_mask = torch.tensor(edge_mask).unsqueeze(0).to(self.device)
        padding = torch.nn.ConstantPad2d((0, NODE_PADDING_SIZE - n_node, 0, NODE_PADDING_SIZE - n_node), 1)
        edge_mask = padding(edge_mask)

        current_in_edge = np.argwhere(current_edge == current_index)[0][0]
        current_edge = torch.tensor(current_edge).unsqueeze(0)
        k_size = current_edge.size()[-1]
        padding = torch.nn.ConstantPad1d((0, K_SIZE - k_size), 0)
        current_edge = padding(current_edge).unsqueeze(-1)

        edge_padding_mask = torch.zeros((1, 1, k_size), dtype=torch.int16).to(self.device)
        edge_padding_mask[0, 0, current_in_edge] = 1
        padding = torch.nn.ConstantPad1d((0, K_SIZE - k_size), 1)
        edge_padding_mask = padding(edge_padding_mask)

        current_index = torch.tensor([current_index]).reshape(1, 1, 1).to(self.device)

        return [node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask]

    def add_node_to_dict(self, coords):
        key = (coords[0], coords[1])
        node = Node(coords)
        self.nodes_dict.insert(point=key, data=node)
        return node

    def initialize_graph(self):
        node_coords = self.get_ground_truth_node_coords(self.ground_truth_map_info)
        for coords in node_coords:
            self.add_node_to_dict(coords)

        for node in self.nodes_dict.__iter__():
            node.data.get_neighbor_nodes(self.ground_truth_map_info, self.nodes_dict)

    def update_graph(self):
        for node in self.node_manager.nodes_dict.__iter__():
            coords = node.data.coords
            ground_truth_node = self.nodes_dict.find(coords.tolist())
            ground_truth_node.data.utility = node.data.utility
            ground_truth_node.data.explored = 1
            ground_truth_node.data.visited = node.data.visited

    def get_ground_truth_node_coords(self, ground_truth_map_info):
        x_min = ground_truth_map_info.map_origin_x
        y_min = ground_truth_map_info.map_origin_y
        x_max = ground_truth_map_info.map_origin_x + (ground_truth_map_info.map.shape[1] - 1) * CELL_SIZE
        y_max = ground_truth_map_info.map_origin_y + (ground_truth_map_info.map.shape[0] - 1) * CELL_SIZE

        if x_min % NODE_RESOLUTION != 0:
            x_min = (x_min // NODE_RESOLUTION + 1) * NODE_RESOLUTION
        if x_max % NODE_RESOLUTION != 0:
            x_max = x_max // NODE_RESOLUTION * NODE_RESOLUTION
        if y_min % NODE_RESOLUTION != 0:
            y_min = (y_min // NODE_RESOLUTION + 1) * NODE_RESOLUTION
        if y_max % NODE_RESOLUTION != 0:
            y_max = y_max // NODE_RESOLUTION * NODE_RESOLUTION

        x_coords = np.arange(x_min, x_max + 0.1, NODE_RESOLUTION)
        y_coords = np.arange(y_min, y_max + 0.1, NODE_RESOLUTION)
        t1, t2 = np.meshgrid(x_coords, y_coords)
        nodes = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
        nodes = np.around(nodes, 1)

        indices = []
        nodes_cells = get_cell_position_from_coords(nodes, ground_truth_map_info).reshape(-1, 2)
        for i, cell in enumerate(nodes_cells):
            assert 0 <= cell[1] < ground_truth_map_info.map.shape[0] and 0 <= cell[0] < ground_truth_map_info.map.shape[1]
            if ground_truth_map_info.map[cell[1], cell[0]] == FREE:
                indices.append(i)
        nodes = nodes[np.array(indices)].reshape(-1, 2)
        return nodes

    def plot_ground_truth_env(self, robot_location):
        import matplotlib.pyplot as plt

        plt.subplot(1, 3, 3)
        plt.imshow(self.ground_truth_map_info.map, cmap="gray")
        plt.axis("off")
        robot = get_cell_position_from_coords(robot_location, self.ground_truth_map_info)
        nodes = get_cell_position_from_coords(self.ground_truth_node_coords, self.ground_truth_map_info)
        plt.imshow(self.ground_truth_map_info.map, cmap="gray")
        plt.scatter(nodes[:, 0], nodes[:, 1], c=self.explored_sign, zorder=2)
        plt.plot(robot[0], robot[1], "mo", markersize=16, zorder=5)


class Node:
    def __init__(self, coords):
        self.coords = coords
        self.utility = -(SENSOR_RANGE * 3.14 // FRONTIER_CELL_SIZE)
        self.explored = 0
        self.visited = 0

        self.neighbor_matrix = -np.ones((5, 5))
        self.neighbor_set = {(self.coords[0], self.coords[1])}

    def get_neighbor_nodes(self, ground_truth_map_info, nodes_dict):
        center_index = self.neighbor_matrix.shape[0] // 2
        for i in range(self.neighbor_matrix.shape[0]):
            for j in range(self.neighbor_matrix.shape[1]):
                if self.neighbor_matrix[i, j] != -1:
                    continue
                if i == center_index and j == center_index:
                    self.neighbor_matrix[i, j] = 1
                    continue

                neighbor_coords = np.around(
                    np.array(
                        [
                            self.coords[0] + (i - center_index) * NODE_RESOLUTION,
                            self.coords[1] + (j - center_index) * NODE_RESOLUTION,
                        ]
                    ),
                    1,
                )
                neighbor_node = nodes_dict.find((neighbor_coords[0], neighbor_coords[1]))
                if neighbor_node is None:
                    continue

                neighbor_node = neighbor_node.data
                collision = check_collision(self.coords, neighbor_coords, ground_truth_map_info)
                neighbor_matrix_x = center_index + (center_index - i)
                neighbor_matrix_y = center_index + (center_index - j)
                if not collision:
                    self.neighbor_matrix[i, j] = 1
                    self.neighbor_set.add((neighbor_coords[0], neighbor_coords[1]))
                    neighbor_node.neighbor_matrix[neighbor_matrix_x, neighbor_matrix_y] = 1
                    neighbor_node.neighbor_set.add((self.coords[0], self.coords[1]))
