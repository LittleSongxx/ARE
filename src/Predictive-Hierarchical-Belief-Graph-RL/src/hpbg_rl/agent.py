from __future__ import annotations

import numpy as np
import torch

from .belief_state import BeliefNodeFeatures, BeliefStateTracker, PredictionResult, zero_belief_features
from .corridor_refinement import build_refined_adjacency_matrix
from .hierarchical_graph import build_hierarchical_graph
from .map_prediction import HeuristicMapPredictor
from .node_manager import NodeManager
from .parameter import (
    CELL_SIZE,
    FRONTIER_CELL_SIZE,
    K_SIZE,
    NODE_INPUT_DIM,
    NODE_PADDING_SIZE,
    NODE_RESOLUTION,
    RuntimeConfig,
    SENSOR_RANGE,
    UPDATING_MAP_SIZE,
    UTILITY_RANGE,
)
from .sparse_visualization import build_sparse_graph_view, build_sparse_graph_view_from_arrays
from .utils import MapInfo, get_cell_position_from_coords, get_frontier_in_map


class Agent:
    def __init__(self, policy_net, device="cpu", plot=False, runtime_config: RuntimeConfig | None = None):
        self.runtime_config = runtime_config or RuntimeConfig()
        self.device = torch.device(device)
        self.policy_net = policy_net
        self.plot = plot

        self.location = None
        self.map_info = None
        self.cell_size = CELL_SIZE
        self.node_resolution = NODE_RESOLUTION
        self.updating_map_size = UPDATING_MAP_SIZE
        self.enable_corridor_graph_compression = self.runtime_config.enable_corridor_graph_compression
        self.enable_corridor_edge_pruning = self.runtime_config.enable_corridor_edge_pruning
        self.corridor_max_width = self.runtime_config.corridor_max_width
        self.corridor_min_length = self.runtime_config.corridor_min_length
        self.updating_map_info = None
        self.frontier = set()
        self.node_manager = NodeManager(plot=self.plot)
        self.node_coords = None
        self.utility = None
        self.guidepost = None
        self.current_index = None
        self.adjacent_matrix = None
        self.neighbor_indices = None
        self.belief_features = None
        self.hierarchical_graph = None
        self.map_predictor = HeuristicMapPredictor(
            sensor_range=SENSOR_RANGE,
            risk_weight=self.runtime_config.hpbg_risk_weight,
        )
        self.belief_tracker = BeliefStateTracker(
            alpha=self.runtime_config.hpbg_belief_ema_alpha,
            risk_weight=self.runtime_config.hpbg_risk_weight,
        )

    def _policy_device(self) -> torch.device:
        try:
            return next(self.policy_net.parameters()).device
        except StopIteration:
            return self.device

    def _align_observation_device(self, observation):
        policy_device = self._policy_device()
        if self.device != policy_device:
            self.device = policy_device
        aligned = []
        for value in observation:
            if isinstance(value, torch.Tensor):
                aligned.append(value.to(policy_device))
            else:
                aligned.append(value)
        return aligned

    def update_map(self, map_info):
        self.map_info = map_info

    def update_updating_map(self, location):
        self.updating_map_info = self.get_updating_map(location)

    def update_location(self, location):
        self.location = location
        node = self.node_manager.nodes_dict.find(location.tolist())
        if self.node_manager.nodes_dict.__len__() != 0 and node is not None:
            node.data.set_visited()

    def update_frontiers(self):
        self.frontier = get_frontier_in_map(self.updating_map_info)

    def get_updating_map(self, location):
        updating_map_origin_x = location[0] - self.updating_map_size / 2
        updating_map_origin_y = location[1] - self.updating_map_size / 2
        updating_map_top_x = updating_map_origin_x + self.updating_map_size
        updating_map_top_y = updating_map_origin_y + self.updating_map_size

        min_x = self.map_info.map_origin_x
        min_y = self.map_info.map_origin_y
        max_x = self.map_info.map_origin_x + self.cell_size * (self.map_info.map.shape[1] - 1)
        max_y = self.map_info.map_origin_y + self.cell_size * (self.map_info.map.shape[0] - 1)

        if updating_map_origin_x < min_x:
            updating_map_origin_x = min_x
        if updating_map_origin_y < min_y:
            updating_map_origin_y = min_y
        if updating_map_top_x > max_x:
            updating_map_top_x = max_x
        if updating_map_top_y > max_y:
            updating_map_top_y = max_y

        updating_map_origin_x = (updating_map_origin_x // self.cell_size + 1) * self.cell_size
        updating_map_origin_y = (updating_map_origin_y // self.cell_size + 1) * self.cell_size
        updating_map_top_x = (updating_map_top_x // self.cell_size) * self.cell_size
        updating_map_top_y = (updating_map_top_y // self.cell_size) * self.cell_size

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

        return MapInfo(updating_map, updating_map_origin_x, updating_map_origin_y, self.cell_size)

    def update_planning_state(self, map_info, location):
        self.update_map(map_info)
        self.update_location(location)
        self.update_updating_map(self.location)
        self.update_frontiers()
        self.node_manager.update_graph(self.location, self.frontier, self.updating_map_info, self.map_info)
        (
            self.node_coords,
            self.utility,
            self.guidepost,
            self.adjacent_matrix,
            self.current_index,
            self.neighbor_indices,
        ) = self.update_observation()

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
                index = np.argwhere(node_coords_to_check == neighbor[0] + neighbor[1] * 1j)[0][0]
                adjacent_matrix[i, index] = 0

        adjacent_matrix = build_refined_adjacency_matrix(
            all_node_coords,
            adjacent_matrix,
            self.updating_map_info,
            enable_edge_pruning=self.enable_corridor_edge_pruning,
            enable_graph_compression=self.enable_corridor_graph_compression,
            corridor_max_width=self.corridor_max_width,
            corridor_min_length=self.corridor_min_length,
        )

        utility = np.array(utility)
        guidepost = np.array(guidepost)
        current_index = np.argwhere(node_coords_to_check == self.location[0] + self.location[1] * 1j)[0][0]
        neighbor_indices = np.argwhere(adjacent_matrix[current_index] == 0).reshape(-1)
        return all_node_coords, utility, guidepost, adjacent_matrix, current_index, neighbor_indices

    def _observed_utility_prediction(self, n_nodes: int) -> PredictionResult:
        if n_nodes == 0:
            empty = np.zeros(0, dtype=np.float32)
            return PredictionResult(empty, empty, empty)
        normalizer = max(float(SENSOR_RANGE * 3.14 // FRONTIER_CELL_SIZE), 1.0)
        observed_utility = np.clip(np.asarray(self.utility, dtype=np.float32).reshape(n_nodes) / normalizer, 0.0, 1.0)
        uncertainty = np.zeros(n_nodes, dtype=np.float32)
        return PredictionResult(observed_utility, uncertainty, observed_utility.copy())

    def _build_actor_belief_features(self):
        n_nodes = int(self.node_coords.shape[0]) if self.node_coords is not None else 0
        self.hierarchical_graph = None
        if n_nodes == 0 or not self.runtime_config.use_hpbg:
            return zero_belief_features(n_nodes)

        if self.runtime_config.use_map_prediction:
            prediction = self.map_predictor.predict(
                self.node_coords,
                self.utility,
                self.guidepost,
                self.updating_map_info,
                frontier=self.frontier,
                utility_normalizer=max(float(SENSOR_RANGE * 3.14 // FRONTIER_CELL_SIZE), 1.0),
            )
        else:
            prediction = self._observed_utility_prediction(n_nodes)

        cluster_prior = np.zeros(n_nodes, dtype=np.float32)
        if self.runtime_config.use_hierarchical_graph:
            self.hierarchical_graph = build_hierarchical_graph(
                self.node_coords,
                self.adjacent_matrix,
                node_scores=prediction.risk_aware_utility,
                cluster_resolution=self.runtime_config.hpbg_cluster_resolution,
                cluster_edge_hops=self.runtime_config.hpbg_cluster_edge_hops,
            )
            cluster_prior = self.hierarchical_graph.node_cluster_prior

        if self.runtime_config.use_belief_state:
            features = self.belief_tracker.update(self.node_coords, prediction, cluster_prior).as_feature_matrix()
        else:
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
        ) / UPDATING_MAP_SIZE / 2
        node_utility = node_utility / (SENSOR_RANGE * 3.14 // FRONTIER_CELL_SIZE)
        self.belief_features = self._build_actor_belief_features()
        node_inputs = np.concatenate((node_coords, node_utility, node_guidepost, self.belief_features), axis=1)
        if node_inputs.shape[1] != NODE_INPUT_DIM:
            raise RuntimeError(f"actor node input dim mismatch: {node_inputs.shape[1]} != {NODE_INPUT_DIM}")
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)

        assert node_coords.shape[0] < NODE_PADDING_SIZE, (node_coords.shape[0], NODE_PADDING_SIZE)
        padding = torch.nn.ZeroPad2d((0, 0, 0, NODE_PADDING_SIZE - n_node))
        node_inputs = padding(node_inputs)

        node_padding_mask = torch.zeros((1, 1, n_node), dtype=torch.int16).to(self.device)
        node_padding = torch.ones((1, 1, NODE_PADDING_SIZE - n_node), dtype=torch.int16).to(self.device)
        node_padding_mask = torch.cat((node_padding_mask, node_padding), dim=-1)

        current_index = torch.tensor([current_index]).reshape(1, 1, 1).to(self.device)

        edge_mask = torch.tensor(edge_mask).unsqueeze(0).to(self.device)
        padding = torch.nn.ConstantPad2d((0, NODE_PADDING_SIZE - n_node, 0, NODE_PADDING_SIZE - n_node), 1)
        edge_mask = padding(edge_mask)

        current_in_edge = np.argwhere(current_edge == self.current_index)[0][0]
        current_edge = torch.tensor(current_edge, dtype=torch.long).unsqueeze(0)
        k_size = current_edge.size()[-1]
        padding = torch.nn.ConstantPad1d((0, K_SIZE - k_size), 0)
        current_edge = padding(current_edge).unsqueeze(-1).to(self.device)

        edge_padding_mask = torch.zeros((1, 1, k_size), dtype=torch.int16).to(self.device)
        edge_padding_mask[0, 0, current_in_edge] = 1
        padding = torch.nn.ConstantPad1d((0, K_SIZE - k_size), 1)
        edge_padding_mask = padding(edge_padding_mask)

        return [node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask]

    def select_next_waypoint(self, observation, greedy=False):
        observation = self._align_observation_device(observation)
        _, _, _, _, current_edge, _ = observation
        with torch.no_grad():
            logp = self.policy_net(*observation)

        if greedy:
            action_index = torch.argmax(logp, dim=1, keepdim=True).long().squeeze(1)
        else:
            action_index = torch.multinomial(logp.exp(), 1).long().squeeze(1)
        next_node_index = current_edge[0, action_index.item(), 0].item()
        next_position = self.node_coords[next_node_index]
        return next_position, action_index

    def plot_env(self):
        import matplotlib.pyplot as plt

        plt.switch_backend("agg")
        plt.figure(figsize=(18, 5))
        plt.subplot(1, 3, 2)
        if self.enable_corridor_edge_pruning or self.enable_corridor_graph_compression:
            sparse_graph = build_sparse_graph_view_from_arrays(
                self.node_coords,
                self.utility,
                self.guidepost,
                self.adjacent_matrix,
                self.current_index,
            )
        else:
            sparse_graph = build_sparse_graph_view(self.node_manager, self.location)
        if len(self.frontier) > 0:
            frontiers = get_cell_position_from_coords(np.array(list(self.frontier)), self.map_info).reshape(-1, 2)
            plt.scatter(frontiers[:, 0], frontiers[:, 1], c="r", s=2)
        robot = get_cell_position_from_coords(self.location, self.map_info)
        plt.imshow(self.map_info.map, cmap="gray")
        plt.axis("off")
        plt.title("Sparse Planning Graph")
        for edge_path in sparse_graph.edge_paths:
            edge_cells = get_cell_position_from_coords(edge_path, self.map_info).reshape(-1, 2)
            plt.plot(edge_cells[:, 0], edge_cells[:, 1], color="tan", linewidth=1.5, zorder=1)
        if sparse_graph.node_coords.size > 0:
            sparse_nodes = get_cell_position_from_coords(sparse_graph.node_coords, self.map_info).reshape(-1, 2)
            node_sizes = np.where(sparse_graph.utility > 0, 42, 24)
            plt.scatter(
                sparse_nodes[:, 0],
                sparse_nodes[:, 1],
                c=sparse_graph.wavelet,
                cmap="viridis",
                s=node_sizes,
                zorder=2,
            )
        plt.plot(robot[0], robot[1], "mo", markersize=16, zorder=5)
