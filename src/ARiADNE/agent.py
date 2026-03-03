from __future__ import annotations

import copy

import matplotlib.pyplot as plt
import numpy as np
import torch

from .node_manager_infer import NodeManager as InferenceNodeManager
from .node_manager_train import NodeManager as TrainNodeManager
from .parameter import (
    FRONTIER_CELL_SIZE,
    K_SIZE,
    NODE_PADDING_SIZE,
    SENSOR_RANGE,
    UPDATING_MAP_SIZE,
    RuntimeConfig,
    get_node_input_dim,
)
from .utils import (
    MapInfo,
    build_wavelet_attn_bias,
    get_cell_position_from_coords,
    get_frontier_in_map,
)


class BaseAgent:
    def __init__(self, policy_net, runtime_config: RuntimeConfig | None = None, device="cpu", plot=False):
        self.runtime_config = runtime_config or RuntimeConfig()
        self.device = device
        self.policy_net = policy_net
        self.plot = plot
        self.location = None
        self.map_info = None
        self.updating_map_info = None
        self.frontier = set()
        self.node_coords = None
        self.utility = None
        self.guidepost = None
        self.wavelet_feat = None
        self.current_index = None
        self.adjacent_matrix = None
        self.neighbor_indices = None

    def update_map(self, map_info):
        self.map_info = map_info

    def update_updating_map(self, location):
        self.updating_map_info = self.get_updating_map(location)

    def update_location(self, location):
        self.location = location
        node = self.node_manager.nodes_dict.find(location.tolist())
        if self.node_manager.nodes_dict.__len__() == 0:
            return
        node.data.set_visited()

    def update_frontiers(self):
        self.frontier = get_frontier_in_map(self.updating_map_info)

    def get_updating_map(self, location):
        updating_map_origin_x = location[0] - UPDATING_MAP_SIZE / 2
        updating_map_origin_y = location[1] - UPDATING_MAP_SIZE / 2
        updating_map_top_x = updating_map_origin_x + UPDATING_MAP_SIZE
        updating_map_top_y = updating_map_origin_y + UPDATING_MAP_SIZE

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

    def select_next_waypoint(self, observation, greedy=False):
        _, _, _, _, current_edge, _, _ = observation
        with torch.no_grad():
            logp = self.policy_net(*observation)

        if greedy:
            action_index = torch.argmax(logp, dim=1).long()
        else:
            action_index = torch.multinomial(logp.exp(), 1).long().squeeze(1)

        next_node_index = current_edge[0, action_index.item(), 0].item()
        next_position = self.node_coords[next_node_index]
        return next_position, action_index

    def _build_node_inputs(self, node_coords, utility, guidepost, current_index, wavelet_feat):
        current_node_coords = node_coords[current_index]
        node_coords = np.concatenate(
            (
                node_coords[:, 0].reshape(-1, 1) - current_node_coords[0],
                node_coords[:, 1].reshape(-1, 1) - current_node_coords[1],
            ),
            axis=-1,
        ) / UPDATING_MAP_SIZE / 2
        node_utility = utility.reshape(-1, 1) / (SENSOR_RANGE * 3.14 // FRONTIER_CELL_SIZE)
        node_guidepost = guidepost.reshape(-1, 1)

        features = [node_coords, node_utility, node_guidepost]
        if self.runtime_config.use_wavelet_feature:
            features.append(wavelet_feat)
        node_inputs = np.concatenate(features, axis=1).astype(np.float32)
        assert node_inputs.shape[1] == get_node_input_dim(self.runtime_config)
        return node_inputs

    def _build_attn_bias(self, wavelet_feat, edge_mask, node_padding_mask=None):
        if not self.runtime_config.use_wavelet_attn_bias:
            return None
        if wavelet_feat.size == 0:
            return None
        wavelet_feat_tensor = torch.as_tensor(wavelet_feat, dtype=torch.float32, device=self.device).unsqueeze(0)
        edge_mask_tensor = torch.as_tensor(edge_mask, device=self.device).unsqueeze(0)
        return build_wavelet_attn_bias(
            wavelet_feat_tensor,
            runtime_config=self.runtime_config,
            edge_mask=edge_mask_tensor,
            node_padding_mask=node_padding_mask,
        )


class Agent(BaseAgent):
    def __init__(self, policy_net, runtime_config: RuntimeConfig | None = None, device="cpu", plot=False):
        super().__init__(policy_net, runtime_config=runtime_config, device=device, plot=plot)
        self.node_manager = TrainNodeManager(runtime_config=self.runtime_config, plot=self.plot)

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
            self.wavelet_feat,
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
        wavelet_feat = []

        n_nodes = all_node_coords.shape[0]
        adjacent_matrix = np.ones((n_nodes, n_nodes)).astype(int)
        node_coords_to_check = all_node_coords[:, 0] + all_node_coords[:, 1] * 1j
        for i, coords in enumerate(all_node_coords):
            node = self.node_manager.nodes_dict.find((coords[0], coords[1])).data
            utility.append(node.utility)
            guidepost.append(node.visited)
            wavelet_feat.append(getattr(node, "wavelet_feat", np.zeros((0,), dtype=np.float32)))
            for neighbor in node.neighbor_set:
                index = np.argwhere(node_coords_to_check == neighbor[0] + neighbor[1] * 1j)
                index = index[0][0]
                adjacent_matrix[i, index] = 0

        utility = np.array(utility)
        guidepost = np.array(guidepost)
        if wavelet_feat and wavelet_feat[0].size > 0:
            wavelet_feat = np.stack(wavelet_feat, axis=0).astype(np.float32)
        else:
            wavelet_feat = np.zeros((n_nodes, 0), dtype=np.float32)
        current_index = np.argwhere(node_coords_to_check == self.location[0] + self.location[1] * 1j)[0][0]
        neighbor_indices = np.argwhere(adjacent_matrix[current_index] == 0).reshape(-1)
        return all_node_coords, utility, guidepost, wavelet_feat, adjacent_matrix, current_index, neighbor_indices

    def get_observation(self):
        node_inputs_np = self._build_node_inputs(
            self.node_coords,
            self.utility,
            self.guidepost,
            self.current_index,
            self.wavelet_feat,
        )
        current_index = self.current_index
        edge_mask = self.adjacent_matrix
        current_edge = self.neighbor_indices
        n_node = node_inputs_np.shape[0]

        node_inputs = torch.FloatTensor(node_inputs_np).unsqueeze(0).to(self.device)
        assert n_node < NODE_PADDING_SIZE, (n_node, NODE_PADDING_SIZE)
        node_inputs = torch.nn.ZeroPad2d((0, 0, 0, NODE_PADDING_SIZE - n_node))(node_inputs)

        unpadded_node_padding_mask = torch.zeros((1, 1, n_node), dtype=torch.int16).to(self.device)
        node_padding_mask = torch.zeros((1, 1, n_node), dtype=torch.int16).to(self.device)
        node_padding = torch.ones((1, 1, NODE_PADDING_SIZE - n_node), dtype=torch.int16).to(self.device)
        node_padding_mask = torch.cat((node_padding_mask, node_padding), dim=-1)

        current_index_tensor = torch.tensor([current_index], dtype=torch.long, device=self.device).reshape(1, 1, 1)
        edge_mask_tensor = torch.tensor(edge_mask, device=self.device).unsqueeze(0)
        edge_mask_tensor = torch.nn.ConstantPad2d(
            (0, NODE_PADDING_SIZE - n_node, 0, NODE_PADDING_SIZE - n_node),
            1,
        )(edge_mask_tensor)

        current_in_edge = np.argwhere(current_edge == current_index)[0][0]
        current_edge_tensor = torch.tensor(current_edge, dtype=torch.long, device=self.device).unsqueeze(0)
        k_size = current_edge_tensor.size()[-1]
        current_edge_tensor = torch.nn.ConstantPad1d((0, K_SIZE - k_size), 0)(current_edge_tensor)
        current_edge_tensor = current_edge_tensor.unsqueeze(-1)

        edge_padding_mask = torch.zeros((1, 1, k_size), dtype=torch.int16).to(self.device)
        edge_padding_mask[0, 0, current_in_edge] = 1
        edge_padding_mask = torch.nn.ConstantPad1d((0, K_SIZE - k_size), 1)(edge_padding_mask)

        attn_bias = self._build_attn_bias(
            self.wavelet_feat,
            self.adjacent_matrix,
            node_padding_mask=unpadded_node_padding_mask,
        )
        if attn_bias is not None:
            attn_bias = torch.nn.ConstantPad2d(
                (0, NODE_PADDING_SIZE - n_node, 0, NODE_PADDING_SIZE - n_node),
                0,
            )(attn_bias[:, 0]).unsqueeze(1)

        return [
            node_inputs,
            node_padding_mask,
            edge_mask_tensor,
            current_index_tensor,
            current_edge_tensor,
            edge_padding_mask,
            attn_bias,
        ]

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


class InferenceAgent(BaseAgent):
    def __init__(self, policy_net, runtime_config: RuntimeConfig | None = None, device="cpu", plot=False):
        super().__init__(policy_net, runtime_config=runtime_config, device=device, plot=plot)
        self.node_manager = InferenceNodeManager(runtime_config=self.runtime_config)
        self.key_node_coords = None
        self.key_utility = None
        self.key_guidepost = None
        self.key_wavelet_feat = None
        self.key_current_index = None
        self.key_adjacent_matrix = None
        self.key_neighbor_indices = None

    def update_planning_state(self, map_info, location):
        self.update_map(map_info)
        self.update_location(location)
        self.update_updating_map(self.location)
        self.update_frontiers()
        self.location = self.node_manager.update_graph(
            self.location,
            self.frontier,
            self.updating_map_info,
            self.map_info,
        )
        (
            self.node_coords,
            self.utility,
            self.guidepost,
            self.wavelet_feat,
            self.adjacent_matrix,
            self.current_index,
            self.neighbor_indices,
        ) = self.update_dense_observation()
        self.node_manager.get_rarefied_graph(self.location, self.map_info)
        (
            self.key_node_coords,
            self.key_utility,
            self.key_guidepost,
            self.key_wavelet_feat,
            self.key_adjacent_matrix,
            self.key_current_index,
            self.key_neighbor_indices,
        ) = self.update_key_node_observation()

    def update_dense_observation(self):
        all_node_coords = []
        for node in self.node_manager.nodes_dict.__iter__():
            all_node_coords.append(node.data.coords)
        all_node_coords = np.array(all_node_coords).reshape(-1, 2)
        utility = []
        guidepost = []
        wavelet_feat = []
        n_nodes = all_node_coords.shape[0]
        adjacent_matrix = np.ones((n_nodes, n_nodes)).astype(int)
        node_coords_to_check = all_node_coords[:, 0] + all_node_coords[:, 1] * 1j
        for i, coords in enumerate(all_node_coords):
            node = self.node_manager.nodes_dict.find((coords[0], coords[1])).data
            utility.append(node.utility)
            guidepost.append(node.visited)
            wavelet_feat.append(getattr(node, "wavelet_feat", np.zeros((0,), dtype=np.float32)))
            for neighbor in node.neighbor_set:
                index = np.argwhere(node_coords_to_check == neighbor[0] + neighbor[1] * 1j)
                index = index[0][0]
                adjacent_matrix[i, index] = 0
        utility = np.array(utility)
        guidepost = np.array(guidepost)
        if wavelet_feat and wavelet_feat[0].size > 0:
            wavelet_feat = np.stack(wavelet_feat, axis=0).astype(np.float32)
        else:
            wavelet_feat = np.zeros((n_nodes, 0), dtype=np.float32)
        current_index = np.argwhere(node_coords_to_check == self.location[0] + self.location[1] * 1j)[0][0]
        neighbor_indices = np.argwhere(adjacent_matrix[current_index] == 0).reshape(-1)
        return all_node_coords, utility, guidepost, wavelet_feat, adjacent_matrix, current_index, neighbor_indices

    def update_key_node_observation(self):
        all_key_node_coords = []
        for key_node_coords in self.node_manager.key_node_dict.keys():
            all_key_node_coords.append(np.array(key_node_coords))
        all_key_node_coords = np.array(all_key_node_coords).reshape(-1, 2)
        utility = []
        guidepost = []
        wavelet_feat = []
        n_nodes = all_key_node_coords.shape[0]
        adjacent_matrix = np.ones((n_nodes, n_nodes)).astype(int)
        node_coords_to_check = all_key_node_coords[:, 0] + all_key_node_coords[:, 1] * 1j
        for i, coords in enumerate(all_key_node_coords):
            node = self.node_manager.key_node_dict[(coords[0], coords[1])]
            utility.append(node.utility)
            guidepost.append(node.visited)
            wavelet_feat.append(getattr(node, "wavelet_feat", np.zeros((0,), dtype=np.float32)))
            for neighbor in node.neighbor_set:
                neighbor = np.array([neighbor[0], neighbor[1]])
                index = np.argwhere(node_coords_to_check == neighbor[0] + neighbor[1] * 1j)
                index = index[0][0]
                adjacent_matrix[i, index] = 0

        utility = np.array(utility)
        guidepost = np.array(guidepost)
        if wavelet_feat and wavelet_feat[0].size > 0:
            wavelet_feat = np.stack(wavelet_feat, axis=0).astype(np.float32)
        else:
            wavelet_feat = np.zeros((n_nodes, 0), dtype=np.float32)
        current_index = np.argwhere(node_coords_to_check == self.location[0] + self.location[1] * 1j)[0][0]
        neighbor_indices = np.argwhere(adjacent_matrix[current_index] == 0).reshape(-1)
        return (
            all_key_node_coords,
            utility,
            guidepost,
            wavelet_feat,
            adjacent_matrix,
            current_index,
            neighbor_indices,
        )

    def get_observation(self, robot_location):
        node_coords = copy.deepcopy(self.key_node_coords)
        node_coords[self.key_current_index] = robot_location
        node_inputs_np = self._build_node_inputs(
            node_coords,
            self.key_utility,
            self.key_guidepost,
            self.key_current_index,
            self.key_wavelet_feat,
        )
        edge_mask = self.key_adjacent_matrix
        current_index = self.key_current_index
        current_edge = self.key_neighbor_indices

        node_inputs = torch.FloatTensor(node_inputs_np).unsqueeze(0).to(self.device)
        edge_mask_tensor = torch.tensor(edge_mask, device=self.device).unsqueeze(0)
        current_in_edge = np.argwhere(current_edge == current_index)[0][0]
        current_edge_tensor = torch.tensor(current_edge, dtype=torch.long, device=self.device).unsqueeze(0)
        k_size = current_edge_tensor.size()[-1]
        current_edge_tensor = current_edge_tensor.unsqueeze(-1)

        current_index_tensor = torch.tensor([current_index], dtype=torch.long, device=self.device).reshape(1, 1, 1)
        edge_padding_mask = torch.zeros((1, 1, k_size), dtype=torch.int16).to(self.device)
        edge_padding_mask[0, 0, current_in_edge] = 1
        attn_bias = self._build_attn_bias(
            self.key_wavelet_feat,
            self.key_adjacent_matrix,
            node_padding_mask=None,
        )
        return [
            node_inputs,
            None,
            edge_mask_tensor,
            current_index_tensor,
            current_edge_tensor,
            edge_padding_mask,
            attn_bias,
        ]

    def select_next_waypoint(self, observation, greedy=False):
        _, _, _, _, current_edge, _, _ = observation
        with torch.no_grad():
            logp = self.policy_net(*observation)

        if greedy:
            action_index = torch.argmax(logp, dim=1).long()
        else:
            action_index = torch.multinomial(logp.exp(), 1).long().squeeze(1)
        next_node_index = current_edge[0, action_index.item(), 0].item()
        next_position = self.key_node_coords[next_node_index]
        return next_position, next_node_index
