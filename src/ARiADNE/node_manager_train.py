from __future__ import annotations

import heapq

import numpy as np

from . import quads
from .parameter import MIN_UTILITY, NODE_RESOLUTION, SENSOR_RANGE, UTILITY_RANGE, RuntimeConfig
from .utils import (
    check_collision,
    compute_wavelet_maps,
    get_frontier_in_map,
    get_updating_node_coords,
    wavelet_feature_at_coords,
    wavelet_scalar_at_coords,
)


class NodeManager:
    def __init__(self, runtime_config: RuntimeConfig | None = None, plot=False):
        self.runtime_config = runtime_config or RuntimeConfig()
        self.nodes_dict = quads.QuadTree((0, 0), 1000, 1000)
        self.plot = plot
        self.frontier = None

    def check_node_exist_in_dict(self, coords):
        return self.nodes_dict.find((coords[0], coords[1]))

    def add_node_to_dict(self, coords, frontiers, updating_map_info):
        node = Node(coords, frontiers, updating_map_info)
        self.nodes_dict.insert(point=(coords[0], coords[1]), data=node)
        return node

    def remove_node_from_dict(self, node):
        for neighbor_coords in node.neighbor_set:
            if neighbor_coords != (node.coords[0], node.coords[1]):
                neighbor_node = self.nodes_dict.find(neighbor_coords)
                neighbor_node.data.neighbor_set.remove(node.coords.tolist())
        self.nodes_dict.remove(node.coords.tolist())

    def _should_compute_wavelet_maps(self):
        return self.runtime_config.use_wavelet_feature or self.runtime_config.use_wavelet_attn_bias

    def update_graph(self, robot_location, frontiers, updating_map_info, map_info):
        node_coords, _ = get_updating_node_coords(robot_location, updating_map_info)
        wavelet_maps = None
        if self._should_compute_wavelet_maps():
            wavelet_maps = compute_wavelet_maps(updating_map_info.map, self.runtime_config)

        if self.frontier is None:
            new_frontier = frontiers
        else:
            new_frontier = frontiers - self.frontier
            new_out_range = []
            for frontier in new_frontier:
                if np.linalg.norm(robot_location - np.array(frontier).reshape(2)) > SENSOR_RANGE:
                    new_out_range.append(frontier)
            for frontier in new_out_range:
                new_frontier.remove(frontier)

        self.frontier = frontiers

        all_node_list = []
        global_frontiers = get_frontier_in_map(map_info)
        for coords in node_coords:
            node = self.check_node_exist_in_dict(coords)
            if node is None:
                node = self.add_node_to_dict(coords, frontiers, updating_map_info)
            else:
                node = node.data
                if node.utility == 0 or np.linalg.norm(node.coords - robot_location) > 2 * SENSOR_RANGE:
                    pass
                else:
                    node.update_node_observable_frontiers(new_frontier, global_frontiers, updating_map_info)

            if wavelet_maps is not None:
                node.wavelet_feat = wavelet_feature_at_coords(
                    node.coords,
                    updating_map_info,
                    wavelet_maps,
                    self.runtime_config,
                )
                node.wavelet_scalar = wavelet_scalar_at_coords(
                    node.coords,
                    updating_map_info,
                    wavelet_maps,
                )
                node.wavelet = float(node.wavelet_scalar)
            else:
                node.wavelet_feat = np.zeros((0,), dtype=np.float32)
                node.wavelet_scalar = 0.0
                node.wavelet = 0.0
            all_node_list.append(node)

        for node in all_node_list:
            if node.need_update_neighbor and np.linalg.norm(node.coords - robot_location) < (
                SENSOR_RANGE + NODE_RESOLUTION
            ):
                node.update_neighbor_nodes(updating_map_info, self.nodes_dict)

    def Dijkstra(self, start, boundary=None):
        q = set()
        dist_dict = {}
        prev_dict = {}

        for node in self.nodes_dict.__iter__():
            coords = node.data.coords
            key = (coords[0], coords[1])
            dist_dict[key] = 1e8
            prev_dict[key] = None
            q.add(key)

        assert (start[0], start[1]) in dist_dict
        dist_dict[(start[0], start[1])] = 0

        while q:
            u = None
            for coords in q:
                if u is None or dist_dict[coords] < dist_dict[u]:
                    u = coords

            q.remove(u)
            node = self.nodes_dict.find(u).data
            for neighbor_node_coords in node.neighbor_set:
                v = (neighbor_node_coords[0], neighbor_node_coords[1])
                if v in q:
                    cost = ((neighbor_node_coords[0] - u[0]) ** 2 + (neighbor_node_coords[1] - u[1]) ** 2) ** 0.5
                    cost = np.round(cost, 2)
                    alt = dist_dict[u] + cost
                    if alt < dist_dict[v]:
                        dist_dict[v] = alt
                        prev_dict[v] = u

        return dist_dict, prev_dict

    def get_Dijkstra_path_and_dist(self, dist_dict, prev_dict, end):
        if (end[0], end[1]) not in dist_dict:
            return [], 1e8

        dist = dist_dict[(end[0], end[1])]
        path = [(end[0], end[1])]
        prev_node = prev_dict[(end[0], end[1])]
        while prev_node is not None:
            path.append(prev_node)
            prev_node = prev_dict[prev_node]
        path.reverse()
        return path[1:], np.round(dist, 2)

    def h(self, coords_1, coords_2):
        return np.linalg.norm(np.array([coords_1[0] - coords_2[0], coords_1[1] - coords_2[1]]))

    def a_star(self, start, destination, max_dist=None):
        if not self.check_node_exist_in_dict(start):
            return [], 1e8
        if not self.check_node_exist_in_dict(destination):
            return [], 1e8
        if start[0] == destination[0] and start[1] == destination[1]:
            return [], 0

        open_list = {(start[0], start[1])}
        closed_list = set()
        g = {(start[0], start[1]): 0}
        parents = {(start[0], start[1]): (start[0], start[1])}
        open_heap = []
        heapq.heappush(open_heap, (0, (start[0], start[1])))

        while open_list:
            _, n = heapq.heappop(open_heap)
            node = self.nodes_dict.find(n).data

            if max_dist is not None and g[n] > max_dist:
                return [], 1e8

            if n[0] == destination[0] and n[1] == destination[1]:
                path = []
                length = g[n]
                while parents[n] != n:
                    path.append(n)
                    n = parents[n]
                path.reverse()
                return path, np.round(length, 2)

            costs = np.linalg.norm(np.array(list(node.neighbor_set)).reshape(-1, 2) - [n[0], n[1]], axis=1)
            for cost, neighbor_node_coords in zip(costs, node.neighbor_set):
                m = (neighbor_node_coords[0], neighbor_node_coords[1])
                if m not in open_list and m not in closed_list:
                    open_list.add(m)
                    parents[m] = n
                    g[m] = g[n] + cost
                    heapq.heappush(open_heap, (g[m], m))
                elif g[m] > g[n] + cost:
                    g[m] = g[n] + cost
                    parents[m] = n

            open_list.remove(n)
            closed_list.add(n)

        return [], 1e8


class Node:
    def __init__(self, coords, frontiers, updating_map_info):
        self.coords = coords
        self.utility_range = UTILITY_RANGE
        self.utility = 0
        self.wavelet = 0.0
        self.wavelet_scalar = 0.0
        self.wavelet_feat = np.zeros((0,), dtype=np.float32)
        self.observable_frontiers = self.initialize_observable_frontiers(frontiers, updating_map_info)
        self.visited = 0

        self.neighbor_matrix = -np.ones((5, 5))
        self.neighbor_set = set()
        self.neighbor_matrix[2, 2] = 1
        self.neighbor_set.add((self.coords[0], self.coords[1]))
        self.need_update_neighbor = True

    def initialize_observable_frontiers(self, frontiers, updating_map_info):
        if len(frontiers) == 0:
            self.utility = 0
            return set()

        observable_frontiers = set()
        frontiers = np.array(list(frontiers)).reshape(-1, 2)
        dist_list = np.linalg.norm(frontiers - self.coords, axis=-1)
        new_frontiers_in_range = frontiers[dist_list < self.utility_range]
        for point in new_frontiers_in_range:
            if not check_collision(self.coords, point, updating_map_info):
                observable_frontiers.add((point[0], point[1]))
        self.utility = len(observable_frontiers)
        if self.utility <= MIN_UTILITY:
            self.utility = 0
            observable_frontiers = set()
        return observable_frontiers

    def update_neighbor_nodes(self, updating_map_info, nodes_dict):
        for i in range(self.neighbor_matrix.shape[0]):
            for j in range(self.neighbor_matrix.shape[1]):
                if self.neighbor_matrix[i, j] != -1:
                    continue

                center_index = self.neighbor_matrix.shape[0] // 2
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
                if not check_collision(self.coords, neighbor_coords, updating_map_info):
                    neighbor_matrix_x = center_index + (center_index - i)
                    neighbor_matrix_y = center_index + (center_index - j)
                    self.neighbor_matrix[i, j] = 1
                    self.neighbor_set.add((neighbor_coords[0], neighbor_coords[1]))
                    neighbor_node.neighbor_matrix[neighbor_matrix_x, neighbor_matrix_y] = 1
                    neighbor_node.neighbor_set.add((self.coords[0], self.coords[1]))

        if self.utility == 0:
            self.need_update_neighbor = False

    def update_node_observable_frontiers(self, new_frontiers, global_frontiers, updating_map_info):
        frontiers_observed = []
        for frontier in self.observable_frontiers:
            if frontier not in global_frontiers:
                frontiers_observed.append(frontier)
        for frontier in frontiers_observed:
            self.observable_frontiers.remove(frontier)

        if len(new_frontiers) > 0:
            new_frontiers = np.array(list(new_frontiers)).reshape(-1, 2)
            dist_list = np.linalg.norm(new_frontiers - self.coords, axis=-1)
            new_frontiers_in_range = new_frontiers[dist_list < self.utility_range]
            for point in new_frontiers_in_range:
                if not check_collision(self.coords, point, updating_map_info):
                    self.observable_frontiers.add((point[0], point[1]))

        self.utility = len(self.observable_frontiers)
        if self.utility <= MIN_UTILITY:
            self.utility = 0
            self.observable_frontiers = set()

    def set_visited(self):
        self.visited = 1
        self.observable_frontiers = set()
        self.utility = 0
