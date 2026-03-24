from __future__ import annotations

import heapq
from copy import deepcopy
import numpy as np

from . import quads
from . import parameter as _param
from .parameter import MIN_UTILITY, NODE_RESOLUTION, SENSOR_RANGE, UTILITY_RANGE
from .utils import (
    check_collision, check_collision_type, get_frontier_in_map,
    get_updating_node_coords, get_cell_position_from_coords, FREE,
    compute_wavelet_energy_map, wavelet_scalar_at_coords, extract_local_map_info,
)


class NodeManager:
    def __init__(self, plot=False):
        self.nodes_dict = quads.QuadTree((0, 0), 1000, 1000)
        self.plot = plot
        self.frontier = None
        self.path_to_nearest_frontier = []

        self.key_node_dict = {}
        self.cluster_center_node_dict = {}
        self.dist_to_nearest_frontier = 1e8

        self._wavelet_map = None
        self._wavelet_map_info = None
        self._wavelet_cache_raw = None
        self._wavelet_cache_geom = None

    def check_node_exist_in_dict(self, coords):
        key = (coords[0], coords[1])
        return self.nodes_dict.find(key)

    def add_node_to_dict(self, coords, frontiers, updating_map_info):
        key = (coords[0], coords[1])
        node = Node(coords, frontiers, updating_map_info)
        self.nodes_dict.insert(point=key, data=node)
        return node

    def remove_node_from_dict(self, node):
        for neighbor_coords in node.neighbor_set:
            if neighbor_coords != (node.coords[0], node.coords[1]):
                neighbor_node = self.nodes_dict.find(neighbor_coords)
                if neighbor_node is not None:
                    neighbor_node.data.neighbor_set.discard((node.coords[0], node.coords[1]))
        self.nodes_dict.remove(node.coords.tolist())

    def check_valid_node(self, robot_location, map_info):
        nodes_to_remove = []
        for node in self.nodes_dict.__iter__():
            coords = node.data.coords
            cell = get_cell_position_from_coords(coords, map_info, check_negative=False)
            if cell[0] < 0 or cell[1] < 0:
                continue
            if cell[0] >= map_info.map.shape[1] or cell[1] >= map_info.map.shape[0]:
                continue
            if map_info.map[cell[1], cell[0]] != FREE:
                if np.linalg.norm(coords - robot_location) < SENSOR_RANGE:
                    nodes_to_remove.append(node.data)

        for node in nodes_to_remove:
            self.remove_node_from_dict(node)

    def update_graph(self, robot_location, frontiers, updating_map_info, map_info):
        node_coords, _ = get_updating_node_coords(robot_location, updating_map_info)

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
            all_node_list.append(node)

        for node in all_node_list:
            if node.need_update_neighbor and np.linalg.norm(node.coords - robot_location) < (
                SENSOR_RANGE + NODE_RESOLUTION
            ):
                node.update_neighbor_nodes(updating_map_info, self.nodes_dict)

        if _param.ENABLE_GRAPH_RAREFACTION:
            self.remove_unconnected_nodes(robot_location)
        self._update_path_to_nearest_frontier(robot_location)

    def _update_path_to_nearest_frontier(self, robot_location):
        self.path_to_nearest_frontier = []
        best_dist = 1e8
        best_target = None

        for node in self.nodes_dict.__iter__():
            if node.data.utility > 0:
                dist = np.linalg.norm(node.data.coords - robot_location)
                if dist < best_dist:
                    best_dist = dist
                    best_target = node.data.coords

        if best_target is None:
            return

        path, dist = self.a_star(robot_location, best_target)
        if dist < 1e8 and len(path) > 0:
            self.path_to_nearest_frontier = [np.array(p) for p in path]

    # ----------------------------------------------------------------
    # Graph rarefaction
    # ----------------------------------------------------------------

    def _build_wavelet_context(self, robot_location, map_info):
        if not _param.WAVELET_ADAPTIVE_DTH:
            self._wavelet_map = None
            self._wavelet_map_info = None
            return None, None

        wmi = extract_local_map_info(robot_location, map_info, _param.WAVELET_LOCAL_MAP_SIZE)
        local_map = wmi.map
        geom = (float(wmi.map_origin_x), float(wmi.map_origin_y),
                int(local_map.shape[0]), int(local_map.shape[1]), float(wmi.cell_size))

        reuse = (
            self._wavelet_map is not None
            and self._wavelet_cache_raw is not None
            and self._wavelet_cache_geom == geom
            and self._wavelet_cache_raw.shape == local_map.shape
            and (float(np.count_nonzero(local_map != self._wavelet_cache_raw)) / max(local_map.size, 1)
                 <= _param.WAVELET_CACHE_CHANGE_RATIO_THRESH)
        )
        if not reuse:
            self._wavelet_map = compute_wavelet_energy_map(local_map)
            self._wavelet_map_info = wmi
            self._wavelet_cache_raw = np.array(local_map, copy=True)
            self._wavelet_cache_geom = geom

        return self._wavelet_map, self._wavelet_map_info

    def _adaptive_distance_threshold(self, ref_coords, map_info, wavelet_map, wavelet_map_info):
        base = _param.THR_NEXT_WAYPOINT + NODE_RESOLUTION
        if not _param.WAVELET_ADAPTIVE_DTH or wavelet_map is None:
            return base
        lmi = wavelet_map_info if wavelet_map_info is not None else map_info
        w = wavelet_scalar_at_coords(np.array(ref_coords), lmi, wavelet_map)
        mult = min(1.0 + _param.WAVELET_DTH_ALPHA * (1.0 - w), _param.WAVELET_DTH_MAX_MULT)
        return base * mult

    def remove_unconnected_nodes(self, origin):
        all_coords = set()
        for node in self.nodes_dict.__iter__():
            c = node.data.coords
            all_coords.add((c[0], c[1]))

        origin_node = self.nodes_dict.find(origin.tolist())
        if origin_node is None:
            return
        connected = set()
        connected.add((origin_node.data.coords[0], origin_node.data.coords[1]))
        to_expand = deepcopy(origin_node.data.neighbor_set)

        while to_expand:
            next_expand = set()
            for nc in to_expand:
                wrapper = self.nodes_dict.find(nc)
                if wrapper is None:
                    continue
                for nb in wrapper.data.neighbor_set:
                    if nb not in all_coords or nb in connected:
                        continue
                    next_expand.add(nb)
                connected.add(nc)
            to_expand = next_expand

        for nc in all_coords - connected:
            wrapper = self.nodes_dict.find(nc)
            if wrapper is not None:
                self.remove_node_from_dict(wrapper.data)

    def update_clustered_nodes(self, center_node, clustered):
        boundary = _get_quad_tree_box(center_node.coords, 2 * _param.CLUSTER_RANGE)
        nearby = self.nodes_dict.within_bb(boundary)
        nz_nearby = {n.data for n in nearby if n.data.utility > 0}

        to_check = set()
        for nc in center_node.neighbor_set:
            if nc == (center_node.coords[0], center_node.coords[1]):
                continue
            w = self.nodes_dict.find(nc)
            if w is not None and w.data in nz_nearby:
                clustered.add(w.data)
                to_check.add(w.data)

        while to_check:
            new_check = set()
            for node in to_check:
                for nc in node.neighbor_set:
                    if nc == (node.coords[0], node.coords[1]):
                        continue
                    w = self.nodes_dict.find(nc)
                    if w is None or w.data in clustered or w.data not in nz_nearby:
                        continue
                    clustered.add(w.data)
                    new_check.add(w.data)
            to_check = new_check

    def _insert_key_nodes_along_path(self, path, ref, ref_node, map_info, wavelet_map, wavelet_map_info):
        first_node = True
        for i, coords in enumerate(path):
            if first_node:
                ref_wrapper = self.nodes_dict.find(ref)
                if ref_wrapper is None:
                    break
                in_neighbor = coords in ref_wrapper.data.neighbor_set
                dist = np.linalg.norm(np.array(coords) - np.array(ref))
                collision = check_collision(np.array(ref), np.array(coords), map_info)
                dth = self._adaptive_distance_threshold(ref, map_info, wavelet_map, wavelet_map_info)

                if not in_neighbor and (collision or dist > dth):
                    if i < 1:
                        break
                    prev = path[i - 1]
                    if prev not in self.key_node_dict:
                        nd = self.nodes_dict.find(prev)
                        if nd is None:
                            break
                        self.key_node_dict[prev] = KeyNode(nd.data.coords, nd.data.utility, nd.data.visited)
                    self.key_node_dict[prev].add_neighbor(ref)
                    ref_node.add_neighbor(prev)
                    first_node = False
                    if coords not in self.key_node_dict:
                        nd = self.nodes_dict.find(coords)
                        if nd is None:
                            continue
                        self.key_node_dict[coords] = KeyNode(nd.data.coords, nd.data.utility, nd.data.visited)

                elif self.nodes_dict.find(coords) is not None and self.nodes_dict.find(coords).data.utility > 0:
                    kc = coords
                    if kc not in self.key_node_dict:
                        nd = self.nodes_dict.find(kc)
                        if nd is None:
                            continue
                        self.key_node_dict[kc] = KeyNode(nd.data.coords, nd.data.utility, nd.data.visited)
                    self.key_node_dict[kc].add_neighbor(ref)
                    ref_node.add_neighbor(kc)
                    first_node = False
            else:
                if coords not in self.key_node_dict:
                    nd = self.nodes_dict.find(coords)
                    if nd is None:
                        continue
                    self.key_node_dict[coords] = KeyNode(nd.data.coords, nd.data.utility, nd.data.visited)

    def get_rarefied_graph(self, robot_location, map_info):
        self.dist_to_nearest_frontier = 1e8
        self.path_to_nearest_frontier = []
        wavelet_map, wavelet_map_info = self._build_wavelet_context(robot_location, map_info)

        nz_nodes = {n.data for n in self.nodes_dict.__iter__() if n.data.utility > 0}
        self.key_node_dict = {}

        current_wrapper = self.nodes_dict.find(robot_location.tolist())
        if current_wrapper is None:
            return
        cn = current_wrapper.data
        ref = (cn.coords[0], cn.coords[1])
        key_current = KeyNode(cn.coords, cn.utility, cn.visited)
        self.key_node_dict[ref] = key_current

        cluster_bb = _get_quad_tree_box(cn.coords, SENSOR_RANGE)
        local_nodes = [n.data for n in self.nodes_dict.within_bb(cluster_bb)]

        for node in local_nodes:
            if node.utility <= 0:
                continue
            nk = (node.coords[0], node.coords[1])
            if nk in self.key_node_dict:
                nz_nodes.discard(node)
                continue
            path, dist = self.a_star(cn.coords, node.coords)
            if dist >= 1e8:
                continue
            if dist < self.dist_to_nearest_frontier:
                self.dist_to_nearest_frontier = dist
                self.path_to_nearest_frontier = [np.array(p) for p in path]
            nz_nodes.discard(node)
            ref_node = self.key_node_dict[ref]
            self._insert_key_nodes_along_path(path, ref, ref_node, map_info, wavelet_map, wavelet_map_info)

        stale = [cn for cn, _ in self.cluster_center_node_dict.values()
                 if cn.utility == 0 or cn in local_nodes
                 or self.nodes_dict.find((cn.coords[0], cn.coords[1])) is None]
        for cn_stale in stale:
            self.cluster_center_node_dict.pop((cn_stale.coords[0], cn_stale.coords[1]), None)

        clustered = set()
        for cc, _ in self.cluster_center_node_dict.values():
            self.update_clustered_nodes(cc, clustered)

        for center_node in nz_nodes:
            if center_node in clustered:
                continue
            self.cluster_center_node_dict[(center_node.coords[0], center_node.coords[1])] = [center_node, None]
            self.update_clustered_nodes(center_node, clustered)

        for center_node, _ in self.cluster_center_node_dict.values():
            path, dist = self.a_star(robot_location, center_node.coords)
            if dist >= 1e8:
                continue
            if dist < self.dist_to_nearest_frontier:
                self.dist_to_nearest_frontier = dist
                self.path_to_nearest_frontier = [np.array(p) for p in path]
            ref_node = self.key_node_dict[ref]
            self._insert_key_nodes_along_path(path, ref, ref_node, map_info, wavelet_map, wavelet_map_info)

        for node in nz_nodes:
            nk = (node.coords[0], node.coords[1])
            if nk not in self.key_node_dict:
                self.key_node_dict[nk] = KeyNode(node.coords, node.utility, node.visited)

        for nk in self.key_node_dict:
            kn = self.key_node_dict[nk]
            wrapper = self.nodes_dict.find(nk)
            if wrapper is None:
                continue
            for nb in wrapper.data.neighbor_set:
                if nb in self.key_node_dict:
                    kn.add_neighbor(nb)

        if self.path_to_nearest_frontier:
            self.path_to_nearest_frontier = [np.array(p) for p in self.path_to_nearest_frontier]

    def Dijkstra(self, start, boundary=None):
        del boundary
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
            node = self.nodes_dict.find(n)
            if node is None:
                open_list.discard(n)
                continue
            node = node.data

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
        self.observable_frontiers = self.initialize_observable_frontiers(frontiers, updating_map_info)
        self.visited = 0

        self.neighbor_matrix = -np.ones((5, 5))
        self.neighbor_set = {(self.coords[0], self.coords[1])}
        self.neighbor_matrix[2, 2] = 1
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
            collision = check_collision(self.coords, point, updating_map_info)
            if not collision:
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
                collision = check_collision(self.coords, neighbor_coords, updating_map_info)
                neighbor_matrix_x = center_index + (center_index - i)
                neighbor_matrix_y = center_index + (center_index - j)
                if not collision:
                    self.neighbor_matrix[i, j] = 1
                    self.neighbor_set.add((neighbor_coords[0], neighbor_coords[1]))
                    neighbor_node.neighbor_matrix[neighbor_matrix_x, neighbor_matrix_y] = 1
                    neighbor_node.neighbor_set.add((self.coords[0], self.coords[1]))

        if self.utility == 0:
            self.need_update_neighbor = False

    def update_node_observable_frontiers(self, new_frontiers, global_frontiers, updating_map_info):
        frontiers_observed = [frontier for frontier in self.observable_frontiers if frontier not in global_frontiers]
        for frontier in frontiers_observed:
            self.observable_frontiers.remove(frontier)

        if len(new_frontiers) > 0:
            new_frontiers = np.array(list(new_frontiers)).reshape(-1, 2)
            dist_list = np.linalg.norm(new_frontiers - self.coords, axis=-1)
            new_frontiers_in_range = new_frontiers[dist_list < self.utility_range]
            for point in new_frontiers_in_range:
                collision = check_collision(self.coords, point, updating_map_info)
                if not collision:
                    self.observable_frontiers.add((point[0], point[1]))

        self.utility = len(self.observable_frontiers)
        if self.utility <= MIN_UTILITY:
            self.utility = 0
            self.observable_frontiers = set()

    def set_visited(self):
        self.visited = 1
        self.observable_frontiers = set()
        self.utility = 0


class KeyNode:
    def __init__(self, coords, utility, visited):
        self.coords = coords
        self.utility = utility
        self.visited = visited
        self.neighbor_set = {(coords[0], coords[1])}

    def add_neighbor(self, coords):
        self.neighbor_set.add((coords[0], coords[1]) if not isinstance(coords, tuple) else coords)


def _get_quad_tree_box(center, size):
    half = size / 2
    return quads.BoundingBox(center[0] - half, center[1] - half, size, size)
