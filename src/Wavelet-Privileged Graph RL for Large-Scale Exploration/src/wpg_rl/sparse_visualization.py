from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _coord_key(coords, digits: int = 3):
    coords = np.asarray(coords, dtype=np.float64).reshape(-1)
    return tuple(np.round(coords[:2], digits).tolist())


@dataclass(frozen=True)
class SparseGraphView:
    node_coords: np.ndarray
    utility: np.ndarray
    wavelet: np.ndarray
    visited: np.ndarray
    edge_paths: tuple[np.ndarray, ...]


def _collect_dense_nodes(node_manager) -> dict[tuple[float, float], object]:
    dense_nodes = {}
    for node in node_manager.nodes_dict.__iter__():
        data = node.data
        dense_nodes[_coord_key(data.coords)] = data
    return dense_nodes


def _build_relevant_subgraph(node_manager, robot_location, dense_nodes):
    current_key = _coord_key(robot_location)
    if current_key not in dense_nodes:
        return current_key, set(), []

    dist_dict, prev_dict = node_manager.Dijkstra(np.asarray(robot_location))
    utility_keys = [
        key
        for key, node in dense_nodes.items()
        if float(getattr(node, "utility", 0.0)) > 0.0 and dist_dict.get(key, 1e8) < 1e8
    ]

    relevant_keys = {current_key}
    for utility_key in utility_keys:
        path, dist = node_manager.get_Dijkstra_path_and_dist(dist_dict, prev_dict, np.asarray(utility_key))
        if dist >= 1e8:
            continue
        relevant_keys.add(utility_key)
        for coords in path:
            relevant_keys.add(_coord_key(coords))

    adjacency = {key: set() for key in relevant_keys}
    for key in relevant_keys:
        node = dense_nodes[key]
        for neighbor in getattr(node, "neighbor_set", set()):
            neighbor_key = _coord_key(neighbor)
            if neighbor_key == key or neighbor_key not in relevant_keys:
                continue
            adjacency[key].add(neighbor_key)

    return current_key, relevant_keys, adjacency


def _contract_subgraph(current_key, relevant_keys, adjacency, dense_nodes):
    if not relevant_keys:
        return {current_key}, ()

    anchor_keys = {current_key}
    for key in relevant_keys:
        node = dense_nodes[key]
        degree = len(adjacency.get(key, ()))
        if float(getattr(node, "utility", 0.0)) > 0.0 or degree != 2:
            anchor_keys.add(key)

    visited_edges = set()
    sparse_keys = set(anchor_keys)
    edge_paths = []

    for start_key in sorted(anchor_keys):
        for neighbor_key in sorted(adjacency.get(start_key, ())):
            edge_id = frozenset((start_key, neighbor_key))
            if edge_id in visited_edges:
                continue

            path = [start_key, neighbor_key]
            prev_key = start_key
            curr_key = neighbor_key

            while curr_key not in anchor_keys:
                next_keys = adjacency.get(curr_key, set()) - {prev_key}
                if len(next_keys) != 1:
                    anchor_keys.add(curr_key)
                    sparse_keys.add(curr_key)
                    break
                next_key = next(iter(next_keys))
                path.append(next_key)
                prev_key, curr_key = curr_key, next_key

            sparse_keys.add(curr_key)
            for edge_start, edge_end in zip(path, path[1:]):
                visited_edges.add(frozenset((edge_start, edge_end)))
            edge_paths.append(np.asarray(path, dtype=np.float32))

    return sparse_keys, tuple(edge_paths)


def build_sparse_graph_view(node_manager, robot_location) -> SparseGraphView:
    dense_nodes = _collect_dense_nodes(node_manager)
    current_key, relevant_keys, adjacency = _build_relevant_subgraph(node_manager, robot_location, dense_nodes)

    if current_key not in dense_nodes:
        empty = np.empty((0, 2), dtype=np.float32)
        empty_scalar = np.empty((0,), dtype=np.float32)
        return SparseGraphView(empty, empty_scalar, empty_scalar, empty_scalar, ())

    sparse_keys, edge_paths = _contract_subgraph(current_key, relevant_keys or {current_key}, adjacency, dense_nodes)

    ordered_keys = [current_key]
    ordered_keys.extend(sorted(key for key in sparse_keys if key != current_key))

    node_coords = np.asarray(ordered_keys, dtype=np.float32).reshape(-1, 2)
    utility = np.asarray([float(getattr(dense_nodes[key], "utility", 0.0)) for key in ordered_keys], dtype=np.float32)
    wavelet = np.asarray([float(getattr(dense_nodes[key], "wavelet", 0.0)) for key in ordered_keys], dtype=np.float32)
    visited = np.asarray([float(getattr(dense_nodes[key], "visited", 0.0)) for key in ordered_keys], dtype=np.float32)

    return SparseGraphView(node_coords, utility, wavelet, visited, edge_paths)


def _dijkstra_on_adjacency(node_coords, adjacency_matrix, start_index: int):
    n_nodes = node_coords.shape[0]
    dist = np.full((n_nodes,), 1e8, dtype=float)
    prev = np.full((n_nodes,), -1, dtype=int)
    remaining = set(range(n_nodes))
    dist[int(start_index)] = 0.0

    while remaining:
        current_index = min(remaining, key=lambda index: dist[index])
        remaining.remove(current_index)
        if dist[current_index] >= 1e8:
            continue

        neighbors = np.argwhere(adjacency_matrix[current_index] == 0).reshape(-1)
        for neighbor_index in neighbors:
            neighbor_index = int(neighbor_index)
            if neighbor_index == current_index or neighbor_index not in remaining:
                continue
            cost = float(np.linalg.norm(node_coords[neighbor_index] - node_coords[current_index]))
            alt = dist[current_index] + cost
            if alt < dist[neighbor_index]:
                dist[neighbor_index] = alt
                prev[neighbor_index] = current_index

    return dist, prev


def build_sparse_graph_view_from_arrays(
    node_coords,
    utility,
    visited,
    adjacency_matrix,
    current_index: int,
) -> SparseGraphView:
    node_coords = np.asarray(node_coords, dtype=np.float32).reshape(-1, 2)
    utility = np.asarray(utility, dtype=np.float32).reshape(-1)
    visited = np.asarray(visited, dtype=np.float32).reshape(-1)
    adjacency_matrix = np.asarray(adjacency_matrix, dtype=int)
    current_index = int(current_index)

    if node_coords.size == 0 or not (0 <= current_index < node_coords.shape[0]):
        empty = np.empty((0, 2), dtype=np.float32)
        empty_scalar = np.empty((0,), dtype=np.float32)
        return SparseGraphView(empty, empty_scalar, empty_scalar, empty_scalar, ())

    dist, prev = _dijkstra_on_adjacency(node_coords, adjacency_matrix, current_index)
    relevant = {current_index}
    utility_indices = np.where((utility > 0.0) & (dist < 1e8))[0]
    for target_index in utility_indices.tolist():
        cursor = int(target_index)
        while cursor >= 0 and cursor not in relevant:
            relevant.add(cursor)
            cursor = int(prev[cursor])
        relevant.add(int(target_index))

    adjacency = {index: set() for index in relevant}
    for index in relevant:
        neighbors = np.argwhere(adjacency_matrix[index] == 0).reshape(-1)
        for neighbor_index in neighbors:
            neighbor_index = int(neighbor_index)
            if neighbor_index == index or neighbor_index not in relevant:
                continue
            adjacency[index].add(neighbor_index)

    anchor_indices = {current_index}
    for index in relevant:
        degree = len(adjacency.get(index, ()))
        if float(utility[index]) > 0.0 or degree != 2:
            anchor_indices.add(index)

    sparse_indices = set(anchor_indices)
    edge_paths = []
    visited_edges = set()
    for start_index in sorted(anchor_indices):
        for neighbor_index in sorted(adjacency.get(start_index, ())):
            edge_id = frozenset((int(start_index), int(neighbor_index)))
            if edge_id in visited_edges:
                continue

            path = [int(start_index), int(neighbor_index)]
            prev_index = int(start_index)
            cursor = int(neighbor_index)
            while cursor not in anchor_indices:
                next_indices = adjacency.get(cursor, set()) - {prev_index}
                if len(next_indices) != 1:
                    anchor_indices.add(cursor)
                    sparse_indices.add(cursor)
                    break
                next_index = int(next(iter(next_indices)))
                path.append(next_index)
                prev_index, cursor = cursor, next_index

            sparse_indices.add(cursor)
            for edge_start, edge_end in zip(path, path[1:]):
                visited_edges.add(frozenset((int(edge_start), int(edge_end))))
            edge_paths.append(node_coords[np.asarray(path, dtype=int)])

    ordered_indices = [current_index]
    ordered_indices.extend(sorted(index for index in sparse_indices if index != current_index))
    ordered_indices = np.asarray(ordered_indices, dtype=int)
    wavelet = np.zeros((ordered_indices.shape[0],), dtype=np.float32)
    return SparseGraphView(
        node_coords[ordered_indices],
        utility[ordered_indices],
        wavelet,
        visited[ordered_indices],
        tuple(np.asarray(path, dtype=np.float32) for path in edge_paths),
    )
