from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HierarchicalGraph:
    cluster_ids: np.ndarray
    cluster_centers: np.ndarray
    cluster_adjacency: np.ndarray
    node_cluster_prior: np.ndarray


def _empty_graph(n_nodes: int) -> HierarchicalGraph:
    return HierarchicalGraph(
        cluster_ids=np.zeros(max(int(n_nodes), 0), dtype=np.int64),
        cluster_centers=np.zeros((0, 2), dtype=np.float32),
        cluster_adjacency=np.zeros((0, 0), dtype=np.int8),
        node_cluster_prior=np.zeros(max(int(n_nodes), 0), dtype=np.float32),
    )


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    v_min = float(np.min(finite))
    v_max = float(np.max(finite))
    if v_max - v_min < 1e-6:
        return np.clip(values, 0.0, 1.0).astype(np.float32)
    return np.clip((values - v_min) / (v_max - v_min), 0.0, 1.0).astype(np.float32)


def build_hierarchical_graph(
    node_coords: np.ndarray,
    adjacency_matrix: np.ndarray,
    node_scores: np.ndarray | None = None,
    cluster_resolution: float = 12.0,
    cluster_edge_hops: int = 1,
) -> HierarchicalGraph:
    node_coords = np.asarray(node_coords, dtype=np.float64).reshape(-1, 2)
    n_nodes = node_coords.shape[0]
    if n_nodes == 0:
        return _empty_graph(0)

    cluster_resolution = max(float(cluster_resolution), 1e-6)
    raw_keys = np.floor(node_coords / cluster_resolution).astype(np.int64)
    unique_keys, cluster_ids = np.unique(raw_keys, axis=0, return_inverse=True)
    n_clusters = unique_keys.shape[0]

    cluster_centers = np.zeros((n_clusters, 2), dtype=np.float32)
    cluster_scores = np.zeros(n_clusters, dtype=np.float32)
    node_scores = np.zeros(n_nodes, dtype=np.float32) if node_scores is None else np.asarray(node_scores, dtype=np.float32).reshape(n_nodes)
    for cluster_index in range(n_clusters):
        mask = cluster_ids == cluster_index
        cluster_centers[cluster_index] = np.mean(node_coords[mask], axis=0)
        cluster_scores[cluster_index] = float(np.mean(node_scores[mask])) if np.any(mask) else 0.0

    cluster_adjacency = np.ones((n_clusters, n_clusters), dtype=np.int8)
    np.fill_diagonal(cluster_adjacency, 0)
    adjacency_matrix = np.asarray(adjacency_matrix).reshape(n_nodes, n_nodes)
    edge_rows, edge_cols = np.where(adjacency_matrix == 0)
    for row, col in zip(edge_rows, edge_cols):
        src = int(cluster_ids[row])
        dst = int(cluster_ids[col])
        cluster_adjacency[src, dst] = 0
        cluster_adjacency[dst, src] = 0

    propagated = cluster_scores.copy()
    hops = max(int(cluster_edge_hops), 0)
    for _ in range(hops):
        next_scores = propagated.copy()
        for cluster_index in range(n_clusters):
            neighbors = np.where(cluster_adjacency[cluster_index] == 0)[0]
            if neighbors.size > 0:
                next_scores[cluster_index] = max(
                    float(propagated[cluster_index]),
                    float(np.mean(propagated[neighbors])),
                )
        propagated = next_scores

    propagated = _normalize(propagated)
    node_cluster_prior = propagated[cluster_ids].astype(np.float32)
    return HierarchicalGraph(
        cluster_ids=cluster_ids.astype(np.int64),
        cluster_centers=cluster_centers.astype(np.float32),
        cluster_adjacency=cluster_adjacency,
        node_cluster_prior=node_cluster_prior,
    )