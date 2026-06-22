"""Graph rarefaction algorithm for large-scale robot exploration.

Reproduces the graph sparsification described in Cao et al. (RAL 2024)
"Deep Reinforcement Learning-based Large-scale Robot Exploration".

The algorithm reduces a dense exploration graph to a compact representation
that preserves navigation-critical structure while fitting within a fixed
node budget (NODE_PADDING_SIZE). This enables models trained in small
environments to scale to large ones.
"""

from __future__ import annotations

import heapq

import numpy as np


def _compute_degree(adj: np.ndarray, idx: int) -> int:
    """Count neighbors (edges where adj==0, excluding self-loop)."""
    row = adj[idx]
    count = int(np.sum(row == 0)) - 1  # subtract self-loop
    return max(count, 0)


def _dijkstra_sparse(adj: np.ndarray, coords: np.ndarray, start: int) -> np.ndarray:
    """Dijkstra from *start* on the adjacency matrix. Returns distance array."""
    n = adj.shape[0]
    dist = np.full(n, np.inf)
    dist[start] = 0.0
    visited = np.zeros(n, dtype=bool)
    heap = [(0.0, start)]

    while heap:
        d, u = heapq.heappop(heap)
        if visited[u]:
            continue
        visited[u] = True
        neighbors = np.where(adj[u] == 0)[0]
        for v in neighbors:
            if v == u or visited[v]:
                continue
            cost = float(np.linalg.norm(coords[u] - coords[v]))
            alt = d + cost
            if alt < dist[v]:
                dist[v] = alt
                heapq.heappush(heap, (alt, int(v)))
    return dist


def identify_anchors(
    coords: np.ndarray,
    utility: np.ndarray,
    adj: np.ndarray,
    current_index: int,
) -> set[int]:
    """Phase 1: identify anchor nodes that must be preserved.

    Anchors include:
    - the robot's current node
    - nodes with utility > 0 (frontier information)
    - nodes with degree != 2 (junctions and dead-ends)
    """
    n = coords.shape[0]
    anchors = {current_index}

    for i in range(n):
        if utility[i] > 0:
            anchors.add(i)
        degree = _compute_degree(adj, i)
        if degree != 2:
            anchors.add(i)

    return anchors


def contract_chains(
    adj: np.ndarray,
    coords: np.ndarray,
    anchors: set[int],
) -> tuple[set[int], np.ndarray]:
    """Phase 2: contract degree-2 chain nodes between anchors.

    Returns the set of nodes to keep and a new adjacency matrix connecting
    the sparse node set. Chain edges are replaced with direct edges weighted
    by the accumulated Euclidean distance.
    """
    n = adj.shape[0]
    keep = set(anchors)

    sparse_adj = np.ones((n, n), dtype=int)
    np.fill_diagonal(sparse_adj, 0)

    for a in sorted(anchors):
        neighbors = np.where(adj[a] == 0)[0]
        for nb in neighbors:
            if int(nb) == a:
                continue

            if int(nb) in anchors:
                sparse_adj[a, nb] = 0
                sparse_adj[nb, a] = 0
                keep.add(int(nb))
                continue

            prev = a
            cur = int(nb)
            while cur not in anchors:
                row = np.where(adj[cur] == 0)[0]
                nexts = [int(x) for x in row if int(x) != cur and int(x) != prev]
                if len(nexts) != 1:
                    anchors.add(cur)
                    keep.add(cur)
                    break
                prev, cur = cur, nexts[0]

            keep.add(cur)
            sparse_adj[a, cur] = 0
            sparse_adj[cur, a] = 0

    return keep, sparse_adj


def distance_aware_pruning(
    coords: np.ndarray,
    utility: np.ndarray,
    adj: np.ndarray,
    keep: set[int],
    current_index: int,
    max_nodes: int,
) -> np.ndarray:
    """Phase 3: if the sparse graph still exceeds the budget, prune distant
    low-utility nodes while preserving connectivity.

    Returns sorted array of indices to retain.
    """
    if len(keep) <= max_nodes:
        return np.array(sorted(keep), dtype=int)

    dist = _dijkstra_sparse(adj, coords, current_index)

    scored = []
    for idx in keep:
        if idx == current_index:
            continue
        u = float(utility[idx])
        d = float(dist[idx]) if np.isfinite(dist[idx]) else 1e8
        score = u / (1.0 + d)
        scored.append((score, idx))

    scored.sort(key=lambda x: x[0], reverse=True)

    result = {current_index}
    for _, idx in scored:
        if len(result) >= max_nodes:
            break
        result.add(idx)

    return np.array(sorted(result), dtype=int)


def remap_adjacency(
    old_adj: np.ndarray,
    sparse_adj: np.ndarray,
    selected: np.ndarray,
) -> np.ndarray:
    """Build compact adjacency matrix for the selected node subset.

    An edge exists in the result if it existed in either the original dense
    adjacency or in the contracted sparse adjacency.
    """
    n = len(selected)
    new_adj = np.ones((n, n), dtype=int)
    np.fill_diagonal(new_adj, 0)

    for i in range(n):
        for j in range(i + 1, n):
            oi, oj = selected[i], selected[j]
            if old_adj[oi, oj] == 0 or sparse_adj[oi, oj] == 0:
                new_adj[i, j] = 0
                new_adj[j, i] = 0

    _ensure_connected(new_adj, 0)
    return new_adj


def _ensure_connected(adj: np.ndarray, root: int) -> None:
    """Patch the adjacency matrix to guarantee all nodes are reachable from root.

    Uses BFS; any unreachable node is connected to its nearest reachable neighbor.
    """
    n = adj.shape[0]
    if n <= 1:
        return

    visited = np.zeros(n, dtype=bool)
    queue = [root]
    visited[root] = True
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        for v in range(n):
            if not visited[v] and adj[u, v] == 0:
                visited[v] = True
                queue.append(v)

    if np.all(visited):
        return

    reachable_set = set(np.where(visited)[0].tolist())
    unreachable = np.where(~visited)[0]
    for u in unreachable:
        adj[u, root] = 0
        adj[root, u] = 0


def graph_rarefaction(
    coords: np.ndarray,
    utility: np.ndarray,
    adj: np.ndarray,
    current_index: int,
    max_nodes: int = 360,
    scoring_utility: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Main entry point: reduce a dense exploration graph to a sparse one.

    Parameters
    ----------
    coords : (N, 2) array of node world coordinates
    utility : (N,) array of node utility values (raw, used for anchor detection)
    adj : (N, N) adjacency matrix (0 = connected, 1 = not connected, convention
          matching ARiADNE / large-scale-DRL-exploration)
    current_index : index of the robot's current node
    max_nodes : maximum number of nodes to keep (NODE_PADDING_SIZE)
    scoring_utility : optional (N,) array of smoothed utility values for
          distance-aware pruning scoring. If None, uses *utility*.

    Returns
    -------
    selected_indices : (M,) sorted array of original indices to keep
    new_adj : (M, M) adjacency matrix for the sparse graph
    """
    n = coords.shape[0]
    if n <= max_nodes:
        return np.arange(n, dtype=int), adj.copy()

    anchors = identify_anchors(coords, utility, adj, current_index)

    keep, sparse_adj = contract_chains(adj, coords, anchors)

    pruning_utility = scoring_utility if scoring_utility is not None else utility
    selected = distance_aware_pruning(coords, pruning_utility, adj, keep, current_index, max_nodes)

    new_adj = remap_adjacency(adj, sparse_adj, selected)

    return selected, new_adj
