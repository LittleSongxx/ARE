"""Tests for the graph rarefaction algorithm."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph_rarefaction import (
    _compute_degree,
    _dijkstra_sparse,
    _ensure_connected,
    contract_chains,
    distance_aware_pruning,
    graph_rarefaction,
    identify_anchors,
    remap_adjacency,
)


def _grid_graph(rows: int, cols: int):
    """Build a 2D grid graph for testing.

    Returns (coords, utility, adj, current_index).
    Adjacency convention: 0 = connected, 1 = not connected (ARiADNE style).
    """
    n = rows * cols
    coords = np.zeros((n, 2), dtype=float)
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            coords[idx] = [c * 4.0, r * 4.0]

    adj = np.ones((n, n), dtype=int)
    np.fill_diagonal(adj, 0)
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            if c + 1 < cols:
                right = r * cols + (c + 1)
                adj[idx, right] = 0
                adj[right, idx] = 0
            if r + 1 < rows:
                down = (r + 1) * cols + c
                adj[idx, down] = 0
                adj[down, idx] = 0

    utility = np.zeros(n)
    return coords, utility, adj, 0


def _chain_graph(length: int):
    """Build a simple linear chain graph: 0 -- 1 -- 2 -- ... -- (length-1).

    Utility > 0 only at the last node.
    """
    n = length
    coords = np.zeros((n, 2), dtype=float)
    for i in range(n):
        coords[i] = [i * 4.0, 0.0]

    adj = np.ones((n, n), dtype=int)
    np.fill_diagonal(adj, 0)
    for i in range(n - 1):
        adj[i, i + 1] = 0
        adj[i + 1, i] = 0

    utility = np.zeros(n)
    utility[-1] = 10.0
    return coords, utility, adj, 0


class TestComputeDegree:
    def test_isolated_node(self):
        adj = np.ones((3, 3), dtype=int)
        np.fill_diagonal(adj, 0)
        assert _compute_degree(adj, 0) == 0

    def test_fully_connected(self):
        adj = np.zeros((4, 4), dtype=int)
        assert _compute_degree(adj, 0) == 3

    def test_chain_endpoints(self):
        _, _, adj, _ = _chain_graph(5)
        assert _compute_degree(adj, 0) == 1
        assert _compute_degree(adj, 2) == 2
        assert _compute_degree(adj, 4) == 1


class TestDijkstra:
    def test_chain_distances(self):
        coords, _, adj, _ = _chain_graph(5)
        dist = _dijkstra_sparse(adj, coords, 0)
        assert dist[0] == pytest.approx(0.0)
        assert dist[1] == pytest.approx(4.0)
        assert dist[4] == pytest.approx(16.0)

    def test_grid_distances(self):
        coords, _, adj, _ = _grid_graph(3, 3)
        dist = _dijkstra_sparse(adj, coords, 0)
        assert dist[0] == pytest.approx(0.0)
        assert dist[8] == pytest.approx(16.0)


class TestIdentifyAnchors:
    def test_chain_anchors(self):
        coords, utility, adj, cur = _chain_graph(5)
        anchors = identify_anchors(coords, utility, adj, cur)
        assert cur in anchors
        assert 4 in anchors  # utility > 0
        assert 0 in anchors  # degree 1

    def test_all_degree2_interior_are_not_anchors(self):
        coords, utility, adj, cur = _chain_graph(10)
        utility[:] = 0
        utility[-1] = 5.0
        anchors = identify_anchors(coords, utility, adj, cur)
        for i in range(1, 9):
            if i not in {cur, 9}:
                assert i not in anchors, f"Interior node {i} should not be an anchor"

    def test_grid_junctions_are_anchors(self):
        coords, utility, adj, cur = _grid_graph(3, 3)
        anchors = identify_anchors(coords, utility, adj, cur)
        assert cur in anchors
        for i in range(9):
            deg = _compute_degree(adj, i)
            if deg != 2:
                assert i in anchors


class TestContractChains:
    def test_chain_contracts_to_endpoints(self):
        coords, utility, adj, cur = _chain_graph(10)
        utility[-1] = 5.0
        anchors = identify_anchors(coords, utility, adj, cur)
        keep, sparse_adj = contract_chains(adj, coords, anchors)
        assert 0 in keep
        assert 9 in keep
        assert sparse_adj[0, 9] == 0 or any(
            sparse_adj[0, k] == 0 and sparse_adj[k, 9] == 0 for k in keep
        )


class TestGraphRarefaction:
    def test_small_graph_passthrough(self):
        coords, utility, adj, cur = _grid_graph(3, 3)
        selected, new_adj = graph_rarefaction(coords, utility, adj, cur, max_nodes=100)
        assert len(selected) == 9
        assert new_adj.shape == (9, 9)

    def test_large_chain_is_compressed(self):
        coords, utility, adj, cur = _chain_graph(50)
        utility[-1] = 5.0
        selected, new_adj = graph_rarefaction(coords, utility, adj, cur, max_nodes=10)
        assert len(selected) <= 10
        assert 0 in set(selected.tolist())
        new_current = int(np.argwhere(selected == cur)[0][0])
        new_last_orig = np.argwhere(selected == 49)
        if len(new_last_orig) > 0:
            new_last = int(new_last_orig[0][0])
            assert utility[selected[new_last]] > 0

    def test_preserves_connectivity(self):
        coords, utility, adj, cur = _grid_graph(5, 5)
        utility[24] = 10.0
        utility[20] = 5.0
        selected, new_adj = graph_rarefaction(coords, utility, adj, cur, max_nodes=10)
        n = len(selected)
        visited = np.zeros(n, dtype=bool)
        new_current = int(np.argwhere(selected == cur)[0][0])
        stack = [new_current]
        visited[new_current] = True
        while stack:
            u = stack.pop()
            for v in range(n):
                if not visited[v] and new_adj[u, v] == 0:
                    visited[v] = True
                    stack.append(v)
        assert np.all(visited), "Graph is not fully connected after rarefaction"

    def test_preserves_utility_nodes(self):
        coords, utility, adj, cur = _chain_graph(100)
        utility[50] = 20.0
        utility[99] = 15.0
        selected, new_adj = graph_rarefaction(coords, utility, adj, cur, max_nodes=10)
        selected_set = set(selected.tolist())
        assert 50 in selected_set or len(selected_set) >= 10
        assert cur in selected_set

    def test_current_index_always_preserved(self):
        coords, utility, adj, _ = _grid_graph(4, 4)
        for cur in [0, 5, 10, 15]:
            selected, _ = graph_rarefaction(coords, utility, adj, cur, max_nodes=5)
            assert cur in set(selected.tolist())

    def test_adjacency_shape_matches_selected(self):
        coords, utility, adj, cur = _chain_graph(30)
        utility[29] = 10.0
        selected, new_adj = graph_rarefaction(coords, utility, adj, cur, max_nodes=8)
        assert new_adj.shape == (len(selected), len(selected))
        assert np.all(np.diag(new_adj) == 0)


class TestEnsureConnected:
    def test_already_connected(self):
        adj = np.ones((4, 4), dtype=int)
        np.fill_diagonal(adj, 0)
        adj[0, 1] = adj[1, 0] = 0
        adj[1, 2] = adj[2, 1] = 0
        adj[2, 3] = adj[3, 2] = 0
        _ensure_connected(adj, 0)
        visited = set()
        stack = [0]
        while stack:
            u = stack.pop()
            visited.add(u)
            for v in range(4):
                if v not in visited and adj[u, v] == 0:
                    stack.append(v)
        assert len(visited) == 4

    def test_disconnected_repaired(self):
        adj = np.ones((4, 4), dtype=int)
        np.fill_diagonal(adj, 0)
        adj[0, 1] = adj[1, 0] = 0
        _ensure_connected(adj, 0)
        visited = set()
        stack = [0]
        while stack:
            u = stack.pop()
            visited.add(u)
            for v in range(4):
                if v not in visited and adj[u, v] == 0:
                    stack.append(v)
        assert len(visited) == 4


class TestRemapAdjacency:
    def test_preserves_original_edges(self):
        _, _, adj, _ = _chain_graph(5)
        sparse_adj = np.ones_like(adj)
        np.fill_diagonal(sparse_adj, 0)
        selected = np.array([0, 1, 2, 3, 4])
        new_adj = remap_adjacency(adj, sparse_adj, selected)
        assert new_adj[0, 1] == 0
        assert new_adj[1, 2] == 0

    def test_subset_remapping(self):
        _, _, adj, _ = _chain_graph(5)
        sparse_adj = adj.copy()
        sparse_adj[0, 4] = 0
        sparse_adj[4, 0] = 0
        selected = np.array([0, 4])
        new_adj = remap_adjacency(adj, sparse_adj, selected)
        assert new_adj.shape == (2, 2)
        assert new_adj[0, 1] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
