from __future__ import annotations

import unittest

import numpy as np

from hpbg_rl.hierarchical_graph import build_hierarchical_graph


class HierarchicalGraphTests(unittest.TestCase):
    def test_cluster_graph_shapes_and_node_prior(self):
        node_coords = np.asarray(
            [
                [0.0, 0.0],
                [4.0, 0.0],
                [12.0, 0.0],
                [16.0, 0.0],
            ],
            dtype=float,
        )
        adjacency = np.ones((4, 4), dtype=int)
        np.fill_diagonal(adjacency, 0)
        adjacency[0, 1] = adjacency[1, 0] = 0
        adjacency[1, 2] = adjacency[2, 1] = 0
        adjacency[2, 3] = adjacency[3, 2] = 0
        scores = np.asarray([0.1, 0.2, 0.9, 1.0], dtype=np.float32)

        graph = build_hierarchical_graph(
            node_coords,
            adjacency,
            node_scores=scores,
            cluster_resolution=8.0,
            cluster_edge_hops=1,
        )

        self.assertEqual(graph.cluster_ids.shape, (4,))
        self.assertEqual(graph.node_cluster_prior.shape, (4,))
        self.assertEqual(graph.cluster_adjacency.shape[0], graph.cluster_centers.shape[0])
        self.assertEqual(graph.cluster_adjacency.shape[1], graph.cluster_centers.shape[0])
        self.assertTrue(np.isfinite(graph.node_cluster_prior).all())
        self.assertTrue(np.all(graph.node_cluster_prior >= 0.0))
        self.assertTrue(np.all(graph.node_cluster_prior <= 1.0))
        self.assertTrue(np.any(graph.cluster_adjacency == 0))

    def test_empty_graph_keeps_node_width_contract(self):
        graph = build_hierarchical_graph(np.zeros((0, 2)), np.zeros((0, 0)))
        self.assertEqual(graph.cluster_ids.shape, (0,))
        self.assertEqual(graph.cluster_centers.shape, (0, 2))
        self.assertEqual(graph.cluster_adjacency.shape, (0, 0))
        self.assertEqual(graph.node_cluster_prior.shape, (0,))


if __name__ == "__main__":
    unittest.main()