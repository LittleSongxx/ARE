from __future__ import annotations

import unittest

import numpy as np

from wpg_rl.sparse_visualization import build_sparse_graph_view


class FakeNode:
    def __init__(self, coords, utility=0.0, wavelet=0.0, visited=0.0):
        self.coords = np.asarray(coords, dtype=np.float32)
        self.utility = float(utility)
        self.wavelet = float(wavelet)
        self.visited = float(visited)
        self.neighbor_set = {tuple(self.coords.tolist())}


class FakeNodeWrapper:
    def __init__(self, data):
        self.data = data


class FakeNodesDict:
    def __init__(self, nodes):
        self._nodes = {
            tuple(np.asarray(node.coords, dtype=np.float32).tolist()): FakeNodeWrapper(node)
            for node in nodes
        }

    def __iter__(self):
        return iter(self._nodes.values())

    def find(self, key):
        key = tuple(np.asarray(key, dtype=np.float32).reshape(-1)[:2].tolist())
        return self._nodes.get(key)


class FakeNodeManager:
    def __init__(self, nodes, dist_dict, prev_dict):
        self.nodes_dict = FakeNodesDict(nodes)
        self._dist_dict = dist_dict
        self._prev_dict = prev_dict

    def Dijkstra(self, start, boundary=None):
        del start, boundary
        return self._dist_dict, self._prev_dict

    def get_Dijkstra_path_and_dist(self, dist_dict, prev_dict, end):
        key = tuple(np.asarray(end, dtype=np.float32).reshape(-1)[:2].tolist())
        if key not in dist_dict:
            return [], 1e8

        dist = dist_dict[key]
        path = [key]
        prev_node = prev_dict[key]
        while prev_node is not None:
            path.append(prev_node)
            prev_node = prev_dict[prev_node]
        path.reverse()
        return path[1:], dist


class SparseVisualizationTests(unittest.TestCase):
    def test_contracts_degree_two_path_nodes_but_keeps_goal(self):
        start = FakeNode((0.0, 0.0), visited=1.0)
        mid_1 = FakeNode((4.0, 0.0), wavelet=0.2)
        mid_2 = FakeNode((8.0, 0.0), wavelet=0.4)
        goal = FakeNode((12.0, 0.0), utility=5.0, wavelet=0.9)

        start.neighbor_set.update({(4.0, 0.0)})
        mid_1.neighbor_set.update({(0.0, 0.0), (8.0, 0.0)})
        mid_2.neighbor_set.update({(4.0, 0.0), (12.0, 0.0)})
        goal.neighbor_set.update({(8.0, 0.0)})

        node_manager = FakeNodeManager(
            [start, mid_1, mid_2, goal],
            dist_dict={
                (0.0, 0.0): 0.0,
                (4.0, 0.0): 4.0,
                (8.0, 0.0): 8.0,
                (12.0, 0.0): 12.0,
            },
            prev_dict={
                (0.0, 0.0): None,
                (4.0, 0.0): (0.0, 0.0),
                (8.0, 0.0): (4.0, 0.0),
                (12.0, 0.0): (8.0, 0.0),
            },
        )

        sparse_graph = build_sparse_graph_view(node_manager, np.array([0.0, 0.0], dtype=np.float32))

        self.assertEqual(sparse_graph.node_coords.tolist(), [[0.0, 0.0], [12.0, 0.0]])
        self.assertEqual(len(sparse_graph.edge_paths), 1)
        self.assertEqual(
            sparse_graph.edge_paths[0].tolist(),
            [[0.0, 0.0], [4.0, 0.0], [8.0, 0.0], [12.0, 0.0]],
        )


if __name__ == "__main__":
    unittest.main()
