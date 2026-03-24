from __future__ import annotations

import unittest

import numpy as np

from wpg_rl.corridor_refinement import (
    build_refined_adjacency_matrix,
    compute_smoothness_penalty,
    refine_neighbor_indices,
)
from wpg_rl.parameter import CELL_SIZE, FREE, OCCUPIED
from wpg_rl.utils import MapInfo


def _build_vertical_corridor_map() -> MapInfo:
    grid = np.ones((41, 41), dtype=int) * OCCUPIED
    grid[:, 10:21] = FREE
    return MapInfo(grid, 0.0, 0.0, CELL_SIZE)


class CorridorRefinementTests(unittest.TestCase):
    def test_refine_neighbor_indices_default_off_preserves_original_actions(self):
        map_info = _build_vertical_corridor_map()
        node_coords = np.asarray(
            [
                [4.0, 8.0],
                [4.0, 12.0],
                [8.0, 12.0],
                [4.0, 4.0],
            ],
            dtype=float,
        )
        raw_neighbors = np.asarray([0, 1, 2, 3], dtype=int)

        refined = refine_neighbor_indices(
            node_coords,
            current_index=0,
            neighbor_indices=raw_neighbors,
            map_info=map_info,
            enable_edge_pruning=False,
            enable_graph_compression=False,
        )

        self.assertTrue(np.array_equal(refined, raw_neighbors))

    def test_edge_pruning_removes_cross_lane_actions_in_narrow_corridor(self):
        map_info = _build_vertical_corridor_map()
        node_coords = np.asarray(
            [
                [4.0, 8.0],
                [4.0, 12.0],
                [8.0, 12.0],
                [4.0, 4.0],
                [8.0, 8.0],
            ],
            dtype=float,
        )
        raw_neighbors = np.asarray([0, 1, 2, 3, 4], dtype=int)

        refined = refine_neighbor_indices(
            node_coords,
            current_index=0,
            neighbor_indices=raw_neighbors,
            map_info=map_info,
            enable_edge_pruning=True,
            enable_graph_compression=False,
        )

        self.assertEqual(refined.tolist(), [0, 1, 3])

    def test_graph_compression_keeps_single_forward_candidate_per_corridor_direction(self):
        map_info = _build_vertical_corridor_map()
        node_coords = np.asarray(
            [
                [4.0, 8.0],
                [4.0, 12.0],
                [8.0, 12.0],
                [4.0, 4.0],
            ],
            dtype=float,
        )
        raw_neighbors = np.asarray([0, 1, 2, 3], dtype=int)

        refined = refine_neighbor_indices(
            node_coords,
            current_index=0,
            neighbor_indices=raw_neighbors,
            map_info=map_info,
            enable_edge_pruning=False,
            enable_graph_compression=True,
        )

        self.assertEqual(refined.tolist(), [0, 1, 3])

    def test_refined_adjacency_matrix_stays_symmetric(self):
        map_info = _build_vertical_corridor_map()
        node_coords = np.asarray(
            [
                [4.0, 8.0],
                [4.0, 12.0],
                [8.0, 12.0],
                [4.0, 4.0],
                [8.0, 8.0],
            ],
            dtype=float,
        )
        adjacency = np.ones((5, 5), dtype=int)
        np.fill_diagonal(adjacency, 0)
        adjacency[0, 1] = adjacency[1, 0] = 0
        adjacency[0, 2] = adjacency[2, 0] = 0
        adjacency[0, 3] = adjacency[3, 0] = 0
        adjacency[0, 4] = adjacency[4, 0] = 0
        adjacency[1, 2] = adjacency[2, 1] = 0
        adjacency[2, 4] = adjacency[4, 2] = 0

        refined = build_refined_adjacency_matrix(
            node_coords,
            adjacency,
            map_info,
            enable_edge_pruning=True,
            enable_graph_compression=True,
        )

        self.assertTrue(np.array_equal(refined, refined.T))
        self.assertEqual(np.argwhere(refined[0] == 0).reshape(-1).tolist(), [0, 1, 3])

    def test_smoothness_penalty_is_zero_for_straight_motion_and_positive_for_turns(self):
        straight = compute_smoothness_penalty(
            [4.0, 0.0],
            [4.0, 4.0],
            [4.0, 8.0],
            turn_penalty_weight=0.5,
            lateral_penalty_weight=0.25,
        )
        turning = compute_smoothness_penalty(
            [4.0, 0.0],
            [4.0, 4.0],
            [8.0, 8.0],
            turn_penalty_weight=0.5,
            lateral_penalty_weight=0.25,
        )

        self.assertAlmostEqual(straight, 0.0)
        self.assertGreater(turning, straight)


if __name__ == "__main__":
    unittest.main()
