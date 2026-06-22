from __future__ import annotations

import unittest

import torch

from hpbg_rl.wavelet_graph import (
    build_adjacency,
    build_overlap_valid_mask,
    build_random_walk,
    decompose_graph_hidden,
    gather_by_index,
    multiscale_wavelet_decompose,
)


class WaveletGraphTests(unittest.TestCase):
    def test_padding_nodes_do_not_leak_into_graph_operator(self):
        edge_mask = torch.tensor(
            [
                [
                    [0, 0, 1, 1],
                    [0, 0, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                ]
            ],
            dtype=torch.bool,
        )
        node_padding_mask = torch.tensor([[[0, 0, 1, 1]]], dtype=torch.bool)
        hidden = torch.randn(1, 4, 8)

        adjacency = build_adjacency(edge_mask, node_padding_mask)
        random_walk = build_random_walk(edge_mask, node_padding_mask)
        low_features, high_features, _, _ = multiscale_wavelet_decompose(hidden, edge_mask, node_padding_mask)

        self.assertEqual(adjacency[0, 2].sum().item(), 0.0)
        self.assertEqual(adjacency[0, :, 2].sum().item(), 0.0)
        self.assertEqual(adjacency[0, 0, 0].item(), 1.0)
        self.assertTrue(torch.isfinite(random_walk).all())
        self.assertEqual(low_features[0].shape, hidden.shape)
        self.assertEqual(high_features[0].shape, hidden.shape)
        self.assertTrue(torch.isfinite(low_features[0]).all())
        self.assertTrue(torch.isfinite(high_features[0]).all())

    def test_batch_safe_gather_supports_negative_one_indices(self):
        features = torch.tensor(
            [
                [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
                [[4.0, 40.0], [5.0, 50.0], [6.0, 60.0]],
            ]
        )
        index_map = torch.tensor([[2, -1, 1], [1, 0, -1]])
        actor_node_padding_mask = torch.tensor(
            [
                [[0, 0, 1]],
                [[0, 0, 0]],
            ],
            dtype=torch.bool,
        )

        gathered = gather_by_index(features, index_map)
        overlap_valid_mask = build_overlap_valid_mask(index_map, actor_node_padding_mask)

        self.assertEqual(gathered.shape, (2, 3, 2))
        self.assertTrue(torch.equal(gathered[0, 0], torch.tensor([3.0, 30.0])))
        self.assertTrue(torch.equal(gathered[0, 1], torch.tensor([0.0, 0.0])))
        self.assertTrue(torch.equal(gathered[1, 2], torch.tensor([0.0, 0.0])))
        self.assertTrue(torch.equal(overlap_valid_mask, torch.tensor([[True, False, False], [True, True, False]])))

    def test_decompose_graph_hidden_returns_concat_raw_features(self):
        edge_mask = torch.zeros(1, 4, 4, dtype=torch.bool)
        node_padding_mask = torch.tensor([[[0, 0, 0, 1]]], dtype=torch.bool)
        hidden = torch.randn(1, 4, 8)

        low_raw, high_raw, low_features, high_features, _, _ = decompose_graph_hidden(
            hidden,
            edge_mask,
            node_padding_mask,
            scales=(1, 2, 4),
        )

        self.assertEqual(low_raw.shape, (1, 4, 24))
        self.assertEqual(high_raw.shape, (1, 4, 24))
        self.assertEqual(len(low_features), 3)
        self.assertEqual(len(high_features), 3)
        self.assertTrue(torch.isfinite(low_raw).all())
        self.assertTrue(torch.isfinite(high_raw).all())


if __name__ == "__main__":
    unittest.main()
