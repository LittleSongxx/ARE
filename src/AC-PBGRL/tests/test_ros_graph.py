import numpy as np

from ac_pbgrl_ros.graph_builder import OccupancyGraphBuilder
from ac_pbgrl_ros.inference import PotentialKF


def test_ros_graph_preserves_candidate_coordinates_and_shapes():
    grid = np.zeros((80, 80), dtype=np.int16)
    grid[:5] = 100
    grid[-5:] = -1
    builder = OccupancyGraphBuilder(nodes=40, candidates=8, local_budget=32)
    graph = builder.build(grid, 0.5, (0.0, 0.0), (20.0, 20.0))
    assert graph.feeds["node_features"].shape == (1, 40, 4)
    assert graph.feeds["candidate_indices"].shape == (1, 8)
    assert graph.feeds["candidate_mask"].any()
    assert np.all(np.isfinite(graph.candidate_xy[graph.feeds["candidate_mask"][0]]))
    valid_nodes = int(graph.feeds["node_mask"].sum())
    assert np.all(np.diag(graph.feeds["adjacency"][0])[:valid_nodes])


def test_ros_graph_emits_visit_event_for_stable_candidate():
    grid = np.zeros((80, 80), dtype=np.int16)
    grid[-5:] = -1
    builder = OccupancyGraphBuilder(nodes=40, candidates=8, local_budget=32)
    first = builder.build(grid, 0.5, (0.0, 0.0), (20.0, 20.0))
    valid = first.feeds["candidate_mask"][0]
    slot = int(np.flatnonzero(valid)[0])
    candidate_id = int(first.candidate_ids[slot])
    builder.mark_visited(*first.candidate_xy[slot])
    second = builder.build(grid, 0.5, (0.0, 0.0), (20.0, 20.0))
    matching = np.flatnonzero(second.candidate_ids == candidate_id)
    assert len(matching) == 1
    assert int(second.candidate_events[int(matching[0])]) & 16


def test_ros_kf_nis_event_reset_and_retire():
    filt = PotentialKF(p0=1.0, q_stable=0.01, q_event=0.25, nis_threshold=0.1)
    assert filt.update(7, 1.0, 0.2, 0) == (1.0, 1.0)
    mean, variance = filt.update(7, 10.0, 0.2, 1)
    assert mean > 1.0 and variance < 1.25
    mean, variance = filt.update(7, 3.0, 0.2, 2, event=16)
    assert (mean, variance) == (3.0, 1.0)
    filt.retire(7)
    assert 7 not in filt.records
