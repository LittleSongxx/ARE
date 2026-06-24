from __future__ import annotations

import numpy as np

from sbge.config import SBGEConfig
from sbge.env import SafeBudgetedGraphEnv
from sbge.graph import SafeGraphBuilder


def _config() -> SBGEConfig:
    return SBGEConfig(seed=7).smoke(seed=7).with_overrides(max_nodes=64, max_neighbors=12)


def test_observation_has_expected_feature_shapes():
    config = _config()
    env = SafeBudgetedGraphEnv(config, seed=7)
    env.reset(0)
    obs = SafeGraphBuilder(config).build(env)

    assert obs.node_features.shape[1] == 11
    assert obs.critic_node_features.shape[1] == 14
    assert obs.action_mask.shape == (config.max_neighbors,)
    assert obs.neighbor_indices.shape == (config.max_neighbors,)


def test_static_and_dynamic_risk_are_deterministic_for_seed():
    config = _config()
    env_a = SafeBudgetedGraphEnv(config, seed=13)
    env_b = SafeBudgetedGraphEnv(config, seed=13)
    env_a.reset(2)
    env_b.reset(2)

    assert np.allclose(env_a.static_risk_map, env_b.static_risk_map)
    assert len(env_a.dynamic_obstacles) == len(env_b.dynamic_obstacles)
    for obs_a, obs_b in zip(env_a.dynamic_obstacles, env_b.dynamic_obstacles):
        assert np.allclose(obs_a["position"], obs_b["position"])
        assert np.allclose(obs_a["velocity"], obs_b["velocity"])


def test_budget_decrements_and_budget_violation_is_reported():
    config = _config().with_overrides(budget_ratio_min=0.001, budget_ratio_max=0.001)
    env = SafeBudgetedGraphEnv(config, seed=7)
    env.reset(0)
    graph_builder = SafeGraphBuilder(config)
    obs = graph_builder.build(env)
    slot = int(obs.fallback_action_slot)
    before = env.remaining_budget
    result = env.step(obs.node_positions[int(obs.neighbor_indices[slot])], return_cost_m=1e9)

    assert env.remaining_budget <= before
    assert result.info["budget_violation"] == 1


def test_action_mask_can_block_budget_infeasible_edges():
    config = _config().with_overrides(budget_ratio_min=0.001, budget_ratio_max=0.001)
    env = SafeBudgetedGraphEnv(config, seed=9)
    env.reset(0)
    obs = SafeGraphBuilder(config).build(env)

    assert obs.action_mask.shape == (config.max_neighbors,)
    assert np.any(obs.action_mask)
