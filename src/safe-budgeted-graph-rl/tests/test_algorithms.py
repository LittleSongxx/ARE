from __future__ import annotations

import numpy as np

from sbge.algorithms import ConstrainedSACAgent, ReplayBuffer, observation_to_tensors, transition_from_observations
from sbge.config import SBGEConfig
from sbge.env import SafeBudgetedGraphEnv
from sbge.graph import SafeGraphBuilder
from sbge.types import Observation


def _config() -> SBGEConfig:
    return SBGEConfig(seed=7).smoke(seed=7).with_overrides(max_nodes=64, max_neighbors=12, batch_size=2)


def test_all_unsafe_observation_uses_fallback():
    config = _config()
    agent = ConstrainedSACAgent(config)
    obs = Observation(
        node_features=np.zeros((2, config.actor_node_dim), dtype=np.float32),
        critic_node_features=np.zeros((2, config.critic_node_dim), dtype=np.float32),
        edge_mask=np.ones((2, 2), dtype=bool),
        action_mask=np.ones(config.max_neighbors, dtype=bool),
        current_index=0,
        neighbor_indices=np.zeros(config.max_neighbors, dtype=np.int64),
        node_positions=np.zeros((2, 2), dtype=float),
        fallback_action_slot=3,
    )

    decision = agent.select_action(obs)

    assert decision.action_slot == 3
    assert decision.shield_intervention == 1
    assert decision.unsafe_action_proposal == 1


def test_network_forward_shapes():
    config = _config()
    env = SafeBudgetedGraphEnv(config, seed=7)
    env.reset(0)
    obs = SafeGraphBuilder(config).build(env)
    agent = ConstrainedSACAgent(config)
    tensors = observation_to_tensors(obs, config, agent.device, batch=True)

    logp = agent.policy(
        tensors["actor_node_features"],
        tensors["node_mask"],
        tensors["current_index"],
        tensors["neighbor_indices"],
        tensors["action_mask"],
    )
    reward_q = agent.reward_q1(
        tensors["critic_node_features"],
        tensors["node_mask"],
        tensors["current_index"],
        tensors["neighbor_indices"],
    )
    cost_q = agent.cost_q1(
        tensors["critic_node_features"],
        tensors["node_mask"],
        tensors["current_index"],
        tensors["neighbor_indices"],
    )

    assert logp.shape == (1, config.max_neighbors)
    assert reward_q.shape == (1, config.max_neighbors)
    assert cost_q.shape == (1, config.max_neighbors)


def test_constrained_sac_single_update_keeps_lambda_nonnegative():
    config = _config()
    env = SafeBudgetedGraphEnv(config, seed=7)
    env.reset(0)
    graph_builder = SafeGraphBuilder(config)
    agent = ConstrainedSACAgent(config)
    replay = ReplayBuffer(8)

    obs = graph_builder.build(env)
    decision = agent.select_action(obs)
    next_idx = int(obs.neighbor_indices[decision.action_slot])
    result = env.step(obs.node_positions[next_idx], graph_builder.edge_return_cost_for_action(obs, decision.action_slot))
    next_obs = graph_builder.build(env)
    transition = transition_from_observations(obs, decision.action_slot, result.reward, result.cost, result.done, next_obs)
    replay.append(transition)
    replay.append(transition)

    metrics = agent.train_step(replay.sample(2))

    assert metrics["lambda"] >= 0.0
    assert "reward_q_loss" in metrics
    assert "cost_q_loss" in metrics
