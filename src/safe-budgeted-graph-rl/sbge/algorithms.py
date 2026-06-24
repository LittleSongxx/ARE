from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .config import SBGEConfig
from .networks import GraphPolicyNet, GraphQNet
from .types import Observation


@dataclass
class ActionDecision:
    action_slot: int
    shield_intervention: int = 0
    unsafe_action_proposal: int = 0
    fallback: int = 0


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self._items: deque[dict[str, Any]] = deque(maxlen=self.capacity)

    def append(self, transition: dict[str, Any]) -> None:
        self._items.append(transition)

    def __len__(self) -> int:
        return len(self._items)

    def sample(self, batch_size: int) -> list[dict[str, Any]]:
        return random.sample(list(self._items), min(int(batch_size), len(self._items)))


class ConstrainedSACAgent:
    def __init__(self, config: SBGEConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.policy = GraphPolicyNet(config.actor_node_dim, config.hidden_dim).to(self.device)
        self.reward_q1 = GraphQNet(config.critic_node_dim, config.hidden_dim).to(self.device)
        self.reward_q2 = GraphQNet(config.critic_node_dim, config.hidden_dim).to(self.device)
        self.target_reward_q1 = GraphQNet(config.critic_node_dim, config.hidden_dim).to(self.device)
        self.target_reward_q2 = GraphQNet(config.critic_node_dim, config.hidden_dim).to(self.device)
        self.cost_q1 = GraphQNet(config.critic_node_dim, config.hidden_dim).to(self.device)
        self.cost_q2 = GraphQNet(config.critic_node_dim, config.hidden_dim).to(self.device)
        self.target_cost_q1 = GraphQNet(config.critic_node_dim, config.hidden_dim).to(self.device)
        self.target_cost_q2 = GraphQNet(config.critic_node_dim, config.hidden_dim).to(self.device)
        self.target_reward_q1.load_state_dict(self.reward_q1.state_dict())
        self.target_reward_q2.load_state_dict(self.reward_q2.state_dict())
        self.target_cost_q1.load_state_dict(self.cost_q1.state_dict())
        self.target_cost_q2.load_state_dict(self.cost_q2.state_dict())

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.lr)
        self.reward_q1_optimizer = torch.optim.Adam(self.reward_q1.parameters(), lr=config.lr)
        self.reward_q2_optimizer = torch.optim.Adam(self.reward_q2.parameters(), lr=config.lr)
        self.cost_q1_optimizer = torch.optim.Adam(self.cost_q1.parameters(), lr=config.lr)
        self.cost_q2_optimizer = torch.optim.Adam(self.cost_q2.parameters(), lr=config.lr)
        self.log_alpha = torch.tensor(-2.0, device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)
        self.log_lambda = torch.tensor(0.0, device=self.device, requires_grad=True)
        self.lambda_optimizer = torch.optim.Adam([self.log_lambda], lr=config.lambda_lr)
        self.entropy_target = 0.05 * (-np.log(1.0 / max(config.max_neighbors, 1)))

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @property
    def lagrange_lambda(self) -> torch.Tensor:
        return self.log_lambda.exp()

    def select_action(self, observation: Observation, greedy: bool = False) -> ActionDecision:
        if not observation.has_safe_action:
            return ActionDecision(
                action_slot=int(observation.fallback_action_slot),
                shield_intervention=1,
                unsafe_action_proposal=1,
                fallback=1,
            )
        tensors = observation_to_tensors(observation, self.config, self.device, batch=False)
        with torch.no_grad():
            logp = self.policy(
                tensors["actor_node_features"].unsqueeze(0),
                tensors["node_mask"].unsqueeze(0),
                tensors["current_index"].unsqueeze(0),
                tensors["neighbor_indices"].unsqueeze(0),
                tensors["action_mask"].unsqueeze(0),
            ).squeeze(0)
        if greedy:
            slot = int(torch.argmax(logp).item())
        else:
            slot = int(torch.distributions.Categorical(logits=logp).sample().item())
        if observation.action_mask[slot]:
            return ActionDecision(
                action_slot=int(observation.fallback_action_slot),
                shield_intervention=1,
                unsafe_action_proposal=1,
                fallback=1,
            )
        return ActionDecision(action_slot=slot)

    def train_step(self, transitions: list[dict[str, Any]]) -> dict[str, float]:
        batch = collate_transitions(transitions, self.config, self.device)
        obs_actor = [
            batch["actor_node_features"],
            batch["node_mask"],
            batch["current_index"],
            batch["neighbor_indices"],
            batch["action_mask"],
        ]
        next_obs_actor = [
            batch["next_actor_node_features"],
            batch["next_node_mask"],
            batch["next_current_index"],
            batch["next_neighbor_indices"],
            batch["next_action_mask"],
        ]
        obs_critic = [
            batch["critic_node_features"],
            batch["node_mask"],
            batch["current_index"],
            batch["neighbor_indices"],
        ]
        next_obs_critic = [
            batch["next_critic_node_features"],
            batch["next_node_mask"],
            batch["next_current_index"],
            batch["next_neighbor_indices"],
        ]
        action = batch["action"].long()
        reward = batch["reward"]
        cost = batch["cost"]
        done = batch["done"]

        with torch.no_grad():
            next_logp = self.policy(*next_obs_actor)
            next_prob = next_logp.exp()
            next_reward_q = torch.min(self.target_reward_q1(*next_obs_critic), self.target_reward_q2(*next_obs_critic))
            next_cost_q = torch.min(self.target_cost_q1(*next_obs_critic), self.target_cost_q2(*next_obs_critic))
            next_reward_v = torch.sum(next_prob * (next_reward_q - self.alpha.detach() * next_logp), dim=1, keepdim=True)
            next_cost_v = torch.sum(next_prob * next_cost_q, dim=1, keepdim=True)
            reward_target = reward + self.config.gamma * (1.0 - done) * next_reward_v
            cost_target = cost + self.config.gamma * (1.0 - done) * next_cost_v

        reward_q1_loss = self._q_loss(self.reward_q1, self.reward_q1_optimizer, obs_critic, action, reward_target)
        reward_q2_loss = self._q_loss(self.reward_q2, self.reward_q2_optimizer, obs_critic, action, reward_target)
        cost_q1_loss = self._q_loss(self.cost_q1, self.cost_q1_optimizer, obs_critic, action, cost_target)
        cost_q2_loss = self._q_loss(self.cost_q2, self.cost_q2_optimizer, obs_critic, action, cost_target)

        logp = self.policy(*obs_actor)
        prob = logp.exp()
        reward_q = torch.min(self.reward_q1(*obs_critic), self.reward_q2(*obs_critic)).detach()
        cost_q = torch.min(self.cost_q1(*obs_critic), self.cost_q2(*obs_critic)).detach()
        policy_loss = torch.sum(
            prob * (self.alpha.detach() * logp - reward_q + self.lagrange_lambda.detach() * cost_q),
            dim=1,
        ).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        entropy = -(prob * logp).sum(dim=1).mean()
        alpha_loss = -(self.log_alpha * (entropy.detach() - self.entropy_target)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        lambda_loss = -(self.log_lambda * (cost.mean().detach() - self.config.cost_limit_per_step))
        self.lambda_optimizer.zero_grad()
        lambda_loss.backward()
        self.lambda_optimizer.step()

        self._soft_update(self.reward_q1, self.target_reward_q1)
        self._soft_update(self.reward_q2, self.target_reward_q2)
        self._soft_update(self.cost_q1, self.target_cost_q1)
        self._soft_update(self.cost_q2, self.target_cost_q2)

        return {
            "reward_q_loss": float(0.5 * (reward_q1_loss + reward_q2_loss)),
            "cost_q_loss": float(0.5 * (cost_q1_loss + cost_q2_loss)),
            "policy_loss": float(policy_loss.item()),
            "alpha": float(self.alpha.detach().item()),
            "lambda": float(self.lagrange_lambda.detach().item()),
            "entropy": float(entropy.item()),
            "batch_reward": float(reward.mean().item()),
            "batch_cost": float(cost.mean().item()),
        }

    def _q_loss(self, net, optimizer, obs_critic, action, target) -> float:
        q = net(*obs_critic).gather(1, action)
        loss = F.mse_loss(q, target.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return float(loss.item())

    def _soft_update(self, source, target) -> None:
        with torch.no_grad():
            for src_param, tgt_param in zip(source.parameters(), target.parameters()):
                tgt_param.data.mul_(1.0 - self.config.tau).add_(src_param.data, alpha=self.config.tau)

    def save(self, checkpoint_path: str | Path) -> None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "reward_q1": self.reward_q1.state_dict(),
                "reward_q2": self.reward_q2.state_dict(),
                "cost_q1": self.cost_q1.state_dict(),
                "cost_q2": self.cost_q2.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                "log_lambda": self.log_lambda.detach().cpu(),
                "config": _serializable_config(self.config),
            },
            path,
        )

    def load(self, checkpoint_path: str | Path) -> None:
        payload = torch.load(checkpoint_path, map_location=self.device)
        self.policy.load_state_dict(payload["policy"])
        self.reward_q1.load_state_dict(payload["reward_q1"])
        self.reward_q2.load_state_dict(payload["reward_q2"])
        self.cost_q1.load_state_dict(payload["cost_q1"])
        self.cost_q2.load_state_dict(payload["cost_q2"])
        self.log_alpha.data.copy_(payload["log_alpha"].to(self.device))
        self.log_lambda.data.copy_(payload["log_lambda"].to(self.device))


def _serializable_config(config: SBGEConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key in ("maps_dir", "result_dir"):
        payload[key] = str(payload[key])
    return payload


def observation_to_tensors(
    observation: Observation,
    config: SBGEConfig,
    device: torch.device,
    batch: bool = False,
) -> dict[str, torch.Tensor]:
    n = min(observation.node_features.shape[0], config.max_nodes)
    actor = np.zeros((config.max_nodes, config.actor_node_dim), dtype=np.float32)
    critic = np.zeros((config.max_nodes, config.critic_node_dim), dtype=np.float32)
    node_mask = np.ones(config.max_nodes, dtype=bool)
    actor[:n] = observation.node_features[:n]
    critic[:n] = observation.critic_node_features[:n]
    node_mask[:n] = False
    neighbor_indices = observation.neighbor_indices.copy()
    neighbor_indices = np.clip(neighbor_indices, 0, config.max_nodes - 1)
    action_mask = observation.action_mask.copy()
    tensors = {
        "actor_node_features": torch.tensor(actor, dtype=torch.float32, device=device),
        "critic_node_features": torch.tensor(critic, dtype=torch.float32, device=device),
        "node_mask": torch.tensor(node_mask, dtype=torch.bool, device=device),
        "current_index": torch.tensor(min(observation.current_index, config.max_nodes - 1), dtype=torch.long, device=device),
        "neighbor_indices": torch.tensor(neighbor_indices, dtype=torch.long, device=device),
        "action_mask": torch.tensor(action_mask, dtype=torch.bool, device=device),
    }
    if batch:
        return {key: value.unsqueeze(0) for key, value in tensors.items()}
    return tensors


def transition_from_observations(
    observation: Observation,
    action_slot: int,
    reward: float,
    cost: float,
    done: bool,
    next_observation: Observation,
) -> dict[str, Any]:
    return {
        "observation": observation,
        "action": int(action_slot),
        "reward": float(reward),
        "cost": float(cost),
        "done": float(done),
        "next_observation": next_observation,
    }


def collate_transitions(transitions: list[dict[str, Any]], config: SBGEConfig, device: torch.device) -> dict[str, torch.Tensor]:
    obs_tensors = [observation_to_tensors(item["observation"], config, device) for item in transitions]
    next_tensors = [observation_to_tensors(item["next_observation"], config, device) for item in transitions]

    def stack(name: str, tensors: list[dict[str, torch.Tensor]]) -> torch.Tensor:
        return torch.stack([item[name] for item in tensors], dim=0)

    batch = {
        "actor_node_features": stack("actor_node_features", obs_tensors),
        "critic_node_features": stack("critic_node_features", obs_tensors),
        "node_mask": stack("node_mask", obs_tensors),
        "current_index": stack("current_index", obs_tensors),
        "neighbor_indices": stack("neighbor_indices", obs_tensors),
        "action_mask": stack("action_mask", obs_tensors),
        "next_actor_node_features": stack("actor_node_features", next_tensors),
        "next_critic_node_features": stack("critic_node_features", next_tensors),
        "next_node_mask": stack("node_mask", next_tensors),
        "next_current_index": stack("current_index", next_tensors),
        "next_neighbor_indices": stack("neighbor_indices", next_tensors),
        "next_action_mask": stack("action_mask", next_tensors),
        "action": torch.tensor([[item["action"]] for item in transitions], dtype=torch.long, device=device),
        "reward": torch.tensor([[item["reward"]] for item in transitions], dtype=torch.float32, device=device),
        "cost": torch.tensor([[item["cost"]] for item in transitions], dtype=torch.float32, device=device),
        "done": torch.tensor([[item["done"]] for item in transitions], dtype=torch.float32, device=device),
    }
    return batch
