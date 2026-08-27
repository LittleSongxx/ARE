from __future__ import annotations

import argparse
import json

import torch

from ac_pbgrl.config import load_config
from ac_pbgrl.learning.distributed import BatchSchedule, DistributedContext
from ac_pbgrl.learning.sac import DiscreteSACLearner
from ac_pbgrl.learning.train import apply_cli_overrides, apply_smoke_overrides, configure_cuda_memory
from ac_pbgrl.state import ExplorationState, PotentialSupervisionBatch, TransitionBatch


def _graph(batch: int, nodes: int, candidates: int, features: int) -> ExplorationState:
    node_features = torch.randn(batch, nodes, features)
    node_xy = torch.randn(batch, nodes, 2)
    node_mask = torch.ones(batch, nodes, dtype=torch.bool)
    adjacency = torch.eye(nodes, dtype=torch.bool).expand(batch, -1, -1).clone()
    diagonal = torch.arange(nodes - 1)
    adjacency[:, diagonal, diagonal + 1] = True
    adjacency[:, diagonal + 1, diagonal] = True
    candidate_count = min(candidates, max(1, nodes - 1))
    candidate_indices = torch.zeros(batch, candidates, dtype=torch.long)
    candidate_indices[:, :candidate_count] = torch.arange(1, candidate_count + 1)
    candidate_mask = torch.zeros(batch, candidates, dtype=torch.bool)
    candidate_mask[:, :candidate_count] = True
    return ExplorationState(
        node_features=node_features,
        node_xy=node_xy,
        node_mask=node_mask,
        adjacency=adjacency,
        stable_ids=torch.arange(nodes).expand(batch, -1).clone(),
        current_index=torch.zeros(batch, dtype=torch.long),
        candidate_indices=candidate_indices,
        candidate_mask=candidate_mask,
        edge_features=torch.randn(batch, candidates, 4),
        candidate_events=torch.zeros(batch, candidates, dtype=torch.int16),
    ).validate()


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="full")
    parser.add_argument("--system", default=None)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--micro-batch", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        print(json.dumps({"ok": False, "reason": "CUDA unavailable"}))
        return 2
    config = load_config(args.config, system=args.system, overrides=args.set)
    if args.smoke:
        config = apply_smoke_overrides(config)
        config = apply_cli_overrides(config, args.set)
    config.train.micro_batch_size = int(args.micro_batch)
    context = DistributedContext(0, 0, 1, torch.device("cuda", 0), False)
    torch.cuda.set_device(0)
    configure_cuda_memory(config, context.device)
    actor_nodes = (
        int(config.graph_context.local_budget) + int(config.graph_context.region_budget)
        if bool(config.method.hierarchy)
        else int(config.environment.node_padding)
    )
    critic_nodes = int(config.environment.critic_node_padding)
    batch_size = int(args.micro_batch)
    candidate_count = int(config.environment.candidate_padding)
    try:
        learner = DiscreteSACLearner(config, context)
        actor = _graph(batch_size, actor_nodes, candidate_count, int(config.environment.node_feature_dim))
        critic = _graph(batch_size, critic_nodes, candidate_count, int(config.environment.critic_feature_dim))
        labels = torch.randn(batch_size, candidate_count)
        transition = TransitionBatch(
            state=actor,
            action=torch.zeros(batch_size, dtype=torch.long),
            reward=torch.randn(batch_size),
            done=torch.zeros(batch_size),
            next_state=actor,
            critic_state=critic,
            critic_next_state=critic,
            future_gain=labels if bool(config.method.potential) else None,
            future_gain_mask=actor.candidate_mask if bool(config.method.potential) else None,
        )
        potential = (
            PotentialSupervisionBatch(actor, labels, actor.candidate_mask)
            if bool(config.method.potential)
            else None
        )
        schedule = BatchSchedule(batch_size, 1, 0, batch_size)
        torch.cuda.reset_peak_memory_stats()
        learner.update_chunks(
            [transition],
            schedule,
            potential_chunks=None if potential is None else [potential],
            potential_schedule=None if potential is None else schedule,
        )
        torch.cuda.synchronize()
        print(
            json.dumps(
                {
                    "ok": True,
                    "micro_batch": batch_size,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                },
                sort_keys=True,
            )
        )
        return 0
    except (torch.OutOfMemoryError, RuntimeError) as exc:
        if "out of memory" not in str(exc).lower():
            raise
        print(json.dumps({"ok": False, "micro_batch": batch_size, "reason": "cuda_oom"}))
        return 42


if __name__ == "__main__":
    raise SystemExit(main())
