from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .algorithms import ConstrainedSACAgent
from .config import SBGEConfig
from .env import SafeBudgetedGraphEnv
from .graph import SafeGraphBuilder
from .train_loop import run_episode, set_global_seeds


def evaluate_checkpoint(
    config: SBGEConfig,
    checkpoint: str | Path,
    episodes: int,
    output_dir: str | Path | None = None,
) -> dict[str, object]:
    set_global_seeds(config.seed)
    agent = ConstrainedSACAgent(config)
    agent.load(checkpoint)
    env = SafeBudgetedGraphEnv(config, seed=config.seed + 10000)
    graph_builder = SafeGraphBuilder(config)
    rows = []
    for episode in range(int(episodes)):
        metrics = run_episode(env, graph_builder, agent, episode, replay=None, train=False, greedy=True)
        rows.append({"episode": episode, **metrics.to_dict()})
        print(f"eval episode={episode + 1}/{episodes} explored={metrics.explored_rate:.3f} cost={metrics.episode_cost:.3f}")
    summary = {
        "episodes": int(episodes),
        "checkpoint": str(Path(checkpoint).resolve()),
        "mean_explored_rate": float(np.mean([row["explored_rate"] for row in rows])) if rows else 0.0,
        "mean_episode_cost": float(np.mean([row["episode_cost"] for row in rows])) if rows else 0.0,
        "rows": rows,
    }
    if output_dir is not None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "eval_results.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary
