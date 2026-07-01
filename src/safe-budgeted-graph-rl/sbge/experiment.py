from __future__ import annotations

from pathlib import Path
from typing import Any

from .baselines import run_baselines
from .config import SBGEConfig
from .eval import evaluate_checkpoint
from .results import METRIC_KEYS, aggregate_metric_rows, config_to_dict, write_csv, write_json
from .train_loop import train


def run_experiment(
    config: SBGEConfig,
    seeds: list[int],
    train_episodes: int,
    eval_episodes: int,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(output_dir) if output_dir is not None else config.result_dir / config.experiment_name
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "config.json", config_to_dict(config))

    all_rows: list[dict[str, Any]] = []
    seed_summaries: list[dict[str, Any]] = []
    for seed in seeds:
        seed_config = config.with_overrides(seed=int(seed))
        train_dir = root / "train" / f"seed_{seed}"
        eval_dir = root / "eval" / f"seed_{seed}"
        baseline_dir = root / "baselines" / f"seed_{seed}"

        train_summary = train(seed_config.with_overrides(split_name="train"), train_episodes, output_dir=train_dir)
        eval_summary = evaluate_checkpoint(
            seed_config.with_overrides(split_name="test"),
            train_summary["checkpoint"],
            eval_episodes,
            output_dir=eval_dir,
        )
        baseline_summary = run_baselines(
            seed_config.with_overrides(split_name="test"),
            episodes=eval_episodes,
            output_dir=baseline_dir,
        )

        for row in eval_summary["rows"]:
            all_rows.append({"method": "sbge", "seed": int(seed), **row})
        for row in baseline_summary["rows"]:
            policy = str(row["policy"])
            all_rows.append({"method": policy, "seed": int(seed), **{k: v for k, v in row.items() if k != "policy"}})
        seed_summaries.append(
            {
                "seed": int(seed),
                "train_checkpoint": train_summary["checkpoint"],
                "train_mean_explored_rate": train_summary["mean_explored_rate"],
                "eval_mean_explored_rate": eval_summary["mean_explored_rate"],
                "eval_mean_episode_cost": eval_summary["mean_episode_cost"],
            }
        )

    aggregate_rows = aggregate_metric_rows(all_rows, group_keys=["method"], metric_keys=METRIC_KEYS)
    write_csv(root / "all_results.csv", all_rows)
    write_csv(root / "aggregate.csv", aggregate_rows)
    summary = {
        "experiment_name": config.experiment_name,
        "seeds": [int(seed) for seed in seeds],
        "train_episodes": int(train_episodes),
        "eval_episodes": int(eval_episodes),
        "seed_summaries": seed_summaries,
        "aggregate": aggregate_rows,
    }
    write_json(root / "aggregate_summary.json", summary)
    return summary
