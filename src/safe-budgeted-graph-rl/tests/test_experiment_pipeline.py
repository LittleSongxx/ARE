from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from sbge.baselines import run_baselines
from sbge.config import SBGEConfig
from sbge.eval import evaluate_checkpoint
from sbge.splits import create_map_split, save_map_split
from sbge.train_loop import train


def _split_file(tmp_path: Path) -> Path:
    split = create_map_split(SBGEConfig().maps_dir, seed=11)
    small_split = {
        "train": split["train"][:3],
        "val": split["val"][:1],
        "test": split["test"][:3],
    }
    return save_map_split(small_split, tmp_path / "split.json")


def test_eval_writes_json_and_csv(tmp_path: Path):
    split_path = _split_file(tmp_path)
    train_config = SBGEConfig(seed=7).smoke(seed=7).with_overrides(
        split_file=split_path,
        split_name="train",
        max_nodes=48,
        max_neighbors=10,
    )
    train_summary = train(train_config, episodes=1, output_dir=tmp_path / "train")
    eval_config = train_config.with_overrides(split_name="test")

    evaluate_checkpoint(eval_config, train_summary["checkpoint"], episodes=1, output_dir=tmp_path / "eval")

    assert (tmp_path / "eval" / "eval_results.json").is_file()
    assert (tmp_path / "eval" / "eval_summary.json").is_file()
    assert (tmp_path / "eval" / "eval_results.csv").is_file()


def test_baseline_four_policies_write_outputs(tmp_path: Path):
    split_path = _split_file(tmp_path)
    config = SBGEConfig(seed=7).smoke(seed=7).with_overrides(
        split_file=split_path,
        split_name="test",
        max_nodes=48,
        max_neighbors=10,
    )

    summary = run_baselines(config, episodes=1, output_dir=tmp_path / "baselines")

    assert summary["policies"] == ["nearest", "utility", "utility_unshielded", "expert"]
    assert len(summary["rows"]) == 4
    assert (tmp_path / "baselines" / "baseline_results.csv").is_file()
    assert (tmp_path / "baselines" / "baseline_summary.json").is_file()


def test_run_experiment_smoke_cli_generates_aggregate(tmp_path: Path):
    split_path = _split_file(tmp_path)
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_experiment.py"
    output_dir = tmp_path / "experiment"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--smoke",
            "--seeds",
            "7",
            "8",
            "--split-file",
            str(split_path),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "aggregate_methods=" in result.stdout
    assert (output_dir / "aggregate.csv").is_file()
    assert (output_dir / "aggregate_summary.json").is_file()
