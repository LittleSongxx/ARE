from __future__ import annotations

from pathlib import Path

from sbge.config import SBGEConfig
from sbge.env import SafeBudgetedGraphEnv
from sbge.map_utils import list_map_files
from sbge.results import aggregate_metric_rows
from sbge.splits import create_map_split, load_map_files_for_config, save_map_split


def test_split_generation_is_deterministic_and_disjoint(tmp_path: Path):
    split_a = create_map_split(SBGEConfig().maps_dir, seed=3)
    split_b = create_map_split(SBGEConfig().maps_dir, seed=3)

    assert split_a == split_b
    all_names = split_a["train"] + split_a["val"] + split_a["test"]
    assert len(all_names) == len(list_map_files(SBGEConfig().maps_dir))
    assert len(all_names) == len(set(all_names))
    assert split_a["train"]
    assert split_a["test"]


def test_env_uses_split_name_and_map_limit(tmp_path: Path):
    full_split = create_map_split(SBGEConfig().maps_dir, seed=5)
    split = {
        "train": full_split["train"][:3],
        "val": full_split["val"][:2],
        "test": full_split["test"][:2],
    }
    split_path = save_map_split(split, tmp_path / "split.json")
    config = SBGEConfig(seed=7).smoke(seed=7).with_overrides(
        split_file=split_path,
        split_name="test",
        map_limit=1,
    )

    files = load_map_files_for_config(config)
    env = SafeBudgetedGraphEnv(config, seed=7)

    assert len(files) == 1
    assert len(env.map_files) == 1
    assert env.map_files[0].name == Path(split["test"][0]).name


def test_aggregate_metric_rows_outputs_mean_std_count():
    rows = [
        {"method": "a", "explored_rate": 0.2, "episode_cost": 1.0},
        {"method": "a", "explored_rate": 0.4, "episode_cost": 3.0},
        {"method": "b", "explored_rate": 0.8, "episode_cost": 2.0},
    ]

    aggregate = aggregate_metric_rows(rows, group_keys=["method"], metric_keys=["explored_rate", "episode_cost"])

    by_method = {row["method"]: row for row in aggregate}
    assert by_method["a"]["count"] == 2
    assert by_method["a"]["explored_rate_mean"] == 0.30000000000000004
    assert by_method["a"]["explored_rate_count"] == 2
    assert by_method["b"]["episode_cost_mean"] == 2.0
