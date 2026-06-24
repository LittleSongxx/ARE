from __future__ import annotations

from pathlib import Path

from sbge.config import SBGEConfig
from sbge.train_loop import train


def test_smoke_training_completes_two_episodes(tmp_path: Path):
    config = SBGEConfig(seed=7).smoke(seed=7).with_overrides(max_nodes=48, max_neighbors=10)
    summary = train(config, episodes=2, output_dir=tmp_path)

    assert summary["episodes"] == 2
    assert Path(summary["checkpoint"]).is_file()
    assert (tmp_path / "episodes.json").is_file()
