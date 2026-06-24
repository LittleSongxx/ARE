from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PACKAGE_ROOT.parent
WORKSPACE_ROOT = SRC_ROOT.parent
DEFAULT_MAPS_DIR = SRC_ROOT / "large-scale-DRL-exploration" / "maps"
DEFAULT_RESULT_DIR = PACKAGE_ROOT / "result"


@dataclass(frozen=True)
class SBGEConfig:
    maps_dir: Path = DEFAULT_MAPS_DIR
    result_dir: Path = DEFAULT_RESULT_DIR
    cell_size_m: float = 0.4
    sensor_range_cells: int = 40
    sensor_rays: int = 120
    sensor_false_free_rate: float = 0.0
    sensor_false_occupied_rate: float = 0.0
    node_stride_cells: int = 10
    max_nodes: int = 256
    max_neighbors: int = 25
    utility_radius_cells: int = 32
    local_patch_radius_cells: int = 5
    max_episode_steps: int = 64
    target_explored_rate: float = 0.95
    budget_ratio_min: float = 6.0
    budget_ratio_max: float = 10.0
    risk_threshold: float = 0.75
    near_miss_clearance_cells: float = 3.0
    clearance_risk_scale_cells: float = 4.0
    negative_obstacle_count: int = 5
    negative_obstacle_radius_cells: int = 8
    dynamic_obstacle_count: int = 3
    dynamic_obstacle_radius_cells: float = 4.0
    dynamic_risk_horizon: int = 3
    reward_completion_bonus: float = 20.0
    reward_return_bonus: float = 5.0
    cost_limit_per_step: float = 0.05
    gamma: float = 1.0
    lr: float = 3e-4
    batch_size: int = 32
    replay_size: int = 5000
    warmup_steps: int = 32
    train_updates_per_step: int = 1
    hidden_dim: int = 128
    tau: float = 0.02
    alpha_lr: float = 1e-4
    lambda_lr: float = 1e-4
    seed: int = 0
    device: str = "cpu"

    @property
    def actor_node_dim(self) -> int:
        return 11

    @property
    def critic_node_dim(self) -> int:
        return 14

    def with_overrides(self, **kwargs) -> "SBGEConfig":
        normalized = {}
        for key, value in kwargs.items():
            if key in {"maps_dir", "result_dir"} and value is not None:
                normalized[key] = Path(value).expanduser().resolve()
            else:
                normalized[key] = value
        return replace(self, **normalized)

    def smoke(self, seed: int | None = None) -> "SBGEConfig":
        return self.with_overrides(
            seed=self.seed if seed is None else seed,
            max_episode_steps=10,
            max_nodes=96,
            batch_size=8,
            replay_size=256,
            warmup_steps=4,
            train_updates_per_step=1,
            dynamic_obstacle_count=1,
            negative_obstacle_count=2,
        )
