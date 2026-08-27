from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ac_pbgrl.state import ExplorationState


@dataclass
class StepResult:
    state: ExplorationState
    critic_state: ExplorationState
    reward: float
    done: bool
    info: dict[str, Any]


class BranchableExplorationEnv(Protocol):
    def reset(self, *args, **kwargs) -> tuple[ExplorationState, ExplorationState]: ...

    def step(self, action_slot: int) -> StepResult: ...

    def clone(self) -> "BranchableExplorationEnv": ...
