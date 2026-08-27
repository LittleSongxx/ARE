from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch


def rng_state() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda"] = torch.cuda.get_rng_state_all()
    return payload


def restore_rng_state(payload: dict[str, Any]) -> None:
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    # ``torch.load(..., map_location=cuda)`` also relocates RNG tensors, while
    # the default CPU generator only accepts a CPU ByteTensor.
    torch.set_rng_state(payload["torch"].cpu())
    if "cuda" in payload and torch.cuda.is_available():
        # The saved run may have used a different number of visible GPUs. Restore
        # the overlapping generators; newly added ranks retain their deterministic
        # rank seed established before checkpoint loading.
        count = torch.cuda.device_count()
        states = [state.cpu() for state in payload["cuda"][:count]]
        if states:
            torch.cuda.set_rng_state_all(states)


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_checkpoint(
    path: Path,
    learner,
    *,
    config: dict,
    replay_manifest: dict,
    temporal_state: dict | None = None,
    run_manifest: dict | None = None,
) -> None:
    payload = {
        "version": 1,
        "learner": learner.state_dict(),
        "rng": rng_state(),
        "config": config,
        "replay": replay_manifest,
        "temporal": temporal_state or {},
        "run_manifest": run_manifest or {},
    }
    atomic_torch_save(payload, path)
    latest = path.parent / "latest.pt"
    atomic_torch_save(payload, latest)


def load_checkpoint(path: Path, learner, temporal=None, map_location="cpu") -> dict:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    learner.load_state_dict(payload["learner"])
    restore_rng_state(payload["rng"])
    if temporal is not None and payload.get("temporal"):
        temporal.load_state_dict(payload["temporal"])
    return payload
