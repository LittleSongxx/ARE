from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import torch

from ac_pbgrl.state import ExplorationState, TransitionBatch
from ac_pbgrl.utils import atomic_write_json


GRAPH_FIELDS = (
    "node_features",
    "node_xy",
    "node_mask",
    "adjacency",
    "stable_ids",
    "current_index",
    "candidate_indices",
    "candidate_mask",
    "edge_features",
    "candidate_events",
    "posterior_mean",
    "posterior_variance",
)


def _torch_to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _graph_arrays(graph: ExplorationState, prefix: str) -> dict[str, np.ndarray]:
    values = {}
    for name in GRAPH_FIELDS:
        value = getattr(graph, name)
        if value is None:
            if name == "candidate_events":
                value = torch.zeros_like(graph.candidate_indices, dtype=torch.int16)
            elif name in {"posterior_mean", "posterior_variance"}:
                value = torch.full_like(graph.candidate_indices, float("nan"), dtype=torch.float32)
            else:
                continue
        values[f"{prefix}.{name}"] = _torch_to_numpy(value)
    return values


def transition_arrays(batch: TransitionBatch) -> dict[str, np.ndarray]:
    arrays = {}
    arrays.update(_graph_arrays(batch.state, "state"))
    arrays.update(_graph_arrays(batch.next_state, "next_state"))
    critic_state = batch.critic_state if batch.critic_state is not None else batch.state
    critic_next = batch.critic_next_state if batch.critic_next_state is not None else batch.next_state
    arrays.update(_graph_arrays(critic_state, "critic_state"))
    arrays.update(_graph_arrays(critic_next, "critic_next_state"))
    arrays["action"] = _torch_to_numpy(batch.action)
    arrays["reward"] = _torch_to_numpy(batch.reward)
    arrays["done"] = _torch_to_numpy(batch.done)
    candidate_shape = batch.state.candidate_indices.shape
    arrays["future_gain"] = (
        np.full(candidate_shape, np.nan, dtype=np.float32)
        if batch.future_gain is None
        else _torch_to_numpy(batch.future_gain).astype(np.float32, copy=False)
    )
    arrays["future_gain_mask"] = (
        np.zeros(candidate_shape, dtype=np.bool_)
        if batch.future_gain_mask is None
        else _torch_to_numpy(batch.future_gain_mask).astype(np.bool_, copy=False)
    )
    arrays["teacher_q"] = (
        np.full(candidate_shape, np.nan, dtype=np.float32)
        if batch.teacher_q is None
        else _torch_to_numpy(batch.teacher_q).astype(np.float32, copy=False)
    )
    return arrays


class PersistentReplayBuffer:
    """Crash-resilient, per-field memory-mapped ring buffer.

    The metadata cursor is committed atomically after array flushes. At worst an
    interrupted write loses the current batch; previously committed samples remain
    available across supervisor restarts and DDP world-size changes.
    """

    def __init__(self, root: Path | str, capacity: int, seed: int = 0) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.capacity = int(capacity)
        self.meta_path = self.root / "metadata.json"
        self.lock = threading.RLock()
        self.rng = np.random.default_rng(seed)
        self.arrays: dict[str, np.memmap] = {}
        self.schema: dict[str, dict] = {}
        self.cursor = 0
        self.size = 0
        self.total_added = 0
        if self.meta_path.is_file():
            self._open_existing()

    def _path_for(self, name: str) -> Path:
        return self.root / f"{name.replace('.', '__')}.npy"

    def _open_existing(self) -> None:
        metadata = json.loads(self.meta_path.read_text(encoding="utf-8"))
        if int(metadata["capacity"]) != self.capacity:
            raise ValueError("replay capacity differs from existing on-disk buffer")
        self.schema = metadata["schema"]
        self.cursor = int(metadata["cursor"])
        self.size = int(metadata["size"])
        self.total_added = int(metadata.get("total_added", self.size))
        for name in self.schema:
            self.arrays[name] = np.load(self._path_for(name), mmap_mode="r+")

    def _initialize(self, sample_arrays: Mapping[str, np.ndarray]) -> None:
        for name, values in sample_arrays.items():
            if values.ndim < 1:
                raise ValueError(f"replay field {name} must include a batch dimension")
            shape = tuple(int(v) for v in values.shape[1:])
            dtype = np.dtype(values.dtype)
            self.schema[name] = {"shape": list(shape), "dtype": dtype.str}
            self.arrays[name] = np.lib.format.open_memmap(
                self._path_for(name), mode="w+", dtype=dtype, shape=(self.capacity,) + shape
            )
        self._commit_metadata()

    def _commit_metadata(self) -> None:
        atomic_write_json(
            self.meta_path,
            {
                "capacity": self.capacity,
                "cursor": self.cursor,
                "size": self.size,
                "total_added": self.total_added,
                "schema": self.schema,
            },
        )

    def add(self, batch: TransitionBatch) -> None:
        self.add_many((batch,))

    def add_many(self, batches: Iterable[TransitionBatch]) -> None:
        encoded = [transition_arrays(batch) for batch in batches]
        if not encoded:
            return
        fields = set(encoded[0])
        if any(set(item) != fields for item in encoded):
            raise ValueError("transition schemas differ within replay batch")
        values = {
            name: np.concatenate([item[name] for item in encoded], axis=0)
            for name in encoded[0]
        }
        batch_size = next(iter(values.values())).shape[0]
        if any(item.shape[0] != batch_size for item in values.values()):
            raise ValueError("all replay fields must share a batch dimension")
        with self.lock:
            if not self.arrays:
                self._initialize(values)
            if set(values) != set(self.arrays):
                raise ValueError("transition schema differs from initialized replay")
            for row in range(batch_size):
                target = self.cursor
                for name, array in self.arrays.items():
                    expected = tuple(self.schema[name]["shape"])
                    if values[name].shape[1:] != expected:
                        raise ValueError(f"shape mismatch for {name}: {values[name].shape[1:]} != {expected}")
                    array[target] = values[name][row]
                self.cursor = (self.cursor + 1) % self.capacity
                self.size = min(self.capacity, self.size + 1)
                self.total_added += 1
            for array in self.arrays.values():
                array.flush()
            self._commit_metadata()

    def __len__(self) -> int:
        return self.size

    def refresh(self) -> None:
        """Refresh the committed cursor after another rank/process appended data."""
        if not self.meta_path.is_file():
            return
        metadata = json.loads(self.meta_path.read_text(encoding="utf-8"))
        if not self.arrays and metadata.get("schema"):
            self._open_existing()
            return
        self.cursor = int(metadata["cursor"])
        self.size = int(metadata["size"])
        self.total_added = int(metadata.get("total_added", self.size))

    def sample(self, batch_size: int, device: str | torch.device = "cpu") -> TransitionBatch:
        if self.size == 0:
            raise ValueError("cannot sample an empty replay buffer")
        indices = self.rng.integers(0, self.size, size=int(batch_size), endpoint=False)
        tensors = {
            name: torch.from_numpy(np.asarray(array[indices]).copy()).to(device)
            for name, array in self.arrays.items()
        }

        def graph(prefix: str) -> ExplorationState:
            candidate_indices = tensors[f"{prefix}.candidate_indices"].long()
            posterior_mean = tensors[f"{prefix}.posterior_mean"].float()
            posterior_variance = tensors[f"{prefix}.posterior_variance"].float()
            if not torch.isfinite(posterior_mean).any():
                posterior_mean = None
                posterior_variance = None
            return ExplorationState(
                node_features=tensors[f"{prefix}.node_features"].float(),
                node_xy=tensors[f"{prefix}.node_xy"].float(),
                node_mask=tensors[f"{prefix}.node_mask"].bool(),
                adjacency=tensors[f"{prefix}.adjacency"].bool(),
                stable_ids=tensors[f"{prefix}.stable_ids"].long(),
                current_index=tensors[f"{prefix}.current_index"].long(),
                candidate_indices=candidate_indices,
                candidate_mask=tensors[f"{prefix}.candidate_mask"].bool(),
                edge_features=tensors[f"{prefix}.edge_features"].float(),
                candidate_events=tensors[f"{prefix}.candidate_events"].to(torch.int16),
                posterior_mean=posterior_mean,
                posterior_variance=posterior_variance,
            ).validate()

        return TransitionBatch(
            state=graph("state"),
            action=tensors["action"].long(),
            reward=tensors["reward"].float(),
            done=tensors["done"].float(),
            next_state=graph("next_state"),
            critic_state=graph("critic_state"),
            critic_next_state=graph("critic_next_state"),
            future_gain=tensors["future_gain"].float(),
            future_gain_mask=tensors["future_gain_mask"].bool(),
            teacher_q=tensors["teacher_q"].float(),
        )

    def manifest(self) -> dict:
        return {
            "root": str(self.root),
            "capacity": self.capacity,
            "size": self.size,
            "cursor": self.cursor,
            "total_added": self.total_added,
            "schema": self.schema,
        }
