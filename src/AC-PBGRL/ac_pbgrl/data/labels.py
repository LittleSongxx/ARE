from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch

from ac_pbgrl.learning.future_gain import FutureGainLabels
from ac_pbgrl.models.context import action_preserving_context
from ac_pbgrl.state import ExplorationState, PotentialSupervisionBatch
from ac_pbgrl.utils import atomic_write_json


STATE_FIELDS = (
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
)


class LabelShardWriter:
    def __init__(
        self,
        root: str | Path,
        split: str,
        shard_size: int = 1024,
        provenance: dict | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = str(split)
        self.shard_size = int(shard_size)
        self.output_dir = self.root / self.split
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pending: list[tuple[ExplorationState, FutureGainLabels, dict]] = []
        self.provenance = dict(provenance or {})
        manifest_path = self.root / f"manifest_{self.split}.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if str(manifest.get("split")) != self.split:
                raise ValueError("existing label manifest uses a different split")
            if self.provenance and manifest.get("provenance", {}) != self.provenance:
                raise ValueError("existing label shards were created from different provenance")
            self.shards = list(manifest.get("shards", []))
            self.total = int(manifest.get("samples", 0))
            self.provenance = dict(manifest.get("provenance", self.provenance))
        else:
            self.shards = []
            self.total = 0

    def append(self, state: ExplorationState, labels: FutureGainLabels, metadata: dict) -> None:
        if state.batch_size != 1:
            raise ValueError("label writer accepts one-state batches")
        self.pending.append((state.detach().to("cpu"), labels, dict(metadata)))
        if len(self.pending) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        index = len(self.shards)
        path = self.output_dir / f"shard_{index:05d}.h5"
        temporary = path.with_suffix(".h5.partial")
        with h5py.File(temporary, "w") as handle:
            for field in STATE_FIELDS:
                values = []
                for state, _, _ in self.pending:
                    value = getattr(state, field)
                    if value is None and field == "candidate_events":
                        value = np.zeros_like(state.candidate_indices.numpy(), dtype=np.int16)
                    elif hasattr(value, "numpy"):
                        value = value.numpy()
                    values.append(np.asarray(value)[0])
                handle.create_dataset(f"state/{field}", data=np.stack(values), compression="gzip", shuffle=True)
            handle.create_dataset("labels/value", data=np.stack([item[1].values for item in self.pending]), compression="gzip")
            handle.create_dataset("labels/mask", data=np.stack([item[1].mask for item in self.pending]), compression="gzip")
            handle.create_dataset(
                "labels/rollout_length",
                data=np.stack([item[1].rollout_lengths for item in self.pending]),
                compression="gzip",
            )
            handle.create_dataset("labels/terminal", data=np.stack([item[1].terminal for item in self.pending]), compression="gzip")
            metadata = [json.dumps(item[2], sort_keys=True) for item in self.pending]
            handle.create_dataset("metadata", data=np.asarray(metadata, dtype=h5py.string_dtype("utf-8")))
        temporary.replace(path)
        count = len(self.pending)
        self.shards.append({"path": str(path.relative_to(self.root)), "samples": count})
        self.total += count
        self.pending.clear()
        self._write_manifest()

    def _write_manifest(self) -> None:
        atomic_write_json(
            self.root / f"manifest_{self.split}.json",
            {
                "version": 2,
                "split": self.split,
                "samples": self.total,
                "shards": self.shards,
                "provenance": self.provenance,
            },
        )

    def close(self) -> None:
        self.flush()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None:
            self.close()


class LabelDataset:
    def __init__(self, root: str | Path, split: str) -> None:
        self.root = Path(root)
        manifest = json.loads((self.root / f"manifest_{split}.json").read_text(encoding="utf-8"))
        self.shards = manifest["shards"]
        self.length = int(manifest["samples"])
        self.cumulative = np.cumsum([int(item["samples"]) for item in self.shards])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict:
        if index < 0:
            index += self.length
        if index < 0 or index >= self.length:
            raise IndexError(index)
        shard_index = int(np.searchsorted(self.cumulative, index, side="right"))
        previous = 0 if shard_index == 0 else int(self.cumulative[shard_index - 1])
        row = index - previous
        path = self.root / self.shards[shard_index]["path"]
        with h5py.File(path, "r") as handle:
            return {
                "state": {field: handle[f"state/{field}"][row] for field in STATE_FIELDS},
                "future_gain": handle["labels/value"][row],
                "future_gain_mask": handle["labels/mask"][row],
                "metadata": json.loads(handle["metadata"].asstr()[row]),
            }

    def _read_rows(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        """Read a random batch while opening every HDF5 shard at most once."""

        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if np.any(indices < 0) or np.any(indices >= self.length):
            raise IndexError("label batch index out of range")
        shard_indices = np.searchsorted(self.cumulative, indices, side="right")
        output: dict[str, np.ndarray] = {}
        for shard_index in np.unique(shard_indices):
            positions = np.flatnonzero(shard_indices == shard_index)
            previous = 0 if shard_index == 0 else int(self.cumulative[shard_index - 1])
            rows = indices[positions] - previous
            unique_rows, inverse = np.unique(rows, return_inverse=True)
            path = self.root / self.shards[int(shard_index)]["path"]
            with h5py.File(path, "r") as handle:
                names = [f"state/{field}" for field in STATE_FIELDS] + ["labels/value", "labels/mask"]
                for name in names:
                    # h5py requires increasing, duplicate-free fancy indices.
                    # Read only unique requested rows, then restore replacement
                    # samples with the inverse map.
                    values = handle[name][unique_rows][inverse]
                    if name not in output:
                        output[name] = np.empty((len(indices),) + values.shape[1:], dtype=values.dtype)
                    output[name][positions] = values
        return output

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
        *,
        hierarchy: bool = False,
        local_budget: int = 192,
        region_budget: int = 32,
        region_size_m: float = 16.0,
    ) -> PotentialSupervisionBatch:
        if self.length <= 0:
            raise ValueError("cannot sample an empty label dataset")
        indices = rng.integers(0, self.length, size=int(batch_size), endpoint=False)
        return self.batch(
            indices,
            hierarchy=hierarchy,
            local_budget=local_budget,
            region_budget=region_budget,
            region_size_m=region_size_m,
        )

    def batch(
        self,
        indices: np.ndarray,
        *,
        hierarchy: bool = False,
        local_budget: int = 192,
        region_budget: int = 32,
        region_size_m: float = 16.0,
    ) -> PotentialSupervisionBatch:
        values = self._read_rows(indices)
        state = ExplorationState(
            node_features=torch.from_numpy(values["state/node_features"]).float(),
            node_xy=torch.from_numpy(values["state/node_xy"]).float(),
            node_mask=torch.from_numpy(values["state/node_mask"]).bool(),
            adjacency=torch.from_numpy(values["state/adjacency"]).bool(),
            stable_ids=torch.from_numpy(values["state/stable_ids"]).long(),
            current_index=torch.from_numpy(values["state/current_index"]).long(),
            candidate_indices=torch.from_numpy(values["state/candidate_indices"]).long(),
            candidate_mask=torch.from_numpy(values["state/candidate_mask"]).bool(),
            edge_features=torch.from_numpy(values["state/edge_features"]).float(),
            candidate_events=torch.from_numpy(values["state/candidate_events"]).to(torch.int16),
        ).validate()
        if hierarchy:
            state = action_preserving_context(
                state,
                local_budget=int(local_budget),
                region_budget=int(region_budget),
                region_size_m=float(region_size_m),
            )
        return PotentialSupervisionBatch(
            state=state,
            future_gain=torch.from_numpy(values["labels/value"]).float(),
            future_gain_mask=torch.from_numpy(values["labels/mask"]).bool(),
        )

    def metadata_batch(self, indices: np.ndarray) -> list[dict]:
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        shard_indices = np.searchsorted(self.cumulative, indices, side="right")
        output: list[dict | None] = [None] * len(indices)
        for shard_index in np.unique(shard_indices):
            positions = np.flatnonzero(shard_indices == shard_index)
            previous = 0 if shard_index == 0 else int(self.cumulative[shard_index - 1])
            rows = indices[positions] - previous
            unique_rows, inverse = np.unique(rows, return_inverse=True)
            path = self.root / self.shards[int(shard_index)]["path"]
            with h5py.File(path, "r") as handle:
                metadata = handle["metadata"].asstr()[unique_rows][inverse]
                for position, value in zip(positions, metadata):
                    output[int(position)] = json.loads(value)
        return [item or {} for item in output]
