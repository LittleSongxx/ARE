from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ac_pbgrl.config import Config
from ac_pbgrl.data.labels import LabelDataset
from ac_pbgrl.evaluation.evaluator import load_actor
from ac_pbgrl.utils import atomic_write_json, seed_everything, sha256_file

from .checkpoint import atomic_torch_save


def default_gru_checkpoint(config: Config) -> Path:
    return (
        Path(config.project.data_root)
        / "temporal"
        / "gru"
        / str(config.project.experiment)
        / f"seed_{int(config.project.seed)}.pt"
    )


def resolve_gru_checkpoint(config: Config) -> Path:
    configured = str(config.filter.get("gru_checkpoint", "auto"))
    return default_gru_checkpoint(config) if configured == "auto" else Path(configured)


@torch.no_grad()
def _extract_sequences(
    config: Config,
    actor_checkpoint: str | Path,
    label_root: str | Path,
    *,
    samples: int,
    batch_size: int,
    device: str | torch.device,
) -> list[tuple[np.ndarray, np.ndarray]]:
    dataset = LabelDataset(label_root, "train")
    actor = load_actor(config, actor_checkpoint, device)
    records: dict[tuple[int, int], list[tuple[int, np.ndarray, float]]] = {}
    limit = min(int(samples), len(dataset))
    for start in range(0, limit, int(batch_size)):
        indices = np.arange(start, min(limit, start + int(batch_size)), dtype=np.int64)
        batch = dataset.batch(
            indices,
            hierarchy=bool(config.method.hierarchy),
            local_budget=int(config.graph_context.local_budget),
            region_budget=int(config.graph_context.region_budget),
            region_size_m=float(config.graph_context.region_size_m),
        )
        metadata = dataset.metadata_batch(indices)
        output = actor(batch.state.to(device))
        region_mean = output.region_mean.float().cpu()
        region_logvar = output.region_log_variance.float().cpu()
        residual = (output.action_mean - output.region_mean).float().cpu()
        pseudo_region_target = batch.future_gain - residual
        candidate_ids = torch.gather(batch.state.stable_ids, 1, batch.state.candidate_indices)
        mask = batch.future_gain_mask & batch.state.candidate_mask & torch.isfinite(pseudo_region_target)
        for row in range(len(indices)):
            episode = int(metadata[row].get("episode", int(indices[row])))
            step = int(metadata[row].get("step", 0))
            for slot in torch.nonzero(mask[row], as_tuple=False).flatten().tolist():
                stable_id = int(candidate_ids[row, slot])
                observation = np.asarray(
                    [float(region_mean[row, slot]), float(region_logvar[row, slot])],
                    dtype=np.float32,
                )
                target = float(pseudo_region_target[row, slot])
                records.setdefault((episode, stable_id), []).append((step, observation, target))
    sequences = []
    for values in records.values():
        values.sort(key=lambda item: item[0])
        if len(values) < 2:
            continue
        sequences.append(
            (
                np.stack([item[1] for item in values]),
                np.asarray([item[2] for item in values], dtype=np.float32),
            )
        )
    if not sequences:
        raise ValueError("no repeated stable-ID sequences were found in the label set")
    return sequences


def _sequence_loss(cell, output, sequences, device) -> torch.Tensor:
    max_length = max(len(item[0]) for item in sequences)
    batch = len(sequences)
    observations = torch.zeros(batch, max_length, 2, device=device)
    targets = torch.zeros(batch, max_length, device=device)
    mask = torch.zeros(batch, max_length, dtype=torch.bool, device=device)
    for index, (inputs, values) in enumerate(sequences):
        length = len(inputs)
        observations[index, :length] = torch.from_numpy(inputs).to(device)
        targets[index, :length] = torch.from_numpy(values).to(device)
        mask[index, :length] = True
    hidden = torch.zeros(batch, cell.hidden_size, device=device)
    losses = []
    for step in range(max_length):
        hidden = cell(observations[:, step], hidden)
        prediction = output(hidden)
        mean = prediction[:, 0]
        logvar = prediction[:, 1].clamp(-8.0, 4.0)
        loss = 0.5 * (torch.exp(-logvar) * (targets[:, step] - mean).square() + logvar)
        losses.append(torch.where(mask[:, step], loss, torch.zeros_like(loss)))
    stacked = torch.stack(losses, dim=1)
    return stacked.sum() / mask.sum().clamp_min(1)


def train_gru_control(
    config: Config,
    actor_checkpoint: str | Path,
    label_root: str | Path,
    output_path: str | Path,
    *,
    samples: int = 20000,
    extraction_batch_size: int = 16,
    sequence_batch_size: int = 64,
    epochs: int = 20,
    learning_rate: float = 1.0e-3,
    device: str | torch.device = "cpu",
) -> dict:
    seed_everything(int(config.project.seed))
    device = torch.device(device)
    sequences = _extract_sequences(
        config,
        actor_checkpoint,
        label_root,
        samples=samples,
        batch_size=extraction_batch_size,
        device=device,
    )
    train_sequences, validation_sequences = [], []
    for index, sequence in enumerate(sequences):
        (validation_sequences if index % 5 == 0 else train_sequences).append(sequence)
    if not train_sequences:
        train_sequences = validation_sequences
    hidden_dim = int(config.filter.get("gru_hidden_dim", 16))
    cell = nn.GRUCell(2, hidden_dim).to(device)
    output = nn.Linear(hidden_dim, 2).to(device)
    optimizer = torch.optim.Adam(list(cell.parameters()) + list(output.parameters()), lr=float(learning_rate))
    rng = np.random.default_rng(int(config.project.seed))
    history = []
    for epoch in range(int(epochs)):
        order = rng.permutation(len(train_sequences))
        losses = []
        cell.train()
        output.train()
        for start in range(0, len(order), int(sequence_batch_size)):
            batch = [train_sequences[int(index)] for index in order[start : start + int(sequence_batch_size)]]
            optimizer.zero_grad(set_to_none=True)
            loss = _sequence_loss(cell, output, batch, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(cell.parameters()) + list(output.parameters()), 10.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
    cell.eval()
    output.eval()
    with torch.no_grad():
        validation_total = 0.0
        validation_observations = 0
        for start in range(0, len(validation_sequences), int(sequence_batch_size)):
            batch = validation_sequences[start : start + int(sequence_batch_size)]
            observations = sum(len(item[0]) for item in batch)
            validation_total += float(_sequence_loss(cell, output, batch, device)) * observations
            validation_observations += observations
        validation_nll = (
            validation_total / validation_observations
            if validation_observations
            else float("nan")
        )
    payload = {
        "version": 1,
        "cell": cell.cpu().state_dict(),
        "output": output.cpu().state_dict(),
        "hidden_dim": hidden_dim,
        "actor_checkpoint": str(Path(actor_checkpoint).resolve()),
        "actor_checkpoint_sha256": sha256_file(Path(actor_checkpoint)),
    }
    output_path = Path(output_path)
    atomic_torch_save(payload, output_path)
    report = {
        "checkpoint": str(output_path),
        "sequences": len(sequences),
        "train_sequences": len(train_sequences),
        "validation_sequences": len(validation_sequences),
        "epochs": int(epochs),
        "train_nll_last": history[-1],
        "validation_nll": validation_nll,
        "history": history,
    }
    atomic_write_json(output_path.with_suffix(".json"), report)
    return report
