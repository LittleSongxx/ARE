from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np

from ac_pbgrl.events import GraphEvent


@dataclass
class BeliefRecord:
    mean: float
    variance: float
    last_step: int
    updates: int = 1


class VarianceCalibrator:
    """Positive scalar temperature fitted by held-out Gaussian NLL."""

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = float(temperature)

    def fit(self, predicted_variance: np.ndarray, residual: np.ndarray) -> float:
        variance = np.asarray(predicted_variance, dtype=np.float64).clip(1.0e-8, None)
        squared = np.asarray(residual, dtype=np.float64) ** 2
        self.temperature = float(np.clip(np.mean(squared / variance), 1.0e-3, 1.0e3))
        return self.temperature

    def __call__(self, variance: float | np.ndarray) -> float | np.ndarray:
        return np.asarray(variance) * self.temperature

    def state_dict(self) -> dict:
        return {"temperature": self.temperature}

    def load_state_dict(self, state: dict) -> None:
        self.temperature = float(state["temperature"])


class NoPotentialMemory:
    def update_many(self, stable_ids, means, variances, events=None, step: int = 0):
        del stable_ids, events, step
        return np.asarray(means, dtype=np.float32), np.asarray(variances, dtype=np.float32)

    def reset(self) -> None:
        return None

    def retire(self, stable_id: int) -> None:
        del stable_id

    def retire_missing(self, present_ids) -> None:
        del present_ids

    def state_dict(self) -> dict:
        return {}


class EMAPotentialMemory:
    def __init__(self, alpha: float = 0.3, ttl_steps: int = 32) -> None:
        self.alpha = float(alpha)
        self.ttl_steps = int(ttl_steps)
        self.records: Dict[int, BeliefRecord] = {}

    def update_many(self, stable_ids, means, variances, events=None, step: int = 0):
        events = np.zeros_like(stable_ids) if events is None else events
        output_mean, output_var = [], []
        for stable_id, mean, variance, event in zip(stable_ids, means, variances, events):
            key = int(stable_id)
            event_flag = GraphEvent(int(event))
            record = self.records.get(key)
            if record is None or event_flag.hard_reset or step - record.last_step > self.ttl_steps:
                record = BeliefRecord(float(mean), float(variance), step)
            else:
                record.mean = self.alpha * float(mean) + (1.0 - self.alpha) * record.mean
                record.variance = self.alpha * float(variance) + (1.0 - self.alpha) * record.variance
                record.last_step = step
                record.updates += 1
            self.records[key] = record
            output_mean.append(record.mean)
            output_var.append(record.variance)
        self._expire(step)
        return np.asarray(output_mean, np.float32), np.asarray(output_var, np.float32)

    def _expire(self, step: int) -> None:
        stale = [key for key, value in self.records.items() if step - value.last_step > self.ttl_steps]
        for key in stale:
            self.records.pop(key, None)

    def reset(self) -> None:
        self.records.clear()

    def retire(self, stable_id: int) -> None:
        self.records.pop(int(stable_id), None)

    def retire_missing(self, present_ids) -> None:
        present = {int(value) for value in present_ids}
        for key in list(self.records):
            if key not in present:
                self.records.pop(key, None)

    def state_dict(self) -> dict:
        return {"records": {str(k): vars(v) for k, v in self.records.items()}}


class AdaptivePotentialKF:
    def __init__(
        self,
        p0: float = 1.0,
        q_stable: float = 0.01,
        q_event: float = 0.25,
        r_min: float = 1.0e-3,
        r_max: float = 10.0,
        nis_threshold: float = 6.63,
        ttl_steps: int = 32,
        calibrator: VarianceCalibrator | None = None,
    ) -> None:
        self.p0 = float(p0)
        self.q_stable = float(q_stable)
        self.q_event = float(q_event)
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.nis_threshold = float(nis_threshold)
        self.ttl_steps = int(ttl_steps)
        self.calibrator = calibrator or VarianceCalibrator()
        self.records: Dict[int, BeliefRecord] = {}
        self.nis_history: list[float] = []
        self.reset_count = 0
        self.event_count = 0

    def update(
        self,
        stable_id: int,
        observation: float,
        predicted_variance: float,
        event: int | GraphEvent = GraphEvent.NONE,
        step: int = 0,
    ) -> Tuple[float, float]:
        key = int(stable_id)
        event_flag = GraphEvent(int(event))
        if event_flag != GraphEvent.NONE:
            self.event_count += 1
        record = self.records.get(key)
        stale = record is not None and step - record.last_step > self.ttl_steps
        if record is None or stale or event_flag.hard_reset:
            record = BeliefRecord(float(observation), self.p0, step)
            self.records[key] = record
            self.reset_count += int(event_flag.hard_reset or stale)
            return record.mean, record.variance

        process_noise = self.q_event if event_flag != GraphEvent.NONE else self.q_stable
        prior_variance = record.variance + process_noise
        measurement_variance = float(np.clip(self.calibrator(predicted_variance), self.r_min, self.r_max))
        innovation = float(observation) - record.mean
        innovation_variance = prior_variance + measurement_variance
        nis = innovation * innovation / max(innovation_variance, 1.0e-12)
        self.nis_history.append(nis)
        if nis > self.nis_threshold:
            prior_variance += self.q_event
            innovation_variance = prior_variance + measurement_variance
        gain = prior_variance / innovation_variance
        record.mean = record.mean + gain * innovation
        record.variance = max((1.0 - gain) * prior_variance, 1.0e-12)
        record.last_step = step
        record.updates += 1
        return record.mean, record.variance

    def update_many(self, stable_ids, means, variances, events=None, step: int = 0):
        events = np.zeros_like(stable_ids) if events is None else events
        values = [
            self.update(int(key), float(mean), float(var), int(event), step)
            for key, mean, var, event in zip(stable_ids, means, variances, events)
        ]
        self._expire(step)
        if not values:
            return np.empty((0,), np.float32), np.empty((0,), np.float32)
        mean, variance = zip(*values)
        return np.asarray(mean, np.float32), np.asarray(variance, np.float32)

    def _expire(self, step: int) -> None:
        stale = [key for key, value in self.records.items() if step - value.last_step > self.ttl_steps]
        for key in stale:
            self.records.pop(key, None)

    def reset(self) -> None:
        self.records.clear()
        self.nis_history.clear()
        self.reset_count = 0
        self.event_count = 0

    def retire(self, stable_id: int) -> None:
        if self.records.pop(int(stable_id), None) is not None:
            self.reset_count += 1

    def retire_missing(self, present_ids) -> None:
        present = {int(value) for value in present_ids}
        for key in list(self.records):
            if key not in present:
                self.retire(key)

    def metrics(self) -> dict[str, float]:
        return {
            "kf/active_records": float(len(self.records)),
            "kf/nis_mean": float(np.mean(self.nis_history[-1000:])) if self.nis_history else 0.0,
            "kf/resets": float(self.reset_count),
            "kf/events": float(self.event_count),
        }

    def state_dict(self) -> dict:
        return {
            "records": {str(k): vars(v) for k, v in self.records.items()},
            "calibrator": self.calibrator.state_dict(),
            "reset_count": self.reset_count,
            "event_count": self.event_count,
        }

    def load_state_dict(self, state: dict) -> None:
        self.records = {int(k): BeliefRecord(**v) for k, v in state.get("records", {}).items()}
        if "calibrator" in state:
            self.calibrator.load_state_dict(state["calibrator"])
        self.reset_count = int(state.get("reset_count", 0))
        self.event_count = int(state.get("event_count", 0))


class GRUPotentialMemory:
    """Learnable recurrent control with the same stable-ID state contract as KF."""

    def __init__(
        self,
        hidden_dim: int = 16,
        ttl_steps: int = 32,
        seed: int = 0,
        checkpoint: str | None = None,
    ) -> None:
        import torch

        torch.manual_seed(seed)
        self.torch = torch
        self.cell = torch.nn.GRUCell(2, hidden_dim)
        self.output = torch.nn.Linear(hidden_dim, 2)
        self.ttl_steps = int(ttl_steps)
        self.hidden: Dict[int, tuple[object, int]] = {}
        if checkpoint is not None:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.cell.load_state_dict(payload["cell"], strict=True)
            self.output.load_state_dict(payload["output"], strict=True)
        self.cell.eval()
        self.output.eval()

    def parameters(self):
        return list(self.cell.parameters()) + list(self.output.parameters())

    def update_many(self, stable_ids, means, variances, events=None, step: int = 0):
        torch = self.torch
        events = np.zeros_like(stable_ids) if events is None else events
        output_mean, output_var = [], []
        for key_raw, mean, variance, event_raw in zip(stable_ids, means, variances, events):
            key = int(key_raw)
            event = GraphEvent(int(event_raw))
            prior = self.hidden.get(key)
            if prior is None or event.hard_reset or step - prior[1] > self.ttl_steps:
                hidden = torch.zeros(1, self.cell.hidden_size)
            else:
                hidden = prior[0]
            observation = torch.tensor([[float(mean), float(np.log(max(variance, 1.0e-8)))]])
            with torch.no_grad():
                hidden = self.cell(observation, hidden)
                result = self.output(hidden)
            self.hidden[key] = (hidden.detach(), step)
            output_mean.append(float(result[0, 0].detach()))
            # The offline objective trains channel 1 as log-variance; use the
            # identical parameterization at inference.
            output_var.append(float(result[0, 1].clamp(-8.0, 4.0).exp().detach()))
        return np.asarray(output_mean, np.float32), np.asarray(output_var, np.float32)

    def reset(self) -> None:
        self.hidden.clear()

    def retire(self, stable_id: int) -> None:
        self.hidden.pop(int(stable_id), None)

    def retire_missing(self, present_ids) -> None:
        present = {int(value) for value in present_ids}
        for key in list(self.hidden):
            if key not in present:
                self.hidden.pop(key, None)

    def state_dict(self) -> dict:
        return {"cell": self.cell.state_dict(), "output": self.output.state_dict()}
