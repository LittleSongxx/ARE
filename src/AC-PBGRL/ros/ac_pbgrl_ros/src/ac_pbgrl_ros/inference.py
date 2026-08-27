from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KFRecord:
    mean: float
    variance: float
    step: int


class PotentialKF:
    def __init__(
        self,
        p0=1.0,
        q_stable=0.01,
        q_event=0.25,
        r_min=1e-3,
        r_max=10.0,
        nis_threshold=6.63,
        ttl=32,
        variance_temperature=1.0,
    ):
        self.p0 = float(p0)
        self.q_stable = float(q_stable)
        self.q_event = float(q_event)
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.nis_threshold = float(nis_threshold)
        self.ttl = int(ttl)
        self.variance_temperature = float(variance_temperature)
        self.records = {}

    def update(self, stable_id, observation, observation_variance, step, event=0):
        key = int(stable_id)
        record = self.records.get(key)
        hard_reset = bool(int(event) & (4 | 8 | 16 | 32))
        if record is None or step - record.step > self.ttl or hard_reset:
            record = KFRecord(float(observation), self.p0, step)
        else:
            prior = record.variance + (self.q_event if int(event) else self.q_stable)
            measurement = float(
                np.clip(observation_variance * self.variance_temperature, self.r_min, self.r_max)
            )
            innovation = float(observation) - record.mean
            innovation_variance = prior + measurement
            nis = innovation * innovation / max(innovation_variance, 1e-12)
            if nis > self.nis_threshold:
                prior += self.q_event
                innovation_variance = prior + measurement
            gain = prior / innovation_variance
            record.mean += gain * innovation
            record.variance = max((1.0 - gain) * prior, 1e-8)
            record.step = step
        self.records[key] = record
        return record.mean, record.variance

    def retire(self, stable_id):
        self.records.pop(int(stable_id), None)

    def retire_missing(self, present_ids):
        present = {int(value) for value in present_ids}
        for key in list(self.records):
            if key not in present:
                self.records.pop(key, None)


class ONNXExplorer:
    def __init__(self, model_path, metadata=None, providers=None):
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            str(model_path), providers=providers or ["CPUExecutionProvider"]
        )
        self.input_names = {item.name for item in self.session.get_inputs()}
        metadata = metadata or {}
        filter_config = metadata.get("filter", {})
        calibration = metadata.get("calibration", {})
        self.region_temperature = float(calibration.get("region_temperature", 1.0))
        self.action_temperature = float(calibration.get("action_temperature", self.region_temperature))
        self.filter = PotentialKF(
            p0=filter_config.get("p0", 1.0),
            q_stable=filter_config.get("q_stable", 0.01),
            q_event=filter_config.get("q_event", 0.25),
            r_min=filter_config.get("r_min", 1e-3),
            r_max=filter_config.get("r_max", 10.0),
            nis_threshold=filter_config.get("nis_threshold", 6.63),
            ttl=filter_config.get("ttl_steps", 32),
            variance_temperature=self.region_temperature,
        )
        self.step = 0

    def reset(self):
        self.filter.records.clear()
        self.step = 0

    def retire(self, stable_id):
        self.filter.retire(stable_id)

    def select(self, graph_input):
        self.filter.retire_missing(graph_input.node_ids)
        feeds = {name: value for name, value in graph_input.feeds.items() if name in self.input_names}
        outputs = self.session.run(None, feeds)
        logits, action_mean, action_logvar = outputs[:3]
        region_mean = outputs[3] if len(outputs) > 3 else action_mean
        region_logvar = outputs[4] if len(outputs) > 4 else action_logvar
        mask = feeds["candidate_mask"][0].astype(bool)
        posterior_mean = action_mean[0].copy()
        posterior_variance = np.exp(action_logvar[0]).copy() * self.action_temperature
        for slot in np.flatnonzero(mask):
            region_posterior, region_variance = self.filter.update(
                graph_input.candidate_ids[slot],
                region_mean[0, slot],
                np.exp(region_logvar[0, slot]),
                self.step,
                graph_input.candidate_events[slot],
            )
            residual_variance = max(
                float(
                    np.exp(action_logvar[0, slot]) * self.action_temperature
                    - np.exp(region_logvar[0, slot]) * self.region_temperature
                ),
                1e-6,
            )
            posterior_mean[slot] = action_mean[0, slot] - region_mean[0, slot] + region_posterior
            posterior_variance[slot] = region_variance + residual_variance
        feeds["posterior_mean"] = posterior_mean[None].astype(np.float32)
        feeds["posterior_variance"] = posterior_variance[None].astype(np.float32)
        logits = self.session.run(["logits"], feeds)[0][0]
        logits[~mask] = -np.inf
        selected = int(np.argmax(logits))
        self.step += 1
        return selected, logits
