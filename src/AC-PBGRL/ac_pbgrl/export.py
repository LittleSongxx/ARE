from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from ac_pbgrl.config import Config
from ac_pbgrl.evaluation.evaluator import load_actor
from ac_pbgrl.state import ExplorationState


class ONNXPolicyWrapper(nn.Module):
    def __init__(self, actor: nn.Module, use_potential: bool) -> None:
        super().__init__()
        self.actor = actor
        self.use_potential = use_potential

    def forward(
        self,
        node_features,
        node_mask,
        adjacency,
        current_index,
        candidate_indices,
        candidate_mask,
        edge_features,
        posterior_mean,
        posterior_variance,
    ):
        batch, nodes, _ = node_features.shape
        state = ExplorationState(
            node_features=node_features,
            node_xy=torch.zeros(batch, nodes, 2, dtype=node_features.dtype, device=node_features.device),
            node_mask=node_mask,
            adjacency=adjacency,
            stable_ids=torch.zeros(batch, nodes, dtype=torch.long, device=node_features.device),
            current_index=current_index,
            candidate_indices=candidate_indices,
            candidate_mask=candidate_mask,
            edge_features=edge_features,
            posterior_mean=posterior_mean if self.use_potential else None,
            posterior_variance=posterior_variance if self.use_potential else None,
        )
        output = self.actor(state)
        if self.use_potential:
            return (
                output.logits,
                output.action_mean,
                output.action_log_variance,
                output.region_mean,
                output.region_log_variance,
            )
        zeros = torch.zeros_like(output.logits)
        return output.logits, zeros, zeros, zeros, zeros


def export_onnx(
    config: Config,
    checkpoint: str | Path,
    output_path: str | Path,
    *,
    opset: int = 17,
    device: str = "cpu",
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    actor = load_actor(config, checkpoint, device)
    wrapper = ONNXPolicyWrapper(actor, bool(config.method.potential)).to(device).eval()
    nodes = (
        int(config.graph_context.local_budget) + int(config.graph_context.region_budget)
        if bool(config.method.hierarchy)
        else int(config.environment.node_padding)
    )
    candidates = int(config.environment.candidate_padding)
    node_dim = int(config.environment.node_feature_dim)
    edge_dim = int(config.environment.edge_feature_dim)
    dummy = (
        torch.zeros(1, nodes, node_dim, device=device),
        torch.ones(1, nodes, dtype=torch.bool, device=device),
        torch.eye(nodes, dtype=torch.bool, device=device).unsqueeze(0),
        torch.zeros(1, dtype=torch.long, device=device),
        torch.arange(candidates, dtype=torch.long, device=device).clamp_max(nodes - 1).unsqueeze(0),
        torch.ones(1, candidates, dtype=torch.bool, device=device),
        torch.zeros(1, candidates, edge_dim, device=device),
        torch.zeros(1, candidates, device=device),
        torch.ones(1, candidates, device=device),
    )
    input_names = [
        "node_features",
        "node_mask",
        "adjacency",
        "current_index",
        "candidate_indices",
        "candidate_mask",
        "edge_features",
        "posterior_mean",
        "posterior_variance",
    ]
    dynamic_axes = {name: {0: "batch"} for name in input_names}
    dynamic_axes.update(
        {
            "logits": {0: "batch"},
            "action_mean": {0: "batch"},
            "action_log_variance": {0: "batch"},
            "region_mean": {0: "batch"},
            "region_log_variance": {0: "batch"},
        }
    )
    torch.onnx.export(
        wrapper,
        dummy,
        str(output_path),
        input_names=input_names,
        output_names=[
            "logits",
            "action_mean",
            "action_log_variance",
            "region_mean",
            "region_log_variance",
        ],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
    )
    calibration = {"region_temperature": 1.0, "action_temperature": 1.0}
    if bool(config.method.potential):
        from ac_pbgrl.learning.calibration import load_variance_temperatures, resolve_calibration_path

        calibration_path = resolve_calibration_path(config)
        if calibration_path.is_file():
            payload = json.loads(calibration_path.read_text(encoding="utf-8"))
            region_temperature, action_temperature = load_variance_temperatures(calibration_path)
            calibration = {
                "region_temperature": region_temperature,
                "action_temperature": action_temperature,
                "source": str(calibration_path.resolve()),
                "checkpoint_sha256": payload.get("checkpoint_sha256"),
            }
        elif str(config.method.temporal) == "kf" and bool(config.filter.get("require_calibration", True)):
            raise FileNotFoundError(f"cannot export KF policy without calibration: {calibration_path}")
    metadata = {
        "format": "ac-pbgrl-onnx-v1",
        "nodes": nodes,
        "candidates": candidates,
        "node_feature_dim": node_dim,
        "edge_feature_dim": edge_dim,
        "method": config.project.experiment,
        "inputs": input_names,
        "calibration": calibration,
        "filter": {
            key: config.filter[key]
            for key in ("p0", "q_stable", "q_event", "r_min", "r_max", "nis_threshold", "ttl_steps")
        },
        "topics": {
            "map": "/projected_map",
            "odometry": "/state_estimation",
            "waypoint": "/way_point",
        },
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def compare_onnx(config: Config, checkpoint: str | Path, onnx_path: str | Path, device: str = "cpu") -> float:
    import numpy as np
    import onnxruntime as ort

    actor = load_actor(config, checkpoint, device)
    wrapper = ONNXPolicyWrapper(actor, bool(config.method.potential)).eval()
    nodes = int(config.graph_context.local_budget) + int(config.graph_context.region_budget) if config.method.hierarchy else int(config.environment.node_padding)
    candidates = int(config.environment.candidate_padding)
    args = (
        torch.randn(1, nodes, int(config.environment.node_feature_dim)),
        torch.ones(1, nodes, dtype=torch.bool),
        torch.eye(nodes, dtype=torch.bool).unsqueeze(0),
        torch.zeros(1, dtype=torch.long),
        torch.arange(candidates).clamp_max(nodes - 1).unsqueeze(0),
        torch.ones(1, candidates, dtype=torch.bool),
        torch.randn(1, candidates, int(config.environment.edge_feature_dim)),
        torch.zeros(1, candidates),
        torch.ones(1, candidates),
    )
    with torch.no_grad():
        expected = wrapper(*args)[0].numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feeds = {item.name: tensor.numpy() for item, tensor in zip(session.get_inputs(), args)}
    actual = session.run(["logits"], feeds)[0]
    return float(np.max(np.abs(expected - actual)))
