from __future__ import annotations

import json
import os
import platform
import socket
import sys
import time
from pathlib import Path

from ac_pbgrl.config import Config, config_fingerprint
from ac_pbgrl.utils import atomic_write_json, git_revision

from .gpu import gpu_inventory


def build_run_manifest(config: Config, project_root: Path, *, selected_gpus=None, micro_batch=None) -> dict:
    selected_indices = [
        int(value)
        for value in os.environ.get("ACPBGRL_SELECTED_GPU_INDICES", "").split(",")
        if value
    ]
    selected_uuids = [value for value in os.environ.get("ACPBGRL_SELECTED_GPU_UUIDS", "").split(",") if value]
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_samples = (int(config.train.global_batch_size) + world_size - 1) // world_size
    accumulation = None if not micro_batch else (local_samples + int(micro_batch) - 1) // int(micro_batch)
    actor_mapping = config.train.get("ray_actors_by_world_size", {})
    rollout_actor_count = int(
        actor_mapping.get(str(world_size), actor_mapping.get(world_size, 1))
    )
    explicit_actor_limit = int(config.train.get("ray_actor_limit", 0))
    if explicit_actor_limit > 0:
        rollout_actor_count = min(rollout_actor_count, explicit_actor_limit)
    payload = {
        "created_unix": time.time(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "git_revision": git_revision(project_root),
        "config_sha256": config_fingerprint(config),
        "config": config.plain(),
        "gpu_inventory": gpu_inventory(),
        "selected_gpus": list(selected_gpus or []),
        "selected_gpu_indices": selected_indices,
        "selected_gpu_uuids": selected_uuids,
        "world_size": world_size,
        "global_batch_size": int(config.train.global_batch_size),
        "micro_batch": micro_batch,
        "gradient_accumulation_steps": accumulation,
        "rollout_actor_count": rollout_actor_count,
    }
    try:
        import torch

        payload["torch"] = torch.__version__
        payload["torch_cuda"] = torch.version.cuda
    except ImportError:
        payload["torch"] = None
    return payload


def save_run_manifest(path: Path, payload: dict) -> None:
    sessions = []
    initial_created = payload["created_unix"]
    if path.is_file():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
            sessions = list(previous.get("resource_sessions", []))
            initial_created = previous.get("initial_created_unix", previous.get("created_unix", initial_created))
        except (OSError, ValueError):
            pass
    sessions.append(
        {
            "started_unix": payload["created_unix"],
            "world_size": payload["world_size"],
            "selected_gpu_indices": payload["selected_gpu_indices"],
            "selected_gpu_uuids": payload["selected_gpu_uuids"],
            "micro_batch": payload["micro_batch"],
            "gradient_accumulation_steps": payload["gradient_accumulation_steps"],
            "rollout_actor_count": payload["rollout_actor_count"],
        }
    )
    payload["initial_created_unix"] = initial_created
    payload["resource_sessions"] = sessions
    atomic_write_json(path, payload)
