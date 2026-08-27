from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import torch

from ac_pbgrl.config import Config, PROJECT_ROOT
from ac_pbgrl.models.policy import ACPolicyNetwork


def actor_count_for_world_size(config: Config, world_size: int) -> int:
    configured = config.train.ray_actors_by_world_size
    count = int(configured.get(str(world_size), configured.get(world_size, 1)))
    explicit = int(config.train.get("ray_actor_limit", 0))
    if explicit > 0:
        count = min(count, explicit)
    return max(1, count)


def _cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in module.state_dict().items()}


class RayRolloutPool:
    """CPU simulator pool whose policy weights are synchronized per rollout wave."""

    def __init__(self, config: Config, world_size: int) -> None:
        try:
            import ray
        except ImportError as exc:  # pragma: no cover - exercised by server doctor
            raise RuntimeError("formal parallel rollout requires ray==2.10.0") from exc

        self.ray = ray
        self.config = config.clone()
        self.count = actor_count_for_world_size(config, world_size)
        self.started_runtime = not ray.is_initialized()
        if self.started_runtime:
            ray_root = Path(config.project.data_root) / "ray"
            ray_root.mkdir(parents=True, exist_ok=True)
            address = str(config.train.get("ray_address", "local"))
            ray.init(
                address=None if address in {"", "local", "none"} else address,
                _temp_dir=str(ray_root),
                num_cpus=self.count,
                num_gpus=0,
                _system_config={"worker_niceness": 0},
                include_dashboard=False,
                log_to_driver=False,
                ignore_reinit_error=True,
            )

        project_path = str(PROJECT_ROOT)

        @ray.remote(num_cpus=float(config.train.get("ray_cpus_per_actor", 1.0)), max_restarts=1)
        class RolloutWorker:
            def __init__(self, config_payload: dict, worker_index: int) -> None:
                import os
                import sys

                if project_path not in sys.path:
                    sys.path.insert(0, project_path)
                # OpenMP may pin the thread that launches Ray to one physical
                # core.  Ray children inherit that two-hyperthread mask even
                # though the node advertises many CPUs.  A worker can safely
                # expand its own mask back to the host/cgroup CPU set; Ray's
                # num_cpus reservation still controls actor concurrency.
                if hasattr(os, "sched_setaffinity"):
                    try:
                        os.sched_setaffinity(0, range(int(os.cpu_count() or 1)))
                    except OSError:
                        pass
                from ac_pbgrl.config import Config
                from ac_pbgrl.learning.rollout import EpisodeCollector
                from ac_pbgrl.learning.train import build_environment, build_q_teacher
                from ac_pbgrl.models.policy import ACPolicyNetwork

                self.config = Config.convert(config_payload)
                self.worker_index = int(worker_index)
                model = self.config.model
                environment = self.config.environment
                method = self.config.method
                self.actor = ACPolicyNetwork(
                    int(environment.node_feature_dim),
                    int(environment.edge_feature_dim),
                    int(model.embedding_dim),
                    int(model.heads),
                    int(model.encoder_layers),
                    float(model.dropout),
                    use_potential=bool(method.potential),
                    use_diffusion=bool(method.graph_diffusion),
                    fuse_uncertainty=bool(method.fuse_uncertainty),
                    logvar_min=float(model.logvar_min),
                    logvar_max=float(model.logvar_max),
                ).cpu().eval()
                self.environment = build_environment(self.config)
                q_teacher = build_q_teacher(self.config, False, torch.device("cpu"))
                self.collector = EpisodeCollector(
                    self.environment,
                    self.actor,
                    self.config,
                    device="cpu",
                    labeler=None,
                    q_teacher=q_teacher,
                )

            def collect(self, actor_state: dict, episode: int, map_path: str | None, max_steps: int):
                self.actor.load_state_dict(actor_state, strict=True)
                self.actor.eval()
                return self.collector.collect(
                    episode=int(episode), map_path=map_path, max_steps=int(max_steps)
                )

        self.workers = [RolloutWorker.remote(config.plain(), index) for index in range(self.count)]

    def collect(
        self,
        actor: torch.nn.Module,
        *,
        episode_start: int,
        count: int,
        maps: Sequence[str | Path] | None,
        step_budget: int | None = None,
    ) -> list[tuple[list, dict[str, float]]]:
        weights = self.ray.put(_cpu_state_dict(actor))
        references = []
        remaining = None if step_budget is None else max(1, int(step_budget))
        for offset in range(int(count)):
            episode = int(episode_start) + offset
            map_path = None if maps is None else str(maps[episode % len(maps)])
            worker = self.workers[offset % len(self.workers)]
            slots_left = int(count) - offset
            maximum = int(self.config.environment.max_episode_steps)
            if remaining is not None:
                maximum = max(1, (remaining + slots_left - 1) // slots_left)
                remaining = max(0, remaining - maximum)
            references.append(worker.collect.remote(weights, episode, map_path, maximum))
        return list(self.ray.get(references))

    def close(self) -> None:
        for worker in self.workers:
            try:
                self.ray.kill(worker, no_restart=True)
            except Exception:
                pass
        self.workers.clear()
        if self.started_runtime and self.ray.is_initialized():
            self.ray.shutdown()
