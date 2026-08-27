from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from ac_pbgrl.config import Config, PROJECT_ROOT


def _restore_cpu_affinity() -> None:
    """Undo narrow OpenMP affinity inherited by Ray worker processes."""

    if hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, range(int(os.cpu_count() or 1)))
        except OSError:
            pass


class RayLabelPool:
    """Parallel, deterministic GT-rollout label generation on CPU simulators."""

    def __init__(self, config: Config, checkpoint: str | Path) -> None:
        try:
            import ray
        except ImportError as exc:
            raise RuntimeError("parallel label generation requires ray==2.10.0") from exc

        self.ray = ray
        count = max(1, int(config.teacher.get("label_actors", 32)))
        self.started_runtime = not ray.is_initialized()
        if self.started_runtime:
            root = Path(config.project.data_root) / "ray-labels"
            root.mkdir(parents=True, exist_ok=True)
            ray.init(
                address=None,
                _temp_dir=str(root),
                num_cpus=count,
                num_gpus=0,
                _system_config={"worker_niceness": 0},
                include_dashboard=False,
                log_to_driver=False,
                ignore_reinit_error=True,
            )
        project_path = str(PROJECT_ROOT)

        @ray.remote(num_cpus=1, max_restarts=1)
        class LabelWorker:
            def __init__(self, payload: dict, checkpoint_path: str) -> None:
                import sys

                if project_path not in sys.path:
                    sys.path.insert(0, project_path)
                from ac_pbgrl.learning.ray_labels import _restore_cpu_affinity

                _restore_cpu_affinity()
                from ac_pbgrl.config import Config
                from ac_pbgrl.learning.future_gain import FutureGainLabeler
                from ac_pbgrl.learning.teacher import FrozenPolicyTeacher
                from ac_pbgrl.learning.train import build_environment

                self.config = Config.convert(payload)
                self.environment = build_environment(self.config)
                teacher = FrozenPolicyTeacher.from_checkpoint(
                    checkpoint_path,
                    node_feature_dim=int(self.config.environment.node_feature_dim),
                    edge_feature_dim=int(self.config.environment.edge_feature_dim),
                    embedding_dim=int(self.config.model.embedding_dim),
                    heads=int(self.config.model.heads),
                    layers=int(self.config.model.encoder_layers),
                    device="cpu",
                )
                self.teacher = teacher
                self.labeler = FutureGainLabeler(
                    teacher,
                    int(self.config.teacher.horizon),
                    float(self.config.teacher.gamma),
                )

            def generate(self, episode: int, map_path: str, maximum: int):
                state, _ = self.environment.reset(episode=int(episode), map_path=map_path)
                records = []
                for step in range(min(int(maximum), int(self.config.environment.max_episode_steps))):
                    labels = self.labeler.label(self.environment, state)
                    records.append(
                        (
                            state.detach().to("cpu"),
                            labels,
                            {"episode": int(episode), "step": step, "map": Path(map_path).name},
                        )
                    )
                    action = self.teacher.select(state)
                    result = self.environment.step(action)
                    state = result.state
                    if result.done:
                        break
                return records

        self.workers = [LabelWorker.remote(config.plain(), str(checkpoint)) for _ in range(count)]

    @property
    def count(self) -> int:
        return len(self.workers)

    def generate(
        self,
        episode_start: int,
        maps: Sequence[str | Path],
        *,
        tasks: int,
        states_per_task: int,
    ) -> list[list[tuple]]:
        references = []
        for offset in range(int(tasks)):
            episode = int(episode_start) + offset
            map_path = str(maps[episode % len(maps)])
            references.append(
                self.workers[offset % len(self.workers)].generate.remote(
                    episode,
                    map_path,
                    int(states_per_task),
                )
            )
        return list(self.ray.get(references))

    def close(self) -> None:
        for worker in self.workers:
            try:
                self.ray.kill(worker, no_restart=True)
            except Exception:
                pass
        if self.started_runtime and self.ray.is_initialized():
            self.ray.shutdown()
