from __future__ import annotations

import sys
from types import ModuleType

from ac_pbgrl.config import Config
from ac_pbgrl.learning import ray_labels


class _FakeRemoteClass:
    def __init__(self, ray, target) -> None:
        self.ray = ray
        self.target = target

    def remote(self, *args, **kwargs):
        actor = object()
        self.ray.created.append((self.target, args, kwargs, actor))
        return actor


class _FakeRay(ModuleType):
    def __init__(self) -> None:
        super().__init__("ray")
        self.initialized = False
        self.init_kwargs = None
        self.remote_options = []
        self.created = []
        self.killed = []
        self.shutdown_calls = 0

    def is_initialized(self):
        return self.initialized

    def init(self, **kwargs):
        self.initialized = True
        self.init_kwargs = kwargs

    def remote(self, **options):
        self.remote_options.append(options)

        def decorate(target):
            return _FakeRemoteClass(self, target)

        return decorate

    def kill(self, actor, no_restart=False):
        self.killed.append((actor, no_restart))

    def shutdown(self):
        self.initialized = False
        self.shutdown_calls += 1


def test_restore_cpu_affinity_expands_to_all_visible_cpus(monkeypatch):
    calls = []
    monkeypatch.setattr(ray_labels.os, "cpu_count", lambda: 6)
    monkeypatch.setattr(ray_labels.os, "sched_setaffinity", lambda pid, cpus: calls.append((pid, list(cpus))))

    ray_labels._restore_cpu_affinity()

    assert calls == [(0, [0, 1, 2, 3, 4, 5])]


def test_label_pool_limits_ray_resources_and_disables_worker_niceness(tmp_path, monkeypatch):
    fake_ray = _FakeRay()
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    config = Config.convert(
        {
            "project": {"data_root": str(tmp_path)},
            "teacher": {"label_actors": 3},
        }
    )

    pool = ray_labels.RayLabelPool(config, tmp_path / "teacher.pt")

    assert fake_ray.init_kwargs == {
        "address": None,
        "_temp_dir": str(tmp_path / "ray-labels"),
        "num_cpus": 3,
        "num_gpus": 0,
        "_system_config": {"worker_niceness": 0},
        "include_dashboard": False,
        "log_to_driver": False,
        "ignore_reinit_error": True,
    }
    assert fake_ray.remote_options == [{"num_cpus": 1, "max_restarts": 1}]
    assert len(fake_ray.created) == 3

    pool.close()

    assert len(fake_ray.killed) == 3
    assert all(no_restart for _, no_restart in fake_ray.killed)
    assert fake_ray.shutdown_calls == 1
