from __future__ import annotations

import os
from pathlib import Path

from .parameter import RuntimeConfig


def _auto_detect_worker_threads(num_meta_agent: int) -> int:
    cpu_count = os.cpu_count() or 1
    max_concurrent = max(1, min(num_meta_agent, cpu_count))
    min_threads = 1 if cpu_count <= 1 else 2
    return max(min_threads, cpu_count // max_concurrent)


def resolve_ray_num_cpus(runtime_config: RuntimeConfig) -> int | None:
    if runtime_config.ray_num_cpus is not None:
        return max(1, int(runtime_config.ray_num_cpus))
    env_override = os.environ.get("HPBG_RL_RAY_NUM_CPUS")
    if env_override:
        return max(1, int(env_override))
    return None


def resolve_ray_worker_num_cpus(runtime_config: RuntimeConfig) -> int:
    if runtime_config.ray_worker_num_cpus is not None:
        return max(1, int(runtime_config.ray_worker_num_cpus))
    env_override = os.environ.get("HPBG_RL_RAY_WORKER_NUM_CPUS")
    if env_override:
        return max(1, int(env_override))
    if runtime_config.worker_num_threads is not None:
        return max(1, int(runtime_config.worker_num_threads))
    env_override = os.environ.get("HPBG_RL_WORKER_NUM_THREADS")
    if env_override:
        return max(1, int(env_override))
    return 1


def resolve_worker_num_threads(runtime_config: RuntimeConfig, worker_num_cpus: int | None = None) -> int:
    del worker_num_cpus
    if runtime_config.worker_num_threads is not None:
        return max(1, int(runtime_config.worker_num_threads))
    env_override = os.environ.get("HPBG_RL_WORKER_NUM_THREADS")
    if env_override:
        return max(1, int(env_override))
    return _auto_detect_worker_threads(runtime_config.num_meta_agent)


def configure_worker_process_threads(num_threads: int) -> None:
    thread_value = str(max(1, int(num_threads)))
    for env_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[env_name] = thread_value


def configure_matplotlib_cache(process_tag: str | None = None) -> Path:
    base_root = os.environ.get("MPLCONFIGDIR")
    if not base_root:
        user = os.environ.get("USER") or "user"
        base_root = f"/tmp/mpl-hpbg-rl-{user}"

    target = Path(base_root).expanduser()
    if process_tag:
        target = target / str(process_tag)
    else:
        target = target / f"pid-{os.getpid()}"

    target.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(target)
    return target
