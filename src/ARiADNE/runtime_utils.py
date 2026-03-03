from __future__ import annotations

import os

from .parameter import RuntimeConfig


def _auto_detect_worker_threads(num_meta_agent: int) -> int:
    """Compute a reasonable per-worker thread count.

    Strategy: give each concurrently-running worker a fair share of the
    available CPU cores, with a minimum of 2 so that PyTorch operations
    (forward passes, tensor ops) are not bottlenecked on a single thread.

    For a server with 128 cores and 63 agents the result is:
        max_concurrent = min(63, 128) = 63
        threads = max(2, 128 // 63) = 2
    For 20 cores and 16 agents:
        max_concurrent = 16, threads = max(2, 20 // 16) = 2
    For 128 cores and 16 agents:
        max_concurrent = 16, threads = max(2, 128 // 16) = 8
    """
    cpu_count = os.cpu_count() or 1
    max_concurrent = max(1, min(num_meta_agent, cpu_count))
    return max(1, cpu_count // max_concurrent)


def resolve_ray_num_cpus(runtime_config: RuntimeConfig) -> int | None:
    """Total CPUs to register with Ray.  None = let Ray auto-detect."""
    if runtime_config.ray_num_cpus is not None:
        return max(1, int(runtime_config.ray_num_cpus))
    env_override = os.environ.get("ARIADNE_RAY_NUM_CPUS")
    if env_override:
        return max(1, int(env_override))
    return None


def resolve_ray_worker_num_cpus(runtime_config: RuntimeConfig) -> int:
    """CPUs each Ray actor *reserves* (resource accounting)."""
    if runtime_config.ray_worker_num_cpus is not None:
        return max(1, int(runtime_config.ray_worker_num_cpus))
    env_override = os.environ.get("ARIADNE_RAY_WORKER_NUM_CPUS")
    if env_override:
        return max(1, int(env_override))
    # Check worker_num_threads for backward compatibility.
    if runtime_config.worker_num_threads is not None:
        return max(1, int(runtime_config.worker_num_threads))
    env_override = os.environ.get("ARIADNE_WORKER_NUM_THREADS")
    if env_override:
        return max(1, int(env_override))
    # Default: 1 CPU per worker so Ray can schedule many workers concurrently.
    return 1


def resolve_worker_num_threads(runtime_config: RuntimeConfig, worker_num_cpus: int | None = None) -> int:
    """PyTorch / OpenMP threads each worker process should use."""
    if runtime_config.worker_num_threads is not None:
        return max(1, int(runtime_config.worker_num_threads))
    env_override = os.environ.get("ARIADNE_WORKER_NUM_THREADS")
    if env_override:
        return max(1, int(env_override))
    # Auto-detect: give each worker enough threads for PyTorch parallelism.
    return _auto_detect_worker_threads(runtime_config.num_meta_agent)


def configure_worker_process_threads(num_threads: int) -> None:
    thread_value = str(max(1, int(num_threads)))
    for env_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[env_name] = thread_value
