from __future__ import annotations

import csv
import fcntl
import io
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence


@dataclass(frozen=True)
class GPUInfo:
    index: int
    uuid: str
    name: str
    memory_total_mib: int
    memory_used_mib: int
    memory_free_mib: int
    utilization_pct: int
    temperature_c: int
    process_count: int = 0

    @property
    def free_gib(self) -> float:
        return self.memory_free_mib / 1024.0

    def is_idle(self, used_limit_mib: int = 512, utilization_limit_pct: int = 10) -> bool:
        return (
            self.memory_used_mib <= used_limit_mib
            and self.utilization_pct <= utilization_limit_pct
            and self.process_count == 0
        )


GPU_QUERY = (
    "index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu"
)


def parse_gpu_csv(text: str, process_counts: dict[str, int] | None = None) -> list[GPUInfo]:
    process_counts = process_counts or {}
    result = []
    for row in csv.reader(io.StringIO(text), skipinitialspace=True):
        if not row or len(row) < 8:
            continue
        values = [item.strip() for item in row]
        result.append(
            GPUInfo(
                index=int(values[0]),
                uuid=values[1],
                name=values[2],
                memory_total_mib=int(values[3]),
                memory_used_mib=int(values[4]),
                memory_free_mib=int(values[5]),
                utilization_pct=int(values[6]),
                temperature_c=int(values[7]),
                process_count=int(process_counts.get(values[1], 0)),
            )
        )
    return result


def parse_gpu_process_csv(text: str) -> dict[str, set[int]]:
    processes: dict[str, set[int]] = {}
    for row in csv.reader(io.StringIO(text), skipinitialspace=True):
        if len(row) < 2:
            continue
        uuid, raw_pid = row[0].strip(), row[1].strip()
        try:
            pid = int(raw_pid)
        except ValueError:
            continue
        processes.setdefault(uuid, set()).add(pid)
    return processes


def query_gpu_processes(
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> dict[str, set[int]]:
    try:
        result = run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return {}
    return parse_gpu_process_csv(result.stdout)


def filter_gpu_allowlist(
    gpus: Sequence[GPUInfo],
    value: str | None = None,
) -> list[GPUInfo]:
    """Restrict automatic scheduling to explicit physical GPU indices."""

    raw = os.environ.get("ACPBGRL_GPU_ALLOWLIST", "") if value is None else value
    raw = str(raw).strip()
    if not raw:
        return list(gpus)
    try:
        allowed = {int(item.strip()) for item in raw.split(",") if item.strip()}
    except ValueError as exc:
        raise ValueError("ACPBGRL_GPU_ALLOWLIST must be comma-separated GPU indices") from exc
    if not allowed:
        raise ValueError("ACPBGRL_GPU_ALLOWLIST must contain at least one GPU index")
    return [gpu for gpu in gpus if gpu.index in allowed]


def query_gpus(run: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> list[GPUInfo]:
    try:
        process = run(
            [
                "nvidia-smi",
                f"--query-gpu={GPU_QUERY}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        app_processes = query_gpu_processes(run)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    counts = {uuid: len(pids) for uuid, pids in app_processes.items()}
    return filter_gpu_allowlist(parse_gpu_csv(process.stdout, counts))


def select_gpus(
    gpus: Sequence[GPUInfo],
    *,
    policy: str = "prefer-idle",
    min_gpus: int = 1,
    max_gpus: int = 4,
    min_free_memory_gib: float = 18.0,
    max_utilization_pct: int = 65,
    max_temperature_c: int = 80,
    idle_used_memory_mib: int = 512,
    idle_utilization_pct: int = 10,
    excluded_uuids: Iterable[str] = (),
) -> list[GPUInfo]:
    if policy not in {"idle-only", "prefer-idle", "shared-ok"}:
        raise ValueError(f"unknown GPU policy: {policy}")
    excluded = set(excluded_uuids)
    eligible = [
        gpu
        for gpu in gpus
        if gpu.uuid not in excluded
        and gpu.free_gib >= min_free_memory_gib
        and gpu.utilization_pct <= max_utilization_pct
        and gpu.temperature_c <= max_temperature_c
    ]
    idle = [gpu for gpu in eligible if gpu.is_idle(idle_used_memory_mib, idle_utilization_pct)]
    shared = [gpu for gpu in eligible if gpu not in idle]
    score = lambda gpu: (
        int(gpu.is_idle(idle_used_memory_mib, idle_utilization_pct)),
        gpu.memory_free_mib,
        -gpu.utilization_pct,
        -gpu.temperature_c,
        -gpu.index,
    )
    idle.sort(key=score, reverse=True)
    shared.sort(key=score, reverse=True)
    if policy == "idle-only":
        pool = idle
    elif policy == "prefer-idle":
        # Do not occupy a shared card merely to reach max_gpus while any useful
        # idle subset already exists.
        pool = idle if idle else shared
    else:
        pool = sorted(eligible, key=score, reverse=True)
    selected = pool[: int(max_gpus)]
    return selected if len(selected) >= int(min_gpus) else []


def recommended_micro_batch(gpus: Sequence[GPUInfo], reserve_gib: float = 6.0) -> int:
    if not gpus:
        return 0
    usable = min(gpu.free_gib - reserve_gib for gpu in gpus)
    # A40-class 48 GB cards can usually train a 64-sample local chunk while
    # preserving the configured reserve.  The disposable real-model probe is
    # still authoritative and falls back before a training process is launched.
    if usable >= 36:
        return 64
    if usable >= 24:
        return 32
    if usable >= 14:
        return 16
    if usable >= 8:
        return 8
    return 4 if usable >= 4 else 0


def calibrate_micro_batch(
    gpus: Sequence[GPUInfo],
    *,
    experiment: str,
    system: str | None,
    overrides: Sequence[str],
    candidates: Sequence[int],
    start: int,
    reserve_gib: float,
    smoke: bool = False,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> tuple[int, list[dict]]:
    """Run a disposable forward/backward probe on the most constrained card."""

    if not gpus or start <= 0:
        return 0, []
    probe_gpu = min(gpus, key=lambda item: item.memory_free_mib)
    ordered = sorted(
        {int(start)} | {int(value) for value in candidates if int(value) <= start},
        reverse=True,
    )
    attempts = []
    for micro_batch in ordered:
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(probe_gpu.index)
        environment["ACPBGRL_MEMORY_RESERVE_GIB"] = str(float(reserve_gib))
        environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
        environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        command = [
            sys.executable,
            "-m",
            "ac_pbgrl.runtime.memory_probe",
            "--config",
            experiment,
            "--micro-batch",
            str(micro_batch),
        ]
        if system:
            command.extend(("--system", system))
        for override in overrides:
            command.extend(("--set", override))
        if smoke:
            command.append("--smoke")
        result = run(command, env=environment, capture_output=True, text=True, check=False)
        record = {
            "micro_batch": micro_batch,
            "return_code": int(result.returncode),
            "stdout": result.stdout.strip()[-2000:],
            "stderr": result.stderr.strip()[-2000:],
        }
        attempts.append(record)
        if result.returncode == 0:
            return micro_batch, attempts
        if result.returncode != 42:
            break
    return 0, attempts


class GPULease:
    """Advisory lock for AC-PBGRL jobs; it never interacts with external jobs."""

    def __init__(self, lock_root: str | Path, gpus: Sequence[GPUInfo]) -> None:
        self.lock_root = Path(lock_root)
        self.gpus = list(gpus)
        self.handles = []

    def acquire(self) -> bool:
        self.lock_root.mkdir(parents=True, exist_ok=True)
        try:
            for gpu in self.gpus:
                path = self.lock_root / f"{gpu.uuid}.lock"
                handle = path.open("a+")
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                handle.seek(0)
                handle.truncate()
                handle.write(f"pid={os.getpid()} index={gpu.index}\n")
                handle.flush()
                self.handles.append(handle)
            return True
        except BlockingIOError:
            self.release()
            return False

    def release(self) -> None:
        for handle in self.handles:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()
        self.handles.clear()

    def __enter__(self):
        if not self.acquire():
            raise RuntimeError("one or more selected GPUs are already leased by AC-PBGRL")
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.release()


def gpu_inventory() -> list[dict]:
    return [asdict(gpu) for gpu in query_gpus()]
