import subprocess
import sys

from ac_pbgrl.runtime.gpu import (
    GPULease,
    GPUInfo,
    calibrate_micro_batch,
    parse_gpu_csv,
    parse_gpu_process_csv,
    recommended_micro_batch,
    select_gpus,
)
from ac_pbgrl.runtime.supervisor import _direct_child_pids


def inventory():
    return [
        GPUInfo(0, "GPU-0", "A40", 46068, 0, 46068, 0, 25, 0),
        GPUInfo(1, "GPU-1", "A40", 46068, 12534, 33534, 30, 56, 1),
        GPUInfo(2, "GPU-2", "A40", 46068, 12362, 33706, 69, 78, 1),
        GPUInfo(3, "GPU-3", "A40", 46068, 0, 46068, 0, 30, 0),
    ]


def test_prefer_idle_matches_shared_server_snapshot():
    selected = select_gpus(inventory(), policy="prefer-idle", min_gpus=1, max_gpus=4)
    assert [gpu.index for gpu in selected] == [0, 3]
    assert recommended_micro_batch(selected, reserve_gib=6) == 64


def test_shared_fallback_and_idle_only():
    shared = inventory()[1:3]
    assert select_gpus(shared, policy="idle-only", min_gpus=1, max_gpus=4) == []
    selected = select_gpus(shared, policy="prefer-idle", min_gpus=1, max_gpus=4)
    assert [gpu.index for gpu in selected] == [1]


def test_zero_one_two_and_four_card_selection_and_safety_thresholds():
    assert select_gpus([], min_gpus=1) == []
    assert len(select_gpus(inventory()[:1], min_gpus=1)) == 1
    assert len(select_gpus([inventory()[0], inventory()[3]], min_gpus=1)) == 2
    assert len(select_gpus(inventory(), policy="shared-ok", min_gpus=1)) == 3
    assert len(
        select_gpus(
            inventory(), policy="shared-ok", min_gpus=1, max_utilization_pct=70
        )
    ) == 4
    hot = GPUInfo(4, "GPU-hot", "A40", 46068, 0, 46068, 0, 81, 0)
    full = GPUInfo(5, "GPU-full", "A40", 46068, 30000, 16068, 0, 30, 0)
    assert select_gpus([hot, full], min_gpus=1) == []


def test_parser_keeps_process_counts():
    text = "0, GPU-a, NVIDIA A40, 46068, 12000, 34068, 20, 55\n"
    parsed = parse_gpu_csv(text, {"GPU-a": 2})
    assert parsed[0].process_count == 2
    assert parsed[0].free_gib > 30
    assert parse_gpu_process_csv("GPU-a, 123\nGPU-a, 456\ninvalid, N/A\n") == {
        "GPU-a": {123, 456}
    }


def test_memory_probe_falls_back_after_injected_oom():
    calls = []

    def fake_run(command, **kwargs):
        del kwargs
        micro = int(command[command.index("--micro-batch") + 1])
        calls.append(micro)
        code = 42 if micro == 64 else 0
        return subprocess.CompletedProcess(command, code, stdout="probe", stderr="")

    selected, attempts = calibrate_micro_batch(
        inventory()[:1],
        experiment="full",
        system=None,
        overrides=[],
        candidates=[64, 32, 16, 8, 4],
        start=64,
        reserve_gib=6,
        run=fake_run,
    )
    assert selected == 32
    assert calls == [64, 32]
    assert [item["return_code"] for item in attempts] == [42, 0]


def test_gpu_uuid_lease_blocks_duplicate_project_job(tmp_path):
    first = GPULease(tmp_path, [inventory()[0]])
    second = GPULease(tmp_path, [inventory()[0]])
    assert first.acquire()
    assert not second.acquire()
    first.release()
    assert second.acquire()
    second.release()


def test_supervisor_can_target_training_children_without_signaling_launcher():
    launcher = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import subprocess,sys,time; "
            "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
            "print(p.pid, flush=True); time.sleep(30)",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert launcher.stdout is not None
        worker_pid = int(launcher.stdout.readline())
        assert worker_pid in _direct_child_pids(launcher.pid)
        assert launcher.pid not in _direct_child_pids(launcher.pid)
    finally:
        for child_pid in _direct_child_pids(launcher.pid):
            subprocess.run(["kill", "-TERM", str(child_pid)], check=False)
        launcher.terminate()
        launcher.wait(timeout=5)
