from __future__ import annotations

import torch
import unittest
from unittest.mock import patch

from ariadne_wavelet.driver import (
    _resolve_learner_device,
    _resolve_learner_gpu_count,
    _resolve_worker_runtime,
)
from ariadne_wavelet.parameter import RuntimeConfig
from ariadne_wavelet.runner import Runner


class TestCudaFallback(unittest.TestCase):
    def test_resolve_worker_runtime_disables_gpu_when_cuda_unusable(self):
        runtime_config = RuntimeConfig()
        with patch("ariadne_wavelet.driver._cuda_runtime_status", return_value=(False, "bad cuda")):
            worker_config, worker_num_gpus = _resolve_worker_runtime(runtime_config)

        self.assertFalse(worker_config.use_gpu)
        self.assertEqual(worker_config.num_gpu, 0)
        self.assertEqual(worker_num_gpus, 0.0)

    def test_resolve_worker_runtime_keeps_workers_on_cpu_even_when_gpu_requested(self):
        runtime_config = RuntimeConfig().with_overrides(use_gpu=True, num_gpu=4)
        worker_config, worker_num_gpus = _resolve_worker_runtime(runtime_config, available_gpus=4.0)

        self.assertFalse(worker_config.use_gpu)
        self.assertEqual(worker_config.num_gpu, 0)
        self.assertEqual(worker_num_gpus, 0.0)

    def test_resolve_learner_device_falls_back_to_cpu_when_cuda_unusable(self):
        runtime_config = RuntimeConfig()
        with patch("ariadne_wavelet.driver._cuda_runtime_status", return_value=(False, "bad cuda")):
            device = _resolve_learner_device(runtime_config)

        self.assertEqual(device.type, "cpu")

    def test_runner_falls_back_to_cpu_when_cuda_unusable(self):
        runtime_config = RuntimeConfig().with_overrides(use_gpu=True)
        with patch("ariadne_wavelet.runner._cuda_runtime_status", return_value=(False, "bad cuda")):
            runner = Runner(0, runtime_config)

        self.assertEqual(runner.device.type, "cpu")

    def test_resolve_learner_gpu_count_uses_visible_devices_when_num_gpu_is_auto(self):
        runtime_config = RuntimeConfig().with_overrides(num_gpu=0)
        with patch("ariadne_wavelet.driver.torch.cuda.device_count", return_value=4):
            gpu_count = _resolve_learner_gpu_count(runtime_config, device=torch.device("cuda"))

        self.assertEqual(gpu_count, 4)

    def test_resolve_learner_gpu_count_caps_to_requested_gpu_count(self):
        runtime_config = RuntimeConfig().with_overrides(num_gpu=2)
        with patch("ariadne_wavelet.driver.torch.cuda.device_count", return_value=4):
            gpu_count = _resolve_learner_gpu_count(runtime_config, device=torch.device("cuda"))

        self.assertEqual(gpu_count, 2)


if __name__ == "__main__":
    unittest.main()
