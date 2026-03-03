from __future__ import annotations

import unittest
from unittest.mock import patch

from ariadne_wavelet_attnbias.driver import _resolve_learner_device, _resolve_worker_runtime
from ariadne_wavelet_attnbias.parameter import RuntimeConfig
from ariadne_wavelet_attnbias.runner import Runner


class TestCudaFallback(unittest.TestCase):
    def test_resolve_worker_runtime_disables_gpu_when_cuda_unusable(self):
        runtime_config = RuntimeConfig()
        with patch("ariadne_wavelet_attnbias.driver._cuda_runtime_status", return_value=(False, "bad cuda")):
            worker_config, worker_num_gpus = _resolve_worker_runtime(runtime_config)

        self.assertFalse(worker_config.use_gpu)
        self.assertEqual(worker_config.num_gpu, 0)
        self.assertEqual(worker_num_gpus, 0.0)

    def test_resolve_learner_device_falls_back_to_cpu_when_cuda_unusable(self):
        runtime_config = RuntimeConfig()
        with patch("ariadne_wavelet_attnbias.driver._cuda_runtime_status", return_value=(False, "bad cuda")):
            device = _resolve_learner_device(runtime_config)

        self.assertEqual(device.type, "cpu")

    def test_runner_falls_back_to_cpu_when_cuda_unusable(self):
        runtime_config = RuntimeConfig().with_overrides(use_gpu=True)
        with patch("ariadne_wavelet_attnbias.runner._cuda_runtime_status", return_value=(False, "bad cuda")):
            runner = Runner(0, runtime_config)

        self.assertEqual(runner.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
