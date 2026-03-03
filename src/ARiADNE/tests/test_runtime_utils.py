import os
import unittest
from unittest import mock

from ARiADNE.parameter import RuntimeConfig, NUM_META_AGENT
from ARiADNE.runtime_utils import (
    _auto_detect_worker_threads,
    resolve_ray_worker_num_cpus,
    resolve_worker_num_threads,
)


class RayWorkerConfigTests(unittest.TestCase):
    def test_driver_defaults_to_single_cpu_and_auto_threads(self):
        """No env vars / no CLI → worker_num_cpus=1, threads auto-detected ≥ 1."""
        config = RuntimeConfig()
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(resolve_ray_worker_num_cpus(config), 1)
            threads = resolve_worker_num_threads(config, 1)
            expected = _auto_detect_worker_threads(NUM_META_AGENT)
            self.assertEqual(threads, expected)
            self.assertGreaterEqual(threads, 1)

    def test_driver_uses_worker_threads_for_worker_cpu_when_cpu_not_set(self):
        config = RuntimeConfig(worker_num_threads=4)
        with mock.patch.dict("os.environ", {}, clear=True):
            worker_num_cpus = resolve_ray_worker_num_cpus(config)
            self.assertEqual(worker_num_cpus, 4)
            self.assertEqual(resolve_worker_num_threads(config, worker_num_cpus), 4)

    def test_environment_overrides_are_respected(self):
        config = RuntimeConfig()
        env = {
            "ARIADNE_RAY_WORKER_NUM_CPUS": "3",
            "ARIADNE_WORKER_NUM_THREADS": "5",
        }
        with mock.patch.dict("os.environ", env, clear=True):
            self.assertEqual(resolve_ray_worker_num_cpus(config), 3)
            self.assertEqual(resolve_worker_num_threads(config, 3), 5)


if __name__ == "__main__":
    unittest.main()
