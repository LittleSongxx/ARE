import unittest
from unittest import mock

from ARiADNE.parameter import RuntimeConfig
from ARiADNE.runner import Runner


class _DummyPolicyNet:
    def __init__(self, *args, **kwargs):
        pass

    def to(self, device):
        return self


class RunnerThreadTests(unittest.TestCase):
    def test_runner_uses_environment_thread_override(self):
        runtime_config = RuntimeConfig(worker_num_threads=None, use_gpu=False)
        env = {"ARIADNE_WORKER_NUM_THREADS": "4"}
        with mock.patch.dict("os.environ", env, clear=True), mock.patch(
            "ARiADNE.runner.PolicyNet", _DummyPolicyNet
        ), mock.patch("ARiADNE.runner.torch.set_num_threads") as set_threads, mock.patch(
            "ARiADNE.runner.torch.set_num_interop_threads"
        ) as set_interop_threads:
            runner = Runner(0, runtime_config)

        self.assertEqual(runner.worker_num_threads, 4)
        set_threads.assert_called_once_with(4)
        set_interop_threads.assert_called_once_with(4)


if __name__ == "__main__":
    unittest.main()
