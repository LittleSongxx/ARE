from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "training_monitor.py"
SPEC = importlib.util.spec_from_file_location("ariadne_wavelet_attnbias.training_monitor", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
TrainingMonitor = MODULE.TrainingMonitor


class TrainingMonitorTests(unittest.TestCase):
    def test_out_of_order_updates_are_sorted_and_deduplicated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor_dir = Path(tmpdir) / "monitor"
            monitor = TrainingMonitor(monitor_dir)

            monitor.update_train(3, {"reward": 3.0, "episode_return": 30.0})
            monitor.update_train(1, {"reward": 1.0, "episode_return": 10.0})
            monitor.update_train(2, {"reward": 2.0, "episode_return": 20.0})
            monitor.update_train(2, {"reward": 2.5, "episode_return": 25.0})

            payload = json.loads((monitor_dir / ".state" / "training_history.json").read_text())
            self.assertEqual(payload["sections"]["train_metrics"]["episodes"], [1, 2, 3])
            self.assertEqual(payload["sections"]["train_metrics"]["history"]["reward"], [1.0, 2.5, 3.0])
            self.assertEqual(payload["sections"]["train_metrics"]["history"]["episode_return"], [10.0, 25.0, 30.0])
            self.assertTrue((monitor_dir / "training_curves.png").is_file())


if __name__ == "__main__":
    unittest.main()
