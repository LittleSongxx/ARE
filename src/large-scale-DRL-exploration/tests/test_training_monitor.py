from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from training_monitor import TrainingMonitor


class TrainingMonitorTests(unittest.TestCase):
    def test_monitor_writes_progress_and_diagnostic_plots(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = TrainingMonitor(Path(tmpdir) / "monitor", window_size=2, snapshot_interval=1)

            monitor.update_train(
                10,
                {
                    "reward": 0.2,
                    "episode_return": 4.0,
                    "travel_dist": 120.0,
                    "episode_steps": 18.0,
                    "policy_loss": 0.5,
                    "q_value_loss": 1.2,
                    "alpha_loss": -0.1,
                    "explored_rate": 0.9,
                    "success_rate": 1.0,
                    "value": 2.0,
                    "entropy": -0.8,
                    "log_alpha": -2.3,
                    "policy_grad_norm": 12.0,
                    "q_value_grad_norm": 55.0,
                },
            )
            monitor.update_system(10, {"buffer_size": 12000})
            monitor.update_eval(
                10,
                {
                    "explored_rate": 0.85,
                    "success_rate": 1.0,
                    "travel_dist": 140.0,
                    "steps_taken": 20.0,
                    "episode_return": 3.5,
                },
            )

            monitor.update_train(
                20,
                {
                    "reward": 0.3,
                    "episode_return": 5.0,
                    "travel_dist": 110.0,
                    "episode_steps": 16.0,
                    "policy_loss": 0.4,
                    "q_value_loss": 1.0,
                    "alpha_loss": -0.15,
                    "explored_rate": 0.95,
                    "success_rate": 1.0,
                    "value": 2.5,
                    "entropy": -0.7,
                    "log_alpha": -2.4,
                    "policy_grad_norm": 10.0,
                    "q_value_grad_norm": 48.0,
                },
            )
            monitor.update_system(20, {"buffer_size": 4000})
            monitor.update_eval(
                20,
                {
                    "explored_rate": 0.9,
                    "success_rate": 1.0,
                    "travel_dist": 130.0,
                    "steps_taken": 18.0,
                    "episode_return": 3.8,
                },
            )

            self.assertTrue((monitor.save_dir / "training_curves.png").is_file())
            self.assertTrue((monitor.save_dir / "training_diagnostics.png").is_file())
            self.assertTrue((monitor.state_dir / "training_history.json").is_file())


if __name__ == "__main__":
    unittest.main()
