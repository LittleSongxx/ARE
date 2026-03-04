from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from ARiADNE.scripts.redraw_monitor import (
    find_latest_monitor_dir,
    get_monitor_history_path,
    redraw_monitor,
    resolve_monitor_dir,
)
from ARiADNE.training_monitor import TrainingMonitor


class RedrawMonitorScriptTests(unittest.TestCase):
    def test_find_latest_monitor_dir_uses_latest_session_with_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result_root = Path(tmpdir) / "result"
            older = result_root / "2026_0304_0101" / "train" / "monitor"
            newer = result_root / "2026_0304_0202" / "train" / "monitor"

            TrainingMonitor(older).update_train(1, {"reward": 1.0, "episode_return": 2.0})
            TrainingMonitor(newer).update_train(1, {"reward": 3.0, "episode_return": 4.0})

            self.assertEqual(find_latest_monitor_dir(result_root), newer.resolve())

    def test_redraw_monitor_rebuilds_plot_from_existing_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor_dir = Path(tmpdir) / "monitor"
            monitor = TrainingMonitor(monitor_dir)
            monitor.update_train(1, {"reward": 1.0, "episode_return": 2.0})
            monitor.plot_file.unlink()

            history_file, plot_file = redraw_monitor(monitor_dir)

            self.assertEqual(history_file, get_monitor_history_path(monitor_dir))
            self.assertEqual(plot_file, monitor.plot_file.resolve())
            self.assertTrue(plot_file.is_file())

    def test_resolve_monitor_dir_from_run_session(self):
        args = Namespace(monitor_dir=None, run_session="2026_0304_1200", checkpoint=None, result_root=None)
        expected = (Path(__file__).resolve().parents[1] / "result" / "2026_0304_1200" / "train" / "monitor").resolve()
        self.assertEqual(resolve_monitor_dir(args), expected)


if __name__ == "__main__":
    unittest.main()
