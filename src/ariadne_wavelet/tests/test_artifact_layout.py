import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ariadne_wavelet import parameter
from ariadne_wavelet.evaluation import save_evaluation_metrics_plot
from ariadne_wavelet.parameter import (
    FOLDER_NAME,
    RuntimeConfig,
    SMOKE_FOLDER_NAME,
    ensure_output_dirs,
    get_checkpoint_path,
    get_latest_checkpoint_path,
    get_model_path,
    get_result_eval_path,
    get_result_gifs_path,
    resolve_resume_checkpoint,
)
from ariadne_wavelet.scripts.train_wavelet import build_runtime_config, parse_args
from ariadne_wavelet.training_monitor import TrainingMonitor


class ArtifactLayoutTests(unittest.TestCase):
    def test_output_dirs_split_train_and_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patch.object(parameter, "result_path", root / "result"), patch.object(parameter, "model_path", root / "model"):
                train_config = RuntimeConfig(run_session="2026_0303_0101")
                smoke_config = RuntimeConfig(run_name=SMOKE_FOLDER_NAME, run_session="2026_0303_0102_smoke")

                ensure_output_dirs(train_config)
                ensure_output_dirs(smoke_config)

                self.assertTrue(get_result_gifs_path(train_config).is_dir())
                self.assertTrue(get_result_eval_path(train_config).is_dir())
                self.assertTrue(get_model_path(train_config).is_dir())
                self.assertEqual(get_checkpoint_path(train_config), root / "model" / "2026_0303_0101" / "checkpoint.pth")

                self.assertTrue(get_result_gifs_path(smoke_config).is_dir())
                self.assertTrue(get_result_eval_path(smoke_config).is_dir())
                self.assertFalse(get_model_path(smoke_config).exists())

    def test_artifact_paths_require_run_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patch.object(parameter, "result_path", root / "result"), patch.object(parameter, "model_path", root / "model"):
                config = RuntimeConfig()
                with self.assertRaises(ValueError):
                    get_result_gifs_path(config)
                with self.assertRaises(ValueError):
                    get_result_eval_path(config)
                with self.assertRaises(ValueError):
                    get_checkpoint_path(config)
                with self.assertRaises(ValueError):
                    ensure_output_dirs(config)

    def test_get_latest_checkpoint_ignores_smoke_and_fake_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_root = Path(tmpdir) / "model"
            (model_root / "2026_0303_0001").mkdir(parents=True)
            (model_root / "2026_0303_0001" / "checkpoint.pth").write_text("ckpt")
            (model_root / "2026_0303_0002_smoke").mkdir(parents=True)
            (model_root / "2026_0303_0002_smoke" / "checkpoint.pth").write_text("smoke")
            (model_root / "checkpoint.pth").mkdir(parents=True)
            (model_root / "2026_0303_0003").mkdir(parents=True)
            (model_root / "2026_0303_0003" / "checkpoint.pth").write_text("latest")

            with patch.object(parameter, "model_path", model_root):
                latest = get_latest_checkpoint_path()

            self.assertEqual(latest, model_root / "2026_0303_0003" / "checkpoint.pth")

    def test_resolve_resume_checkpoint_requires_normal_model_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_root = Path(tmpdir) / "model"
            checkpoint_file = model_root / "2026_0303_0003" / "checkpoint.pth"
            checkpoint_file.parent.mkdir(parents=True)
            checkpoint_file.write_text("ckpt")

            with patch.object(parameter, "model_path", model_root):
                resolved_path, run_session = resolve_resume_checkpoint(checkpoint_file)

            self.assertEqual(resolved_path, checkpoint_file.resolve())
            self.assertEqual(run_session, "2026_0303_0003")

    def test_training_monitor_persists_hidden_state_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor_dir = Path(tmpdir) / "eval"
            monitor = TrainingMonitor(monitor_dir)
            monitor.update_train(1, {"reward": 1.0, "episode_return": 2.0})

            resumed_monitor = TrainingMonitor(monitor_dir)
            resumed_monitor.update_train(2, {"reward": 3.0, "episode_return": 4.0})

            self.assertTrue((monitor_dir / "training_curves.png").is_file())
            self.assertTrue((monitor_dir / ".state" / "training_history.json").is_file())
            self.assertFalse((monitor_dir / "latest_metrics.json").exists())
            self.assertFalse((monitor_dir / "summary.txt").exists())
            self.assertFalse((monitor_dir / "snapshots").exists())

            payload = json.loads((monitor_dir / ".state" / "training_history.json").read_text())
            self.assertEqual(payload["sections"]["train_metrics"]["episodes"], [1, 2])
            self.assertEqual(payload["sections"]["train_metrics"]["history"]["reward"], [1.0, 3.0])

    def test_eval_plot_is_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_path = save_evaluation_metrics_plot(
                [
                    {
                        "episode": 0,
                        "explored_rate": 0.5,
                        "travel_dist": 10.0,
                        "success": True,
                        "episode_return": 1.5,
                        "steps_taken": 7,
                    },
                    {
                        "episode": 1,
                        "explored_rate": 0.7,
                        "travel_dist": 12.0,
                        "success": False,
                        "episode_return": 0.5,
                        "steps_taken": 9,
                    },
                ],
                Path(tmpdir),
            )
            self.assertEqual(plot_path, Path(tmpdir) / "evaluation_metrics.png")
            self.assertTrue(plot_path.is_file())

    def test_train_parse_args_rejects_resume_conflicts(self):
        with patch.object(sys, "argv", ["train_wavelet.py", "--smoke", "--resume-from", "/tmp/checkpoint.pth"]), patch(
            "ariadne_wavelet.scripts.train_wavelet.resolve_resume_checkpoint",
            return_value=(Path("/tmp/checkpoint.pth"), "2026_0303_0003"),
        ), patch(
            "ariadne_wavelet.scripts.train_wavelet.get_run_identity_from_checkpoint",
            return_value=(FOLDER_NAME, "2026_0303_0003"),
        ):
            with self.assertRaises(SystemExit):
                parse_args()

    def test_train_parse_args_accepts_resume(self):
        with patch.object(sys, "argv", ["train_wavelet.py", "--resume-from", "/tmp/checkpoint.pth"]), patch(
            "ariadne_wavelet.scripts.train_wavelet.resolve_resume_checkpoint",
            return_value=(Path("/tmp/checkpoint.pth"), "2026_0303_0003"),
        ), patch(
            "ariadne_wavelet.scripts.train_wavelet.get_run_identity_from_checkpoint",
            return_value=(FOLDER_NAME, "2026_0303_0003"),
        ):
            args = parse_args()

        self.assertEqual(args.resume_from, "/tmp/checkpoint.pth")
        self.assertEqual(args.resume_session, "2026_0303_0003")
        self.assertEqual(args.resume_run_name, FOLDER_NAME)

    def test_smoke_config_keeps_default_gpu_settings(self):
        baseline = RuntimeConfig()
        with patch.object(sys, "argv", ["train_wavelet.py", "--smoke"]):
            args = parse_args()
        config = build_runtime_config(args)

        self.assertEqual(config.use_gpu, baseline.use_gpu)
        self.assertEqual(config.use_gpu_global, baseline.use_gpu_global)
        self.assertEqual(config.num_gpu, baseline.num_gpu)
        self.assertEqual(config.run_name, SMOKE_FOLDER_NAME)


if __name__ == "__main__":
    unittest.main()
