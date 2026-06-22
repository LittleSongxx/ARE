from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import imageio.v2 as imageio
import numpy as np

from hpbg_rl.evaluation import (
    _normalize_eval_device,
    get_evaluated_episodes,
    load_evaluation_detail_history,
    load_evaluation_history,
    resolve_eval_maps,
    resolve_fixed_eval_maps,
    save_evaluation_summary,
    summarize_eval_results,
)
from hpbg_rl.map_splits import MapSplitError
from hpbg_rl.parameter import RuntimeConfig


def _write_map(path: Path, value: int = 255) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((32, 32), value, dtype=np.uint8)
    image[[0, -1], :] = 0
    image[:, [0, -1]] = 0
    imageio.imwrite(path, image)


class EvaluationTests(unittest.TestCase):
    def test_normalize_eval_device_adds_explicit_cuda_index(self):
        with mock.patch("hpbg_rl.evaluation.torch.cuda.current_device", return_value=0):
            device = _normalize_eval_device("cuda")

        self.assertEqual(device.type, "cuda")
        self.assertEqual(device.index, 0)

    def test_resolve_eval_maps_defaults_to_validation_split_and_blocks_train(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "maps"
            _write_map(root / "train" / "train_map.png")
            _write_map(root / "val" / "val_map.png")
            _write_map(root / "test" / "test_map.png")
            config = RuntimeConfig(maps_dir=str(root), val_map_count=1, test_map_count=1)

            default_selected = resolve_eval_maps(config)
            test_selected = resolve_eval_maps(config, split="test")

            self.assertEqual([path.parent.name for path in default_selected], ["val"])
            self.assertEqual([path.name for path in default_selected], ["val_map.png"])
            self.assertEqual([path.parent.name for path in test_selected], ["test"])
            with self.assertRaises(MapSplitError):
                resolve_eval_maps(config, split="train", count=1)

    def test_evaluation_summary_records_protocol_and_per_map_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "validation"
            protocol = {
                "split": "val",
                "manifest_hash": "abc123",
                "split_manifest_path": "/tmp/split_manifest.json",
                "step_budget": 8,
            }
            results = [
                {
                    "episode": 12,
                    "split": "val",
                    "manifest_hash": "abc123",
                    "map_slot": 1,
                    "map_label": "map_1",
                    "map_name": "val_map.png",
                    "map_path": "/tmp/val_map.png",
                    "size_bucket": "small",
                    "free_area": 900,
                    "free_ratio": 0.8,
                    "explored_rate": 0.75,
                    "travel_dist": 10.0,
                    "success": True,
                    "episode_return": 1.0,
                    "steps_taken": 8,
                    "completion_steps": 8,
                    "completion_travel_dist": 10.0,
                    "normalized_exploration_efficiency": 0.075,
                    "exploration_auc": 0.62,
                    "gif_path": "/tmp/val.gif",
                    "last_frame_path": "/tmp/val.png",
                }
            ]
            summary = summarize_eval_results(results)

            artifact_paths = save_evaluation_summary(
                output_dir,
                episode_number=12,
                results=results,
                summary=summary,
                bucket_size=10,
                protocol=protocol,
            )

            detail_payload = json.loads(artifact_paths["detail_path"].read_text())
            self.assertEqual(detail_payload["protocol"]["split"], "val")
            self.assertEqual(detail_payload["protocol"]["manifest_hash"], "abc123")
            self.assertEqual(detail_payload["summary"]["split"], "val")
            self.assertEqual(detail_payload["summary"]["manifest_hash"], "abc123")
            self.assertEqual(detail_payload["summary"]["by_size_bucket"]["small"]["maps"], 1)
            self.assertTrue(artifact_paths["per_map_csv_path"].is_file())
            per_map_csv = artifact_paths["per_map_csv_path"].read_text()
            self.assertIn("normalized_exploration_efficiency", per_map_csv)
            self.assertIn("exploration_auc", per_map_csv)
            self.assertIn("val_map.png", per_map_csv)

    def test_resolve_fixed_eval_maps_uses_stable_sorted_subset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name in ("c.png", "a.png", "b.png"):
                (root / name).write_text("map")

            selected = resolve_fixed_eval_maps(2, maps_dir=root)

            self.assertEqual([path.name for path in selected], ["a.png", "b.png"])

    def test_save_evaluation_summary_writes_raw_history_and_metric_plots(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "eval"
            results = [
                {
                    "episode": 137,
                    "map_name": "map_a.png",
                    "map_path": "/tmp/map_a.png",
                    "explored_rate": 0.8,
                    "travel_dist": 12.0,
                    "success": True,
                    "episode_return": 1.5,
                    "steps_taken": 9,
                    "completion_steps": 9,
                    "completion_travel_dist": 12.0,
                    "gif_path": "/tmp/a.gif",
                    "last_frame_path": "/tmp/a.png",
                },
                {
                    "episode": 137,
                    "map_name": "map_b.png",
                    "map_path": "/tmp/map_b.png",
                    "explored_rate": 0.6,
                    "travel_dist": 18.0,
                    "success": False,
                    "episode_return": 0.5,
                    "steps_taken": 16,
                    "completion_steps": None,
                    "completion_travel_dist": None,
                    "gif_path": "/tmp/b.gif",
                    "last_frame_path": "/tmp/b.png",
                },
            ]
            summary = summarize_eval_results(results)

            artifact_paths = save_evaluation_summary(
                output_dir,
                episode_number=137,
                results=results,
                summary=summary,
                bucket_size=100,
            )

            self.assertEqual(
                artifact_paths["detail_path"],
                output_dir / "raw" / "episode_200" / "eval_episode_00137.json",
            )
            self.assertTrue(artifact_paths["detail_path"].is_file())
            self.assertTrue(artifact_paths["history_path"].is_file())
            self.assertTrue(artifact_paths["history_csv_path"].is_file())
            self.assertTrue(artifact_paths["maps_manifest_path"].is_file())
            self.assertEqual(
                sorted(artifact_paths["metric_plot_paths"].keys()),
                [
                    "completion_steps",
                    "completion_travel_dist",
                    "explored_rate",
                    "success_rate",
                ],
            )
            for plot_path in artifact_paths["metric_plot_paths"].values():
                self.assertTrue(plot_path.is_file())

            history = load_evaluation_history(output_dir)
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]["episode"], 137)
            self.assertEqual(get_evaluated_episodes(output_dir), {137})

            detail_payload = json.loads(artifact_paths["detail_path"].read_text())
            self.assertEqual(detail_payload["episode"], 137)
            self.assertEqual(len(detail_payload["results"]), 2)
            self.assertEqual(detail_payload["results"][0]["map_label"], "map_1")
            self.assertEqual(detail_payload["results"][1]["map_label"], "map_2")

            manifest_lines = artifact_paths["maps_manifest_path"].read_text().strip().splitlines()
            self.assertEqual(manifest_lines[0].split("\t")[0], "map_1")
            self.assertEqual(manifest_lines[1].split("\t")[0], "map_2")

    def test_save_evaluation_summary_overwrites_same_episode_and_builds_detail_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "eval"
            results_episode_100 = [
                {
                    "episode": 100,
                    "map_slot": 1,
                    "map_label": "map_1",
                    "map_name": "map_a.png",
                    "map_path": "/tmp/map_a.png",
                    "explored_rate": 0.7,
                    "travel_dist": 15.0,
                    "success": True,
                    "episode_return": 1.0,
                    "steps_taken": 10,
                    "completion_steps": 10,
                    "completion_travel_dist": 15.0,
                    "gif_path": "/tmp/a.gif",
                    "last_frame_path": "/tmp/a.png",
                },
                {
                    "episode": 100,
                    "map_slot": 2,
                    "map_label": "map_2",
                    "map_name": "map_b.png",
                    "map_path": "/tmp/map_b.png",
                    "explored_rate": 0.5,
                    "travel_dist": 20.0,
                    "success": False,
                    "episode_return": 0.2,
                    "steps_taken": 16,
                    "completion_steps": None,
                    "completion_travel_dist": None,
                    "gif_path": "/tmp/b.gif",
                    "last_frame_path": "/tmp/b.png",
                },
            ]
            results_episode_200 = [
                {
                    "episode": 200,
                    "map_slot": 1,
                    "map_label": "map_1",
                    "map_name": "map_a.png",
                    "map_path": "/tmp/map_a.png",
                    "explored_rate": 0.9,
                    "travel_dist": 11.0,
                    "success": True,
                    "episode_return": 1.8,
                    "steps_taken": 8,
                    "completion_steps": 8,
                    "completion_travel_dist": 11.0,
                    "gif_path": "/tmp/a2.gif",
                    "last_frame_path": "/tmp/a2.png",
                },
                {
                    "episode": 200,
                    "map_slot": 2,
                    "map_label": "map_2",
                    "map_name": "map_b.png",
                    "map_path": "/tmp/map_b.png",
                    "explored_rate": 0.8,
                    "travel_dist": 18.0,
                    "success": True,
                    "episode_return": 1.2,
                    "steps_taken": 12,
                    "completion_steps": 12,
                    "completion_travel_dist": 18.0,
                    "gif_path": "/tmp/b2.gif",
                    "last_frame_path": "/tmp/b2.png",
                },
            ]

            save_evaluation_summary(
                output_dir,
                episode_number=100,
                results=results_episode_100,
                summary=summarize_eval_results(results_episode_100),
                bucket_size=100,
            )
            save_evaluation_summary(
                output_dir,
                episode_number=200,
                results=results_episode_200,
                summary=summarize_eval_results(results_episode_200),
                bucket_size=100,
            )

            updated_episode_200 = list(results_episode_200)
            updated_episode_200[1] = {
                **updated_episode_200[1],
                "explored_rate": 0.85,
                "completion_steps": 11,
                "completion_travel_dist": 17.0,
            }
            save_evaluation_summary(
                output_dir,
                episode_number=200,
                results=updated_episode_200,
                summary=summarize_eval_results(updated_episode_200),
                bucket_size=100,
            )

            detail_history = load_evaluation_detail_history(output_dir)
            self.assertEqual([item["episode"] for item in detail_history], [100, 200])
            self.assertEqual(detail_history[1]["results"][1]["explored_rate"], 0.85)
            self.assertEqual(detail_history[0]["results"][1]["completion_steps"], None)
            self.assertTrue((output_dir / "completion_steps" / "completion_steps_history.png").is_file())
            self.assertTrue((output_dir / "completion_travel_dist" / "completion_travel_dist_history.png").is_file())
            self.assertEqual(get_evaluated_episodes(output_dir), {100, 200})


if __name__ == "__main__":
    unittest.main()
