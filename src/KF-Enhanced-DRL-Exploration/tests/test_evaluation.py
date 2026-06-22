from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from evaluation import (
    _normalize_eval_device,
    get_evaluated_episodes,
    load_evaluation_detail_history,
    load_evaluation_history,
    resolve_fixed_eval_maps,
    save_evaluation_summary,
    summarize_eval_results,
)


class EvaluationTests(unittest.TestCase):
    def test_normalize_eval_device_adds_explicit_cuda_index(self):
        with mock.patch("evaluation.torch.cuda.current_device", return_value=0):
            device = _normalize_eval_device("cuda")

        self.assertEqual(device.type, "cuda")
        self.assertEqual(device.index, 0)

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
                    "success_rate": 1.0,
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
                    "success_rate": 0.0,
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
                    "distance_efficiency",
                    "explored_rate",
                    "mean_planning_time_ms",
                    "success_rate",
                    "time_efficiency",
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

            manifest_lines = (
                artifact_paths["maps_manifest_path"].read_text().strip().splitlines()
            )
            self.assertEqual(manifest_lines[0].split("\t")[0], "map_1")
            self.assertEqual(manifest_lines[1].split("\t")[0], "map_2")

    def test_save_evaluation_summary_overwrites_same_episode_and_builds_detail_history(
        self,
    ):
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
                    "success_rate": 1.0,
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
                    "success_rate": 0.0,
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
                    "success_rate": 1.0,
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
                    "success_rate": 1.0,
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
            self.assertTrue(
                (
                    output_dir / "completion_steps" / "completion_steps_history.png"
                ).is_file()
            )
            self.assertTrue(
                (
                    output_dir
                    / "completion_travel_dist"
                    / "completion_travel_dist_history.png"
                ).is_file()
            )
            self.assertEqual(get_evaluated_episodes(output_dir), {100, 200})

    def test_summarize_eval_results_includes_benchmark_metrics(self):
        results = [
            {
                "explored_rate": 0.8,
                "travel_dist": 12.0,
                "success": True,
                "success_rate": 1.0,
                "episode_return": 1.5,
                "steps_taken": 9,
                "completion_steps": 9,
                "completion_travel_dist": 12.0,
                "distance_efficiency": 0.4,
                "step_efficiency": 0.2,
                "time_efficiency": 0.1,
                "episode_wall_time_sec": 2.0,
                "planning_time_sec": 1.0,
                "graph_update_time_sec": 0.5,
                "observation_time_sec": 0.2,
                "policy_inference_time_sec": 0.3,
                "env_step_time_sec": 1.0,
                "node_manager_update_time_sec": 0.25,
                "dense_graph_build_time_sec": 0.1,
                "graph_rarefaction_time_sec": 0.05,
                "mean_step_wall_time_ms": 20.0,
                "mean_planning_time_ms": 10.0,
                "mean_graph_update_time_ms": 5.0,
                "mean_observation_time_ms": 2.0,
                "mean_policy_inference_time_ms": 3.0,
                "mean_env_step_time_ms": 10.0,
                "mean_node_manager_update_time_ms": 2.5,
                "mean_dense_graph_build_time_ms": 1.0,
                "mean_graph_rarefaction_time_ms": 0.5,
            },
            {
                "explored_rate": 0.6,
                "travel_dist": 18.0,
                "success": False,
                "success_rate": 0.0,
                "episode_return": 0.5,
                "steps_taken": 15,
                "completion_steps": None,
                "completion_travel_dist": None,
                "distance_efficiency": 0.2,
                "step_efficiency": 0.1,
                "time_efficiency": 0.05,
                "episode_wall_time_sec": 4.0,
                "planning_time_sec": 2.0,
                "graph_update_time_sec": 1.0,
                "observation_time_sec": 0.4,
                "policy_inference_time_sec": 0.6,
                "env_step_time_sec": 2.0,
                "node_manager_update_time_sec": 0.5,
                "dense_graph_build_time_sec": 0.2,
                "graph_rarefaction_time_sec": 0.1,
                "mean_step_wall_time_ms": 26.0,
                "mean_planning_time_ms": 13.0,
                "mean_graph_update_time_ms": 6.0,
                "mean_observation_time_ms": 2.5,
                "mean_policy_inference_time_ms": 4.0,
                "mean_env_step_time_ms": 12.0,
                "mean_node_manager_update_time_ms": 3.0,
                "mean_dense_graph_build_time_ms": 1.5,
                "mean_graph_rarefaction_time_ms": 0.7,
            },
        ]

        summary = summarize_eval_results(results)

        self.assertAlmostEqual(summary["distance_efficiency"], 0.3)
        self.assertAlmostEqual(summary["time_efficiency"], 0.075)
        self.assertAlmostEqual(summary["mean_planning_time_ms"], 11.5)
        self.assertAlmostEqual(summary["completion_steps"], 9.0)
        self.assertAlmostEqual(summary["completion_travel_dist"], 12.0)


if __name__ == "__main__":
    unittest.main()
