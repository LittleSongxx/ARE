from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ARiADNE.env import resolve_curriculum_map_files, select_map_path_for_episode
from ARiADNE.parameter import RuntimeConfig, get_curriculum_level, get_curriculum_level_index


class CurriculumSamplerTests(unittest.TestCase):
    def _create_maps_dir(self, root: Path, names: tuple[str, ...]) -> Path:
        maps_dir = root / "maps"
        maps_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            (maps_dir / name).write_text(name, encoding="utf-8")
        return maps_dir

    def _write_manifest(self, root: Path, records: list[dict[str, object]]) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        manifest_path = root / "difficulty_manifest.json"
        manifest_path.write_text(
            json.dumps({"generated_at_utc": "2026-03-04T00:00:00Z", "records": records}, indent=2),
            encoding="utf-8",
        )
        return manifest_path

    def test_curriculum_level_helpers_follow_two_bucket_milestones(self):
        config = RuntimeConfig(
            enable_curriculum=True,
            curriculum_source="/tmp/ariadne_curriculum",
            curriculum_milestones=(0, 5),
            curriculum_levels=("easy", "hard"),
        )
        self.assertEqual(get_curriculum_level_index(0, config), 0)
        self.assertEqual(get_curriculum_level_index(4, config), 0)
        self.assertEqual(get_curriculum_level_index(5, config), 1)
        self.assertEqual(get_curriculum_level(0, config), "easy")
        self.assertEqual(get_curriculum_level(9, config), "hard")

    def test_manifest_source_selects_bucket_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            maps_dir = self._create_maps_dir(root, ("easy_a.png", "easy_b.png", "hard_a.png", "base.png"))
            output_dir = root / "difficulty_output"
            self._write_manifest(
                output_dir,
                [
                    {
                        "map_name": "easy_a.png",
                        "map_path": str((maps_dir / "easy_a.png").resolve()),
                        "difficulty_bucket": "easy",
                    },
                    {
                        "map_name": "easy_b.png",
                        "map_path": str((maps_dir / "easy_b.png").resolve()),
                        "difficulty_bucket": "easy",
                    },
                    {
                        "map_name": "hard_a.png",
                        "map_path": str((maps_dir / "hard_a.png").resolve()),
                        "difficulty_bucket": "hard",
                    },
                ],
            )
            config = RuntimeConfig(
                enable_curriculum=True,
                curriculum_source=str(output_dir),
                curriculum_milestones=(0, 2),
                curriculum_levels=("easy", "hard"),
                curriculum_mix_window=0,
            )

            easy_files, level, level_index = resolve_curriculum_map_files(maps_dir, 1, config)
            self.assertEqual(level, "easy")
            self.assertEqual(level_index, 0)
            self.assertEqual([path.name for path in easy_files], ["easy_a.png", "easy_b.png"])

            map_path, level, level_index = select_map_path_for_episode(maps_dir, 1, config)
            self.assertEqual(map_path.name, "easy_b.png")
            self.assertEqual(level, "easy")
            self.assertEqual(level_index, 0)

            hard_files, level, level_index = resolve_curriculum_map_files(maps_dir, 3, config)
            self.assertEqual(level, "hard")
            self.assertEqual(level_index, 1)
            self.assertEqual([path.name for path in hard_files], ["hard_a.png"])

    def test_bucket_directory_source_selects_bucket_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            maps_dir = self._create_maps_dir(root, ("easy_map.png", "hard_map.png", "base.png"))
            bucket_root = root / "buckets"
            (bucket_root / "easy").mkdir(parents=True, exist_ok=True)
            (bucket_root / "hard").mkdir(parents=True, exist_ok=True)
            (bucket_root / "easy" / "easy_map.png").write_text("easy", encoding="utf-8")
            (bucket_root / "hard" / "hard_map.png").write_text("hard", encoding="utf-8")

            config = RuntimeConfig(
                enable_curriculum=True,
                curriculum_source=str(bucket_root),
                curriculum_milestones=(0, 3),
                curriculum_levels=("easy", "hard"),
                curriculum_mix_window=0,
            )

            easy_path, level, level_index = select_map_path_for_episode(maps_dir, 0, config)
            self.assertEqual(easy_path.name, "easy_map.png")
            self.assertEqual(level, "easy")
            self.assertEqual(level_index, 0)

            hard_path, level, level_index = select_map_path_for_episode(maps_dir, 4, config)
            self.assertEqual(hard_path.name, "hard_map.png")
            self.assertEqual(level, "hard")
            self.assertEqual(level_index, 1)

    def test_mix_window_interleaves_previous_and_current_levels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            maps_dir = self._create_maps_dir(root, ("easy_a.png", "hard_a.png"))
            output_dir = root / "difficulty_output"
            self._write_manifest(
                output_dir,
                [
                    {
                        "map_name": "easy_a.png",
                        "map_path": str((maps_dir / "easy_a.png").resolve()),
                        "difficulty_bucket": "easy",
                    },
                    {
                        "map_name": "hard_a.png",
                        "map_path": str((maps_dir / "hard_a.png").resolve()),
                        "difficulty_bucket": "hard",
                    },
                ],
            )
            config = RuntimeConfig(
                enable_curriculum=True,
                curriculum_source=str(output_dir),
                curriculum_milestones=(0, 2),
                curriculum_levels=("easy", "hard"),
                curriculum_mix_window=2,
            )

            map_path, level, level_index = select_map_path_for_episode(maps_dir, 2, config)
            self.assertEqual(map_path.name, "easy_a.png")
            self.assertEqual(level, "easy")
            self.assertEqual(level_index, 0)

            map_path, level, level_index = select_map_path_for_episode(maps_dir, 3, config)
            self.assertEqual(map_path.name, "hard_a.png")
            self.assertEqual(level, "hard")
            self.assertEqual(level_index, 1)

            map_path, level, level_index = select_map_path_for_episode(maps_dir, 4, config)
            self.assertEqual(map_path.name, "hard_a.png")
            self.assertEqual(level, "hard")
            self.assertEqual(level_index, 1)

    def test_missing_bucket_falls_back_to_base_maps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            maps_dir = self._create_maps_dir(root, ("base_a.png", "base_b.png"))
            output_dir = root / "difficulty_output"
            self._write_manifest(
                output_dir,
                [
                    {
                        "map_name": "base_b.png",
                        "map_path": str((maps_dir / "base_b.png").resolve()),
                        "difficulty_bucket": "hard",
                    },
                ],
            )
            config = RuntimeConfig(
                enable_curriculum=True,
                curriculum_source=str(output_dir),
                curriculum_milestones=(0, 5),
                curriculum_levels=("easy", "hard"),
            )

            map_files, level, level_index = resolve_curriculum_map_files(maps_dir, 0, config)
            self.assertEqual(level, "easy")
            self.assertEqual(level_index, 0)
            self.assertEqual([path.name for path in map_files], ["base_a.png", "base_b.png"])

    def test_curriculum_override_false_uses_base_maps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            maps_dir = self._create_maps_dir(root, ("base_a.png", "base_b.png", "hard_only.png"))
            output_dir = root / "difficulty_output"
            self._write_manifest(
                output_dir,
                [
                    {
                        "map_name": "hard_only.png",
                        "map_path": str((maps_dir / "hard_only.png").resolve()),
                        "difficulty_bucket": "hard",
                    },
                ],
            )
            config = RuntimeConfig(
                enable_curriculum=True,
                curriculum_source=str(output_dir),
                curriculum_milestones=(0, 1),
                curriculum_levels=("easy", "hard"),
            )

            map_files, level, level_index = resolve_curriculum_map_files(
                maps_dir,
                4,
                config,
                curriculum_override=False,
            )
            self.assertIsNone(level)
            self.assertIsNone(level_index)
            self.assertEqual([path.name for path in map_files], ["base_a.png", "base_b.png", "hard_only.png"])


if __name__ == "__main__":
    unittest.main()
