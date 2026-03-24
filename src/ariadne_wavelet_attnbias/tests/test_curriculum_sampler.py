import tempfile
import unittest
from pathlib import Path

from ariadne_wavelet_attnbias.env import resolve_curriculum_map_files, select_map_path_for_episode
from ariadne_wavelet_attnbias.parameter import RLOptions, get_curriculum_level, get_curriculum_level_index


class CurriculumSamplerTests(unittest.TestCase):
    def _touch(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("map")

    def test_curriculum_level_boundaries(self):
        rl_options = RLOptions(
            use_curriculum=True,
            curriculum_milestones=(0, 2000, 5000),
            curriculum_levels=("easy", "medium", "hard"),
        )

        self.assertEqual(get_curriculum_level(0, rl_options), "easy")
        self.assertEqual(get_curriculum_level(1999, rl_options), "easy")
        self.assertEqual(get_curriculum_level(2000, rl_options), "medium")
        self.assertEqual(get_curriculum_level(4999, rl_options), "medium")
        self.assertEqual(get_curriculum_level(5000, rl_options), "hard")
        self.assertEqual(get_curriculum_level_index(5000, rl_options), 2)

    def test_dir_mode_is_deterministic_and_falls_back_to_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._touch(root / "root_a.png")
            self._touch(root / "root_b.png")
            self._touch(root / "easy" / "easy_a.png")
            self._touch(root / "easy" / "easy_b.png")

            rl_options = RLOptions(
                use_curriculum=True,
                curriculum_mode="dir",
                curriculum_milestones=(0, 2, 4),
                curriculum_levels=("easy", "medium", "hard"),
                curriculum_dirs=(("easy", "easy"), ("medium", "medium"), ("hard", "hard")),
            )

            bucket, level, level_index = resolve_curriculum_map_files(root, 0, rl_options)
            self.assertEqual(level, "easy")
            self.assertEqual(level_index, 0)
            self.assertEqual([path.name for path in bucket], ["easy_a.png", "easy_b.png"])

            selected_once, fallback_level, fallback_index = select_map_path_for_episode(root, 3, rl_options)
            selected_twice, _, _ = select_map_path_for_episode(root, 3, rl_options)
            self.assertEqual(fallback_level, "medium")
            self.assertEqual(fallback_index, 1)
            self.assertEqual(selected_once.name, "root_b.png")
            self.assertEqual(selected_once, selected_twice)

    def test_pattern_mode_matches_and_falls_back(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._touch(root / "maze_easy_01.png")
            self._touch(root / "maze_medium_01.png")
            self._touch(root / "maze_misc_01.png")

            rl_options = RLOptions(
                use_curriculum=True,
                curriculum_mode="pattern",
                curriculum_milestones=(0, 1, 2),
                curriculum_levels=("easy", "medium", "hard"),
                curriculum_patterns=(
                    ("easy", ("*easy*",)),
                    ("medium", ("*medium*",)),
                    ("hard", ("*hard*",)),
                ),
            )

            bucket, level, level_index = resolve_curriculum_map_files(root, 1, rl_options)
            self.assertEqual(level, "medium")
            self.assertEqual(level_index, 1)
            self.assertEqual([path.name for path in bucket], ["maze_medium_01.png"])

            selected, fallback_level, fallback_index = select_map_path_for_episode(root, 2, rl_options)
            self.assertEqual(fallback_level, "hard")
            self.assertEqual(fallback_index, 2)
            self.assertEqual(selected.name, "maze_misc_01.png")


if __name__ == "__main__":
    unittest.main()
