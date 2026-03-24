from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evaluation import ensure_eval_map_dir
from utils import ensure_episode_bucket_dir, get_episode_bucket_name


class ArtifactLayoutTest(unittest.TestCase):
    def test_episode_bucket_name_uses_bucket_end(self):
        self.assertEqual(get_episode_bucket_name(1, bucket_size=100), "episode_100")
        self.assertEqual(get_episode_bucket_name(100, bucket_size=100), "episode_100")
        self.assertEqual(get_episode_bucket_name(101, bucket_size=100), "episode_200")

    def test_ensure_episode_bucket_dir_creates_expected_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir) / "gifs"
            bucket_dir = ensure_episode_bucket_dir(base_dir, 137, bucket_size=50)

            self.assertEqual(bucket_dir, base_dir / "episode_150")
            self.assertTrue(bucket_dir.is_dir())

    def test_eval_map_dir_uses_eval_bucket_and_map_subdir(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir) / "test" / "gifs"
            map_dir = ensure_eval_map_dir(base_dir, episode_number=100, map_number=2, bucket_size=100)

            self.assertEqual(map_dir, base_dir / "episode_100" / "map_2")
            self.assertTrue(map_dir.is_dir())


if __name__ == "__main__":
    unittest.main()
