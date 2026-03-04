import tempfile
import unittest
from pathlib import Path

import numpy as np

from ARiADNE.map_difficulty import (
    assign_difficulty_buckets,
    analyze_occupancy_map,
    score_map_records,
    write_difficulty_outputs,
)
from ARiADNE.parameter import FREE, OCCUPIED


def _open_room_map(size: int = 21) -> np.ndarray:
    occupancy = np.full((size, size), FREE, dtype=np.int16)
    occupancy[0, :] = OCCUPIED
    occupancy[-1, :] = OCCUPIED
    occupancy[:, 0] = OCCUPIED
    occupancy[:, -1] = OCCUPIED
    return occupancy


def _maze_like_map(size: int = 21) -> np.ndarray:
    occupancy = _open_room_map(size)
    gap_rows = {4: size - 3, 8: 2, 12: size - 3, 16: 2}
    for wall_col, gap_row in gap_rows.items():
        occupancy[1:-1, wall_col] = OCCUPIED
        occupancy[gap_row, wall_col] = FREE

    occupancy[6:10, 2] = OCCUPIED
    occupancy[11:15, 6] = OCCUPIED
    occupancy[3:7, 10] = OCCUPIED
    occupancy[14:18, 14] = OCCUPIED
    return occupancy


class MapDifficultyTests(unittest.TestCase):
    def test_maze_like_map_scores_harder_than_open_room(self):
        easy_record = analyze_occupancy_map(
            _open_room_map(),
            robot_cell=np.array([1, 1], dtype=np.int64),
            map_path="easy.png",
        )
        hard_record = analyze_occupancy_map(
            _maze_like_map(),
            robot_cell=np.array([1, 1], dtype=np.int64),
            map_path="hard.png",
        )

        scored = score_map_records([easy_record, hard_record])
        scores = {record["map_name"]: float(record["difficulty_score"]) for record in scored}

        self.assertGreater(scores["hard.png"], scores["easy.png"])

    def test_bucket_assignment_uses_sorted_difficulty_order(self):
        records = [
            {"map_name": "a.png", "map_path": "/tmp/a.png", "difficulty_score": 0.1},
            {"map_name": "b.png", "map_path": "/tmp/b.png", "difficulty_score": 0.5},
            {"map_name": "c.png", "map_path": "/tmp/c.png", "difficulty_score": 0.9},
        ]

        bucketed = assign_difficulty_buckets(records)
        buckets = {record["map_name"]: record["difficulty_bucket"] for record in bucketed}

        self.assertEqual(buckets["a.png"], "easy")
        self.assertEqual(buckets["b.png"], "medium")
        self.assertEqual(buckets["c.png"], "hard")

    def test_bucket_assignment_supports_two_bucket_curriculum(self):
        records = [
            {"map_name": "a.png", "map_path": "/tmp/a.png", "difficulty_score": 0.1},
            {"map_name": "b.png", "map_path": "/tmp/b.png", "difficulty_score": 0.4},
            {"map_name": "c.png", "map_path": "/tmp/c.png", "difficulty_score": 0.9},
            {"map_name": "d.png", "map_path": "/tmp/d.png", "difficulty_score": 1.1},
        ]

        bucketed = assign_difficulty_buckets(records, bucket_names=("easy", "hard"))
        buckets = {record["map_name"]: record["difficulty_bucket"] for record in bucketed}

        self.assertEqual(buckets["a.png"], "easy")
        self.assertEqual(buckets["b.png"], "easy")
        self.assertEqual(buckets["c.png"], "hard")
        self.assertEqual(buckets["d.png"], "hard")

    def test_write_outputs_materializes_csv_json_and_bucket_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_dir = root / "maps"
            source_dir.mkdir()
            for name in ("easy.png", "medium.png", "hard.png"):
                (source_dir / name).write_text(name)

            records = [
                {"map_name": "easy.png", "map_path": str((source_dir / "easy.png").resolve()), "difficulty_score": 0.1, "difficulty_bucket": "easy"},
                {"map_name": "medium.png", "map_path": str((source_dir / "medium.png").resolve()), "difficulty_score": 0.5, "difficulty_bucket": "medium"},
                {"map_name": "hard.png", "map_path": str((source_dir / "hard.png").resolve()), "difficulty_score": 0.9, "difficulty_bucket": "hard"},
            ]

            outputs = write_difficulty_outputs(records, root / "output", clear_output=False)

            self.assertTrue(outputs["csv_path"].is_file())
            self.assertTrue(outputs["json_path"].is_file())
            self.assertTrue(outputs["summary_path"].is_file())
            self.assertTrue((outputs["bucket_root"] / "easy" / "easy.png").exists())
            self.assertTrue((outputs["bucket_root"] / "medium" / "medium.png").exists())
            self.assertTrue((outputs["bucket_root"] / "hard" / "hard.png").exists())


if __name__ == "__main__":
    unittest.main()
