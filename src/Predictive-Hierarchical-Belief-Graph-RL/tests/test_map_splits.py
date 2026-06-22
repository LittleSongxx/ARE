from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from hpbg_rl.evaluation import resolve_eval_maps
from hpbg_rl.map_splits import (
    MapSplitError,
    load_split_manifest,
    materialize_split_manifest,
    resolve_map_split_paths,
    save_split_manifest,
)
from hpbg_rl.parameter import RuntimeConfig


def _write_map(path: Path, value: int = 255) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((32, 32), value, dtype=np.uint8)
    image[[0, -1], :] = 0
    image[:, [0, -1]] = 0
    imageio.imwrite(path, image)


def _manifest_payload(root: Path) -> dict[str, object]:
    return {
        "version": 1,
        "root": str(root),
        "seed": 123,
        "splits": {
            "train": ["train_map.png"],
            "val": ["val_map.png"],
            "test": ["test_map.png"],
        },
    }


class MapSplitProtocolTests(unittest.TestCase):
    def test_manifest_load_validates_disjoint_splits_and_records_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "maps"
            for name in ("train_map.png", "val_map.png", "test_map.png"):
                _write_map(root / name)
            manifest_path = Path(tmpdir) / "split_manifest.json"
            manifest_path.write_text(json.dumps(_manifest_payload(root)))

            manifest = load_split_manifest(manifest_path)
            reloaded = load_split_manifest(save_split_manifest(manifest, Path(tmpdir) / "saved_manifest.json"))

            self.assertEqual([path.name for path in manifest.split_paths("train")], ["train_map.png"])
            self.assertEqual([path.name for path in manifest.split_paths("validation")], ["val_map.png"])
            self.assertEqual([path.name for path in manifest.split_paths("test")], ["test_map.png"])
            self.assertEqual(manifest.content_hash(), reloaded.content_hash())
            self.assertEqual({entry.split for entry in manifest.entries}, {"train", "val", "test"})
            for entry in manifest.entries:
                self.assertEqual(entry.width, 32)
                self.assertEqual(entry.height, 32)
                self.assertEqual(len(entry.sha256), 64)
                self.assertGreater(entry.free_ratio, 0.0)
                self.assertIn(entry.size_bucket, {"small", "medium", "large"})

    def test_duplicate_map_path_across_splits_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "maps"
            _write_map(root / "shared.png")
            manifest_path = Path(tmpdir) / "split_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "root": str(root),
                        "splits": {
                            "train": ["shared.png"],
                            "val": ["shared.png"],
                            "test": [],
                        },
                    }
                )
            )

            with self.assertRaises(MapSplitError):
                load_split_manifest(manifest_path)

    def test_single_dir_generation_is_deterministic_and_disjoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "maps"
            for index in range(8):
                _write_map(root / f"map_{index}.png")
            config = RuntimeConfig(maps_dir=str(root), split_seed=7)

            first = materialize_split_manifest(config)
            second = materialize_split_manifest(config)

            first_by_split = {
                split: [path.name for path in first.split_paths(split)]
                for split in ("train", "val", "test")
            }
            second_by_split = {
                split: [path.name for path in second.split_paths(split)]
                for split in ("train", "val", "test")
            }
            self.assertEqual(first_by_split, second_by_split)
            self.assertEqual(first.content_hash(), second.content_hash())

            split_sets = {split: set(paths) for split, paths in first_by_split.items()}
            self.assertTrue(split_sets["train"].isdisjoint(split_sets["val"]))
            self.assertTrue(split_sets["train"].isdisjoint(split_sets["test"]))
            self.assertTrue(split_sets["val"].isdisjoint(split_sets["test"]))
            self.assertEqual(sum(len(paths) for paths in first_by_split.values()), 8)
            self.assertGreater(len(first_by_split["val"]), 0)
            self.assertGreater(len(first_by_split["test"]), 0)

    def test_split_dirs_allow_training_maps_but_forbid_train_eval_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "maps"
            _write_map(root / "train" / "train_map.png")
            _write_map(root / "val" / "val_map.png")
            _write_map(root / "test" / "test_map.png")
            config = RuntimeConfig(maps_dir=str(root), val_map_count=1, test_map_count=1)

            train_paths = resolve_map_split_paths(config, "train", allow_train_eval=True)
            val_paths = resolve_eval_maps(config, split="val")
            test_paths = resolve_eval_maps(config, split="test")

            self.assertEqual([path.parent.name for path in train_paths], ["train"])
            self.assertEqual([path.parent.name for path in val_paths], ["val"])
            self.assertEqual([path.parent.name for path in test_paths], ["test"])
            with self.assertRaises(MapSplitError):
                resolve_eval_maps(config, split="train", count=1)

            train_eval_config = config.with_overrides(allow_train_split_eval=True)
            self.assertEqual(
                [path.name for path in resolve_eval_maps(train_eval_config, split="train", count=1)],
                ["train_map.png"],
            )


if __name__ == "__main__":
    unittest.main()