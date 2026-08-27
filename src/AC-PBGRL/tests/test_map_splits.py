from pathlib import Path

import numpy as np
from skimage import io

from ac_pbgrl.data.map_splits import canonical_d4_hash, create_map_splits


def test_d4_hash_groups_rotations():
    image = np.zeros((9, 7), dtype=bool)
    image[1:5, 2] = True
    image[4, 2:6] = True
    assert canonical_d4_hash(image) == canonical_d4_hash(np.rot90(image))


def test_split_has_no_equivalent_leakage(tmp_path: Path):
    maps = tmp_path / "maps"
    maps.mkdir()
    for index in range(20):
        image = np.zeros((24, 24), dtype=np.uint8)
        image[2:-2, 2:-2] = 255
        image[4 + index % 8 : 7 + index % 8, 5 + index % 10] = 0
        io.imsave(str(maps / f"{index:03d}.png"), image, check_contrast=False)
    rotated = np.rot90(io.imread(str(maps / "000.png")))
    io.imsave(str(maps / "rotation.png"), rotated, check_contrast=False)
    manifest = create_map_splits(maps, tmp_path / "splits.json", seed=3)
    assignment = {item["name"]: item["split"] for item in manifest["records"]}
    assert assignment["000.png"] == assignment["rotation.png"]
    split_groups = {}
    for item in manifest["records"]:
        split_groups.setdefault(item["canonical_d4"], set()).add(item["split"])
    assert all(len(splits) == 1 for splits in split_groups.values())
