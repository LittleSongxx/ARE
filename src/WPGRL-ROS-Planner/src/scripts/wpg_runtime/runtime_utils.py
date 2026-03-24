from __future__ import annotations

import os

import numpy as np
import torch


def remap_ros_occupancy_to_wpg(
    ros_map: np.ndarray,
    ros_free: int = 0,
    ros_occupied: int = 100,
    ros_unknown: int = -1,
    wpg_free: int = 255,
    wpg_occupied: int = 1,
    wpg_unknown: int = 127,
) -> np.ndarray:
    """Convert ROS OccupancyGrid values to WPG training values.

    ROS convention:  free=0,   occupied=100, unknown=-1
    WPG convention:  free=255, occupied=1,   unknown=127
    """
    out = np.full_like(ros_map, wpg_unknown, dtype=np.int16)
    out[ros_map == ros_free] = wpg_free
    out[ros_map == ros_occupied] = wpg_occupied
    return out


def load_policy_checkpoint(
    model: torch.nn.Module,
    ckpt_path: str,
    device: str = "cpu",
) -> torch.nn.Module:
    """Load policy_model weights from a checkpoint file.

    Raises clear errors when the file is missing or the key is absent.
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)

    if "policy_model" not in checkpoint:
        available = list(checkpoint.keys())
        raise KeyError(
            f"Checkpoint does not contain 'policy_model'. Available keys: {available}"
        )

    state_dict = checkpoint["policy_model"]
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Shape mismatch loading checkpoint. Ensure the model architecture "
            f"matches the training configuration.\nOriginal error: {exc}"
        ) from exc

    model.eval()
    return model


def resolve_model_path(
    package_path: str,
    rosparam_model_path: str | None = None,
    default_subpath: str = "scripts/model/checkpoint.pth",
) -> str:
    """Resolve the checkpoint path: rosparam override > default location."""
    if rosparam_model_path and os.path.isfile(rosparam_model_path):
        return rosparam_model_path

    default_path = os.path.join(package_path, default_subpath)
    if os.path.isfile(default_path):
        return default_path

    if rosparam_model_path:
        return rosparam_model_path
    return default_path
