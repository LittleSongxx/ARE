from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from hpbg_rl.agent import Agent
from hpbg_rl.driver import create_learner_state, train_step
from hpbg_rl.ground_truth_node_manager import GroundTruthNodeManager
from hpbg_rl.map_splits import materialize_split_manifest
from hpbg_rl.model import PolicyNet
from hpbg_rl.parameter import (
    BASE_NODE_INPUT_DIM,
    EMBEDDING_DIM,
    NODE_INPUT_DIM,
    RuntimeConfig,
    SENSOR_RANGE,
    MIN_UTILITY,
    MAX_EPISODES,
    REPLAY_SIZE,
    MINIMUM_BUFFER_SIZE,
    BATCH_SIZE,
    LR,
    NUM_META_AGENT,
    ENABLE_CORRIDOR_EDGE_PRUNING,
    ENABLE_CORRIDOR_GRAPH_COMPRESSION,
)
from tests.test_wavelet_distillation import build_training_batch


def _write_map(path: Path, value: int = 255) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((32, 32), value, dtype=np.uint8)
    image[[0, -1], :] = 0
    image[:, [0, -1]] = 0
    imageio.imwrite(path, image)


class BaselineCompatibilityTests(unittest.TestCase):
    def test_hpbg_off_defaults_match_baseline_training_protocol(self):
        self.assertEqual(float(SENSOR_RANGE), 16.0)
        self.assertEqual(int(MIN_UTILITY), 2)
        self.assertEqual(int(MAX_EPISODES), 10000)
        self.assertEqual(int(REPLAY_SIZE), 10000)
        self.assertEqual(int(MINIMUM_BUFFER_SIZE), 2000)
        self.assertEqual(int(BATCH_SIZE), 128)
        self.assertEqual(float(LR), 1e-5)
        self.assertEqual(int(NUM_META_AGENT), 16)
        self.assertFalse(ENABLE_CORRIDOR_EDGE_PRUNING)
        self.assertFalse(ENABLE_CORRIDOR_GRAPH_COMPRESSION)

    def test_hpbg_off_and_full_share_same_map_manifest_and_splits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "maps"
            for name in ("train_a.png", "val_a.png", "test_a.png"):
                _write_map(root / name)
            manifest_path = Path(tmpdir) / "split_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "root": str(root),
                        "seed": 5,
                        "splits": {
                            "train": ["train_a.png"],
                            "val": ["val_a.png"],
                            "test": ["test_a.png"],
                        },
                    }
                )
            )
            baseline_config = RuntimeConfig(
                split_manifest_path=str(manifest_path),
                use_hpbg=False,
                use_belief_state=False,
                use_map_prediction=False,
                use_hierarchical_graph=False,
                use_expert_reward=False,
                use_belief_distillation=False,
                use_privileged_wavelet_distillation=False,
            )
            hpbg_config = RuntimeConfig(split_manifest_path=str(manifest_path), use_hpbg=True)

            baseline_manifest = materialize_split_manifest(baseline_config)
            hpbg_manifest = materialize_split_manifest(hpbg_config)

            self.assertEqual(baseline_manifest.content_hash(), hpbg_manifest.content_hash())
            for split in ("train", "val", "test"):
                self.assertEqual(
                    [path.name for path in baseline_manifest.split_paths(split)],
                    [path.name for path in hpbg_manifest.split_paths(split)],
                )

    def test_all_hpbg_switches_off_keep_training_step_finite(self):
        runtime_config = RuntimeConfig(
            use_hpbg=False,
            use_belief_state=False,
            use_map_prediction=False,
            use_hierarchical_graph=False,
            use_expert_reward=False,
            use_belief_distillation=False,
            use_privileged_wavelet_distillation=False,
            use_lf_attention_hf_residual=False,
            wavelet_distill_weight=0.0,
            hpbg_belief_distill_weight=0.0,
        )
        learner_state = create_learner_state(runtime_config, torch.device("cpu"))
        batch = build_training_batch(batch_size=2, actor_nodes=5, critic_nodes=7)

        metrics = train_step(batch, learner_state, runtime_config)

        for key in ("policy_loss", "q_value_loss", "alpha_loss", "wavelet_loss", "belief_loss"):
            self.assertTrue(math.isfinite(float(metrics[key])), key)
        self.assertEqual(float(metrics["wavelet_weighted_loss"]), 0.0)
        self.assertEqual(float(metrics["belief_weighted_loss"]), 0.0)
    def test_direct_agent_runtime_config_disables_hpbg_features(self):
        runtime_config = RuntimeConfig(
            use_hpbg=False,
            use_belief_state=False,
            use_map_prediction=False,
            use_hierarchical_graph=False,
            use_expert_reward=False,
            use_belief_distillation=False,
            use_privileged_wavelet_distillation=False,
            use_lf_attention_hf_residual=False,
        )
        policy = PolicyNet(
            NODE_INPUT_DIM,
            EMBEDDING_DIM,
            use_lf_attention_hf_residual=False,
            use_privileged_wavelet_distillation=False,
            use_hierarchical_context=False,
        )
        agent = Agent(policy, runtime_config=runtime_config)
        agent.node_coords = np.array([[0.0, 0.0], [4.0, 0.0], [8.0, 0.0]], dtype=np.float32)
        agent.utility = np.array([4.0, 2.0, 1.0], dtype=np.float32)
        agent.guidepost = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        agent.adjacent_matrix = np.zeros((3, 3), dtype=np.int64)
        agent.updating_map_info = None

        features = agent._build_actor_belief_features()

        self.assertEqual(features.shape, (3, NODE_INPUT_DIM - BASE_NODE_INPUT_DIM))
        self.assertTrue(np.allclose(features, 0.0))

    def test_hpbg_off_critic_preserves_baseline_explored_sign(self):
        runtime_config = RuntimeConfig(use_hpbg=False)
        manager = GroundTruthNodeManager.__new__(GroundTruthNodeManager)
        manager.runtime_config = runtime_config
        manager.ground_truth_map_info = None
        node_coords = np.array([[0.0, 0.0], [4.0, 0.0], [8.0, 0.0]], dtype=np.float32)
        explored = np.array([1.0, 0.0, 1.0], dtype=np.float32)

        privileged = manager._build_privileged_features(node_coords, explored, belief_map_info=None)

        self.assertTrue(np.allclose(privileged[:, 0], explored))
        self.assertTrue(np.allclose(privileged[:, 1:], 0.0))


if __name__ == "__main__":
    unittest.main()