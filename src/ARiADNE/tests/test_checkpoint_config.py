import unittest

from ARiADNE.driver import _resume_counters_from_checkpoint
from ARiADNE.parameter import (
    RuntimeConfig,
    checkpoint_model_config,
    ensure_checkpoint_compatible,
    runtime_config_from_checkpoint,
)


class CheckpointConfigTests(unittest.TestCase):
    def test_runtime_config_roundtrip_from_checkpoint_snapshot(self):
        config = RuntimeConfig(
            use_wavelet_feature=True,
            wavelet_feature_mode="scales_orient",
            wavelet_scales_auto=False,
            wavelet_scales=(1, 2),
            use_wavelet_attn_bias=True,
        )
        payload = {"config_snapshot": checkpoint_model_config(config)}
        restored = runtime_config_from_checkpoint(payload, RuntimeConfig(run_session="x"))
        self.assertTrue(restored.use_wavelet_feature)
        self.assertEqual(restored.wavelet_feature_mode, "scales_orient")
        self.assertEqual(restored.wavelet_scales, (1, 2))
        self.assertTrue(restored.use_wavelet_attn_bias)

    def test_incompatible_checkpoint_raises(self):
        config = RuntimeConfig(use_wavelet_feature=True, wavelet_feature_mode="scales")
        payload = {"config_snapshot": checkpoint_model_config(config)}
        with self.assertRaises(ValueError):
            ensure_checkpoint_compatible(payload, RuntimeConfig(use_wavelet_feature=False))

    def test_rl_runtime_overrides_are_not_part_of_model_snapshot(self):
        payload = {"config_snapshot": checkpoint_model_config(RuntimeConfig())}
        restored = runtime_config_from_checkpoint(
            payload,
            RuntimeConfig(
                run_session="x",
                enable_per=True,
                policy_delay=3,
                replay_ratio=2.0,
                enable_reward_decomposition=True,
                reward_terminal_bonus=9.0,
                enable_curriculum=True,
                curriculum_source="/tmp/curriculum",
                curriculum_milestones=(0, 10),
                curriculum_levels=("easy", "hard"),
                curriculum_mix_window=12,
            ),
        )
        self.assertTrue(restored.enable_per)
        self.assertEqual(restored.policy_delay, 3)
        self.assertEqual(restored.replay_ratio, 2.0)
        self.assertTrue(restored.enable_reward_decomposition)
        self.assertEqual(restored.reward_terminal_bonus, 9.0)
        self.assertTrue(restored.enable_curriculum)
        self.assertEqual(restored.curriculum_source, "/tmp/curriculum")
        self.assertEqual(restored.curriculum_milestones, (0, 10))
        self.assertEqual(restored.curriculum_levels, ("easy", "hard"))
        self.assertEqual(restored.curriculum_mix_window, 12)

    def test_resume_counters_default_for_legacy_checkpoint(self):
        self.assertEqual(_resume_counters_from_checkpoint({"episode": 7}), (0, 0, 1))
        self.assertEqual(
            _resume_counters_from_checkpoint(
                {
                    "episode": 7,
                    "learner_update_step": 11,
                    "policy_update_step": 5,
                    "target_q_update_counter": 9,
                }
            ),
            (11, 5, 9),
        )


if __name__ == "__main__":
    unittest.main()
