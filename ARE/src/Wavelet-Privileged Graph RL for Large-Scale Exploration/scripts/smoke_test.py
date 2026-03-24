#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/mpl-wpg-rl-{os.environ.get('USER', 'user')}")

from wpg_rl.parameter import RuntimeConfig, apply_runtime_config, build_run_session
from wpg_rl.runtime_utils import configure_matplotlib_cache


def main():
    configure_matplotlib_cache("smoke-main")
    base_config = apply_runtime_config(
        RuntimeConfig(
            run_name="wpg_rl_smoke",
            run_session=build_run_session("wpg_rl_smoke"),
            max_episodes=1,
            num_meta_agent=1,
            max_episode_step=8,
            minimum_buffer_size=1,
            batch_size=4,
            replay_size=16,
            save_img_gap=999999,
            summary_window=1,
            train_updates_per_iter=1,
            use_gpu=False,
            use_gpu_global=False,
        )
    )

    from wpg_rl.driver import create_learner_state, sample_rollout_batch, stack_batch_tensors, train_step
    from wpg_rl.worker import Worker

    device = torch.device("cpu")
    combos = [
        ("baseline", False, False),
        ("distill_only", False, True),
        ("lf_attn_only", True, False),
        ("both", True, True),
    ]

    for name, use_lf, use_distill in combos:
        combo_config = base_config.with_overrides(
            use_lf_attention_hf_residual=use_lf,
            use_privileged_wavelet_distillation=use_distill,
            run_session=build_run_session(name),
        )
        learner_state = create_learner_state(combo_config, device)
        worker = Worker(0, learner_state.policy_net, global_step=1, device=device, save_image=False)
        worker.run_episode()

        rollouts = sample_rollout_batch(worker.episode_buffer, combo_config.batch_size, replace=True)
        batch = stack_batch_tensors(rollouts, device)
        metrics = train_step(batch, learner_state, combo_config)

        for key in ("policy_loss", "q_value_loss", "alpha_loss", "wavelet_loss"):
            value = float(metrics[key])
            if not math.isfinite(value):
                raise RuntimeError(f"{name}: {key} is not finite: {value}")

        print(
            f"[{name}] steps={worker.perf_metrics['episode_steps']} "
            f"buffer={len(worker.episode_buffer[0])} "
            f"policy_loss={metrics['policy_loss']:.6f} "
            f"q_value_loss={metrics['q_value_loss']:.6f} "
            f"alpha_loss={metrics['alpha_loss']:.6f} "
            f"wavelet_loss={metrics['wavelet_loss']:.6f}"
        )


if __name__ == "__main__":
    main()
