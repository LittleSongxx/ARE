#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import sys
import tempfile
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/mpl-hpbg-rl-{os.environ.get('USER', 'user')}")

from hpbg_rl.evaluation import resolve_eval_maps
from hpbg_rl.map_splits import MapSplitError
from hpbg_rl.parameter import (
    BASE_NODE_INPUT_DIM,
    HPBG_CRITIC_ORACLE_UTILITY_INDEX,
    HPBG_CRITIC_EXPERT_POTENTIAL_INDEX,
    NODE_INPUT_DIM,
    RuntimeConfig,
    apply_runtime_config,
    build_run_session,
)
from hpbg_rl.runtime_utils import configure_matplotlib_cache


def _write_smoke_map(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    map_array = np.ones((64, 64), dtype=np.uint8)
    map_array[[0, -1], :] = 0
    map_array[:, [0, -1]] = 0
    map_array[30:34, 18:46] = 0
    map_array[8:16, 8:16] = 48
    imageio.imwrite(path, map_array)


def _build_smoke_maps_dir() -> str:
    maps_dir = Path(tempfile.mkdtemp(prefix="hpbg-rl-smoke-maps-"))
    _write_smoke_map(maps_dir / "train" / "smoke_train.png")
    _write_smoke_map(maps_dir / "val" / "smoke_val.png")
    _write_smoke_map(maps_dir / "test" / "smoke_test.png")
    return str(maps_dir)


def main():
    configure_matplotlib_cache("smoke-main")
    smoke_maps_dir = _build_smoke_maps_dir()
    base_config = apply_runtime_config(
        RuntimeConfig(
            run_name="hpbg_rl_smoke",
            run_session=build_run_session("hpbg_rl_smoke"),
            maps_dir=smoke_maps_dir,
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

    val_eval_maps = resolve_eval_maps(base_config, split="val", count=1)
    test_eval_maps = resolve_eval_maps(base_config, split="test", count=1)
    if [path.parent.name for path in val_eval_maps] != ["val"]:
        raise RuntimeError("smoke split protocol: validation eval must use val split")
    if [path.parent.name for path in test_eval_maps] != ["test"]:
        raise RuntimeError("smoke split protocol: final test eval must use test split")
    try:
        resolve_eval_maps(base_config, split="train", count=1)
    except MapSplitError:
        pass
    else:
        raise RuntimeError("smoke split protocol: train split must not be available to eval by default")

    from hpbg_rl.driver import create_learner_state, sample_rollout_batch, stack_batch_tensors, train_step
    from hpbg_rl.worker import Worker

    device = torch.device("cpu")
    combos = [
        (
            "baseline_compat",
            {
                "use_hpbg": False,
                "use_belief_state": False,
                "use_map_prediction": False,
                "use_hierarchical_graph": False,
                "use_expert_reward": False,
                "use_belief_distillation": False,
                "use_lf_attention_hf_residual": False,
                "use_privileged_wavelet_distillation": False,
                "wavelet_distill_weight": 0.0,
                "hpbg_belief_distill_weight": 0.0,
            },
        ),
        (
            "hpbg_core",
            {
                "use_lf_attention_hf_residual": False,
                "use_privileged_wavelet_distillation": False,
                "use_belief_distillation": True,
            },
        ),
        (
            "hpbg_distill",
            {
                "use_lf_attention_hf_residual": False,
                "use_privileged_wavelet_distillation": True,
                "use_belief_distillation": True,
            },
        ),
        (
            "hpbg_full",
            {
                "use_lf_attention_hf_residual": True,
                "use_privileged_wavelet_distillation": True,
                "use_belief_distillation": True,
            },
        ),
    ]

    for name, overrides in combos:
        combo_config = apply_runtime_config(
            base_config.with_overrides(
                **overrides,
                run_session=build_run_session(name),
            )
        )
        learner_state = create_learner_state(combo_config, device)
        worker = Worker(
            0,
            learner_state.policy_net,
            global_step=1,
            device=device,
            save_image=False,
            runtime_config=combo_config,
        )
        if Path(worker.env.map_path).parent.name != "train":
            raise RuntimeError("smoke split protocol: training worker must use train split")
        worker.run_episode()

        if not combo_config.use_hpbg:
            actor_inputs = worker.episode_buffer[0][0]
            critic_inputs = worker.episode_buffer[15][0]
            actor_extra = actor_inputs[..., BASE_NODE_INPUT_DIM:NODE_INPUT_DIM]
            critic_online_extra = critic_inputs[..., BASE_NODE_INPUT_DIM:NODE_INPUT_DIM]
            critic_oracle_extra = critic_inputs[
                ...,
                HPBG_CRITIC_ORACLE_UTILITY_INDEX : HPBG_CRITIC_EXPERT_POTENTIAL_INDEX + 1,
            ]
            if torch.any(actor_extra != 0) or torch.any(critic_online_extra != 0) or torch.any(critic_oracle_extra != 0):
                raise RuntimeError("baseline_compat: HPBG-off features are not neutral")
            if worker._state_potential() != 0.0:
                raise RuntimeError("baseline_compat: expert potential must be disabled")

        rollouts = sample_rollout_batch(worker.episode_buffer, combo_config.batch_size, replace=True)
        batch = stack_batch_tensors(rollouts, device)
        metrics = train_step(batch, learner_state, combo_config)
        if not combo_config.use_hpbg:
            if metrics["wavelet_weighted_loss"] != 0.0 or metrics["belief_weighted_loss"] != 0.0:
                raise RuntimeError("baseline_compat: auxiliary losses must be zero when HPBG is disabled")

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
