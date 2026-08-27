from pathlib import Path

import pytest
import yaml

from ac_pbgrl.cli import (
    _minimum_pilot_environment_steps,
    _pilot_evidence_level,
    _suite_entries,
    build_parser,
)
from ac_pbgrl.config import CONFIG_ROOT, config_fingerprint, load_config, parse_overrides
from ac_pbgrl.runtime.manifest import build_run_manifest, save_run_manifest


def test_experiment_merge_and_dotted_override():
    config = load_config("ariadne", overrides=["project.seed=9", "train.global_batch_size=64"])
    assert config.project.seed == 9
    assert config.train.global_batch_size == 64
    assert config.method.potential is False
    assert config.method.privileged_critic is False
    assert Path(config.project.maps_dir).name == "maps"


def test_all_declared_experiments_load():
    names = {path.stem for path in (CONFIG_ROOT / "experiments").glob("*.yaml")}
    assert names >= {
        "ariadne",
        "ariadne_pi",
        "q_distillation",
        "potential_mse",
        "potential_nll",
        "potential_nll_rank",
        "potential_kf",
        "potential_diffusion",
        "full",
        "ema_control",
        "gru_control",
    }
    fingerprints = {config_fingerprint(load_config(name)) for name in names}
    assert len(fingerprints) == len(names)


def test_suite_groups_can_run_main_before_ablations():
    suite = yaml.safe_load((CONFIG_ROOT / "suites" / "main.yaml").read_text(encoding="utf-8"))
    main = _suite_entries(suite, ("main",))
    ablations = _suite_entries(suite, ("ablations",))

    assert len(main) == 20
    assert len(ablations) == 21
    assert {method for method, _ in main} == {
        "ariadne",
        "ariadne_pi",
        "q_distillation",
        "full",
    }
    assert not ({method for method, _ in main} & {method for method, _ in ablations})
    assert _suite_entries(suite) == main + ablations


def test_credibility_pilot_defaults_cover_full_auxiliary_schedule():
    args = build_parser().parse_args(["paper", "--pilot-only"])
    config = load_config("full")

    assert args.pilot_seeds == 3
    assert args.single_run_screening is False
    assert args.pilot_early_steps == 200000
    assert args.pilot_steps == 500000
    assert _minimum_pilot_environment_steps(config) == 480000
    assert args.pilot_early_steps < _minimum_pilot_environment_steps(config) <= args.pilot_steps


def test_single_run_screening_is_explicit_and_exactly_one_seed():
    args = build_parser().parse_args(
        ["paper", "--pilot-only", "--single-run-screening", "--pilot-seeds", "1"]
    )

    assert args.single_run_screening is True
    assert _pilot_evidence_level(
        args.pilot_seeds,
        single_run_screening=args.single_run_screening,
        available_seed_count=5,
    ) == "single_run_directional_screening"
    with pytest.raises(ValueError, match="at least three independent seeds"):
        _pilot_evidence_level(1, single_run_screening=False, available_seed_count=5)
    with pytest.raises(ValueError, match="exactly one"):
        _pilot_evidence_level(2, single_run_screening=True, available_seed_count=5)


def test_invalid_override_is_rejected():
    try:
        parse_overrides(["not-an-assignment"])
    except ValueError as exc:
        assert "key=value" in str(exc)
    else:
        raise AssertionError("invalid override was accepted")


def test_run_manifest_retains_dynamic_gpu_sessions(tmp_path, monkeypatch):
    config = load_config("full", overrides=[f"project.data_root={tmp_path}"])
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("ACPBGRL_SELECTED_GPU_INDICES", "0,3")
    monkeypatch.setenv("ACPBGRL_SELECTED_GPU_UUIDS", "GPU-0,GPU-3")
    path = tmp_path / "run_manifest.json"
    first = build_run_manifest(config, tmp_path, selected_gpus=[], micro_batch=32)
    save_run_manifest(path, first)
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("ACPBGRL_SELECTED_GPU_INDICES", "0")
    monkeypatch.setenv("ACPBGRL_SELECTED_GPU_UUIDS", "GPU-0")
    second = build_run_manifest(config, tmp_path, selected_gpus=[], micro_batch=16)
    save_run_manifest(path, second)
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert [item["world_size"] for item in payload["resource_sessions"]] == [2, 1]
    assert payload["resource_sessions"][0]["gradient_accumulation_steps"] == 2
    assert payload["resource_sessions"][1]["gradient_accumulation_steps"] == 8
    assert [item["rollout_actor_count"] for item in payload["resource_sessions"]] == [48, 32]
