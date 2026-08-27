from pathlib import Path

import pytest
import yaml

from ac_pbgrl.cli import (
    _minimum_pilot_environment_steps,
    _pilot_evidence_level,
    _suite_entries,
    build_parser,
    command_paper,
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


def test_single_run_screening_plans_only_seed_zero_and_descriptive_figures(
    tmp_path, monkeypatch
):
    from ac_pbgrl import cli

    config = load_config("full", overrides=[f"project.data_root={tmp_path}"])
    teacher = tmp_path / "teachers/ariadne_pi" / f"step_{config.teacher.checkpoint_step}.pt"
    teacher.parent.mkdir(parents=True)
    teacher.touch()
    (tmp_path / "map_splits.json").write_text("{}", encoding="utf-8")
    calls = []

    def fake_call(command, **kwargs):
        arguments = list(command[1:])
        calls.append(arguments)
        if arguments[0] == "calibrate":
            artifact = tmp_path / "calibration/full/seed_0.json"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text("{}", encoding="utf-8")
        return 0

    monkeypatch.setattr(cli, "load_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(cli.subprocess, "call", fake_call)
    args = build_parser().parse_args(
        [
            "paper",
            "--pilot-only",
            "--single-run-screening",
            "--pilot-seeds",
            "1",
            "--pilot-early-steps",
            "200000",
            "--pilot-steps",
            "500000",
            "--gpus",
            "cpu",
        ]
    )

    assert command_paper(args) == 0
    assert all("project.seed=1" not in call for call in calls)
    assert all("seed_1" not in item for call in calls for item in call)

    supervise = [call for call in calls if call[0] == "supervise"]
    stages = [
        (
            next(item for item in call if item.startswith("project.run_name=")),
            next(item for item in call if item.startswith("train.max_environment_steps=")),
        )
        for call in supervise
    ]
    assert stages == [
        ("project.run_name=ariadne_pi/seed_0", "train.max_environment_steps=200000"),
        ("project.run_name=full/seed_0", "train.max_environment_steps=30000"),
        ("project.run_name=full/seed_0", "train.max_environment_steps=200000"),
        ("project.run_name=ariadne_pi/seed_0", "train.max_environment_steps=500000"),
        ("project.run_name=full/seed_0", "train.max_environment_steps=500000"),
    ]

    evaluations = [call for call in calls if call[0] == "evaluate"]
    assert len(evaluations) == 4
    assert all(call[call.index("--seeds") + 1] == "0" for call in evaluations)
    figures = [call for call in calls if call[0] == "figures"]
    assert len(figures) == 2
    assert all(
        call[call.index("--evidence-level") + 1] == "single_run_directional_screening"
        for call in figures
    )
    assert not any(call[0] == "ablate" for call in calls)

    for step, kind in ((200000, "single_run_early_diagnostic"), (500000, "single_run_screening")):
        manifest = yaml.safe_load((tmp_path / f"pilot/step_{step}/manifest.json").read_text())
        assert manifest["kind"] == kind
        assert manifest["evidence_level"] == "single_run_directional_screening"
        assert manifest["seeds"] == [0]
        assert manifest["methods"] == ["ariadne_pi", "full"]
        assert manifest["statistical_claims_supported"] is False


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
