import json
from pathlib import Path

import numpy as np
import pytest


def test_paper_figure_pipeline_uses_paired_artifacts(tmp_path: Path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    from ac_pbgrl.evaluation.figures import generate_paper_figures

    runs = tmp_path / "runs"
    methods = ("ariadne_pi", "full")
    for method_index, method in enumerate(methods):
        evaluation = runs / method / "seed_0" / "evaluation"
        evaluation.mkdir(parents=True)
        rows = []
        for map_index in range(3):
            rows.append(
                {
                    "method": method,
                    "seed": 0,
                    "split": "iid_test",
                    "map_id": f"map-{map_index}",
                    "episode/completion_distance": 100.0 - 5 * method_index + map_index,
                    "episode/coverage_95_distance": 90.0 - 4 * method_index + map_index,
                    "episode/explored_rate": 0.95 + 0.01 * method_index,
                    "episode/success": 1.0,
                    "episode/backtracks": 3 - method_index,
                    "episode/repeated_edges": 2 - method_index,
                    "episode/direction_switches": 4 - method_index,
                    "system/planning_latency_mean_ms": 8.0 + method_index,
                    "graph/nodes_mean": 180 + 10 * method_index,
                    "potential/spearman": 0.6 + 0.1 * method_index,
                    "potential/kendall": 0.5 + 0.1 * method_index,
                    "potential/pairwise_accuracy": 0.7 + 0.1 * method_index,
                }
            )
        pd.DataFrame(rows).to_csv(evaluation / "episodes.csv", index=False)
        with (evaluation / "paths.jsonl").open("w", encoding="utf-8") as handle:
            for map_index in range(3):
                handle.write(
                    json.dumps(
                        {
                            "method": method,
                            "seed": 0,
                            "split": "iid_test",
                            "map_id": f"map-{map_index}",
                            "xy": [[0, 0], [1 + method_index, map_index]],
                        }
                    )
                    + "\n"
                )
        values = np.linspace(0.1, 1.0, 20)
        np.savez_compressed(
            evaluation / "potential_samples.npz",
            mean=values + 0.02 * method_index,
            variance=np.full(20, 0.1 + 0.01 * method_index),
            target=values,
            group=np.repeat(np.arange(10), 2),
        )

    output = tmp_path / "figures"
    generated = generate_paper_figures(runs, output)
    names = {path.name for path in generated}
    assert {
        "completion_distance.pdf",
        "ablation_forest.pdf",
        "uncertainty_calibration.pdf",
        "potential_ranking.pdf",
        "path_behavior.pdf",
        "representative_paths.pdf",
    }.issubset(names)
    effects = pd.read_csv(output / "paired_effects.csv")
    assert set(effects["reference"]) == {"ariadne_pi"}
    assert effects["n"].min() > 0
    manifest = json.loads((output / "figure_manifest.json").read_text(encoding="utf-8"))
    assert manifest["paired_keys"] == ["seed", "map_id", "split"]
    assert manifest["training_seed_count"] == 1
    assert manifest["statistical_claims_supported"] is False

    screening_output = tmp_path / "screening_figures"
    generate_paper_figures(
        runs,
        screening_output,
        evidence_level="single_run_directional_screening",
    )
    screening_effects = pd.read_csv(screening_output / "paired_effects.csv")
    assert "wilcoxon_p" not in screening_effects
    assert "holm_p" not in screening_effects
    screening_manifest = json.loads(
        (screening_output / "figure_manifest.json").read_text(encoding="utf-8")
    )
    assert screening_manifest["evidence_level"] == "single_run_directional_screening"
    assert screening_manifest["inferential_statistics_included"] is False
    assert screening_manifest["multiple_testing"] == "none"
    assert screening_manifest["paired_uncertainty_scope"] == (
        "map_variation_conditional_on_one_training_seed"
    )
