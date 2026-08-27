from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .statistics import holm_adjust, paired_comparison


def generate_paper_figures(
    runs_root: str | Path,
    output_dir: str | Path,
    *,
    evidence_level: str = "multi_seed_formal",
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    runs_root = Path(runs_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_files = list(runs_root.rglob("evaluation/episodes.csv")) + list(runs_root.rglob("episodes.csv"))
    frames = []
    for path in evaluation_files:
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if "method" in frame:
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"no evaluation CSV files found below {runs_root}")
    data = pd.concat(frames, ignore_index=True).drop_duplicates()
    supported_evidence_levels = {
        "multi_seed_formal",
        "multi_seed_credibility_pilot",
        "single_run_directional_screening",
    }
    if evidence_level not in supported_evidence_levels:
        raise ValueError(f"unsupported evidence level: {evidence_level}")
    single_run_screening = evidence_level == "single_run_directional_screening"
    training_seed_count = int(data["seed"].nunique()) if "seed" in data else 0
    sns.set_theme(style="whitegrid", context="paper")
    generated = []

    # Learning curves use environment transitions/update steps rather than wall
    # time, so 1/2/4-GPU runs remain directly comparable.
    training_frames = []
    for path in runs_root.rglob("metrics/train.jsonl"):
        try:
            frame = pd.read_json(path, lines=True)
        except (ValueError, OSError):
            continue
        config_path = path.parents[1] / "config_resolved.yaml"
        method = path.parents[2].name
        if config_path.is_file():
            try:
                import yaml

                method = str(yaml.safe_load(config_path.read_text(encoding="utf-8"))["project"]["experiment"])
            except Exception:
                pass
        frame["method"] = method
        training_frames.append(frame)
    if training_frames:
        training = pd.concat(training_frames, ignore_index=True)
        for metric, label, filename in (
            ("episode/explored_rate", "Explored ratio", "learning_explored_rate.pdf"),
            ("episode/return", "Episode return", "learning_return.pdf"),
            ("loss/policy_sac", "SAC actor loss", "learning_actor_loss.pdf"),
        ):
            if metric not in training or training[metric].notna().sum() == 0:
                continue
            x_column = (
                "train/environment_steps"
                if "train/environment_steps" in training
                and training.loc[training[metric].notna(), "train/environment_steps"].notna().any()
                else "step"
            )
            subset = training[["method", x_column, metric]].dropna().sort_values(x_column)
            subset["smoothed"] = subset.groupby("method")[metric].transform(
                lambda values: values.rolling(25, min_periods=1).mean()
            )
            figure, axis = plt.subplots(figsize=(6.4, 3.8))
            sns.lineplot(data=subset, x=x_column, y="smoothed", hue="method", errorbar=None, ax=axis)
            axis.set_xlabel("Environment transitions" if x_column != "step" else "Recorded step")
            axis.set_ylabel(label)
            figure.tight_layout()
            path = output_dir / filename
            figure.savefig(path, bbox_inches="tight")
            plt.close(figure)
            generated.append(path)

    metric_pairs = [
        ("episode/completion_distance", "Completion travel distance (successful runs)", "completion_distance.pdf"),
        ("episode/coverage_95_distance", "Distance to 95% coverage", "coverage_95_distance.pdf"),
        ("episode/explored_rate", "Explored ratio", "explored_rate.pdf"),
        ("episode/success", "Success rate", "success_rate.pdf"),
        ("system/planning_latency_mean_ms", "Planning latency (ms)", "planning_latency.pdf"),
    ]
    for metric, label, filename in metric_pairs:
        if metric not in data:
            continue
        figure, axis = plt.subplots(figsize=(6.4, 3.8))
        sns.pointplot(data=data, x="method", y=metric, errorbar=("ci", 95), capsize=0.15, ax=axis)
        axis.set_xlabel("")
        axis.set_ylabel(label)
        axis.tick_params(axis="x", rotation=25)
        figure.tight_layout()
        path = output_dir / filename
        figure.savefig(path, bbox_inches="tight")
        plt.close(figure)
        generated.append(path)

    if {"episode/completion_distance", "system/planning_latency_mean_ms"}.issubset(data.columns):
        figure, axis = plt.subplots(figsize=(5.5, 4.2))
        summary = data.groupby("method", as_index=False)[
            ["episode/completion_distance", "system/planning_latency_mean_ms"]
        ].mean()
        sns.scatterplot(
            data=summary,
            x="system/planning_latency_mean_ms",
            y="episode/completion_distance",
            hue="method",
            s=90,
            ax=axis,
        )
        axis.set_xlabel("Planning latency (ms)")
        axis.set_ylabel("Completion distance")
        figure.tight_layout()
        path = output_dir / "distance_latency_pareto.pdf"
        figure.savefig(path, bbox_inches="tight")
        plt.close(figure)
        generated.append(path)

    # Paired effect sizes and corrected significance tests use identical
    # seed/map/split keys, preventing an unpaired map-composition advantage.
    reference = "ariadne_pi"
    if reference in set(data["method"]):
        comparisons = []
        paired_keys = [key for key in ("seed", "map_id", "split") if key in data]
        effect_metrics = [
            metric
            for metric in ("episode/completion_distance", "episode/success", "episode/explored_rate")
            if metric in data
        ]
        baseline = data[data.method == reference]
        for method in sorted(set(data.method) - {reference}):
            candidate = data[data.method == method]
            joined = candidate.merge(baseline, on=paired_keys, suffixes=("_method", "_baseline"))
            for metric in effect_metrics:
                result = paired_comparison(
                    joined[f"{metric}_method"],
                    joined[f"{metric}_baseline"],
                )
                if result["n"]:
                    if single_run_screening:
                        result.pop("wilcoxon_p", None)
                    comparisons.append({"method": method, "reference": reference, "metric": metric, **result})
        if comparisons:
            if not single_run_screening:
                adjusted = holm_adjust([item["wilcoxon_p"] for item in comparisons])
                for item, value in zip(comparisons, adjusted):
                    item["holm_p"] = value
            effect_frame = pd.DataFrame(comparisons)
            effect_frame.to_csv(output_dir / "paired_effects.csv", index=False)
            figure, axis = plt.subplots(figsize=(7.2, max(3.8, 0.34 * len(effect_frame))))
            labels = [f"{row.method}: {row.metric.split('/')[-1]}" for row in effect_frame.itertuples()]
            positions = np.arange(len(effect_frame))
            axis.errorbar(
                effect_frame["difference"],
                positions,
                xerr=np.vstack(
                    (
                        effect_frame["difference"] - effect_frame["ci_low"],
                        effect_frame["ci_high"] - effect_frame["difference"],
                    )
                ),
                fmt="o",
                capsize=3,
            )
            axis.axvline(0.0, color="black", linewidth=1)
            axis.set_yticks(positions, labels)
            axis.set_xlabel(
                f"Descriptive paired-map difference vs {reference} (conditional on one training seed)"
                if single_run_screening
                else f"Paired difference vs {reference} (95% bootstrap CI)"
            )
            figure.tight_layout()
            path = output_dir / "ablation_forest.pdf"
            figure.savefig(path, bbox_inches="tight")
            plt.close(figure)
            generated.append(path)

    if {"graph/nodes_mean", "system/planning_latency_mean_ms"}.issubset(data.columns):
        figure, axis = plt.subplots(figsize=(5.8, 4.2))
        sns.scatterplot(
            data=data,
            x="graph/nodes_mean",
            y="system/planning_latency_mean_ms",
            hue="method",
            alpha=0.65,
            ax=axis,
        )
        axis.set_xlabel("Graph nodes")
        axis.set_ylabel("Planning latency (ms)")
        figure.tight_layout()
        path = output_dir / "latency_graph_scale.pdf"
        figure.savefig(path, bbox_inches="tight")
        plt.close(figure)
        generated.append(path)

    sample_frames = []
    for path in runs_root.rglob("potential_samples.npz"):
        try:
            values = np.load(path)
            sibling = path.parent / "episodes.csv"
            method = pd.read_csv(sibling, nrows=1)["method"].iloc[0] if sibling.is_file() else path.parents[2].name
            sample_frames.append(
                pd.DataFrame(
                    {
                        "method": method,
                        "mean": values["mean"],
                        "variance": values["variance"],
                        "target": values["target"],
                    }
                )
            )
        except Exception:
            continue
    if sample_frames:
        samples = pd.concat(sample_frames, ignore_index=True)
        calibration_rows = []
        for method, frame in samples.groupby("method"):
            frame = frame[np.isfinite(frame[["mean", "variance", "target"]]).all(axis=1)].copy()
            if len(frame) < 5:
                continue
            if frame["variance"].nunique() < 2:
                frame["bin"] = "all"
            else:
                frame["bin"] = pd.qcut(
                    frame["variance"], q=min(10, len(frame)), duplicates="drop"
                )
            for _, group in frame.groupby("bin", observed=True):
                calibration_rows.append(
                    {
                        "method": method,
                        "predicted_std": float(np.sqrt(group["variance"].mean())),
                        "empirical_rmse": float(np.sqrt(np.mean((group["target"] - group["mean"]) ** 2))),
                        "count": len(group),
                    }
                )
        if calibration_rows:
            calibration = pd.DataFrame(calibration_rows)
            calibration.to_csv(output_dir / "calibration_bins.csv", index=False)
            figure, axis = plt.subplots(figsize=(5.2, 4.4))
            sns.lineplot(
                data=calibration,
                x="predicted_std",
                y="empirical_rmse",
                hue="method",
                marker="o",
                ax=axis,
            )
            limit = float(max(calibration.predicted_std.max(), calibration.empirical_rmse.max()))
            axis.plot([0, limit], [0, limit], "k--", linewidth=1, label="ideal")
            axis.set_xlabel("Predicted standard deviation")
            axis.set_ylabel("Empirical RMSE")
            figure.tight_layout()
            path = output_dir / "uncertainty_calibration.pdf"
            figure.savefig(path, bbox_inches="tight")
            plt.close(figure)
            generated.append(path)
        plot_samples = samples.sample(min(5000, len(samples)), random_state=4777)
        figure, axis = plt.subplots(figsize=(5.8, 4.5))
        sns.scatterplot(data=plot_samples, x="target", y="mean", hue="method", alpha=0.3, s=16, ax=axis)
        axis.set_xlabel("GT rollout future gain")
        axis.set_ylabel("Predicted action potential")
        figure.tight_layout()
        path = output_dir / "potential_ranking_scatter.pdf"
        figure.savefig(path, bbox_inches="tight")
        plt.close(figure)
        generated.append(path)

    ranking_columns = [
        value
        for value in ("potential/spearman", "potential/kendall", "potential/pairwise_accuracy")
        if value in data.columns
    ]
    if ranking_columns:
        ranking = data[["method", *ranking_columns]].melt(
            id_vars="method", var_name="metric", value_name="score"
        ).dropna()
        if len(ranking):
            figure, axis = plt.subplots(figsize=(6.5, 4.0))
            sns.pointplot(data=ranking, x="method", y="score", hue="metric", errorbar=("ci", 95), ax=axis)
            axis.set_xlabel("")
            axis.set_ylabel("Candidate ranking quality")
            axis.tick_params(axis="x", rotation=25)
            figure.tight_layout()
            path = output_dir / "potential_ranking.pdf"
            figure.savefig(path, bbox_inches="tight")
            plt.close(figure)
            generated.append(path)

    behavior_columns = [
        value
        for value in (
            "episode/backtracks",
            "episode/repeated_edges",
            "episode/direction_switches",
        )
        if value in data.columns
    ]
    if behavior_columns:
        behavior = data[["method", *behavior_columns]].melt(
            id_vars="method", var_name="metric", value_name="count"
        ).dropna()
        if len(behavior):
            figure, axis = plt.subplots(figsize=(6.8, 4.0))
            sns.pointplot(data=behavior, x="method", y="count", hue="metric", errorbar=("ci", 95), ax=axis)
            axis.set_xlabel("")
            axis.set_ylabel("Per-episode count")
            axis.tick_params(axis="x", rotation=25)
            figure.tight_layout()
            path = output_dir / "path_behavior.pdf"
            figure.savefig(path, bbox_inches="tight")
            plt.close(figure)
            generated.append(path)

    path_records = []
    for path in runs_root.rglob("paths.jsonl"):
        try:
            path_records.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
        except Exception:
            continue
    if path_records:
        figure, axis = plt.subplots(figsize=(6.0, 5.0))
        seen = set()
        coverage = {}
        for record in path_records:
            key = (record.get("map_id"), record.get("seed"), record.get("split"))
            coverage.setdefault(key, set()).add(record.get("method"))
        representative = max(coverage, key=lambda key: (len(coverage[key]), key == ("synthetic", 0, "iid_test")))
        for record in path_records:
            key = (record.get("map_id"), record.get("seed"), record.get("split"))
            if key != representative or record["method"] in seen or not record.get("xy"):
                continue
            seen.add(record["method"])
            xy = np.asarray(record["xy"], dtype=float)
            axis.plot(xy[:, 0], xy[:, 1], marker=".", linewidth=1.2, label=record["method"])
        axis.set_aspect("equal", adjustable="datalim")
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.set_title(f"Paired map: {representative[0]}, seed {representative[1]}")
        axis.legend()
        figure.tight_layout()
        path = output_dir / "representative_paths.pdf"
        figure.savefig(path, bbox_inches="tight")
        plt.close(figure)
        generated.append(path)

    (output_dir / "figure_manifest.json").write_text(
        json.dumps(
            {
                "files": [str(path) for path in generated],
                "paired_keys": ["seed", "map_id", "split"],
                "reference": "ariadne_pi",
                "confidence": 0.95,
                "evidence_level": evidence_level,
                "training_seed_count": training_seed_count,
                "inferential_statistics_included": not single_run_screening,
                "statistical_claims_supported": (
                    evidence_level == "multi_seed_formal" and training_seed_count >= 3
                ),
                "paired_uncertainty_scope": (
                    "map_variation_conditional_on_one_training_seed"
                    if single_run_screening
                    else "pooled_seed_map_pairs"
                ),
                "multiple_testing": "none" if single_run_screening else "Holm",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return generated
