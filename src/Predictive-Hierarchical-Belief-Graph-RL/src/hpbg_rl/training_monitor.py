from __future__ import annotations

import json
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class TrainingMonitor:
    def __init__(self, save_dir: str | Path, window_size: int = 10, snapshot_interval: int = 10):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir = self.save_dir / ".state"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.window_size = max(int(window_size), 1)
        self.snapshot_interval = max(int(snapshot_interval), 1)
        self.data_file = self.state_dir / "training_history.json"
        self.plot_file = self.save_dir / "training_curves.png"
        self.diagnostic_plot_file = self.save_dir / "training_diagnostics.png"
        self.convergence_plot_file = self.save_dir / "convergence_curves.png"
        self.generalization_plot_file = self.save_dir / "generalization_gap.png"
        self.loss_decomposition_plot_file = self.save_dir / "loss_decomposition.png"
        self.reward_decomposition_plot_file = self.save_dir / "reward_decomposition.png"
        self.eval_distribution_plot_file = self.save_dir / "eval_distribution.png"
        self.stability_plot_file = self.save_dir / "stability_diagnostics.png"

        self.sections = {
            "train_metrics": {"episodes": [], "history": defaultdict(list)},
            "eval_metrics": {"episodes": [], "history": defaultdict(list)},
            "system_metrics": {"episodes": [], "history": defaultdict(list)},
        }
        self._trim_pending = set(self.sections.keys())
        self._load_history()

        self.progress_plot_groups = [
            (
                "Fixed Eval Rates",
                [
                    ("eval_metrics", "explored_rate", "Val explored"),
                    ("eval_metrics", "success_rate", "Val success"),
                    ("eval_metrics", "exploration_auc", "Val AUC"),
                ],
                "Rate",
            ),
            (
                "Fixed Eval Efficiency",
                [
                    ("eval_metrics", "travel_dist", "Eval travel dist"),
                    ("eval_metrics", "steps_taken", "Eval steps"),
                    ("eval_metrics", "episode_return", "Eval return"),
                ],
                "Value",
            ),
            (
                "Train Return",
                [
                    ("train_metrics", "episode_raw_return", "Raw return"),
                    ("train_metrics", "episode_return", "Shaped return"),
                    ("train_metrics", "expert_reward_delta", "Expert delta"),
                    ("train_metrics", "reward", "Batch reward"),
                ],
                "Value",
            ),
            (
                "Train Efficiency",
                [
                    ("train_metrics", "travel_dist", "Train travel dist"),
                    ("train_metrics", "episode_steps", "Train steps"),
                ],
                "Value",
            ),
            (
                "Optimization Losses",
                [
                    ("train_metrics", "policy_loss_original", "Policy RL loss"),
                    ("train_metrics", "wavelet_weighted_loss", "Wavelet weighted"),
                    ("train_metrics", "belief_weighted_loss", "Belief weighted"),
                    ("train_metrics", "q_value_loss", "Q loss"),
                ],
                "Loss",
            ),
        ]
        self.diagnostic_plot_groups = [
            (
                "Train Rates",
                [
                    ("train_metrics", "explored_rate", "Train explored"),
                    ("train_metrics", "success_rate", "Train success"),
                ],
                "Rate",
            ),
            (
                "Value / Temperature",
                [
                    ("train_metrics", "value", "Value"),
                    ("train_metrics", "entropy", "Entropy proxy"),
                    ("train_metrics", "log_alpha", "Log alpha"),
                ],
                "Value",
            ),
            (
                "Grad Norms",
                [
                    ("train_metrics", "policy_grad_norm", "Policy grad"),
                    ("train_metrics", "q_value_grad_norm", "Q grad"),
                ],
                "Norm",
            ),
            (
                "Grad / TD Stability",
                [
                    ("train_metrics", "policy_q_grad_ratio", "Policy/Q grad ratio"),
                    ("train_metrics", "td_error_mean", "TD error mean"),
                    ("train_metrics", "td_error_std", "TD error std"),
                ],
                "Value",
            ),
            (
                "System / Buffer",
                [
                    ("system_metrics", "buffer_size", "Replay buffer size"),
                    ("system_metrics", "completed_episodes", "Completed episodes"),
                    ("train_metrics", "update_step", "Update step"),
                ],
                "Transitions",
            ),
        ]
        self.convergence_plot_groups = [
            (
                "Validation Composite",
                [
                    ("eval_metrics", "validation_score", "Validation score"),
                    ("eval_metrics", "explored_rate", "Val explored"),
                    ("eval_metrics", "success_rate", "Val success"),
                ],
                "Score / Rate",
            ),
            (
                "Validation Efficiency",
                [
                    ("eval_metrics", "normalized_exploration_efficiency", "Efficiency"),
                    ("eval_metrics", "exploration_auc", "AUC"),
                    ("eval_metrics", "travel_dist", "Travel dist"),
                ],
                "Value",
            ),
            (
                "Train Progress",
                [
                    ("train_metrics", "episode_raw_return", "Raw return"),
                    ("train_metrics", "episode_return", "Shaped return"),
                    ("train_metrics", "explored_rate", "Train explored"),
                ],
                "Value",
            ),
        ]
        self.generalization_plot_groups = [
            (
                "Explored Rate Gap",
                [
                    ("train_metrics", "explored_rate", "Train explored"),
                    ("eval_metrics", "explored_rate", "Val explored"),
                    ("eval_metrics", "generalization_gap_explored_rate", "Train - val gap"),
                ],
                "Rate",
            ),
            (
                "Success Rate Gap",
                [
                    ("train_metrics", "success_rate", "Train success"),
                    ("eval_metrics", "success_rate", "Val success"),
                    ("eval_metrics", "success_rate_std", "Val success std"),
                ],
                "Rate",
            ),
            (
                "Efficiency / AUC",
                [
                    ("eval_metrics", "normalized_exploration_efficiency", "Val efficiency"),
                    ("eval_metrics", "exploration_auc", "Val AUC"),
                    ("eval_metrics", "explored_rate_std", "Val explored std"),
                ],
                "Value",
            ),
        ]
        self.loss_decomposition_plot_groups = [
            (
                "Policy / Auxiliary Loss",
                [
                    ("train_metrics", "policy_loss_original", "Policy RL loss"),
                    ("train_metrics", "wavelet_weighted_loss", "Wavelet weighted"),
                    ("train_metrics", "belief_weighted_loss", "Belief weighted"),
                    ("train_metrics", "policy_loss", "Policy total"),
                ],
                "Loss",
            ),
            (
                "Q / TD Loss",
                [
                    ("train_metrics", "q_value_loss", "Q loss"),
                    ("train_metrics", "td_error_mean", "TD mean"),
                    ("train_metrics", "td_error_std", "TD std"),
                ],
                "Value",
            ),
            (
                "Wavelet Distillation",
                [
                    ("train_metrics", "wavelet_lf_loss", "LF loss"),
                    ("train_metrics", "wavelet_hf_loss", "HF loss"),
                    ("train_metrics", "wavelet_loss", "Total loss"),
                    ("train_metrics", "wavelet_lambda_eff", "Lambda eff"),
                ],
                "Value",
            ),
            (
                "Belief Distillation",
                [
                    ("train_metrics", "belief_loss", "Belief loss"),
                    ("train_metrics", "belief_explored_loss", "Explored target"),
                    ("train_metrics", "belief_oracle_loss", "Oracle target"),
                    ("train_metrics", "belief_potential_loss", "Potential target"),
                    ("train_metrics", "belief_lambda_eff", "Lambda eff"),
                ],
                "Value",
            ),
        ]
        self.reward_decomposition_plot_groups = [
            (
                "Episode Return",
                [
                    ("train_metrics", "episode_raw_return", "Raw return"),
                    ("train_metrics", "episode_return", "Shaped return"),
                    ("train_metrics", "expert_reward_delta", "Expert delta"),
                ],
                "Return",
            ),
            (
                "Batch Reward / Value",
                [
                    ("train_metrics", "reward", "Batch reward"),
                    ("train_metrics", "value", "Value"),
                    ("train_metrics", "entropy", "Entropy proxy"),
                ],
                "Value",
            ),
            (
                "Travel Cost",
                [
                    ("train_metrics", "travel_dist", "Train travel dist"),
                    ("train_metrics", "episode_steps", "Train steps"),
                    ("eval_metrics", "travel_dist", "Val travel dist"),
                    ("eval_metrics", "steps_taken", "Val steps"),
                ],
                "Value",
            ),
        ]
        self.eval_distribution_plot_groups = [
            (
                "Explored Distribution",
                [
                    ("eval_metrics", "explored_rate", "Mean"),
                    ("eval_metrics", "explored_rate_std", "Std"),
                    ("eval_metrics", "best_explored_rate", "Best"),
                    ("eval_metrics", "worst_explored_rate", "Worst"),
                ],
                "Rate",
            ),
            (
                "Success / Completion",
                [
                    ("eval_metrics", "success_rate", "Success mean"),
                    ("eval_metrics", "success_rate_std", "Success std"),
                    ("eval_metrics", "completion_steps", "Completion steps"),
                    ("eval_metrics", "completion_steps_std", "Completion steps std"),
                ],
                "Value",
            ),
            (
                "Travel Distribution",
                [
                    ("eval_metrics", "travel_dist", "Travel mean"),
                    ("eval_metrics", "travel_dist_std", "Travel std"),
                    ("eval_metrics", "completion_travel_dist", "Completion dist"),
                    ("eval_metrics", "completion_travel_dist_std", "Completion dist std"),
                ],
                "Distance",
            ),
        ]
        self.stability_plot_groups = [
            (
                "Gradient Balance",
                [
                    ("train_metrics", "policy_q_grad_ratio", "Policy/Q grad ratio"),
                    ("train_metrics", "policy_grad_norm", "Policy grad"),
                    ("train_metrics", "q_value_grad_norm", "Q grad"),
                ],
                "Value",
            ),
            (
                "TD Stability",
                [
                    ("train_metrics", "td_error_mean", "TD mean"),
                    ("train_metrics", "td_error_std", "TD std"),
                    ("train_metrics", "q_value_loss", "Q loss"),
                ],
                "Value",
            ),
            (
                "Entropy / Temperature",
                [
                    ("train_metrics", "entropy", "Entropy proxy"),
                    ("train_metrics", "log_alpha", "Log alpha"),
                    ("train_metrics", "alpha_loss", "Alpha loss"),
                ],
                "Value",
            ),
        ]
        self.series_colors = {
            ("eval_metrics", "explored_rate"): "#2ca02c",
            ("eval_metrics", "success_rate"): "#d62728",
            ("eval_metrics", "exploration_auc"): "#17becf",
            ("eval_metrics", "normalized_exploration_efficiency"): "#bcbd22",
            ("eval_metrics", "generalization_gap_explored_rate"): "#8c564b",
            ("eval_metrics", "travel_dist"): "#1f77b4",
            ("eval_metrics", "steps_taken"): "#ff7f0e",
            ("eval_metrics", "episode_return"): "#9467bd",
            ("train_metrics", "episode_return"): "#ff7f0e",
            ("train_metrics", "episode_raw_return"): "#2ca02c",
            ("train_metrics", "expert_reward_delta"): "#d62728",
            ("train_metrics", "reward"): "#8c564b",
            ("train_metrics", "travel_dist"): "#1f77b4",
            ("train_metrics", "episode_steps"): "#ff7f0e",
            ("train_metrics", "policy_loss"): "#1f77b4",
            ("train_metrics", "policy_loss_original"): "#1f77b4",
            ("train_metrics", "q_value_loss"): "#ff7f0e",
            ("train_metrics", "alpha_loss"): "#2ca02c",
            ("train_metrics", "wavelet_loss"): "#9467bd",
            ("train_metrics", "wavelet_weighted_loss"): "#9467bd",
            ("train_metrics", "belief_weighted_loss"): "#e377c2",
            ("train_metrics", "explored_rate"): "#2ca02c",
            ("train_metrics", "success_rate"): "#d62728",
            ("train_metrics", "value"): "#1f77b4",
            ("train_metrics", "entropy"): "#ff7f0e",
            ("train_metrics", "log_alpha"): "#2ca02c",
            ("train_metrics", "policy_grad_norm"): "#1f77b4",
            ("train_metrics", "q_value_grad_norm"): "#ff7f0e",
            ("train_metrics", "policy_q_grad_ratio"): "#d62728",
            ("train_metrics", "td_error_mean"): "#8c564b",
            ("train_metrics", "td_error_std"): "#e377c2",
            ("train_metrics", "update_step"): "#17becf",
            ("train_metrics", "wavelet_lf_loss"): "#9467bd",
            ("train_metrics", "wavelet_hf_loss"): "#8c564b",
            ("train_metrics", "wavelet_lambda_eff"): "#bcbd22",
            ("train_metrics", "belief_loss"): "#e377c2",
            ("train_metrics", "belief_explored_loss"): "#2ca02c",
            ("train_metrics", "belief_oracle_loss"): "#d62728",
            ("train_metrics", "belief_potential_loss"): "#17becf",
            ("train_metrics", "belief_lambda_eff"): "#bcbd22",
            ("eval_metrics", "validation_score"): "#1f77b4",
            ("eval_metrics", "validation_score_best"): "#2ca02c",
            ("eval_metrics", "validation_score_delta"): "#ff7f0e",
            ("eval_metrics", "validation_plateau"): "#d62728",
            ("eval_metrics", "explored_rate_std"): "#98df8a",
            ("eval_metrics", "best_explored_rate"): "#2ca02c",
            ("eval_metrics", "worst_explored_rate"): "#d62728",
            ("eval_metrics", "success_rate_std"): "#ff9896",
            ("eval_metrics", "travel_dist_std"): "#aec7e8",
            ("eval_metrics", "completion_steps"): "#9467bd",
            ("eval_metrics", "completion_steps_std"): "#c5b0d5",
            ("eval_metrics", "completion_travel_dist"): "#8c564b",
            ("eval_metrics", "completion_travel_dist_std"): "#c49c94",
            ("system_metrics", "buffer_size"): "#1f77b4",
            ("system_metrics", "completed_episodes"): "#17becf",
        }
        if any(section["episodes"] for section in self.sections.values()):
            self._save_data()
            self._save_plots()

    def update_train(self, episode: int, metrics: dict[str, object]) -> None:
        self._update_section("train_metrics", episode, metrics)

    def update_eval(self, episode: int, metrics: dict[str, object]) -> None:
        self._update_section("eval_metrics", episode, metrics)

    def update_system(self, episode: int, metrics: dict[str, object]) -> None:
        self._update_section("system_metrics", episode, metrics)

    def _load_history(self) -> None:
        if not self.data_file.exists():
            return
        try:
            data = json.loads(self.data_file.read_text())
        except (OSError, json.JSONDecodeError):
            return

        sections = data.get("sections", {})
        for section_name, section_data in sections.items():
            if section_name not in self.sections:
                continue
            self.sections[section_name]["episodes"] = list(section_data.get("episodes", []))
            history = defaultdict(list)
            for key, values in section_data.get("history", {}).items():
                history[key] = list(values)
            self.sections[section_name]["history"] = history
            self._normalize_section(section_name)

    def _normalize_value(self, value: object) -> float | int | bool | None:
        if value is None:
            return None
        if hasattr(value, "item"):
            try:
                value = value.item()
            except (TypeError, ValueError):
                pass
        if isinstance(value, (np.bool_, bool)):
            return bool(value)
        if isinstance(value, (np.integer, int)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            return float(value)
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _trim_section(self, section_name: str, start_episode: int) -> None:
        episodes = self.sections[section_name]["episodes"]
        trim_idx = len(episodes)
        for idx, episode in enumerate(episodes):
            if episode >= start_episode:
                trim_idx = idx
                break
        if trim_idx == len(episodes):
            return
        self.sections[section_name]["episodes"] = episodes[:trim_idx]
        history = self.sections[section_name]["history"]
        for key in list(history.keys()):
            history[key] = history[key][:trim_idx]

    def _normalize_section(self, section_name: str) -> None:
        section = self.sections[section_name]
        episodes = [int(episode) for episode in section["episodes"]]
        history = section["history"]
        for key in list(history.keys()):
            values = list(history[key])
            if len(values) < len(episodes):
                values.extend([None] * (len(episodes) - len(values)))
            elif len(values) > len(episodes):
                values = values[: len(episodes)]
            history[key] = values

        order = sorted(range(len(episodes)), key=lambda idx: (episodes[idx], idx))
        if order != list(range(len(episodes))):
            section["episodes"] = [episodes[idx] for idx in order]
            for key in list(history.keys()):
                history[key] = [history[key][idx] for idx in order]
        else:
            section["episodes"] = episodes

    def _update_section(self, section_name: str, episode: int, metrics: dict[str, object]) -> None:
        episode = int(episode)
        normalized_metrics = {}
        for key, value in metrics.items():
            normalized = self._normalize_value(value)
            if normalized is None:
                continue
            normalized_metrics[key] = normalized
        if not normalized_metrics:
            return

        if section_name in self._trim_pending:
            self._trim_section(section_name, episode)
            self._trim_pending.remove(section_name)

        section = self.sections[section_name]
        history = section["history"]
        for key in normalized_metrics:
            if key not in history:
                history[key] = [None] * len(section["episodes"])

        insert_idx = bisect_left(section["episodes"], episode)
        if insert_idx == len(section["episodes"]) or section["episodes"][insert_idx] != episode:
            section["episodes"].insert(insert_idx, episode)
            for key in list(history.keys()):
                history[key].insert(insert_idx, None)

        for key, value in normalized_metrics.items():
            history[key][insert_idx] = value

        self._sync_derived_metrics(section_name)
        self._save_data()
        self._save_plots()

    def _sync_derived_metrics(self, section_name: str) -> None:
        if section_name != "eval_metrics":
            return
        history = self.sections[section_name]["history"]
        validation_scores = history.get("validation_score")
        if not validation_scores:
            return

        best_scores = []
        deltas = []
        plateau = []
        best_value = None
        previous_value = None
        stagnant_count = 0
        for value in validation_scores:
            if value is None:
                best_scores.append(best_value)
                deltas.append(None)
                plateau.append(None)
                continue
            value = float(value)
            if best_value is None or value > best_value:
                best_value = value
                stagnant_count = 0
            else:
                stagnant_count += 1
            if previous_value is None:
                deltas.append(0.0)
            else:
                deltas.append(value - previous_value)
            best_scores.append(best_value)
            plateau.append(1.0 if stagnant_count >= max(self.window_size, 1) else 0.0)
            previous_value = value

        history["validation_score_best"] = best_scores
        history["validation_score_delta"] = deltas
        history["validation_plateau"] = plateau

    def _smooth(self, values: list[float | int | bool]) -> list[float]:
        if not values:
            return []
        smoothed = []
        for idx in range(len(values)):
            start = max(0, idx - self.window_size + 1)
            smoothed.append(float(np.mean(values[start : idx + 1])))
        return smoothed

    def _rolling_std(self, values: list[float | int | bool]) -> list[float]:
        if not values:
            return []
        spread = []
        for idx in range(len(values)):
            start = max(0, idx - self.window_size + 1)
            spread.append(float(np.std(values[start : idx + 1])))
        return spread

    def _series_points(self, section_name: str, key: str) -> tuple[list[int], list[float]]:
        section = self.sections[section_name]
        episodes = []
        values = []
        for episode, value in zip(section["episodes"], section["history"].get(key, [])):
            if value is None:
                continue
            if isinstance(value, (np.floating, float)) and np.isnan(float(value)):
                continue
            episodes.append(int(episode))
            values.append(float(value))
        return episodes, values

    def _latest_point(self, section_name: str, key: str) -> tuple[int, float] | None:
        episodes, values = self._series_points(section_name, key)
        if not values:
            return None
        return episodes[-1], values[-1]

    def _best_point(self, section_name: str, key: str, mode: str) -> tuple[int, float] | None:
        episodes, values = self._series_points(section_name, key)
        if not values:
            return None
        if mode == "min":
            index = int(np.argmin(values))
        else:
            index = int(np.argmax(values))
        return episodes[index], values[index]

    def _detect_resume_markers(self) -> list[int]:
        episodes, buffer_sizes = self._series_points("system_metrics", "buffer_size")
        markers = []
        for idx in range(1, len(buffer_sizes)):
            previous = buffer_sizes[idx - 1]
            current = buffer_sizes[idx]
            if previous < 1_000:
                continue
            drop_ratio = current / max(previous, 1.0)
            drop_amount = previous - current
            if drop_ratio <= 0.65 and drop_amount >= max(2_000.0, 0.2 * previous):
                episode = episodes[idx]
                if markers and episode - markers[-1] <= self.snapshot_interval:
                    markers[-1] = episode
                else:
                    markers.append(episode)
        return markers

    def _add_resume_markers(self, ax: plt.Axes, resume_markers: list[int]) -> None:
        for episode in resume_markers:
            ax.axvline(episode, color="#7f7f7f", linestyle="--", linewidth=1.0, alpha=0.3)

    def _plot_series_group(
        self,
        ax: plt.Axes,
        title: str,
        series_list: list[tuple[str, str, str]],
        ylabel: str,
        resume_markers: list[int],
        empty_message: str,
    ) -> None:
        has_data = False
        for section_name, key, label in series_list:
            episodes, values = self._series_points(section_name, key)
            if not values:
                continue
            color = self.series_colors.get((section_name, key))
            smoothed = np.asarray(self._smooth(values), dtype=float)
            spread = np.asarray(self._rolling_std(values), dtype=float)
            values_array = np.asarray(values, dtype=float)

            ax.plot(episodes, values_array, color=color, linewidth=1.0, alpha=0.16)
            if len(spread) > 1 and np.any(spread > 0):
                ax.fill_between(
                    episodes,
                    smoothed - spread,
                    smoothed + spread,
                    color=color,
                    alpha=0.10,
                    linewidth=0,
                )
            ax.plot(episodes, smoothed, color=color, linewidth=2.2, label=label)
            has_data = True

        self._add_resume_markers(ax, resume_markers)
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if has_data:
            ax.legend(fontsize=8, loc="best")
        else:
            ax.text(0.5, 0.5, empty_message, ha="center", va="center", fontsize=11, color="#666666")

    def _format_summary_value(self, point: tuple[int, float] | None, precision: int = 3) -> str:
        if point is None:
            return "n/a"
        episode, value = point
        return f"{value:.{precision}f} @ ep {episode}"

    def _render_summary_panel(self, ax: plt.Axes, resume_markers: list[int]) -> None:
        ax.set_title("Run Summary")
        ax.axis("off")

        train_return = self._latest_point("train_metrics", "episode_return")
        train_travel = self._latest_point("train_metrics", "travel_dist")
        train_steps = self._latest_point("train_metrics", "episode_steps")
        eval_success = self._latest_point("eval_metrics", "success_rate")
        eval_explored = self._latest_point("eval_metrics", "explored_rate")
        eval_travel = self._latest_point("eval_metrics", "travel_dist")
        eval_steps = self._latest_point("eval_metrics", "steps_taken")
        best_eval_travel = self._best_point("eval_metrics", "travel_dist", mode="min")
        best_eval_return = self._best_point("eval_metrics", "episode_return", mode="max")
        latest_buffer = self._latest_point("system_metrics", "buffer_size")

        resume_text = "none"
        if resume_markers:
            preview = ", ".join(str(episode) for episode in resume_markers[:4])
            if len(resume_markers) > 4:
                preview = f"{preview}, ..."
            resume_text = preview

        lines = [
            "Main progress",
            f"Latest train return: {self._format_summary_value(train_return)}",
            f"Latest train dist:   {self._format_summary_value(train_travel, precision=2)}",
            f"Latest train steps:  {self._format_summary_value(train_steps, precision=1)}",
            "",
            "Fixed eval",
            f"Latest eval success: {self._format_summary_value(eval_success)}",
            f"Latest eval explore: {self._format_summary_value(eval_explored)}",
            f"Latest eval dist:    {self._format_summary_value(eval_travel, precision=2)}",
            f"Latest eval steps:   {self._format_summary_value(eval_steps, precision=1)}",
            f"Best eval dist:      {self._format_summary_value(best_eval_travel, precision=2)}",
            f"Best eval return:    {self._format_summary_value(best_eval_return)}",
            "",
            "Run state",
            f"Latest buffer size:  {self._format_summary_value(latest_buffer, precision=0)}",
            f"Resume markers:      {len(resume_markers)}",
            f"Resume episodes:     {resume_text}",
        ]
        ax.text(
            0.02,
            0.98,
            "\n".join(lines),
            ha="left",
            va="top",
            fontsize=10.5,
            family="monospace",
        )

    def _save_data(self) -> None:
        payload = {"sections": {}}
        for section_name, section in self.sections.items():
            payload["sections"][section_name] = {
                "episodes": list(section["episodes"]),
                "history": {key: list(values) for key, values in section["history"].items()},
            }
        self.data_file.write_text(json.dumps(payload, indent=2))

    def _save_group_figure(
        self,
        plot_groups: list[tuple[str, list[tuple[str, str, str]], str]],
        output_file: Path,
        title: str,
        resume_markers: list[int],
        columns: int = 2,
        figsize_per_row: float = 4.2,
    ) -> None:
        if not plot_groups:
            return
        columns = max(int(columns), 1)
        rows = int(np.ceil(len(plot_groups) / columns))
        fig, axes = plt.subplots(rows, columns, figsize=(7 * columns, figsize_per_row * rows))
        fig.suptitle(title, fontsize=16, fontweight="bold")
        axes_flat = np.asarray(axes, dtype=object).reshape(-1)

        for idx, (group_title, series_list, ylabel) in enumerate(plot_groups):
            self._plot_series_group(
                axes_flat[idx],
                group_title,
                series_list,
                ylabel,
                resume_markers,
                empty_message="No data yet",
            )
        for idx in range(len(plot_groups), len(axes_flat)):
            axes_flat[idx].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _save_progress_plot(self, resume_markers: list[int]) -> None:
        progress_fig, progress_axes = plt.subplots(3, 2, figsize=(14, 13))
        progress_fig.suptitle("Training Progress", fontsize=16, fontweight="bold")
        progress_axes_flat = np.asarray(progress_axes, dtype=object).reshape(-1)

        for idx, (title, series_list, ylabel) in enumerate(self.progress_plot_groups):
            self._plot_series_group(
                progress_axes_flat[idx],
                title,
                series_list,
                ylabel,
                resume_markers,
                empty_message="No data yet",
            )
        self._render_summary_panel(progress_axes_flat[-1], resume_markers)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        progress_fig.savefig(self.plot_file, dpi=150, bbox_inches="tight")
        plt.close(progress_fig)

    def _save_plots(self) -> None:
        resume_markers = self._detect_resume_markers()
        self._save_progress_plot(resume_markers)
        self._save_group_figure(
            self.diagnostic_plot_groups,
            self.diagnostic_plot_file,
            "Training Diagnostics",
            resume_markers,
        )
        self._save_group_figure(
            self.convergence_plot_groups,
            self.convergence_plot_file,
            "Convergence Curves",
            resume_markers,
        )
        self._save_group_figure(
            self.generalization_plot_groups,
            self.generalization_plot_file,
            "Generalization Gap",
            resume_markers,
        )
        self._save_group_figure(
            self.loss_decomposition_plot_groups,
            self.loss_decomposition_plot_file,
            "Loss Decomposition",
            resume_markers,
        )
        self._save_group_figure(
            self.reward_decomposition_plot_groups,
            self.reward_decomposition_plot_file,
            "Reward Decomposition",
            resume_markers,
        )
        self._save_group_figure(
            self.eval_distribution_plot_groups,
            self.eval_distribution_plot_file,
            "Evaluation Distribution",
            resume_markers,
        )
        self._save_group_figure(
            self.stability_plot_groups,
            self.stability_plot_file,
            "Stability Diagnostics",
            resume_markers,
        )
