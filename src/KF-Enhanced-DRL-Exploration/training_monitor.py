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
    def __init__(
        self, save_dir: str | Path, window_size: int = 10, snapshot_interval: int = 10
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir = self.save_dir / ".state"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.window_size = max(int(window_size), 1)
        self.snapshot_interval = max(int(snapshot_interval), 1)
        self.data_file = self.state_dir / "training_history.json"
        self.plot_file = self.save_dir / "training_curves.png"
        self.diagnostic_plot_file = self.save_dir / "training_diagnostics.png"

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
                    ("eval_metrics", "explored_rate", "Eval explored"),
                    ("eval_metrics", "success_rate", "Eval success"),
                ],
                "Rate",
            ),
            (
                "Fixed Eval Efficiency",
                [
                    ("eval_metrics", "distance_efficiency", "Eval dist efficiency"),
                    ("eval_metrics", "time_efficiency", "Eval time efficiency"),
                    ("eval_metrics", "travel_dist", "Eval travel dist"),
                    ("eval_metrics", "steps_taken", "Eval steps"),
                    ("eval_metrics", "mean_planning_time_ms", "Eval planning ms"),
                ],
                "Value",
            ),
            (
                "Train Return",
                [
                    ("train_metrics", "episode_return", "Train return"),
                    ("train_metrics", "reward", "Batch reward"),
                ],
                "Value",
            ),
            (
                "Train Efficiency",
                [
                    ("train_metrics", "distance_efficiency", "Train dist efficiency"),
                    ("train_metrics", "travel_dist", "Train travel dist"),
                    ("train_metrics", "episode_steps", "Train steps"),
                    ("train_metrics", "mean_planning_time_ms", "Train planning ms"),
                ],
                "Value",
            ),
            (
                "Optimization Losses",
                [
                    ("train_metrics", "policy_loss", "Policy loss"),
                    ("train_metrics", "q_value_loss", "Q loss"),
                    ("train_metrics", "alpha_loss", "Alpha loss"),
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
                "System / Buffer",
                [("system_metrics", "buffer_size", "Replay buffer size")],
                "Transitions",
            ),
        ]
        self.series_colors = {
            ("eval_metrics", "explored_rate"): "#2ca02c",
            ("eval_metrics", "success_rate"): "#d62728",
            ("eval_metrics", "travel_dist"): "#1f77b4",
            ("eval_metrics", "steps_taken"): "#ff7f0e",
            ("eval_metrics", "episode_return"): "#9467bd",
            ("eval_metrics", "distance_efficiency"): "#17becf",
            ("eval_metrics", "time_efficiency"): "#8c564b",
            ("eval_metrics", "mean_planning_time_ms"): "#bcbd22",
            ("train_metrics", "episode_return"): "#ff7f0e",
            ("train_metrics", "reward"): "#8c564b",
            ("train_metrics", "travel_dist"): "#1f77b4",
            ("train_metrics", "episode_steps"): "#ff7f0e",
            ("train_metrics", "distance_efficiency"): "#17becf",
            ("train_metrics", "mean_planning_time_ms"): "#bcbd22",
            ("train_metrics", "policy_loss"): "#1f77b4",
            ("train_metrics", "q_value_loss"): "#ff7f0e",
            ("train_metrics", "alpha_loss"): "#2ca02c",
            ("train_metrics", "explored_rate"): "#2ca02c",
            ("train_metrics", "success_rate"): "#d62728",
            ("train_metrics", "value"): "#1f77b4",
            ("train_metrics", "entropy"): "#ff7f0e",
            ("train_metrics", "log_alpha"): "#2ca02c",
            ("train_metrics", "policy_grad_norm"): "#1f77b4",
            ("train_metrics", "q_value_grad_norm"): "#ff7f0e",
            ("system_metrics", "buffer_size"): "#1f77b4",
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
            self.sections[section_name]["episodes"] = list(
                section_data.get("episodes", [])
            )
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
        return float(value)

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

    def _update_section(
        self, section_name: str, episode: int, metrics: dict[str, object]
    ) -> None:
        episode = int(episode)
        if section_name in self._trim_pending:
            self._trim_section(section_name, episode)
            self._trim_pending.remove(section_name)

        section = self.sections[section_name]
        history = section["history"]
        for key in metrics:
            if key not in history:
                history[key] = [None] * len(section["episodes"])

        insert_idx = bisect_left(section["episodes"], episode)
        if (
            insert_idx == len(section["episodes"])
            or section["episodes"][insert_idx] != episode
        ):
            section["episodes"].insert(insert_idx, episode)
            for key in list(history.keys()):
                history[key].insert(insert_idx, None)

        for key, value in metrics.items():
            history[key][insert_idx] = self._normalize_value(value)

        self._save_data()
        self._save_plots()

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

    def _series_points(
        self, section_name: str, key: str
    ) -> tuple[list[int], list[float]]:
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

    def _best_point(
        self, section_name: str, key: str, mode: str
    ) -> tuple[int, float] | None:
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
            ax.axvline(
                episode, color="#7f7f7f", linestyle="--", linewidth=1.0, alpha=0.3
            )

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
            ax.text(
                0.5,
                0.5,
                empty_message,
                ha="center",
                va="center",
                fontsize=11,
                color="#666666",
            )

    def _format_summary_value(
        self, point: tuple[int, float] | None, precision: int = 3
    ) -> str:
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
        best_eval_return = self._best_point(
            "eval_metrics", "episode_return", mode="max"
        )
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
                "history": {
                    key: list(values) for key, values in section["history"].items()
                },
            }
        self.data_file.write_text(json.dumps(payload, indent=2))

    def _save_plots(self) -> None:
        resume_markers = self._detect_resume_markers()

        progress_fig, progress_axes = plt.subplots(3, 2, figsize=(14, 13))
        progress_fig.suptitle("Training Progress", fontsize=16, fontweight="bold")
        progress_axes = np.asarray(progress_axes).reshape(3, 2)
        progress_axes_flat = progress_axes.flatten()

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

        diagnostic_fig, diagnostic_axes = plt.subplots(2, 2, figsize=(14, 9))
        diagnostic_fig.suptitle("Training Diagnostics", fontsize=16, fontweight="bold")
        diagnostic_axes = np.asarray(diagnostic_axes).reshape(2, 2)
        diagnostic_axes_flat = diagnostic_axes.flatten()
        for idx, (title, series_list, ylabel) in enumerate(self.diagnostic_plot_groups):
            self._plot_series_group(
                diagnostic_axes_flat[idx],
                title,
                series_list,
                ylabel,
                resume_markers,
                empty_message="No data yet",
            )
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        diagnostic_fig.savefig(self.diagnostic_plot_file, dpi=150, bbox_inches="tight")
        plt.close(diagnostic_fig)
