from __future__ import annotations

import json
import shutil
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
        self.window_size = max(int(window_size), 1)
        self.snapshot_interval = max(int(snapshot_interval), 1)
        self.data_file = self.save_dir / "training_history.json"
        self.latest_metrics_file = self.save_dir / "latest_metrics.json"
        self.summary_file = self.save_dir / "summary.txt"
        self.plot_file = self.save_dir / "training_curves.png"
        self.snapshot_root = self.save_dir / "snapshots"

        self.sections = {
            "train_metrics": {"episodes": [], "history": defaultdict(list)},
            "eval_metrics": {"episodes": [], "history": defaultdict(list)},
            "system_metrics": {"episodes": [], "history": defaultdict(list)},
        }
        self._trim_pending = set(self.sections.keys())
        self._load_history()

        self.plot_groups = [
            ("Reward / Return", [("train_metrics", "reward"), ("train_metrics", "episode_return"), ("eval_metrics", "episode_return")], "Value"),
            (
                "Reward Components",
                [
                    ("train_metrics", "reward_info"),
                    ("train_metrics", "reward_dist"),
                    ("train_metrics", "reward_safe"),
                    ("train_metrics", "reward_terminal"),
                ],
                "Value",
            ),
            ("Losses", [("train_metrics", "policy_loss"), ("train_metrics", "q_value_loss"), ("train_metrics", "alpha_loss")], "Loss"),
            ("Value / Entropy", [("train_metrics", "value"), ("train_metrics", "entropy"), ("train_metrics", "log_alpha")], "Value"),
            ("Grad Norms", [("train_metrics", "policy_grad_norm"), ("train_metrics", "q_value_grad_norm")], "Norm"),
            ("Exploration / Success", [("train_metrics", "explored_rate"), ("train_metrics", "success_rate"), ("eval_metrics", "explored_rate"), ("eval_metrics", "success_rate")], "Rate"),
            ("Distance / Steps", [("train_metrics", "travel_dist"), ("train_metrics", "episode_steps"), ("eval_metrics", "travel_dist"), ("eval_metrics", "steps_taken")], "Value"),
            ("System", [("system_metrics", "buffer_size"), ("system_metrics", "completed_episodes")], "Value"),
        ]

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

    def _normalize_value(self, value: object) -> float | int | bool:
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
        return float(value)  # best effort for tensor-like scalars

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

    def _update_section(self, section_name: str, episode: int, metrics: dict[str, object]) -> None:
        episode = int(episode)
        if section_name in self._trim_pending:
            self._trim_section(section_name, episode)
            self._trim_pending.remove(section_name)

        section = self.sections[section_name]
        section["episodes"].append(episode)
        for key, value in metrics.items():
            section["history"][key].append(self._normalize_value(value))

        self._save_all()
        if episode % self.snapshot_interval == 0:
            self._save_snapshot(episode)

    def _smooth(self, values: list[float | int | bool]) -> list[float]:
        if not values:
            return []
        smoothed = []
        for idx in range(len(values)):
            start = max(0, idx - self.window_size + 1)
            smoothed.append(float(np.mean(values[start : idx + 1])))
        return smoothed

    def _save_all(self) -> None:
        self._save_data()
        self._save_latest_metrics()
        self._save_summary()
        self._save_plots()

    def _save_data(self) -> None:
        payload = {"sections": {}}
        for section_name, section in self.sections.items():
            payload["sections"][section_name] = {
                "episodes": list(section["episodes"]),
                "history": {key: list(values) for key, values in section["history"].items()},
            }
        self.data_file.write_text(json.dumps(payload, indent=2))

    def _save_latest_metrics(self) -> None:
        payload = {"latest_episode": {}, "latest_metrics": {}}
        for section_name, section in self.sections.items():
            payload["latest_episode"][section_name] = section["episodes"][-1] if section["episodes"] else None
            payload["latest_metrics"][section_name] = {}
            for key, values in section["history"].items():
                if values:
                    payload["latest_metrics"][section_name][key] = values[-1]
        self.latest_metrics_file.write_text(json.dumps(payload, indent=2))

    def _save_summary(self) -> None:
        lines = ["Training Monitor Summary", ""]
        for section_name, section in self.sections.items():
            if not section["episodes"]:
                continue
            lines.append(f"[{section_name}]")
            lines.append(
                f"episodes: {section['episodes'][0]} -> {section['episodes'][-1]}  points={len(section['episodes'])}"
            )
            for key, values in section["history"].items():
                if not values:
                    continue
                recent = values[-min(self.window_size, len(values)) :]
                lines.append(
                    f"{key}: latest={float(values[-1]):.6f} "
                    f"mean(last{len(recent)})={float(np.mean(recent)):.6f} "
                    f"min={float(np.min(values)):.6f} max={float(np.max(values)):.6f}"
                )
            lines.append("")
        self.summary_file.write_text("\n".join(lines).rstrip() + "\n")

    def _save_plots(self) -> None:
        num_groups = len(self.plot_groups)
        ncols = 2
        nrows = (num_groups + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows))
        fig.suptitle("Training Progress", fontsize=16, fontweight="bold")
        axes = np.array(axes).reshape(nrows, ncols)

        for idx, (title, series_list, ylabel) in enumerate(self.plot_groups):
            row, col = divmod(idx, ncols)
            ax = axes[row, col]
            has_data = False
            for section_name, key in series_list:
                section = self.sections[section_name]
                values = section["history"].get(key, [])
                if not values:
                    continue
                episodes = section["episodes"][: len(values)]
                smoothed = self._smooth(values)
                label = f"{section_name.replace('_metrics', '')}:{key}"
                ax.plot(episodes, values, alpha=0.2, linewidth=0.8)
                ax.plot(episodes[: len(smoothed)], smoothed, linewidth=2.0, label=label)
                has_data = True
            ax.set_title(title)
            ax.set_xlabel("Episode")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            if has_data:
                ax.legend(fontsize=8, loc="best")
            else:
                ax.set_visible(False)

        for idx in range(num_groups, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row, col].set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(self.plot_file, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _save_snapshot(self, episode: int) -> None:
        snapshot_dir = self.snapshot_root / f"episode_{episode}"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        for source in (self.plot_file, self.data_file, self.latest_metrics_file, self.summary_file):
            if source.exists():
                shutil.copy2(source, snapshot_dir / source.name)
