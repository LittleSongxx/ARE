from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs"


class Config(dict):
    """Dictionary with recursive attribute access and deterministic serialization."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    @classmethod
    def convert(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return cls({str(k): cls.convert(v) for k, v in value.items()})
        if isinstance(value, list):
            return [cls.convert(v) for v in value]
        return value

    def plain(self) -> dict[str, Any]:
        def unwrap(value: Any) -> Any:
            if isinstance(value, Mapping):
                return {str(k): unwrap(v) for k, v in value.items()}
            if isinstance(value, list):
                return [unwrap(v) for v in value]
            if isinstance(value, Path):
                return str(value)
            return value

        return unwrap(self)

    def clone(self) -> "Config":
        return Config.convert(copy.deepcopy(self.plain()))


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload or {}


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _set_dotted(payload: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    if any(not key for key in keys):
        raise ValueError(f"invalid override key: {dotted!r}")
    cursor = payload
    for key in keys[:-1]:
        current = cursor.get(key)
        if current is None:
            current = {}
            cursor[key] = current
        if not isinstance(current, dict):
            raise ValueError(f"cannot override child of scalar key: {dotted!r}")
        cursor = current
    cursor[keys[-1]] = value


def parse_overrides(items: Iterable[str]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        key, raw = item.split("=", 1)
        _set_dotted(payload, key, yaml.safe_load(raw))
    return payload


def default_data_root() -> Path:
    configured = os.environ.get("ACPBGRL_DATA_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    server_root = Path("/mnt/songensheng/ac-pbgrl")
    if server_root.parent.exists() and os.access(str(server_root.parent), os.W_OK):
        return server_root
    return PROJECT_ROOT / ".runtime"


def load_config(
    experiment: str = "full",
    *,
    system: str | None = None,
    overrides: Iterable[str] = (),
) -> Config:
    payload = _read_yaml(CONFIG_ROOT / "base.yaml")
    experiment_path = Path(experiment)
    if not experiment_path.suffix:
        experiment_path = CONFIG_ROOT / "experiments" / f"{experiment}.yaml"
    elif not experiment_path.is_absolute():
        experiment_path = PROJECT_ROOT / experiment_path
    payload = deep_merge(payload, _read_yaml(experiment_path))
    if system:
        system_path = Path(system)
        if not system_path.suffix:
            system_path = CONFIG_ROOT / "system" / f"{system}.yaml"
        elif not system_path.is_absolute():
            system_path = PROJECT_ROOT / system_path
        payload = deep_merge(payload, _read_yaml(system_path))
    payload = deep_merge(payload, parse_overrides(overrides))

    if str(payload["project"].get("data_root", "auto")) == "auto":
        payload["project"]["data_root"] = str(default_data_root())
    maps_dir = Path(str(payload["project"].get("maps_dir", "maps")))
    if not maps_dir.is_absolute():
        maps_dir = PROJECT_ROOT / maps_dir
    payload["project"]["maps_dir"] = str(maps_dir.resolve())
    payload["project"]["experiment"] = Path(experiment_path).stem
    return Config.convert(payload)


def save_resolved_config(config: Config, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config.plain(), sort_keys=True), encoding="utf-8")


def config_fingerprint(config: Config) -> str:
    import hashlib

    encoded = json.dumps(config.plain(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
