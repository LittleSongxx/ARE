from __future__ import annotations

from .parameter import RuntimeConfig

__all__ = ["Agent", "InferenceAgent", "RuntimeConfig", "main"]


def __getattr__(name: str):
    if name in {"Agent", "InferenceAgent"}:
        from .agent import Agent, InferenceAgent

        return {"Agent": Agent, "InferenceAgent": InferenceAgent}[name]
    if name == "main":
        from .driver import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
