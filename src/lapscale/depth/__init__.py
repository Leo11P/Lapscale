"""Stage 2: depth from a single frame (MoGe)."""
from .base import DepthBackend, DepthPrediction

__all__ = ["DepthBackend", "DepthPrediction", "make_backend"]


def make_backend(name: str = "moge", **kw) -> DepthBackend:
    if name == "moge":
        from .moge import MoGeBackend
        return MoGeBackend(**kw)
    raise ValueError(f"unknown depth backend {name!r}")
