"""Stage 2: per-frame depth from a monocular network (MoGe), in the network's units.

  base.py   DepthBackend protocol, DepthPrediction
  moge.py   MoGe v2 / v3 backend (lazy import)
  view.py   PinholeView: undistortion, optical-disc mask, FOV passed to the network
  run.py    run_depth: a video -> cached depth maps (io/cache.py), resumable, timed
"""
from .base import DepthBackend, DepthPrediction
from .view import PinholeView

__all__ = ["DepthBackend", "DepthPrediction", "PinholeView", "make_backend", "run_depth"]


def make_backend(name: str = "moge", **kw) -> DepthBackend:
    if name == "moge":
        from .moge import MoGeBackend
        return MoGeBackend(**kw)
    raise ValueError(f"unknown depth backend {name!r}")


def run_depth(*args, **kw):
    from .run import run_depth as _run
    return _run(*args, **kw)
