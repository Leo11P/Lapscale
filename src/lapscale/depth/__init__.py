"""Stage 2: per-frame depth from a monocular network (MoGe), in the network's units.

  base.py   DepthBackend protocol, DepthPrediction
  moge.py   MoGe v2 / v3 backend (lazy import)
  da3.py    Depth Anything 3 backend (lazy import)
  folder.py precomputed depth maps (any external model, or ground truth)
  view.py   PinholeView: undistortion, optical-disc mask, FOV passed to the network
  run.py    run_depth: a video -> cached depth maps (io/cache.py), resumable, timed
"""
from .base import DepthBackend, DepthPrediction
from .view import PinholeView

__all__ = ["DepthBackend", "DepthPrediction", "PinholeView", "make_backend", "run_depth"]


BACKENDS = ("moge", "da3", "folder")


def make_backend(name: str = "moge", **kw) -> DepthBackend:
    if name == "moge":
        from .moge import MoGeBackend
        return MoGeBackend(**kw)
    if name == "da3":
        from .da3 import DA3Backend
        return DA3Backend(**kw)
    if name == "folder":
        from .folder import FolderBackend
        return FolderBackend(**kw)
    raise ValueError(f"unknown depth backend {name!r}; available: {BACKENDS}")


def run_depth(*args, **kw):
    from .run import run_depth as _run
    return _run(*args, **kw)
