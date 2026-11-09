"""Stage 2 over a video: undistort, run the depth backend, cache one map per frame."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..camera import Camera
from ..io.cache import DepthStore
from ..io.video import FrameSource
from .base import DepthBackend
from .view import PinholeView


def run_depth(video: str | Path, camera: Camera | None, backend: DepthBackend, run_dir: str | Path,
              tag: str | None = None, indices: list[int] | None = None, config: dict | None = None,
              log=print) -> dict:
    """``camera`` None = no calibration (the backend estimates the focal length per frame, no
    undistortion). Frames already in the cache are skipped. Returns a timing summary."""
    src = FrameSource(video)
    view = PinholeView(camera, src.size)
    tag = tag or backend.name.replace(":", "_").replace("/", "_")
    store = DepthStore(run_dir, tag)
    store.write_meta(dict(backend=backend.name, video=str(video), size=list(src.size),
                          camera=None if camera is None else camera.to_dict(),
                          K_view=None if view.K_view is None else view.K_view.tolist(),
                          fov_x_deg=view.fov_x_deg, undistorted=view.maps is not None, config=config or {},
                          output_kind=getattr(backend, "output_kind", "depth"),
                          native_unit_mm=getattr(backend, "native_unit_mm", None)))
    indices = src.sample_indices() if indices is None else indices
    todo = [i for i in indices if not store.has(i)]
    log(f"depth [{backend.name}] on {Path(video).name}: {len(todo)} frames to do, "
        f"{len(indices) - len(todo)} cached; fov_x {view.fov_x_deg if view.fov_x_deg else 'estimated'}")
    lat = []
    w, h = src.size
    for k, (i, frame) in enumerate(src.iter(todo), 1):
        img = view.apply(frame)
        if hasattr(backend, "set_frame_id"):          # precomputed maps
            backend.set_frame_id(f"{Path(video).stem}_{i:06d}")
        t0 = time.perf_counter()
        pred = backend.predict(img, fov_x_deg=view.fov_x_deg)
        ms = 1000 * (time.perf_counter() - t0)
        lat.append(ms)
        if pred.depth.shape != (h, w):
            raise ValueError(f"backend returned {pred.depth.shape[::-1]}, frame is {w}x{h}")
        valid = pred.valid & view.mask & np.isfinite(pred.depth) & (pred.depth > 0)
        if view.K_view is not None:
            K = view.K_view
        else:                                    # estimated per frame, principal point at the centre
            fx = pred.fx_px or float("nan")
            K = np.array([[fx, 0, (w - 1) / 2], [0, pred.fy_px or fx, (h - 1) / 2], [0, 0, 1]])
        store.save(i, pred.depth, valid, K, ms, pred.fx_px, pred.fy_px)
        if k % 25 == 0:
            log(f"  {k}/{len(todo)} frames, {np.median(lat):.0f} ms/frame")
    summary = dict(tag=tag, n_done=len(lat), n_cached=len(indices) - len(todo),
                   latency_ms_median=round(float(np.median(lat)), 2) if lat else None,
                   fps=round(1000 / float(np.median(lat)), 2) if lat else None, dir=str(store.dir))
    log(f"done: {summary}")
    return summary
