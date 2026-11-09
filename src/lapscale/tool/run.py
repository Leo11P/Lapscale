"""Stage 3 over a video: segmentation -> shaft edges -> cylinder -> metric samples of the shaft pixels.

Per frame and instrument the cache keeps the rays of (a subsample of) the shaft pixels with their depth
in mm, which is all stage 4 needs, plus the edges for overlays and the uncertainty of the shaft depth.
All instruments are assumed to have the same diameter (the one given): a frame cannot tell which
instrument is which.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..camera import Camera
from ..io.cache import ShaftSamples, ToolStore
from ..io.video import FrameSource
from .base import SegBackend
from .edges import EdgeConfig, extract_shaft_edges, shaft_region
from .geometry import shaft_from_edges, to_normalized
from .uncertainty import shaft_uncertainty


@dataclass
class ToolConfig:
    diameter_mm: float = 5.0
    max_samples: int = 1500           # shaft pixels kept per instrument
    region_inset_px: float = 3.0
    mc_draws: int = 100               # 0 = no Monte Carlo (faster, no uncertainty gating)
    point_sd_px: float = 0.5
    bias_sd_px: float = 1.0           # assumed segmentation edge bias (systematic), see segbench
    min_score: float = 0.0


def process_frame(frame: np.ndarray, camera: Camera, backend: SegBackend, cfg: ToolConfig,
                  edge_cfg: EdgeConfig | None = None, frame_index: int = 0, rng=None
                  ) -> tuple[list[dict], list[ShaftSamples], float]:
    """One frame -> (csv rows, samples of the valid instruments, segmentation latency ms)."""
    rng = rng if rng is not None else np.random.default_rng(frame_index)
    t0 = time.perf_counter()
    instances = backend.predict(frame)
    lat = 1000 * (time.perf_counter() - t0)
    rows, samples = [], []
    focal_sd = float(camera.quality.get("focal_std_pct", 0.0) or 0.0) / 100
    for k, inst in enumerate(instances):
        row = dict(frame=frame_index, instrument=k, score=round(inst.score, 4), diameter_mm=cfg.diameter_mm,
                   latency_ms=round(lat, 2))
        if inst.score < cfg.min_score:
            rows.append(dict(row, status="rejected", reason="low score"))
            continue
        res = extract_shaft_edges(inst.prob, camera, edge_cfg)
        if res.reason != "ok":
            rows.append(dict(row, status="rejected", reason=res.reason))
            continue
        e = res.edges
        try:
            est = shaft_from_edges(e.a_raw, e.b_raw, camera, cfg.diameter_mm)
        except ValueError as err:
            rows.append(dict(row, status="rejected", reason=f"geometry: {err}"))
            continue
        region = shaft_region(e, camera, inst.prob, cfg.region_inset_px)
        yy, xx = np.nonzero(region)
        if len(xx) > cfg.max_samples:
            pick = rng.choice(len(xx), cfg.max_samples, replace=False)
            xx, yy = xx[pick], yy[pick]
        rays = to_normalized(np.column_stack([xx, yy]).astype(np.float64), camera)
        z = est.depth_at(rays)
        keep = np.isfinite(z)
        rays, z = rays[keep], z[keep]
        unc = None
        if cfg.mc_draws:
            unc = shaft_uncertainty(e.a_raw, e.b_raw, camera, cfg.diameter_mm, cfg.point_sd_px, cfg.bias_sd_px,
                                    focal_sd, n=cfg.mc_draws, seed=frame_index * 31 + k)
        rel_tot = unc.rel_sd_total_pct if unc else np.nan
        samples.append(ShaftSamples(rays, z, e.a_raw, e.b_raw, est.depth_mid, rel_tot))
        rows.append(dict(row, status="ok", reason="ok", depth_mid_mm=round(est.depth_mid, 3),
                         width_px=round(est.width_px, 2), n_samples=int(len(z)),
                         rel_sd_random_pct=round(unc.rel_sd_random_pct, 3) if unc else "",
                         rel_sd_total_pct=round(rel_tot, 3) if unc else "",
                         bias_sensitivity_pct_per_px=round(unc.bias_sensitivity_pct_per_px, 3) if unc else ""))
    if not instances:
        rows.append(dict(frame=frame_index, instrument=-1, status="none", reason="no instance",
                         latency_ms=round(lat, 2)))
    return rows, samples, lat


def run_tool(video: str | Path, camera: Camera, backend: SegBackend, run_dir: str | Path,
             cfg: ToolConfig | None = None, edge_cfg: EdgeConfig | None = None, tag: str | None = None,
             indices: list[int] | None = None, log=print) -> dict:
    cfg = cfg or ToolConfig()
    src = FrameSource(video)
    cam = camera.for_size(src.size)
    tag = tag or backend.name.replace(":", "_").replace("/", "_").replace("|", "+")
    store = ToolStore(run_dir, tag)
    store.write_meta(dict(backend=backend.name, video=str(video), size=list(src.size), config=cfg.__dict__,
                          edges=(edge_cfg or EdgeConfig()).__dict__, camera=cam.to_dict()))
    indices = src.sample_indices() if indices is None else indices
    todo = [i for i in indices if not store.has(i)]
    log(f"tool [{backend.name}] on {Path(video).name}: {len(todo)} frames to do, {len(indices) - len(todo)} cached")
    lats, n_ok = [], 0
    for k, (i, frame) in enumerate(src.iter(todo), 1):
        if hasattr(backend, "set_frame_id"):
            backend.set_frame_id(f"{Path(video).stem}_{i:06d}")
        rows, samples, lat = process_frame(frame, cam, backend, cfg, edge_cfg, i)
        store.save(i, rows, samples)
        lats.append(lat)
        n_ok += bool(samples)
        if k % 25 == 0:
            log(f"  {k}/{len(todo)} frames, {n_ok} with a valid shaft")
    summary = dict(tag=tag, n_done=len(lats), frames_with_shaft=n_ok,
                   seg_latency_ms_median=round(float(np.median(lats)), 2) if lats else None)
    log(f"done: {summary}")
    return summary
