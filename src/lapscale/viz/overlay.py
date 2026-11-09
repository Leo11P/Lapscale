"""Overlays: metric depth on the undistorted view, shaft edges, status text."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..camera import Camera
from ..depth.view import PinholeView
from ..io.cache import ToolStore
from ..io.video import FrameSource


def colorize(mm: np.ndarray, valid: np.ndarray, rng_mm: tuple[float, float]) -> np.ndarray:
    lo, hi = rng_mm
    t = np.clip((mm - lo) / (hi - lo), 0, 1)
    col = cv2.applyColorMap((255 * (1 - t)).astype(np.uint8), cv2.COLORMAP_TURBO)
    col[~valid] = 0
    return col


def draw_frame(view_img: np.ndarray, mm: np.ndarray | None, valid: np.ndarray | None, samples,
               view: PinholeView, rng_mm=(20.0, 200.0), text: str = "") -> np.ndarray:
    vis = view_img.copy()
    if mm is not None and valid is not None:
        ok = valid & np.isfinite(mm)
        vis = np.where(ok[..., None], (0.45 * vis + 0.55 * colorize(mm, ok, rng_mm)).astype(np.uint8), vis)
    if view.K_view is not None and view.camera is not None:
        from ..tool.geometry import to_normalized
        K = view.K_view
        for s in samples or []:
            for e, col in ((s.edge_a, (0, 0, 255)), (s.edge_b, (255, 0, 0))):
                xy = to_normalized(np.asarray(e, np.float64), view.camera)
                px = xy * [K[0, 0], K[1, 1]] + K[:2, 2]
                for p in px[:: max(1, len(px) // 60)]:
                    cv2.circle(vis, (int(round(p[0])), int(round(p[1]))), 1, col, -1)
            if np.isfinite(s.depth_mid_mm):
                c = np.round(px.mean(0)).astype(int)
                cv2.putText(vis, f"{s.depth_mid_mm:.0f} mm", (int(c[0]) + 6, int(c[1])), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1, cv2.LINE_AA)
    lo, hi = rng_mm
    cv2.putText(vis, f"{text} | depth colours {lo:.0f}-{hi:.0f} mm", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def render_run(run_dir: str | Path, video: str | Path, camera: Camera, depth_tag: str, tool_tag: str,
               variant: str = "scale_only", out: str | Path | None = None, rng_mm=(20.0, 200.0), log=print) -> Path:
    """Video of the frames processed by the offline run, with metric depth and shaft edges."""
    from ..scale.run import load_scale, metric_depth
    src = FrameSource(video)
    view = PinholeView(camera.for_size(src.size), src.size)
    tool = ToolStore(run_dir, tool_tag)
    scales = load_scale(run_dir, depth_tag, tool_tag, variant)
    frames = sorted(scales)
    out = Path(out) if out else Path(run_dir) / f"overlay_{variant}.mp4"
    writer = None
    for i, frame in src.iter(frames):
        md = metric_depth(run_dir, depth_tag, tool_tag, variant, i)
        a, _, st = scales[i]
        vis = draw_frame(view.apply(frame), None if md is None else md.depth, None if md is None else md.valid,
                         tool.load(i) if tool.has(i) else [], view, rng_mm,
                         f"frame {i} | {variant} a={a:.4g} ({st})" if np.isfinite(a) else f"frame {i} | no scale")
        if writer is None:
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), min(src.info.fps or 25, 10),
                                     (vis.shape[1], vis.shape[0]))
        writer.write(vis)
    if writer is not None:
        writer.release()
    log(f"overlay: {out}")
    return out
