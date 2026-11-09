"""``lapscale run --mode stream``: real time.

Segmentation, edges and geometry run on every frame (cheap); the depth network runs in a worker thread
on the LATEST frame whenever it is free (expensive); when a depth map arrives it is fitted against the
shaft samples of the same frame and a causal smoother (running median) updates the scale. Every frame is
shown with the latest depth map times the current scale. Latencies are logged per frame.
"""
from __future__ import annotations

import csv
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from ..camera import Camera
from ..depth.base import DepthBackend
from ..depth.view import PinholeView
from ..io.cache import DepthFrame
from ..io.video import FrameSource
from ..scale.fit import ScaleConfig, TrackState, apply_scale, fit_frame
from ..scale.run import frame_pairs
from ..tool.base import SegBackend
from ..tool.edges import EdgeConfig
from ..tool.run import ToolConfig, process_frame


class _DepthWorker(threading.Thread):
    def __init__(self, backend: DepthBackend, view: PinholeView):
        super().__init__(daemon=True)
        self.backend, self.view = backend, view
        self.inbox: queue.Queue = queue.Queue(maxsize=1)
        self.outbox: queue.Queue = queue.Queue()
        self.stop = threading.Event()

    def submit(self, index: int, image: np.ndarray, frame_id: str) -> bool:
        try:
            self.inbox.put_nowait((index, image, frame_id))
            return True
        except queue.Full:
            return False                       # busy: this frame gets no depth

    def run(self):
        while not self.stop.is_set():
            try:
                index, img, fid = self.inbox.get(timeout=0.05)
            except queue.Empty:
                continue
            if hasattr(self.backend, "set_frame_id"):
                self.backend.set_frame_id(fid)
            t0 = time.perf_counter()
            pred = self.backend.predict(img, fov_x_deg=self.view.fov_x_deg)
            ms = 1000 * (time.perf_counter() - t0)
            valid = pred.valid & self.view.mask & np.isfinite(pred.depth) & (pred.depth > 0)
            K = self.view.K_view if self.view.K_view is not None else np.array(
                [[pred.fx_px, 0, (img.shape[1] - 1) / 2], [0, pred.fy_px or pred.fx_px, (img.shape[0] - 1) / 2],
                 [0, 0, 1]])
            self.outbox.put((index, DepthFrame(index, pred.depth.astype(np.float32), valid, K), ms))


def run_stream(source: str | Path, camera: Camera, seg: SegBackend, depth: DepthBackend, variant: str = "scale_only",
               tool_cfg: ToolConfig | None = None, edge_cfg: EdgeConfig | None = None,
               scale_cfg: ScaleConfig | None = None, out_video: str | Path | None = None,
               log_csv: str | Path | None = None, show: bool = False, max_frames: int | None = None,
               depth_range_mm: tuple[float, float] = (20.0, 200.0), log=print) -> dict:
    tool_cfg = tool_cfg or ToolConfig(mc_draws=0)
    scale_cfg = scale_cfg or ScaleConfig()
    src = FrameSource(source)
    cam = camera.for_size(src.size)
    view = PinholeView(cam, src.size)
    worker = _DepthWorker(depth, view)
    worker.start()
    track = TrackState(scale_cfg.smooth_window, scale_cfg.max_hold)
    samples_buf: dict[int, list] = {}
    latest_depth: DepthFrame | None = None
    a = b = np.nan
    writer = None
    rows = []
    stem = Path(source).stem
    t_start = time.perf_counter()
    n = 0
    try:
        for i, frame in src.iter():
            if max_frames is not None and n >= max_frames:
                break
            t0 = time.perf_counter()
            if hasattr(seg, "set_frame_id"):
                seg.set_frame_id(f"{stem}_{i:06d}")
            _, samples, seg_ms = process_frame(frame, cam, seg, tool_cfg, edge_cfg, i)
            samples_buf[i] = samples
            for old in [k for k in samples_buf if k < i - 300]:
                del samples_buf[old]
            img = view.apply(frame)
            worker.submit(i, img, f"{stem}_{i:06d}")
            status = "waiting"
            while not worker.outbox.empty():
                j, df, depth_ms = worker.outbox.get()
                latest_depth = df
                d, z, used = frame_pairs(df, samples_buf.get(j, []), scale_cfg)
                fit = fit_frame(d, z, variant, scale_cfg) if used else None
                a, b, status = track.update(j, fit)
            mm = None
            if latest_depth is not None and np.isfinite(a):
                mm = apply_scale(latest_depth.depth, a, b, scale_cfg.output_kind)
            if out_video or show:
                vis = _overlay(img, mm, latest_depth, samples, view, a, depth_range_mm)
                if out_video:
                    if writer is None:
                        writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"),
                                                 src.info.fps or 25, (vis.shape[1], vis.shape[0]))
                    writer.write(vis)
                if show:  # pragma: no cover - interactive
                    cv2.imshow("lapscale stream", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            rows.append(dict(frame=i, seg_ms=round(seg_ms, 2), loop_ms=round(1000 * (time.perf_counter() - t0), 2),
                             depth_frame=latest_depth.index if latest_depth is not None else "",
                             n_shafts=len(samples), a=a if np.isfinite(a) else "", b=b if np.isfinite(b) else "",
                             status=status))
            n += 1
        # drain the last depth result so short runs are not lost
        deadline = time.perf_counter() + 30
        while (worker.inbox.qsize() or not worker.outbox.empty()) and time.perf_counter() < deadline:
            try:
                j, df, _ = worker.outbox.get(timeout=0.1)
            except queue.Empty:
                continue
            d, z, used = frame_pairs(df, samples_buf.get(j, []), scale_cfg)
            a, b, _ = track.update(j, fit_frame(d, z, variant, scale_cfg) if used else None)
    finally:
        worker.stop.set()
        if writer is not None:
            writer.release()
        if show:  # pragma: no cover
            cv2.destroyAllWindows()
    elapsed = time.perf_counter() - t_start
    if log_csv:
        with open(log_csv, "w", newline="") as fh:
            wr = csv.DictWriter(fh, list(rows[0].keys()) if rows else ["frame"])
            wr.writeheader()
            wr.writerows(rows)
    summary = dict(frames=n, fps=round(n / elapsed, 2) if elapsed > 0 else None,
                   seg_ms_median=round(float(np.median([r["seg_ms"] for r in rows])), 2) if rows else None,
                   frames_with_scale=int(sum(r["a"] != "" for r in rows)), final_a=a, final_b=b)
    log(f"stream: {summary}")
    return summary


def _overlay(img, mm, depth: DepthFrame | None, samples, view: PinholeView, a, rng_mm):
    from ..viz.overlay import draw_frame
    return draw_frame(img, mm, depth.valid if depth is not None else None, samples, view, rng_mm,
                      f"scale a={a:.4g}" if np.isfinite(a) else "scale: waiting for a valid shaft")
