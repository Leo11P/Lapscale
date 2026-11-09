"""Calibration entry points used by the CLI (and usable from Python)."""
from __future__ import annotations

from pathlib import Path

from ..camera import Camera
from ..io.video import FrameSource
from .calibrate import CalibConfig, calibrate_views, collect_views, select_views
from .importers import import_opencv
from .optical_mask import detect_optical_circle
from .report import write_report
from .targets import TargetDetector, TargetSpec


def detect_circle_on(source: FrameSource, n_frames: int = 60, log=print):
    idx = source.sample_indices(n=n_frames)
    circle, diag = detect_optical_circle(f for _, f in source.iter(idx))
    if circle is None:
        log(f"optical disc: {diag['result']}")
    else:
        log(f"optical disc: centre ({circle.cx:.1f}, {circle.cy:.1f}) r {circle.r:.1f} px, fit rms "
            f"{diag['fit_rms_px']} px on {100 * diag['arc_fraction']:.0f}% of the circumference")
    return circle, diag


def run_board(input_path: str | Path, spec: TargetSpec, out: str | Path, cfg: CalibConfig | None = None,
              stride: int | None = None, max_candidates: int = 600, detect_circle: bool = True,
              report_dir: str | Path | None = None, log=print) -> Camera:
    cfg = cfg or CalibConfig()
    src = FrameSource(input_path)
    log(f"source: {src.info.path}  {src.size[0]}x{src.size[1]}, {len(src)} frames")
    circle, circle_diag = detect_circle_on(src, log=log) if detect_circle else (None, {})
    mask = circle.mask(src.size, margin_px=2) if circle is not None else None

    idx = src.sample_indices(stride=stride) if stride else src.sample_indices(n=max_candidates)
    log(f"detecting {spec.kind} {spec.cols}x{spec.rows} ({spec.square_mm} mm) in {len(idx)} frames ...")
    views = collect_views(src, TargetDetector(spec), idx, mask, progress=log)
    log(f"target found in {len(views)} frames")
    sel = select_views(views, src.size, cfg)
    log(f"{len(sel)} views kept after blur filter and de-duplication")
    res = calibrate_views(sel, src.size, cfg, spec.kind, circle, log=log)
    res.camera.notes = (f"{spec.kind} {spec.cols}x{spec.rows}, square {spec.square_mm} mm"
                        + (f", marker {spec.marker_mm} mm {spec.dictionary}" if spec.kind == "charuco" else "")
                        + f", from {Path(input_path).name}")
    path = res.camera.save(out)
    rdir = Path(report_dir) if report_dir else Path(out).with_suffix("").parent / (Path(out).stem + "_report")
    ref = next((f for _, f in src.iter([sel[len(sel) // 2].frame])), None)
    files = write_report(rdir, res, sel, ref, extra=dict(optical_circle_detection=circle_diag,
                                                         target=spec.__dict__, config=cfg.__dict__))
    log("\n" + res.camera.summary())
    for w in res.warnings:
        log(f"WARNING: {w}")
    log(f"\nsaved {path}\nreport: {', '.join(str(f) for f in files)}")
    return res.camera


def run_import(file: str | Path, out: str | Path, side: str = "mono", image_size=None, crop_offset=(0, 0),
               key_K=None, key_D=None, video: str | Path | None = None, log=print) -> Camera:
    cam = import_opencv(file, side, image_size, crop_offset, key_K, key_D)
    if video is not None:
        src = FrameSource(video)
        if src.size != cam.image_size:
            cam = cam.for_size(src.size)
            log(f"intrinsics rescaled to the video resolution {src.size[0]}x{src.size[1]}")
        cam.optical_circle, _ = detect_circle_on(src, log=log)
    path = cam.save(out)
    log(cam.summary())
    for w in cam.quality.get("warnings", []):
        log(f"WARNING: {w}")
    log(f"saved {path}")
    return cam


def run_moge(video: str | Path, out: str | Path, n_frames: int = 20, backend_kw: dict | None = None,
             log=print) -> Camera:
    from ..depth import make_backend
    from .moge_fov import estimate_focal
    src = FrameSource(video)
    circle, _ = detect_circle_on(src, log=log)
    backend = make_backend("moge", **(backend_kw or {}))
    cam = estimate_focal(src, backend, src.sample_indices(n=n_frames), circle, log=log)
    path = cam.save(out)
    log(cam.summary())
    log(f"saved {path}")
    return cam


def run_mask(camera: str | Path, video: str | Path, out: str | Path | None = None, log=print) -> Camera:
    cam = Camera.load(camera)
    src = FrameSource(video)
    if src.size != cam.image_size:
        raise ValueError(f"camera is {cam.image_size}, video is {src.size}")
    cam.optical_circle, _ = detect_circle_on(src, log=log)
    path = cam.save(out or camera)
    log(f"saved {path}")
    return cam


def parse_size(s: str | None):
    if not s:
        return None
    w, h = s.lower().replace(",", "x").split("x")
    return int(w), int(h)


def parse_pair(s: str | None, default=(0.0, 0.0)):
    if not s:
        return default
    a, b = s.split(",")
    return float(a), float(b)


__all__ = ["run_board", "run_import", "run_moge", "run_mask", "parse_size", "parse_pair"]
