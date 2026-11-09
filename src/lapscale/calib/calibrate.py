"""Intrinsic calibration from views of a planar target.

Pipeline: detect the target in many frames -> drop blurred and near-duplicate views (consecutive
video frames are almost identical and would just overweight one pose) -> hold out ~20% of the views
-> OpenCV calibration with iterative rejection of bad views -> evaluate on the held-out views.

What matters downstream is the focal length: in monocular metric reconstruction a relative error on
f is the same relative error on every depth. So the report states the focal-length standard
deviation (from OpenCV's covariance) as a percentage, i.e. directly as a scale uncertainty, and
checks the two things that make f identifiable: tilted views and coverage of the image.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..camera import Camera, OpticalCircle
from ..io.video import FrameSource
from .targets import Detection, TargetDetector

DIST_MODELS = {
    "k1k2": cv2.CALIB_FIX_K3,                 # 5 coefficients, k3 = 0: few or central-only views
    "standard": 0,                            # k1 k2 p1 p2 k3
    "rational": cv2.CALIB_RATIONAL_MODEL,     # 8 coefficients: strong wide-angle distortion
}


@dataclass
class View:
    frame: int
    det: Detection
    sharpness: float


@dataclass
class CalibConfig:
    dist_model: str = "standard"
    zero_tangent: bool = False
    max_views: int = 60
    min_motion_frac: float = 0.02      # min median corner motion between kept views (fraction of diagonal)
    min_sharpness_pct: float = 20.0    # drop the blurriest X% of the detected views
    holdout_fraction: float = 0.2
    reject_factor: float = 3.0         # views with RMS > factor x median are rejected
    reject_min_px: float = 1.0         # ... but never below this RMS
    reject_iters: int = 3


@dataclass
class CalibResult:
    camera: Camera
    train_frames: list[int]
    holdout_frames: list[int]
    rejected_frames: list[int]
    per_view_rms: dict[int, float]
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------------- detection
def sharpness(image: np.ndarray, mask: np.ndarray | None = None) -> float:
    g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    lap = cv2.Laplacian(g, cv2.CV_64F)
    return float(lap[mask].var() if mask is not None else lap.var())


def collect_views(source: FrameSource, detector: TargetDetector, indices: list[int],
                  mask: np.ndarray | None = None, progress=None) -> list[View]:
    views = []
    for k, (i, frame) in enumerate(source.iter(indices)):
        det = detector.detect(frame, mask)
        if det is not None:
            views.append(View(i, det, sharpness(frame, mask)))
        if progress and (k + 1) % 50 == 0:
            progress(f"  scanned {k + 1}/{len(indices)} frames, {len(views)} with the target")
    return views


def select_views(views: list[View], size: tuple[int, int], cfg: CalibConfig) -> list[View]:
    """Sharpness filter, then greedy de-duplication (sharpest first), then even spread over time."""
    if not views:
        return []
    s = np.array([v.sharpness for v in views])
    thr = np.percentile(s, cfg.min_sharpness_pct) if len(views) >= 10 else -np.inf
    cand = sorted((v for v in views if v.sharpness >= thr), key=lambda v: -v.sharpness)
    min_motion = cfg.min_motion_frac * float(np.hypot(*size))
    kept: list[View] = []
    for v in cand:
        if all(_motion(v.det, u.det) > min_motion for u in kept):
            kept.append(v)
    kept.sort(key=lambda v: v.frame)
    if len(kept) > cfg.max_views:
        idx = np.unique(np.round(np.linspace(0, len(kept) - 1, cfg.max_views)).astype(int))
        kept = [kept[i] for i in idx]
    return kept


def _motion(a: Detection, b: Detection) -> float:
    """Median displacement of the corners two views share (inf if they share fewer than 4)."""
    common, ia, ib = np.intersect1d(a.ids, b.ids, return_indices=True)
    if len(common) < 4:
        return np.inf
    return float(np.median(np.linalg.norm(a.image_points[ia] - b.image_points[ib], axis=1)))


# ---------------------------------------------------------------------------------- calibration
def _calibrate(views: list[View], size: tuple[int, int], flags: int):
    obj = [v.det.object_points for v in views]
    img = [v.det.image_points for v in views]
    crit = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 200, 1e-10)
    out = cv2.calibrateCameraExtended(obj, img, size, None, None, flags=flags, criteria=crit)
    rms, K, D, rvecs, tvecs, std_in, _, per_view = out
    return float(rms), K, D.ravel(), rvecs, tvecs, std_in.ravel(), per_view.ravel()


def _view_rms(det: Detection, K, D, rvec, tvec) -> float:
    proj, _ = cv2.projectPoints(det.object_points, rvec, tvec, K, D)
    return float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - det.image_points) ** 2, axis=1))))


def holdout_rms(views: list[View], K, D) -> list[float]:
    """Pose re-estimated per held-out view with the calibrated intrinsics, then reprojection RMS.
    Slightly optimistic (the pose is fitted to the same points) but tests the intrinsics on views
    they have never seen."""
    out = []
    for v in views:
        ok, rvec, tvec = cv2.solvePnP(v.det.object_points, v.det.image_points, K, D)
        if ok:
            out.append(_view_rms(v.det, K, D, rvec, tvec))
    return out


def tilt_deg(rvec) -> float:
    """Angle between the board normal and the optical axis (0 = fronto-parallel)."""
    R, _ = cv2.Rodrigues(rvec)
    return float(np.degrees(np.arccos(min(1.0, abs(R[2, 2])))))


def coverage(views: list[View], size: tuple[int, int], circle: OpticalCircle | None,
             grid: int = 8) -> dict:
    """Share of the valid image area (grid cells) and of its outer ring that contain corners."""
    w, h = size
    pts = np.vstack([v.det.image_points for v in views])
    cell_w, cell_h = w / grid, h / grid
    cx_ = (np.arange(grid) + 0.5) * cell_w
    cy_ = (np.arange(grid) + 0.5) * cell_h
    gx, gy = np.meshgrid(cx_, cy_)
    if circle is not None:
        valid = (gx - circle.cx) ** 2 + (gy - circle.cy) ** 2 <= circle.r ** 2
        ccx, ccy, R = circle.cx, circle.cy, circle.r
    else:
        valid = np.ones_like(gx, bool)
        ccx, ccy, R = (w - 1) / 2, (h - 1) / 2, np.hypot(w, h) / 2
    hit = np.zeros((grid, grid), bool)
    ix = np.clip((pts[:, 0] / cell_w).astype(int), 0, grid - 1)
    iy = np.clip((pts[:, 1] / cell_h).astype(int), 0, grid - 1)
    hit[iy, ix] = True
    rad = np.hypot(pts[:, 0] - ccx, pts[:, 1] - ccy) / R
    return dict(grid_cells_covered=round(float(hit[valid].mean()), 3),
                outer_ring_corner_fraction=round(float((rad > 0.7).mean()), 3),
                max_radius_fraction=round(float(rad.max()), 3),
                grid=hit.astype(int).tolist())


def calibrate_views(views: list[View], size: tuple[int, int], cfg: CalibConfig,
                    source_kind: str, circle: OpticalCircle | None = None, log=print) -> CalibResult:
    if cfg.dist_model not in DIST_MODELS:
        raise ValueError(f"dist_model must be one of {list(DIST_MODELS)}")
    flags = DIST_MODELS[cfg.dist_model] | (cv2.CALIB_ZERO_TANGENT_DIST if cfg.zero_tangent else 0)
    if len(views) < 4:
        raise RuntimeError(f"only {len(views)} usable views: at least 4 needed (10-30 recommended)")

    n_hold = int(round(cfg.holdout_fraction * len(views))) if len(views) >= 10 else 0
    hold_idx = set(np.round(np.linspace(0, len(views) - 1, n_hold + 2)[1:-1]).astype(int)) if n_hold else set()
    train = [v for k, v in enumerate(views) if k not in hold_idx]
    hold = [v for k, v in enumerate(views) if k in hold_idx]

    rejected: list[int] = []
    for it in range(cfg.reject_iters + 1):
        rms, K, D, rvecs, tvecs, std_in, per_view = _calibrate(train, size, flags)
        lim = max(cfg.reject_min_px, cfg.reject_factor * float(np.median(per_view)))
        bad = [k for k, e in enumerate(per_view) if e > lim]
        if not bad or it == cfg.reject_iters or len(train) - len(bad) < 4:
            break
        log(f"  iteration {it}: rms {rms:.3f} px, rejecting {len(bad)} view(s) above {lim:.2f} px")
        rejected += [train[k].frame for k in bad]
        train = [v for k, v in enumerate(train) if k not in set(bad)]

    hold_err = holdout_rms(hold, K, D) if hold else []
    tilts = [tilt_deg(r) for r in rvecs]
    cov = coverage(train, size, circle)
    fx, fy = K[0, 0], K[1, 1]
    q = dict(
        reprojection_rms_px=round(rms, 4),
        holdout_rms_px=round(float(np.sqrt(np.mean(np.square(hold_err)))), 4) if hold_err else None,
        n_views_train=len(train), n_views_holdout=len(hold), n_views_rejected=len(rejected),
        n_corners=int(sum(len(v.det.ids) for v in train)),
        fx_std_px=round(float(std_in[0]), 4), fy_std_px=round(float(std_in[1]), 4),
        focal_std_pct=round(100 * float(max(std_in[0] / fx, std_in[1] / fy)), 4),
        tilt_deg_median=round(float(np.median(tilts)), 2), tilt_deg_max=round(float(np.max(tilts)), 2),
        dist_model=cfg.dist_model,
        coverage=cov,
    )
    warnings = []
    if len(train) < 10:
        warnings.append(f"only {len(train)} training views: 15-30 recommended")
    if rms > 1.0:
        warnings.append(f"reprojection RMS {rms:.2f} px > 1 px: blur, wrong board size, or distortion model too simple")
    if hold_err and q["holdout_rms_px"] > 1.5 * rms + 0.2:
        warnings.append(f"held-out RMS {q['holdout_rms_px']:.2f} px >> training {rms:.2f} px: overfitting "
                        "(try --dist-model k1k2 or more varied views)")
    if q["tilt_deg_median"] < 15:
        warnings.append(f"median board tilt {q['tilt_deg_median']:.0f} deg: mostly fronto-parallel views leave "
                        "the focal length poorly determined; tilt the board 20-45 deg")
    if cov["grid_cells_covered"] < 0.6:
        warnings.append(f"corners cover {100 * cov['grid_cells_covered']:.0f}% of the image: distortion is "
                        "extrapolated in the empty areas")
    if cov["outer_ring_corner_fraction"] < 0.05:
        warnings.append("almost no corners near the edge of the field: the distortion there is a guess "
                        "(a ChArUco board can be pushed to the edge)")
    if q["focal_std_pct"] > 0.5:
        warnings.append(f"focal length std {q['focal_std_pct']:.2f}%: this is directly a scale uncertainty")
    q["focal_std_note"] = ("from OpenCV's covariance: assumes a perfect model and independent corner noise, so "
                           "treat it as a lower bound (synthetic tests: true error up to ~2x this value)")
    if abs(fx / fy - 1) > 0.02:
        warnings.append(f"fx/fy = {fx / fy:.3f}: unusual for a laparoscope (anamorphic video? wrong resize?)")
    q["warnings"] = warnings

    cam = Camera(K, D, size, source=source_kind, optical_circle=circle, quality=q)
    per_rms = {v.frame: round(float(_view_rms(v.det, K, D, r, t)), 4)
               for v, r, t in zip(train, rvecs, tvecs, strict=True)}
    return CalibResult(cam, [v.frame for v in train], [v.frame for v in hold], rejected, per_rms, warnings)
