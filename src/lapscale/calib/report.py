"""Calibration report: JSON with everything needed to judge the result, and two images to eyeball it."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from ..camera import Camera, _plain
from .calibrate import CalibResult, View


def write_report(out_dir: Path, result: CalibResult, views: list[View], frame: np.ndarray | None,
                 extra: dict | None = None) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cam = result.camera
    rep = dict(camera=cam.to_dict(), train_frames=result.train_frames, holdout_frames=result.holdout_frames,
               rejected_frames=result.rejected_frames, per_view_rms_px=result.per_view_rms,
               warnings=result.warnings, **(extra or {}))
    paths = [out_dir / "calib_report.json"]
    paths[0].write_text(json.dumps(_plain(rep), indent=2))
    if frame is not None:
        paths.append(out_dir / "coverage.png")
        cv2.imwrite(str(paths[-1]), coverage_image(frame, cam, views, result))
        paths.append(out_dir / "undistort_check.png")
        cv2.imwrite(str(paths[-1]), undistort_image(frame, cam))
    return paths


def coverage_image(frame: np.ndarray, cam: Camera, views: list[View], result: CalibResult) -> np.ndarray:
    """All corners on one frame: green = training, orange = held-out, red = rejected views."""
    img = (frame * 0.5).astype(np.uint8)
    train, hold, rej = set(result.train_frames), set(result.holdout_frames), set(result.rejected_frames)
    for v in views:
        col = (0, 220, 0) if v.frame in train else (0, 140, 255) if v.frame in hold else \
              (0, 0, 255) if v.frame in rej else None
        if col is None:
            continue
        for x, y in v.det.image_points:
            cv2.circle(img, (int(round(x)), int(round(y))), 2, col, -1, cv2.LINE_AA)
    if cam.optical_circle is not None:
        c = cam.optical_circle
        cv2.circle(img, (int(round(c.cx)), int(round(c.cy))), int(round(c.r)), (0, 255, 255), 1, cv2.LINE_AA)
    q = cam.quality
    txt = (f"rms {q.get('reprojection_rms_px')} px | holdout {q.get('holdout_rms_px')} px | "
           f"f std {q.get('focal_std_pct')}% | views {q.get('n_views_train')}")
    cv2.putText(img, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def undistort_image(frame: np.ndarray, cam: Camera) -> np.ndarray:
    """Original | undistorted, with a straight-line grid: straight edges must look straight on the right."""
    m1, m2, _ = cam.undistort_maps(alpha=0.0)
    und = cv2.remap(frame, m1, m2, cv2.INTER_LINEAR)
    both = np.hstack([frame, und])
    h, w = frame.shape[:2]
    for k in range(1, 8):
        x = w + k * w // 8
        cv2.line(both, (x, 0), (x, h - 1), (0, 255, 255), 1)
        y = k * h // 8
        cv2.line(both, (w, y), (2 * w - 1, y), (0, 255, 255), 1)
    return both
