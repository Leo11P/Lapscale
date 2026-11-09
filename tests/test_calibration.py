"""End-to-end calibration on rendered views with known intrinsics."""
import cv2
import numpy as np
import pytest
from synthetic import CHARUCO, CHECKER, CIRCLE_TRUE, D_TRUE, K_TRUE, SIZE, make_views

from lapscale.calib.calibrate import CalibConfig
from lapscale.calib.run import run_board
from lapscale.camera import Camera


def _write(frames, folder):
    folder.mkdir()
    for k, f in enumerate(frames):
        cv2.imwrite(str(folder / f"{k:04d}.png"), f)
    return folder


def _check(cam: Camera, focal_tol=0.01, dist_tol=1.5):
    assert cam.fx == pytest.approx(K_TRUE[0, 0], rel=focal_tol)
    assert cam.fy == pytest.approx(K_TRUE[1, 1], rel=focal_tol)
    assert cam.cx == pytest.approx(K_TRUE[0, 2], abs=5)
    assert cam.cy == pytest.approx(K_TRUE[1, 2], abs=5)
    # distortion: compare the effect, not the coefficients (they trade off against each other)
    pts = np.array([[x, y] for x in np.linspace(250, 710, 6) for y in np.linspace(80, 460, 5)], np.float32)
    u_true = cv2.undistortPoints(pts.reshape(-1, 1, 2), K_TRUE, D_TRUE, P=K_TRUE).reshape(-1, 2)
    u_est = cv2.undistortPoints(pts.reshape(-1, 1, 2), cam.K, cam.dist, P=K_TRUE).reshape(-1, 2)
    assert np.abs(u_true - u_est).max() < dist_tol
    q = cam.quality
    assert q["reprojection_rms_px"] < 0.5
    assert q["holdout_rms_px"] is not None and q["holdout_rms_px"] < 0.8
    assert 0 < q["focal_std_pct"] < 1.0


def test_charuco(tmp_path):
    folder = _write(make_views(CHARUCO, 30, seed=0), tmp_path / "imgs")
    cam = run_board(folder, CHARUCO, tmp_path / "cam.yaml", CalibConfig(), log=lambda *_: None)
    _check(cam)
    c = cam.optical_circle
    assert c is not None and np.hypot(c.cx - CIRCLE_TRUE[0], c.cy - CIRCLE_TRUE[1]) < 3
    assert (tmp_path / "cam_report" / "calib_report.json").exists()
    assert (tmp_path / "cam_report" / "undistort_check.png").exists()
    assert Camera.load(tmp_path / "cam.yaml").source == "charuco"


def test_checkerboard(tmp_path):
    folder = _write(make_views(CHECKER, 30, seed=3, spread=0.3), tmp_path / "imgs")
    cam = run_board(folder, CHECKER, tmp_path / "cam.yaml", CalibConfig(), log=lambda *_: None)
    # central-only views (the full board must be visible): distortion at the rim is extrapolated
    _check(cam, focal_tol=0.015, dist_tol=2.5)


def test_video_input_and_duplicates(tmp_path):
    """Video with each view repeated 4 times: de-duplication must keep one per pose."""
    frames = [f for f in make_views(CHARUCO, 16, seed=5) for _ in range(4)]
    path = tmp_path / "calib.avi"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, SIZE)
    for f in frames:
        vw.write(f)
    vw.release()
    cam = run_board(path, CHARUCO, tmp_path / "cam.yaml", CalibConfig(min_sharpness_pct=0),
                    log=lambda *_: None)
    q = cam.quality
    assert q["n_views_train"] + q["n_views_holdout"] + q["n_views_rejected"] <= 16
    assert cam.fx == pytest.approx(K_TRUE[0, 0], rel=0.015)


def test_fronto_parallel_views_warn(tmp_path):
    folder = _write(make_views(CHARUCO, 14, seed=7, tilt_deg=(0.0, 4.0)), tmp_path / "imgs")
    cam = run_board(folder, CHARUCO, tmp_path / "cam.yaml", CalibConfig(), log=lambda *_: None)
    assert any("fronto-parallel" in w for w in cam.quality["warnings"])
