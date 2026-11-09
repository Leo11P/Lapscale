"""Stage 2 plumbing with a fake backend: undistortion, FOV, cache, and the bridge to the shaft geometry
(sampling the depth map at the normalised coordinates of RAW pixels)."""
import cv2
import numpy as np
import pytest
from synthetic_shaft import CAMERA

from lapscale.camera import Camera, OpticalCircle
from lapscale.depth.base import DepthPrediction
from lapscale.depth.run import run_depth
from lapscale.depth.view import PinholeView
from lapscale.io.cache import DepthStore
from lapscale.tool.geometry import to_normalized

PLANE_N = np.array([0.15, -0.1, 1.0])     # plane n . X = D, tilted
PLANE_D = 90.0
SCALE = 0.013                            # the "network units" differ from mm by an unknown factor


def plane_depth(xy):
    return PLANE_D / (np.column_stack([xy, np.ones(len(xy))]) @ PLANE_N)


class PlaneBackend:
    """Returns the exact depth of a plane for the pixels of the image it receives, assuming that image
    is a pinhole view with intrinsics ``K`` (what the real network is told through the FOV)."""

    name = "fake:plane"

    def __init__(self, K, estimate_focal=False):
        self.K, self.estimate, self.calls = K, estimate_focal, 0

    def predict(self, img, fov_x_deg=None):
        self.calls += 1
        h, w = img.shape[:2]
        if fov_x_deg is not None:
            assert fov_x_deg == pytest.approx(Camera(self.K, [], (w, h)).fov_x_deg(), abs=1e-6)
        yy, xx = np.mgrid[0:h, 0:w]
        xy = np.column_stack([(xx.ravel() - self.K[0, 2]) / self.K[0, 0], (yy.ravel() - self.K[1, 2]) / self.K[1, 1]])
        depth = (SCALE * plane_depth(xy)).reshape(h, w).astype(np.float32)
        fx = self.K[0, 0] if self.estimate else None
        return DepthPrediction(depth, np.ones((h, w), bool), fx, fx)


def _video(path, n=6):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, CAMERA.image_size)
    for _ in range(n):
        vw.write(np.full((CAMERA.image_size[1], CAMERA.image_size[0], 3), 128, np.uint8))
    vw.release()
    return path


def test_view_undistorts_where_the_geometry_expects():
    view = PinholeView(CAMERA, CAMERA.image_size)
    raw = np.zeros((540, 960, 3), np.uint8)
    p_raw = np.array([[820.0, 120.0]])                  # far from the centre: strong distortion
    cv2.circle(raw, tuple(int(v) for v in p_raw[0]), 3, (255, 255, 255), -1)
    und = view.apply(raw)[..., 0].astype(float)
    yy, xx = np.nonzero(und > 100)
    w = und[yy, xx]
    found = np.array([np.sum(xx * w) / w.sum(), np.sum(yy * w) / w.sum()])
    xy = to_normalized(p_raw, CAMERA)[0]
    expected = xy * [view.K_view[0, 0], view.K_view[1, 1]] + view.K_view[:2, 2]
    assert np.linalg.norm(found - expected) < 1.0


def test_optical_mask_follows_the_undistortion():
    cam = Camera(CAMERA.K, CAMERA.dist, CAMERA.image_size, optical_circle=OpticalCircle(481, 270, 300))
    view = PinholeView(cam, cam.image_size)
    assert view.mask[268, 482] and not view.mask[5, 5]
    assert 0.3 < view.mask.mean() < 0.95


def test_run_cache_and_ray_sampling(tmp_path):
    video = _video(tmp_path / "v.avi")
    view = PinholeView(CAMERA, CAMERA.image_size)
    be = PlaneBackend(view.K_view)
    s = run_depth(video, CAMERA, be, tmp_path / "run", "plane", [0, 2, 4], log=lambda *_: None)
    assert s["n_done"] == 3 and be.calls == 3
    store = DepthStore(tmp_path / "run", "plane")
    assert store.frames() == [0, 2, 4] and store.meta()["undistorted"]
    # the bridge: raw pixels -> normalised rays (as the shaft geometry does) -> depth of the network
    rng = np.random.default_rng(0)
    raw = np.column_stack([rng.uniform(100, 860, 300), rng.uniform(60, 480, 300)])
    xy = to_normalized(raw, CAMERA)
    d = store.load(2).sample(xy)
    ok = np.isfinite(d)
    assert ok.mean() > 0.95
    assert np.allclose(d[ok] / SCALE, plane_depth(xy[ok]), rtol=2e-3)   # float16 + bilinear
    # resume: nothing recomputed; another configuration in the same tag is refused
    s2 = run_depth(video, CAMERA, be, tmp_path / "run", "plane", [0, 2, 4], log=lambda *_: None)
    assert s2["n_done"] == 0 and be.calls == 3
    with pytest.raises(ValueError, match="different"):
        run_depth(video, CAMERA, be, tmp_path / "run", "plane", [1], config={"x": 1}, log=lambda *_: None)


def test_without_calibration_the_estimated_focal_is_stored(tmp_path):
    video = _video(tmp_path / "v.avi", n=2)
    K = np.array([[650.0, 0, 479.5], [0, 650.0, 269.5], [0, 0, 1]])
    run_depth(video, None, PlaneBackend(K, estimate_focal=True), tmp_path / "run", "nocal", [0, 1],
              log=lambda *_: None)
    f = DepthStore(tmp_path / "run", "nocal").load(1)
    assert f.K[0, 0] == pytest.approx(650.0) and f.K[0, 2] == pytest.approx(479.5)


def test_sampling_refuses_invalid_neighbours():
    from lapscale.io.cache import DepthFrame
    depth = np.ones((10, 10), np.float32)
    valid = np.ones((10, 10), bool)
    valid[:, 5:] = False
    f = DepthFrame(0, depth, valid, np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1]]))
    d = f.sample(np.array([[2.0, 2.0], [4.5, 2.0], [7.0, 2.0]]))
    assert d[0] == pytest.approx(1.0) and np.isnan(d[1]) and np.isnan(d[2])


def test_cli_depth(tmp_path, monkeypatch):
    import lapscale.depth as depth_pkg
    from lapscale.cli import main
    video = _video(tmp_path / "v.avi", n=4)
    cam = CAMERA.save(tmp_path / "cam.yaml")
    view = PinholeView(CAMERA, CAMERA.image_size)
    monkeypatch.setattr(depth_pkg, "make_backend", lambda name, **kw: PlaneBackend(view.K_view))
    assert main(["depth", "--video", str(video), "--camera", str(cam), "--run-dir", str(tmp_path / "run"),
                 "--tag", "t", "--max-frames", "2"]) == 0
    assert DepthStore(tmp_path / "run", "t").frames() == [0, 3]
