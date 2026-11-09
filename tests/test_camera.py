import numpy as np
import pytest

from lapscale.camera import Camera, OpticalCircle, resolve_camera


def cam():
    K = [[1000, 0, 639.5], [0, 1000, 359.5], [0, 0, 1]]
    return Camera(K, [-0.2, 0.05, 0, 0, 0], (1280, 720), source="import", optical_circle=OpticalCircle(640, 360, 350))


def test_yaml_roundtrip(tmp_path):
    c = cam()
    c.quality = {"reprojection_rms_px": np.float64(0.31), "grid": np.eye(2, dtype=int)}
    c2 = Camera.load(c.save(tmp_path / "c.yaml"))
    assert np.allclose(c.K, c2.K) and np.allclose(c.dist, c2.dist)
    assert c2.image_size == (1280, 720) and c2.optical_circle.r == pytest.approx(350)
    assert c2.quality["reprojection_rms_px"] == pytest.approx(0.31)


def test_resize_keeps_fov_and_pixel_centres():
    c = cam()
    d = c.for_size((1920, 1080))
    assert d.fov_x_deg() == pytest.approx(c.fov_x_deg(), abs=1e-9)
    assert d.cx == pytest.approx(959.5) and d.cy == pytest.approx(539.5)
    assert d.fx == pytest.approx(1500)
    assert d.optical_circle.r == pytest.approx(525)


def test_resize_refuses_other_aspect_ratio():
    with pytest.raises(ValueError, match="aspect ratio"):
        cam().for_size((1280, 960))


def test_resolve(tmp_path):
    p = cam().save(tmp_path / "c.yaml")
    assert resolve_camera("moge") is None
    assert resolve_camera(p, (640, 360)).fx == pytest.approx(500)


def test_bad_schema(tmp_path):
    p = tmp_path / "x.yaml"
    p.write_text("K: [[1,0,0],[0,1,0],[0,0,1]]\n")
    with pytest.raises(ValueError, match="schema"):
        Camera.load(p)
