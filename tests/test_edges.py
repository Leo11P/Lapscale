"""Edge extraction from instrument masks rendered in the distorted frame (shaft + jaw)."""
import cv2
import numpy as np
import pytest
from synthetic_scene import render_instrument, true_edge_lines_undistorted
from synthetic_shaft import CAMERA

from lapscale.camera import Camera, OpticalCircle
from lapscale.tool.edges import extract_shaft_edges, shaft_region, undistorted_px
from lapscale.tool.geometry import Cylinder, shaft_from_edges

POSES = [  # axis point, direction, radius (5 and 10 mm shafts, 60-110 mm, widths ~34-132 px)
    (np.array([8.0, 4.0, 80.0]), np.array([0.8, 0.35, 0.35]), 2.5),
    (np.array([-5.0, 2.0, 60.0]), np.array([-0.9, 0.1, 0.4]), 5.0),
    (np.array([0.0, -3.0, 110.0]), np.array([0.3, -0.9, 0.3]), 2.5),
]


@pytest.fixture(scope="module")
def scenes():
    out = []
    for a, u, r in POSES:
        u = u / np.linalg.norm(u)
        out.append((a, u, r, render_instrument(a, u, r)))
    return out


def _true_depth(a, u, r, xy):
    p = a - (a @ u) * u
    return float(Cylinder(p, u if u[2] >= 0 else -u, r, 0.0).surface_depth(xy[None])[0])


def _outward_offsets(edges, a, u, r):
    """Mean signed distance (undistorted px) of the extracted edge points from the true edges, positive
    = outward (mask wider than the truth)."""
    lines = true_edge_lines_undistorted(a, u, r)
    out = []
    for pts in (edges.a_raw, edges.b_raw):
        und = undistorted_px(pts, CAMERA)
        k = int(np.argmin([np.median(np.abs(und @ ln[:2] + ln[2])) for ln in lines]))
        ln, other = lines[k], lines[1 - k]
        sign = -np.sign(ln[:2] @ (-other[2] * other[:2]) + ln[2])
        out.append(float(np.mean((und @ ln[:2] + ln[2]) * sign)))
    return out


@pytest.mark.parametrize("kind", ["soft", "binary"])
def test_edges_unbiased_and_depth_accurate(scenes, kind):
    for a, u, r, soft in scenes:
        m = soft if kind == "soft" else (soft > 0.5).astype(np.float32)
        res = extract_shaft_edges(m, CAMERA)
        assert res.reason == "ok", res.reason
        off = _outward_offsets(res.edges, a, u, r)
        assert max(abs(o) for o in off) < 0.1          # no 0.5 px contour bias
        est = shaft_from_edges(res.edges.a_raw, res.edges.b_raw, CAMERA, 2 * r)
        assert est.depth_mid == pytest.approx(_true_depth(a, u, r, est.mid_xy), rel=0.003)


def test_dilated_mask_is_measured_not_absorbed(scenes):
    """A mask 1 px too wide must show up as ~1-1.4 px outward offset and the matching depth error."""
    for a, u, r, soft in scenes:
        m = cv2.dilate((soft > 0.5).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
        res = extract_shaft_edges(m, CAMERA)
        off = np.mean(_outward_offsets(res.edges, a, u, r))
        assert 0.9 < off < 1.5
        est = shaft_from_edges(res.edges.a_raw, res.edges.b_raw, CAMERA, 2 * r)
        err = est.depth_mid / _true_depth(a, u, r, est.mid_xy) - 1
        assert err == pytest.approx(-2 * off / res.edges.width_px, rel=0.3)


def test_jaw_ignored_and_region_inside(scenes):
    a, u, r, soft = scenes[1]                       # 10 mm shaft with a large jaw blob
    res = extract_shaft_edges(soft, CAMERA)
    reg = shaft_region(res.edges, CAMERA, soft, inset_px=3)
    assert reg.sum() > 1000
    assert soft[reg].min() > 0.99                   # strictly inside the instrument
    tip = a + 8.0 * u + 1.2 * r * u                 # jaw sphere centre
    px = cv2.projectPoints(tip[None], np.zeros(3), np.zeros(3), CAMERA.K, CAMERA.dist)[0].ravel()
    yy, xx = np.nonzero(reg)
    jaw_r_px = 1.7 * r * CAMERA.fx / tip[2]
    assert np.mean(np.hypot(xx - px[0], yy - px[1]) < jaw_r_px) < 0.05          # (almost) not on the jaw


def test_rim_of_the_field_is_not_an_edge(scenes):
    """Instrument cut by the optical disc: the cut must not be taken as an edge."""
    a, u, r, soft = scenes[0]
    cam = Camera(CAMERA.K, CAMERA.dist, CAMERA.image_size, optical_circle=OpticalCircle(480, 270, 250))
    m = soft * cam.mask()
    res = extract_shaft_edges(m, cam)
    if res.reason == "ok":
        assert max(abs(o) for o in _outward_offsets(res.edges, a, u, r)) < 0.2


def test_rejections():
    h, w = CAMERA.image_size[1], CAMERA.image_size[0]
    blob = np.zeros((h, w), np.float32)
    cv2.circle(blob, (480, 270), 60, 1.0, -1)
    assert extract_shaft_edges(blob, CAMERA).edges is None
    assert extract_shaft_edges(np.zeros((h, w), np.float32), CAMERA).reason == "empty mask"
    with pytest.raises(ValueError):
        extract_shaft_edges(np.zeros((10, 10), np.float32), CAMERA)
