"""Shaft geometry against forward-generated synthetic shafts (known pose, known camera with distortion)."""
import numpy as np
import pytest
from synthetic_shaft import (
    CAMERA,
    edges,
    in_image,
    project,
    random_shaft,
    true_distance,
    visible_surface_points,
)

from lapscale.tool.geometry import Cylinder, fit_line, shaft_from_edges, to_normalized
from lapscale.tool.uncertainty import shaft_uncertainty, shift_edges


def _true_cylinder(a, u, r):
    p = a - (a @ u) * u                       # axis point closest to the camera
    return Cylinder(p, u if u[2] >= 0 else -u, r, 0.0)


def _true_depth(a, u, r, xy):
    """Depth of the true cylinder along the ray of a normalised point (intersection code is checked
    independently by test_surface_depth_matches_forward_points)."""
    return float(_true_cylinder(a, u, r).surface_depth(np.asarray(xy)[None])[0])


def _poses(n, seed, diameter=5.0, **kw):
    """Random shafts whose edges are fully inside the image."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        a, u = random_shaft(rng, **kw)
        ea, eb = edges(a, u, diameter / 2)
        if in_image(ea) and in_image(eb):
            out.append((a, u))
    return out


@pytest.mark.parametrize("diameter", [5.0, 10.0])
def test_exact_recovery(diameter):
    for a, u in _poses(40, seed=1, diameter=diameter):
        ea, eb = edges(a, u, diameter / 2)
        est = shaft_from_edges(ea, eb, CAMERA, diameter)
        cyl = est.cylinder
        assert cyl.distance == pytest.approx(true_distance(a, u), rel=1e-6)
        assert abs(abs(cyl.s @ u) - 1) < 1e-9                      # same axis direction
        # the recovered closest point lies on the true axis
        w = (cyl.p - a) - ((cyl.p - a) @ u) * u
        assert np.linalg.norm(w) < 1e-5


def test_edge_order_and_two_point_edges():
    a, u = _poses(1, seed=2)[0]
    ea, eb = edges(a, u, 2.5)
    d1 = shaft_from_edges(ea, eb, CAMERA, 5.0).distance
    d2 = shaft_from_edges(eb[::-1], ea, CAMERA, 5.0).distance           # swapped, reversed
    d3 = shaft_from_edges(ea[[0, -1]], eb[[5, 40]], CAMERA, 5.0).distance  # two "clicks" per edge
    assert d1 == pytest.approx(d2, rel=1e-9)
    assert d1 == pytest.approx(d3, rel=1e-6)


def test_surface_depth_matches_forward_points():
    rng = np.random.default_rng(3)
    for a, u in _poses(15, seed=3):
        cyl = shaft_from_edges(*edges(a, u, 2.5), CAMERA, 5.0).cylinder
        P = visible_surface_points(a, u, 2.5, rng)
        px = project(P)
        z = cyl.surface_depth(to_normalized(px, CAMERA))
        assert np.nanmax(np.abs(z - P[:, 2])) < 1e-4
        assert np.isfinite(z).all()


def test_rays_outside_the_shaft_miss():
    a, u = _poses(1, seed=4)[0]
    ea, eb = edges(a, u, 2.5)
    cyl = shaft_from_edges(ea, eb, CAMERA, 5.0).cylinder
    na, nb = to_normalized(ea, CAMERA), to_normalized(eb, CAMERA)
    away = 2 * na - 0.5 * (na + nb)            # beyond edge A, on the side away from B
    assert np.isnan(cyl.surface_depth(away)).all()


def test_axis_projection_and_tip():
    a, u = _poses(1, seed=5)[0]
    cyl = shaft_from_edges(*edges(a, u, 2.5), CAMERA, 5.0).cylinder
    axis_pts = np.array([a + t * u for t in np.linspace(-8, 8, 9)])
    xy = to_normalized(project(axis_pts), CAMERA)
    line = cyl.axis_image_line()
    assert np.abs(np.column_stack([xy, np.ones(len(xy))]) @ line).max() < 1e-8
    tip = a + 6.0 * u
    rec = cyl.closest_axis_point(to_normalized(project(tip[None]), CAMERA))[0]
    assert np.linalg.norm(rec - tip) < 1e-5


def test_degenerate_edges_raise():
    a, u = _poses(1, seed=6)[0]
    ea, _ = edges(a, u, 2.5)
    with pytest.raises(ValueError):
        shaft_from_edges(ea, ea + 1e-9, CAMERA, 5.0)


def test_fit_line_residual():
    rng = np.random.default_rng(0)
    x = np.linspace(-0.3, 0.3, 200)
    pts = np.column_stack([x, 0.2 * x + 0.05]) + rng.normal(0, 1e-3, (200, 2))
    lf = fit_line(pts)
    assert lf.rms == pytest.approx(1e-3, rel=0.2)
    assert lf.span == pytest.approx(0.6 * np.sqrt(1.04), rel=0.02)


def test_noise_error_and_bias_sensitivity():
    """0.5 px noise on 60-point edges: small depth error; 1 px outward shift: depth down by ~2/width."""
    rng = np.random.default_rng(7)
    errs = []
    for a, u in _poses(30, seed=7):
        ea, eb = edges(a, u, 2.5, rng=rng, noise_px=0.5)
        est = shaft_from_edges(ea, eb, CAMERA, 5.0)
        errs.append(est.depth_mid / _true_depth(a, u, 2.5, est.mid_xy) - 1)
        sa, sb = shift_edges(ea, eb, 1.0)
        sens = shaft_from_edges(sa, sb, CAMERA, 5.0).depth_at(est.mid_xy[None])[0] / est.depth_mid - 1
        assert sens == pytest.approx(-2.0 / est.width_px, rel=0.25)
    assert np.median(np.abs(errs)) < 0.005


def test_depth_is_better_conditioned_than_axis_distance():
    """Tilted shafts: the visible depth is stable while the distance to the infinite axis is not."""
    rng = np.random.default_rng(10)
    poses = [p for p in _poses(60, seed=10) if abs(p[1][2]) > np.sin(np.radians(45))][:10]
    ratios = []
    for k, (a, u) in enumerate(poses):
        ea, eb = edges(a, u, 2.5, rng=rng, noise_px=0.3)
        unc = shaft_uncertainty(ea, eb, CAMERA, 5.0, point_sd_px=0.3, n=100, seed=k)
        ratios.append((unc.distance_axis_sd_random / unc.distance_axis) / (unc.sd_random / unc.depth))
    assert np.median(ratios) > 3


def test_monte_carlo_coverage():
    """Predicted random sd of the depth must cover the true error about 68% of the time."""
    rng = np.random.default_rng(8)
    z = []
    for k, (a, u) in enumerate(_poses(80, seed=8)):
        ea, eb = edges(a, u, 2.5, n=20, rng=rng, noise_px=1.0)
        unc = shaft_uncertainty(ea, eb, CAMERA, 5.0, point_sd_px=1.0, n=150, seed=k)
        mid = shaft_from_edges(ea, eb, CAMERA, 5.0).mid_xy
        z.append((unc.depth - _true_depth(a, u, 2.5, mid)) / unc.sd_random)
    cover = np.mean(np.abs(z) <= 1)
    assert 0.55 <= cover <= 0.82, cover


def test_systematic_sources_widen_the_interval():
    a, u = _poses(1, seed=9)[0]
    ea, eb = edges(a, u, 2.5)
    unc = shaft_uncertainty(ea, eb, CAMERA, 5.0, point_sd_px=0.3, bias_sd_px=1.0, focal_rel_sd=0.01, n=200)
    assert unc.sd_total > 2 * unc.sd_random
    assert unc.p5 < unc.depth < unc.p95
    assert unc.bias_sensitivity_pct_per_px < 0
