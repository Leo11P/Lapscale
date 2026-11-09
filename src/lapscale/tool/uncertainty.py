"""Uncertainty of the shaft depth (and hence of the metric scale).

The reported quantity is the depth of the shaft surface at a FIXED reference ray (the middle of the
visible edges), which is what the scale fit uses. The distance to the infinite axis is also given,
for diagnostics only: it is poorly conditioned for tilted shafts.

Three error sources, kept apart because they behave differently over a video:

* ``point_sd_px``: independent noise on each edge point (clicks, contour jitter). Random: averages out
  over frames, and over the many points of an automatic contour.
* ``bias_sd_px``: common outward/inward shift of BOTH edges (a mask systematically too wide or too
  narrow). For a given segmentation model this is mostly SYSTEMATIC: it does not average out over
  frames. Positive shift = shaft looks wider = depth underestimated.
* ``focal_rel_sd``: relative error of the focal length, from the calibration. Also systematic.

``sd_random`` contains only the first; ``sd_total`` all three. The bias sensitivity (% of depth per
pixel of outward shift of both edges) is reported deterministically: it says how good the
segmentation edges must be.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..camera import Camera
from .geometry import shaft_from_edges


@dataclass
class ShaftUncertainty:
    depth: float                    # mm, surface depth at the reference ray, unperturbed edges
    sd_random: float                # mm
    sd_total: float                 # mm
    p5: float                       # mm, 5th / 95th percentile with all sources
    p95: float
    bias_sensitivity_pct_per_px: float   # % change of depth for +1 px outward shift of both edges
    distance_axis: float            # mm, camera -> infinite axis (diagnostic)
    distance_axis_sd_random: float  # mm (diagnostic; large for tilted shafts)
    valid_fraction: float

    @property
    def rel_sd_total_pct(self) -> float:
        return 100 * self.sd_total / self.depth

    @property
    def rel_sd_random_pct(self) -> float:
        return 100 * self.sd_random / self.depth


def _outward_normals(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit normals (pixel space) of the two edges, each pointing away from the other edge."""
    out = []
    for e, other in ((a, b), (b, a)):
        c = e.mean(0)
        _, _, vt = np.linalg.svd(e - c)
        n = vt[1]
        if n @ (other.mean(0) - c) > 0:
            n = -n
        out.append(n)
    return out[0], out[1]


def shift_edges(a: np.ndarray, b: np.ndarray, shift_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Move both edges outward (shift > 0) or inward (shift < 0) by ``shift_px`` pixels."""
    na, nb = _outward_normals(a, b)
    return a + shift_px * na, b + shift_px * nb


def _robust_sd(x: np.ndarray) -> float:
    q16, q84 = np.percentile(x, [16, 84])
    return float(0.5 * (q84 - q16))


def shaft_uncertainty(edge_a_px: np.ndarray, edge_b_px: np.ndarray, camera: Camera, diameter_mm: float,
                      point_sd_px: float = 0.5, bias_sd_px: float = 0.0, focal_rel_sd: float = 0.0,
                      n: int = 300, seed: int = 0) -> ShaftUncertainty:
    a = np.asarray(edge_a_px, np.float64).reshape(-1, 2)
    b = np.asarray(edge_b_px, np.float64).reshape(-1, 2)
    rng = np.random.default_rng(seed)
    est0 = shaft_from_edges(a, b, camera, diameter_mm)
    ref = est0.mid_xy[None]                         # the same ray for every draw
    z0 = est0.depth_mid

    sa, sb = shift_edges(a, b, 1.0)
    sens = 100 * (float(shaft_from_edges(sa, sb, camera, diameter_mm).depth_at(ref)[0]) / z0 - 1)

    def draw(with_systematic: bool) -> tuple[np.ndarray, np.ndarray]:
        z = np.full(n, np.nan)
        rho = np.full(n, np.nan)
        for k in range(n):
            pa = a + rng.normal(0, point_sd_px, a.shape)
            pb = b + rng.normal(0, point_sd_px, b.shape)
            cam = camera
            if with_systematic:
                if bias_sd_px > 0:
                    pa, pb = shift_edges(pa, pb, rng.normal(0, bias_sd_px))
                if focal_rel_sd > 0:
                    K = camera.K.copy()
                    e = 1 + rng.normal(0, focal_rel_sd)
                    K[0, 0] *= e
                    K[1, 1] *= e
                    cam = Camera(K, camera.dist, camera.image_size, camera.source)
            try:
                est = shaft_from_edges(pa, pb, cam, diameter_mm)
            except ValueError:
                continue
            # the reference ray is fixed in PIXELS: re-normalise it with the perturbed camera
            if cam is not camera:
                px = ref[0] * [camera.fx, camera.fy] + [camera.cx, camera.cy]
                xy = (px - [cam.cx, cam.cy]) / [cam.fx, cam.fy]
                z[k] = est.depth_at(xy[None])[0]
            else:
                z[k] = est.depth_at(ref)[0]
            rho[k] = est.distance
        return z, rho

    z_r, rho_r = draw(False)
    z_t, _ = draw(True)
    ok_r, ok_t = np.isfinite(z_r), np.isfinite(z_t)
    valid = float(min(ok_r.mean(), ok_t.mean()))
    if valid < 0.5:
        nan = float("nan")
        return ShaftUncertainty(z0, nan, nan, nan, nan, float(sens), est0.distance, nan, valid)
    p5, p95 = np.percentile(z_t[ok_t], [5, 95])
    return ShaftUncertainty(z0, _robust_sd(z_r[ok_r]), _robust_sd(z_t[ok_t]), float(p5), float(p95),
                            float(sens), est0.distance, _robust_sd(rho_r[np.isfinite(rho_r)]), valid)
