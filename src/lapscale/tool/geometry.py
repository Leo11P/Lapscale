"""Metric pose of an instrument shaft from ONE image: two straight edges + known diameter + intrinsics.

Geometry (exact for a cylinder seen by a pinhole camera; same model as Doignon & de Mathelin 2007 and
Wei et al. 2024, written with planes instead of Pluecker algebra):

* Each visible edge of the shaft back-projects to a plane through the camera centre. With undistorted
  normalised coordinates (x, y, 1), the image line a*x + b*y + c = 0 is the plane n . X = 0 with
  n = (a, b, c): no intrinsics left in the formula.
* Both planes are tangent to the cylinder, and a plane tangent to a cylinder is parallel to its axis,
  so the axis direction is s = n1 x n2 (normalised).
* In the cross-section perpendicular to s the two planes are two lines through the camera centre with
  an angle gamma between them (the angular width of the shaft), and the axis is the point at distance r
  from both, inside the wedge: rho = r / sin(gamma / 2) is the distance from the camera centre to the
  axis. Solving n1 . p = r, n2 . p = r, s . p = 0 gives p, the point of the axis closest to the camera.
* A pixel of the shaft gets its metric depth by intersecting its ray with the cylinder (front surface).

Conditioning (measured on synthetic shafts, see tests): the depth of the VISIBLE shaft is set by its
apparent width, so its relative error equals the relative error of the width (~2% per pixel of total
width error on a ~100 px wide shaft). The distance to the infinite axis (rho) is NOT well conditioned
when the shaft is tilted towards the camera: its closest point is then an extrapolation. Use
``ShaftEstimate.depth_mid`` / ``depth_at``, not ``distance``, downstream. There is no degenerate edge
orientation (the near-horizontal problem was specific to stereo).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..camera import Camera


# ------------------------------------------------------------------------------------- image lines
def to_normalized(points_px: np.ndarray, camera: Camera) -> np.ndarray:
    """Pixels of the RAW (distorted) frame -> undistorted normalised coordinates (x, y), z = 1."""
    pts = np.asarray(points_px, np.float64).reshape(-1, 1, 2)
    if pts.shape[0] == 0:
        return np.zeros((0, 2))
    if not camera.has_distortion():
        return cv2.undistortPoints(pts, camera.K, None).reshape(-1, 2)
    # the default 5 iterations leave ~1e-5 normalised error with strong barrel distortion
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-12)
    if hasattr(cv2, "undistortPointsIter"):                                   # OpenCV 4.x
        out = cv2.undistortPointsIter(pts, camera.K, camera.dist, None, None, crit)
    else:                                                                      # OpenCV 5.x
        out = cv2.undistortPoints(pts, camera.K, camera.dist, R=None, P=None, criteria=crit)
    return out.reshape(-1, 2)


@dataclass
class LineFit:
    line: np.ndarray        # (a, b, c), a^2 + b^2 = 1, in normalised coordinates
    rms: float              # orthogonal residual RMS, normalised units (x focal length = pixels)
    n: int
    span: float             # extent of the points along the line, normalised units


def fit_line(xy: np.ndarray) -> LineFit:
    """Total-least-squares line through 2D points (needs at least 2 distinct points)."""
    xy = np.asarray(xy, np.float64).reshape(-1, 2)
    if len(xy) < 2:
        raise ValueError("a line needs at least 2 points")
    c0 = xy.mean(0)
    _, sv, vt = np.linalg.svd(xy - c0, full_matrices=False)
    direction, normal = vt[0], vt[1]
    if sv[0] < 1e-12:
        raise ValueError("edge points coincide")
    t = (xy - c0) @ direction
    rms = float(sv[1] / np.sqrt(len(xy))) if len(sv) > 1 else 0.0
    line = np.array([normal[0], normal[1], -normal @ c0])
    return LineFit(line, rms, len(xy), float(t.max() - t.min()))


# ------------------------------------------------------------------------------------- cylinder
@dataclass
class Cylinder:
    """Infinite cylinder in camera coordinates (mm, OpenCV axes: x right, y down, z forward)."""

    p: np.ndarray           # point of the axis closest to the camera centre
    s: np.ndarray           # unit axis direction, oriented away from the camera (s_z >= 0)
    radius: float
    gamma: float            # angular width of the shaft as seen from the camera, rad

    @property
    def distance(self) -> float:
        """Distance camera centre -> axis (rho)."""
        return float(np.linalg.norm(self.p))

    @property
    def tangent_range(self) -> float:
        """Distance camera centre -> silhouette lines (where the edges are), sqrt(rho^2 - r^2)."""
        return float(np.sqrt(max(self.distance**2 - self.radius**2, 0.0)))

    def axis_image_line(self) -> np.ndarray:
        """Projection of the axis in normalised image coordinates, as (a, b, c) with a^2 + b^2 = 1."""
        n = np.cross(self.p, self.s)
        return n / np.linalg.norm(n[:2])

    def surface_depth(self, xy: np.ndarray) -> np.ndarray:
        """Depth Z (mm) of the FRONT surface along the rays of normalised points (N, 2); NaN if the ray
        misses the cylinder."""
        d = np.column_stack([np.asarray(xy, np.float64).reshape(-1, 2), np.ones(len(xy))])
        # |(t d - p) - ((t d - p) . s) s|^2 = r^2  ->  A t^2 + B t + C = 0
        dp = d - np.outer(d @ self.s, self.s)                 # d projected off the axis
        pp = -self.p + (self.p @ self.s) * self.s              # -p projected off the axis (p . s = 0)
        A = np.einsum("ij,ij->i", dp, dp)
        B = 2 * dp @ pp
        C = pp @ pp - self.radius**2
        disc = B**2 - 4 * A * C
        t = np.full(len(d), np.nan)
        ok = (disc >= 0) & (A > 1e-15)
        t[ok] = (-B[ok] - np.sqrt(disc[ok])) / (2 * A[ok])     # nearest intersection
        t[t <= 0] = np.nan
        return t                                              # d_z = 1, so t is the depth Z

    def closest_axis_point(self, xy: np.ndarray) -> np.ndarray:
        """3D point of the axis closest to the ray of each normalised point (N, 2) -> (N, 3).
        Used for the tip: its pixel lies on the projected axis."""
        d = np.column_stack([np.asarray(xy, np.float64).reshape(-1, 2), np.ones(len(xy))])
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        # minimise |p + u s - v d| over u, v
        sd = d @ self.s
        pd = d @ self.p
        ps = self.p @ self.s                                   # = 0 by construction
        denom = 1 - sd**2
        u = np.where(denom > 1e-12, (sd * pd - ps) / np.where(denom > 1e-12, denom, 1), 0.0)
        return self.p + np.outer(u, self.s)


def cylinder_from_planes(n1: np.ndarray, n2: np.ndarray, radius: float, inside: np.ndarray,
                         min_gamma_rad: float = 1e-4) -> Cylinder:
    """``n1``, ``n2``: normals of the two edge planes (through the camera centre); ``inside``: any
    direction inside the silhouette (e.g. the ray of a point between the edges) to fix the signs."""
    n1 = np.asarray(n1, np.float64) / np.linalg.norm(n1)
    n2 = np.asarray(n2, np.float64) / np.linalg.norm(n2)
    inside = np.asarray(inside, np.float64)
    if n1 @ inside < 0:
        n1 = -n1
    if n2 @ inside < 0:
        n2 = -n2
    s = np.cross(n1, n2)
    ns = np.linalg.norm(s)
    if ns < 1e-12:
        raise ValueError("the two edges back-project to the same plane: not two sides of a shaft")
    s /= ns
    if s[2] < 0 or (abs(s[2]) < 1e-12 and s[0] < 0):
        s = -s
    # wedge angle between the planes, measured inside the silhouette: the inward normals of the two
    # sides make an angle pi - gamma
    gamma = float(np.pi - np.arccos(np.clip(n1 @ n2, -1.0, 1.0)))
    if gamma < min_gamma_rad:
        raise ValueError(f"angular width {gamma:.2e} rad: edges (almost) coincide")
    p = np.linalg.solve(np.vstack([n1, n2, s]), np.array([radius, radius, 0.0]))
    if p @ inside < 0:
        raise ValueError("axis behind the camera: wrong edges")
    return Cylinder(p, s, float(radius), gamma)


# ------------------------------------------------------------------------------------- end to end
@dataclass
class ShaftEstimate:
    cylinder: Cylinder
    edge_a: LineFit
    edge_b: LineFit
    mid_xy: np.ndarray       # normalised point midway between the two edges (inside the silhouette)
    width_px: float          # apparent width there (pixels of the undistorted image)

    @property
    def distance(self) -> float:
        """Distance camera -> INFINITE axis. Poorly conditioned for shafts tilted towards the camera
        (the closest axis point can lie far outside the visible segment): not the quantity to trust."""
        return self.cylinder.distance

    @property
    def depth_mid(self) -> float:
        """Depth Z (mm) of the shaft surface at the middle of the visible edges. Well conditioned:
        this is what the scale fit relies on."""
        return float(self.cylinder.surface_depth(self.mid_xy[None])[0])

    def depth_at(self, xy: np.ndarray) -> np.ndarray:
        return self.cylinder.surface_depth(xy)


def shaft_from_edges(edge_a_px: np.ndarray, edge_b_px: np.ndarray, camera: Camera,
                     diameter_mm: float) -> ShaftEstimate:
    """Edges as points (>= 2 each) in pixels of the RAW frame (distortion is removed here)."""
    if diameter_mm <= 0:
        raise ValueError("diameter must be > 0")
    a = to_normalized(edge_a_px, camera)
    b = to_normalized(edge_b_px, camera)
    la, lb = fit_line(a), fit_line(b)
    # a point inside the silhouette: midpoint between the two edges' centroids
    mid = 0.5 * (a.mean(0) + b.mean(0))
    inside = np.array([mid[0], mid[1], 1.0])
    cyl = cylinder_from_planes(la.line, lb.line, diameter_mm / 2, inside)
    f = float(np.sqrt(camera.fx * camera.fy))
    width = _width_at(la.line, lb.line, mid) * f
    return ShaftEstimate(cyl, la, lb, mid, width)


def _width_at(l1: np.ndarray, l2: np.ndarray, pt: np.ndarray) -> float:
    """Distance between the two lines measured through ``pt`` along the normal of their bisector."""
    n1, n2 = l1[:2], l2[:2]
    if n1 @ n2 < 0:
        n2 = -n2
    nb = n1 + n2
    nb /= np.linalg.norm(nb)
    # intersections of the line pt + t*nb with l1 and l2
    t1 = -(l1 @ np.append(pt, 1.0)) / (l1[:2] @ nb)
    t2 = -(l2 @ np.append(pt, 1.0)) / (l2[:2] @ nb)
    return float(abs(t1 - t2))
