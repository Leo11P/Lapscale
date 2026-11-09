"""Synthetic instrument masks in the RAW (distorted) frame: a finite shaft entering from the border of the
field plus a jaw-like blob at its tip, rendered by ray casting with supersampling (soft mask = coverage)."""
from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
from synthetic_shaft import CAMERA, silhouette_points

SS = 3   # supersampling per axis


@lru_cache(maxsize=4)
def _rays(ss: int = SS):
    """Normalised ray (x, y) of every sub-pixel of the raw frame."""
    w, h = CAMERA.image_size
    off = (np.arange(ss) + 0.5) / ss - 0.5
    xs = (np.arange(w)[:, None] + off[None]).ravel()
    ys = (np.arange(h)[:, None] + off[None]).ravel()
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], 1).reshape(-1, 1, 2).astype(np.float64)
    n = cv2.undistortPoints(pts, CAMERA.K, CAMERA.dist).reshape(-1, 2)
    return n, (h * ss, w * ss)


def _hit_cylinder(n, a, u, r, t_range):
    """Ray (through normalised n) hits the finite cylinder a + t u, t in t_range."""
    d = np.column_stack([n, np.ones(len(n))])
    dp = d - np.outer(d @ u, u)
    ap = -a + (a @ u) * u                       # -a projected off the axis
    A = np.einsum("ij,ij->i", dp, dp)
    B = 2 * dp @ ap
    C = ap @ ap - r**2
    disc = B**2 - 4 * A * C
    hit = np.zeros(len(n), bool)
    ok = disc >= 0
    sq = np.sqrt(np.where(ok, disc, 0))
    for sgn in (-1, 1):
        lam = (-B + sgn * sq) / (2 * A)
        X = lam[:, None] * d
        t = (X - a) @ u
        hit |= ok & (lam > 0) & (t >= t_range[0]) & (t <= t_range[1])
    return hit


def _hit_sphere(n, c, r):
    d = np.column_stack([n, np.ones(len(n))])
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    b = d @ c
    return (b > 0) & (c @ c - b**2 <= r**2)


def render_instrument(a, u, r, t_range=(-60.0, 8.0), jaw=True):
    """Soft mask (coverage in [0, 1]) of a shaft from a + t0 u to a + t1 u, jaw sphere at the t1 end."""
    n, (H, W) = _rays()
    hit = _hit_cylinder(n, a, u, r, t_range)
    if jaw:
        tip = a + t_range[1] * u
        hit |= _hit_sphere(n, tip + 1.2 * r * u, 1.7 * r)
    cov = hit.reshape(H, W).astype(np.float32)
    h, w = H // SS, W // SS
    return cov.reshape(h, SS, w, SS).mean(axis=(1, 3))


def true_edge_lines_undistorted(a, u, r):
    """Exact edge lines (undistorted pixels, (a, b, c) with a^2 + b^2 = 1) from the silhouette generators."""
    lines = []
    for g in silhouette_points(a, u, r, np.linspace(-30, 30, 2)):
        xn = g[:, :2] / g[:, 2:3]
        px = xn * [CAMERA.fx, CAMERA.fy] + [CAMERA.cx, CAMERA.cy]
        d = px[1] - px[0]
        nrm = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        lines.append(np.array([nrm[0], nrm[1], -nrm @ px[0]]))
    return lines
