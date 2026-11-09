"""Synthetic instrument shaft: a cylinder with known pose seen by a camera with known intrinsics and
distortion. Edges are generated FORWARD (exact silhouette generators projected with distortion), so the
tests check the inverse geometry against an independent construction."""
from __future__ import annotations

import cv2
import numpy as np

from lapscale.camera import Camera

K = np.array([[700.0, 0, 482.0], [0, 700.0, 268.0], [0, 0, 1]])
D = np.array([-0.28, 0.09, 0.0005, -0.0003, 0.0])
SIZE = (960, 540)
CAMERA = Camera(K, D, SIZE, source="manual")


def random_shaft(rng, distance=(30.0, 150.0), max_tilt_deg=75.0):
    """Axis point ``a`` (where the visible segment is centred) and unit direction ``u``."""
    z = rng.uniform(*distance)
    a = np.array([rng.uniform(-0.25, 0.25) * z, rng.uniform(-0.15, 0.15) * z, z])
    ang = rng.uniform(0, 2 * np.pi)
    tilt = np.radians(rng.uniform(0, max_tilt_deg))             # angle out of the image plane
    u = np.array([np.cos(ang) * np.cos(tilt), np.sin(ang) * np.cos(tilt), np.sin(tilt)])
    return a, u


def true_distance(a, u):
    """Distance camera centre -> axis line."""
    w = -a + (a @ u) * u
    return float(np.linalg.norm(w))


def silhouette_points(a, u, r, ts):
    """3D points on the two silhouette generators (tangent lines seen from the origin)."""
    w = -a + (a @ u) * u                    # from the axis towards the camera, perpendicular to u
    rho = np.linalg.norm(w)
    wh = w / rho
    vh = np.cross(u, wh)
    phi = np.arccos(r / rho)
    gens = []
    for sgn in (1, -1):
        q = a + r * (np.cos(phi) * wh + sgn * np.sin(phi) * vh)
        gens.append(np.array([q + t * u for t in ts]))
    return gens


def project(P, camera=CAMERA):
    pix, _ = cv2.projectPoints(np.asarray(P, np.float64), np.zeros(3), np.zeros(3), camera.K, camera.dist)
    return pix.reshape(-1, 2)


def edges(a, u, r, n=60, half_len=10.0, rng=None, noise_px=0.0, camera=CAMERA):
    """Pixel points of the two edges (raw, distorted image), optional isotropic noise."""
    ts = np.linspace(-half_len, half_len, n)
    ea, eb = (project(g, camera) for g in silhouette_points(a, u, r, ts))
    if noise_px and rng is not None:
        ea = ea + rng.normal(0, noise_px, ea.shape)
        eb = eb + rng.normal(0, noise_px, eb.shape)
    return ea, eb


def visible_surface_points(a, u, r, rng, n=200, half_len=10.0):
    """Random 3D points on the part of the surface facing the camera."""
    w = -a + (a @ u) * u
    wh = w / np.linalg.norm(w)
    vh = np.cross(u, wh)
    pts = []
    while len(pts) < n:
        th = rng.uniform(-np.pi, np.pi)
        t = rng.uniform(-half_len, half_len)
        normal = np.cos(th) * wh + np.sin(th) * vh
        q = a + t * u + r * normal
        if normal @ (-q) > 1e-3:            # faces the camera
            pts.append(q)
    return np.array(pts)


def in_image(px, size=SIZE, margin=5):
    w, h = size
    return np.all((px[:, 0] > margin) & (px[:, 0] < w - margin) & (px[:, 1] > margin) & (px[:, 1] < h - margin))
