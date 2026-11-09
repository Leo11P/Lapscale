"""Instrument mask -> the two straight edges of the shaft, with sub-pixel precision.

Steps (each one there for a measured reason, see docs/decisions.md):

1. Boundary of the instance (prob > 0.5), minus the points on the image border or on the rim of the
   optical disc: there the mask is cut by the field of view, not by the instrument.
2. **Sub-pixel refinement**: each boundary point is moved along its local normal to the 0.5 crossing of
   the soft mask (bilinear sampling). For a binary mask this is the midpoint between the last inside and
   first outside pixel: without it, OpenCV contours (centres of the inside pixels) bias both edges
   inward by 0.5 px, i.e. ~1% of depth on a 100 px wide shaft.
3. Points are **undistorted** before fitting: in the raw frame the edges are curves.
4. **Sequential RANSAC** extracts up to ``max_lines`` straight lines; the shaft is the pair that is nearly
   parallel, on opposite sides of the mask and side by side along the shaft. Jaws, tip and clutter are
   not straight and drop out without part labels.
5. Quality gate with an explicit reason when the frame is rejected.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..camera import Camera
from .geometry import fit_line, to_normalized


@dataclass
class EdgeConfig:
    border_margin_px: float = 4.0        # drop boundary points this close to the frame / optical rim
    refine_halfwidth_px: float = 3.0     # search range of the 0.5 crossing along the normal
    refine_step_px: float = 0.25
    tangent_k: int = 3                   # neighbours used for the local tangent
    max_lines: int = 4
    ransac_iters: int = 400
    ransac_tol_px: float = 1.5           # inlier distance, undistorted pixels
    max_gap_px: float = 8.0              # an edge is one contiguous run of boundary points
    min_inliers: int = 30
    min_span_px: float = 50.0            # length of each edge
    min_overlap: float = 0.4             # edges side by side over >= this fraction of the shorter one
    max_edge_angle_deg: float = 25.0     # convergence of the two edges (perspective)
    min_width_px: float = 6.0
    max_rms_px: float = 0.8              # straightness of each edge after refinement
    seed: int = 0


@dataclass
class ShaftEdges:
    a_raw: np.ndarray          # (N, 2) refined inlier points, raw pixels  -> input of shaft_from_edges
    b_raw: np.ndarray
    line_a: np.ndarray         # (a, b, c), a^2 + b^2 = 1, undistorted pixels
    line_b: np.ndarray
    rms_a: float               # undistorted pixels
    rms_b: float
    span_a: float
    span_b: float
    angle_deg: float           # between the two edges
    width_px: float            # at the middle of the overlap


@dataclass
class EdgeResult:
    edges: ShaftEdges | None
    reason: str = "ok"
    diagnostics: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------------------- helpers
def undistorted_px(points_raw: np.ndarray, camera: Camera) -> np.ndarray:
    """Raw pixels -> pixels of the ideal pinhole image with the same K."""
    n = to_normalized(points_raw, camera)
    return n * [camera.fx, camera.fy] + [camera.cx, camera.cy]


def _valid_region(shape: tuple[int, int], camera: Camera, margin: float) -> np.ndarray:
    h, w = shape
    valid = np.zeros((h, w), bool)
    m = int(np.ceil(margin))
    valid[m:h - m, m:w - m] = True
    if camera.optical_circle is not None and camera.image_size == (w, h):
        valid &= camera.optical_circle.mask((w, h), margin_px=margin)
    return valid


def _refine_subpixel(prob: np.ndarray, pts: np.ndarray, cfg: EdgeConfig) -> np.ndarray:
    """Move contour points along their outward normal to the 0.5 crossing. Returns (M, 2), M <= N."""
    n = len(pts)
    if n < 2 * cfg.tangent_k + 1:
        return np.zeros((0, 2))
    k = cfg.tangent_k
    t = np.roll(pts, -k, axis=0) - np.roll(pts, k, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    nrm = np.column_stack([-t[:, 1], t[:, 0]])
    s = np.arange(-cfg.refine_halfwidth_px, cfg.refine_halfwidth_px + 1e-9, cfg.refine_step_px)
    p = pts.astype(np.float32)
    mx = (p[:, 0:1] + nrm[:, 0:1] * s[None]).astype(np.float32)
    my = (p[:, 1:2] + nrm[:, 1:2] * s[None]).astype(np.float32)
    v = cv2.remap(prob.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    # orient the normal outward: values must decrease with s
    flip = v[:, -1] > v[:, 0]
    v[flip] = v[flip, ::-1]
    nrm[flip] = -nrm[flip]
    above = v >= 0.5
    cross = above[:, :-1] & ~above[:, 1:]                  # inside -> outside between j and j+1
    out = []
    centre = len(s) // 2
    for i in np.flatnonzero(cross.any(1)):
        js = np.flatnonzero(cross[i])
        j = js[np.argmin(np.abs(js + 0.5 - centre))]
        v0, v1 = v[i, j], v[i, j + 1]
        frac = (v0 - 0.5) / (v0 - v1) if v0 != v1 else 0.5
        out.append(pts[i] + nrm[i] * (s[j] + frac * cfg.refine_step_px))
    return np.array(out) if out else np.zeros((0, 2))


def _ransac_line(xy: np.ndarray, cfg: EdgeConfig, rng) -> np.ndarray | None:
    """Indices of the inliers of the best line, refitted with TLS."""
    n = len(xy)
    if n < cfg.min_inliers:
        return None
    i = rng.integers(0, n, cfg.ransac_iters)
    j = rng.integers(0, n, cfg.ransac_iters)
    d = xy[j] - xy[i]
    norm = np.linalg.norm(d, axis=1)
    ok = norm > 5.0
    if not ok.any():
        return None
    i, d, norm = i[ok], d[ok], norm[ok]
    nrm = np.column_stack([-d[:, 1], d[:, 0]]) / norm[:, None]
    c = -np.einsum("ij,ij->i", nrm, xy[i])
    dist = np.abs(xy @ nrm.T + c)                          # (n, iters)
    counts = (dist < cfg.ransac_tol_px).sum(0)
    best = int(np.argmax(counts))
    inl = np.flatnonzero(dist[:, best] < cfg.ransac_tol_px)
    for _ in range(2):                                     # TLS refit, re-select
        if len(inl) < 2:
            return None
        lf = fit_line(xy[inl])
        inl = np.flatnonzero(np.abs(xy @ lf.line[:2] + lf.line[2]) < cfg.ransac_tol_px)
        inl = _longest_run(xy, inl, lf.line, cfg.max_gap_px)
    return inl if len(inl) >= cfg.min_inliers else None


def _longest_run(xy: np.ndarray, inl: np.ndarray, line: np.ndarray, max_gap: float) -> np.ndarray:
    """Keep the largest group of inliers that is contiguous along the line: points of other contour
    parts (e.g. the jaw) that happen to cross the line's extension are dropped."""
    if len(inl) < 2:
        return inl
    d = np.array([-line[1], line[0]])
    t = xy[inl] @ d
    order = np.argsort(t)
    gaps = np.flatnonzero(np.diff(t[order]) > max_gap)
    bounds = np.concatenate([[0], gaps + 1, [len(order)]])
    k = int(np.argmax(np.diff(bounds)))
    return inl[order[bounds[k]:bounds[k + 1]]]


def _pair_score(xa, xb, la, lb, cfg: EdgeConfig, inside) -> tuple[float, dict] | tuple[None, str]:
    na, nb = la.line[:2], lb.line[:2]
    cosang = abs(float(na @ nb))
    angle = float(np.degrees(np.arccos(min(1.0, cosang))))
    if angle > cfg.max_edge_angle_deg:
        return None, "edges not parallel enough"
    # common direction and overlap along it
    da = np.array([-na[1], na[0]])
    ta, tb = xa @ da, xb @ da
    lo, hi = max(ta.min(), tb.min()), min(ta.max(), tb.max())
    shorter = min(np.ptp(ta), np.ptp(tb))
    overlap = (hi - lo) / shorter if shorter > 0 else 0.0
    if overlap < cfg.min_overlap:
        return None, "edges not side by side"
    mid_t = 0.5 * (lo + hi)
    pa = xa[np.argmin(np.abs(ta - mid_t))]
    pb = xb[np.argmin(np.abs(tb - mid_t))]
    width = abs(float(lb.line[:2] @ pa + lb.line[2]))
    if width < cfg.min_width_px:
        return None, "edges too close"
    if not inside(0.5 * (pa + pb)):              # the instrument must lie BETWEEN the two edges
        return None, "edges do not enclose the instrument"
    return float(len(xa) + len(xb)), dict(angle_deg=angle, width_px=width, overlap=overlap)


# ----------------------------------------------------------------------------------------- main
def extract_shaft_edges(prob: np.ndarray, camera: Camera, cfg: EdgeConfig | None = None) -> EdgeResult:
    cfg = cfg or EdgeConfig()
    h, w = prob.shape
    if camera.image_size != (w, h):
        raise ValueError(f"mask is {w}x{h}, camera is {camera.image_size}")
    binary = (prob > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return EdgeResult(None, "empty mask")
    cnt = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    diag = dict(n_contour=len(cnt))

    refined = _refine_subpixel(prob, cnt, cfg)
    if len(refined) == 0:
        return EdgeResult(None, "no sub-pixel boundary", diag)
    valid = _valid_region((h, w), camera, cfg.border_margin_px)
    xi = np.clip(np.round(refined[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(refined[:, 1]).astype(int), 0, h - 1)
    refined = refined[valid[yi, xi]]
    diag["n_boundary"] = len(refined)
    if len(refined) < 2 * cfg.min_inliers:
        return EdgeResult(None, "boundary too short inside the field of view", diag)

    und = undistorted_px(refined, camera)
    rng = np.random.default_rng(cfg.seed)
    remaining = np.arange(len(und))
    lines = []                                              # (indices into refined)
    for _ in range(cfg.max_lines):
        inl = _ransac_line(und[remaining], cfg, rng)
        if inl is None:
            break
        lines.append(remaining[inl])
        remaining = np.delete(remaining, inl)
    diag["n_lines"] = len(lines)
    if len(lines) < 2:
        return EdgeResult(None, "fewer than two straight segments", diag)

    def inside(pt_und: np.ndarray) -> bool:
        xn = (pt_und - [camera.cx, camera.cy]) / [camera.fx, camera.fy]
        raw, _ = cv2.projectPoints(np.array([[xn[0], xn[1], 1.0]]), np.zeros(3), np.zeros(3), camera.K, camera.dist)
        x, y = np.round(raw.ravel()).astype(int)
        return bool(0 <= x < w and 0 <= y < h and prob[y, x] > 0.5)

    best, best_info, reasons = None, None, []
    fits = [fit_line(und[ix]) for ix in lines]
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            if fits[i].span < cfg.min_span_px or fits[j].span < cfg.min_span_px:
                reasons.append("edge too short")
                continue
            sc, info = _pair_score(und[lines[i]], und[lines[j]], fits[i], fits[j], cfg, inside)
            if sc is None:
                reasons.append(info)
                continue
            if best is None or sc > best[0]:
                best, best_info = (sc, i, j), info
    if best is None:
        return EdgeResult(None, reasons[0] if reasons else "no valid pair", diag)
    _, i, j = best
    la, lb = fits[i], fits[j]
    edges = ShaftEdges(refined[lines[i]], refined[lines[j]], la.line, lb.line, la.rms, lb.rms, la.span, lb.span,
                       best_info["angle_deg"], best_info["width_px"])
    diag.update(best_info)
    if max(la.rms, lb.rms) > cfg.max_rms_px:
        return EdgeResult(edges, f"edge not straight (rms {max(la.rms, lb.rms):.2f} px)", diag)
    return EdgeResult(edges, "ok", diag)


def shaft_region(edges: ShaftEdges, camera: Camera, prob: np.ndarray, inset_px: float = 3.0,
                 n_vertices: int = 24) -> np.ndarray:
    """Boolean mask of the shaft pixels between the two edges (inset from them), inside the instance
    mask: where stage 4 reads the depth of the network (never on the boundary)."""
    und_a = undistorted_px(edges.a_raw, camera)
    und_b = undistorted_px(edges.b_raw, camera)
    na = edges.line_a[:2]
    d = np.array([-na[1], na[0]])
    ta, tb = und_a @ d, und_b @ d
    lo, hi = max(ta.min(), tb.min()), min(ta.max(), tb.max())
    ts = np.linspace(lo, hi, n_vertices)

    def on_line(line, other_pt):
        n, c = line[:2], line[2]
        base = -c * n                                      # point of the line closest to the origin
        dirv = np.array([-n[1], n[0]])
        pts = base[None] + ((ts - base @ d) / (dirv @ d))[:, None] * dirv[None]
        inward = np.sign(n @ other_pt + c) * n             # towards the other edge
        return pts + inset_px * inward[None]

    pa = on_line(edges.line_a, und_b.mean(0))
    pb = on_line(edges.line_b, und_a.mean(0))
    poly_und = np.vstack([pa, pb[::-1]])
    # back to raw pixels
    xn = (poly_und - [camera.cx, camera.cy]) / [camera.fx, camera.fy]
    P = np.column_stack([xn, np.ones(len(xn))])
    raw, _ = cv2.projectPoints(P, np.zeros(3), np.zeros(3), camera.K, camera.dist)
    region = np.zeros(prob.shape, np.uint8)
    cv2.fillPoly(region, [np.round(raw.reshape(-1, 2)).astype(np.int32)], 1)
    inner = cv2.erode((prob > 0.5).astype(np.uint8), np.ones((3, 3), np.uint8),
                      iterations=max(1, int(round(inset_px))))
    return (region & inner).astype(bool)
