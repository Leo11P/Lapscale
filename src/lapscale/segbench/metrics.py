"""Edge-accuracy metrics: how far the predicted shaft edges are from the hand-annotated ones, and what
that does to the metric depth.

Everything is measured in UNDISTORTED pixels (the space where edges are straight). For each ground-truth
edge, points are sampled along its annotated extent; at each one the offset of the predicted edge is
measured along the ground-truth outward normal (positive = predicted edge outside = mask too wide).

Per instrument:
* ``bias_px``: mean outward offset over both edges (systematic part);
* ``offset_sd_px``: spread of the offsets along the edges;
* ``width_err_px`` / ``width_err_pct`` at the middle of the annotated edges;
* ``depth_err_pct``: depth of the shaft at the middle ray, predicted edges vs annotated edges, through
  the full cylinder geometry. Independent of the diameter (both scale with it), so it is defined even
  when the diameter is not annotated. THIS is the number that matters for the metric scale.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..camera import Camera
from ..tool.edges import undistorted_px
from ..tool.geometry import fit_line, shaft_from_edges

N_SAMPLES = 20


@dataclass
class EdgePairMetrics:
    match_cost_px: float
    bias_px: float
    offset_a_px: float
    offset_b_px: float
    offset_sd_px: float
    max_abs_offset_px: float
    angle_err_deg: float
    width_gt_px: float
    width_err_px: float
    width_err_pct: float
    depth_err_pct: float


def _gt_geometry(gt_a_raw: np.ndarray, gt_b_raw: np.ndarray, camera: Camera):
    """GT lines in undistorted px, sample points along each edge, outward normals."""
    ua, ub = undistorted_px(gt_a_raw, camera), undistorted_px(gt_b_raw, camera)
    out = []
    for u, other in ((ua, ub), (ub, ua)):
        lf = fit_line(u)
        n = lf.line[:2]
        if n @ other.mean(0) + lf.line[2] > 0:          # outward = away from the other edge
            n = -n
            line = -lf.line
        else:
            line = lf.line
        d = np.array([-n[1], n[0]])
        t = u @ d
        foot = u - np.outer(u @ line[:2] + line[2], line[:2])            # projections on the line
        p0 = foot[np.argmin(t)]
        p1 = foot[np.argmax(t)]
        samples = p0[None] + np.linspace(0, 1, N_SAMPLES)[:, None] * (p1 - p0)[None]
        out.append((line, n, samples))
    return out


def _offsets(samples: np.ndarray, normal: np.ndarray, pred_line: np.ndarray) -> np.ndarray:
    """Signed distance along ``normal`` from each sample to the predicted line."""
    denom = pred_line[:2] @ normal
    if abs(denom) < 1e-9:
        return np.full(len(samples), np.nan)
    return -(samples @ pred_line[:2] + pred_line[2]) / denom


def compare_edges(gt_a_raw, gt_b_raw, pred_a_raw, pred_b_raw, camera: Camera) -> EdgePairMetrics:
    gt = _gt_geometry(np.asarray(gt_a_raw, float), np.asarray(gt_b_raw, float), camera)
    pl = [fit_line(undistorted_px(np.asarray(p, float), camera)).line for p in (pred_a_raw, pred_b_raw)]
    pr = [np.asarray(pred_a_raw, float), np.asarray(pred_b_raw, float)]

    def assign(order):
        offs = [_offsets(gt[k][2], gt[k][1], pl[order[k]]) for k in (0, 1)]
        return offs, float(np.nanmean(np.abs(np.concatenate(offs))))

    offs, cost = assign((0, 1))
    offs2, cost2 = assign((1, 0))
    order = (0, 1)
    if cost2 < cost:
        offs, cost, order = offs2, cost2, (1, 0)
    all_off = np.concatenate(offs)
    ang = []
    for k in (0, 1):
        c = abs(float(gt[k][0][:2] @ pl[order[k]][:2]))
        ang.append(np.degrees(np.arccos(min(1.0, c))))
    # width at the middle of the annotated edges
    mid = N_SAMPLES // 2
    pa = gt[0][2][mid]
    width_gt = float(abs(gt[1][0][:2] @ pa + gt[1][0][2]))
    width_err = float(offs[0][mid] + offs[1][mid])
    # depth at the same ray (middle between the annotated edges), predicted vs annotated edges
    gt_est = shaft_from_edges(np.asarray(gt_a_raw, float), np.asarray(gt_b_raw, float), camera, 1.0)
    pred_est = shaft_from_edges(pr[order[0]], pr[order[1]], camera, 1.0)
    z_gt = gt_est.depth_mid
    z_pr = float(pred_est.depth_at(gt_est.mid_xy[None])[0])
    return EdgePairMetrics(
        match_cost_px=cost, bias_px=float(np.nanmean(all_off)),
        offset_a_px=float(np.nanmean(offs[0])), offset_b_px=float(np.nanmean(offs[1])),
        offset_sd_px=float(np.nanstd(all_off)), max_abs_offset_px=float(np.nanmax(np.abs(all_off))),
        angle_err_deg=float(max(ang)), width_gt_px=width_gt, width_err_px=width_err,
        width_err_pct=100 * width_err / width_gt, depth_err_pct=100 * (z_pr / z_gt - 1))
