"""Run a segmentation backend on the annotated frames and score its shaft edges; compare backends.

Outputs in ``<bench>/results/<name>/``:
  per_instrument.csv   one row per annotated shaft (matched or not, with the reason)
  summary.json         the numbers to compare backends (see ``summarise``)
  overlays/<frame>.png annotated edges (green) vs predicted edges (red / blue) and the depth error
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from ..camera import Camera
from ..tool.base import SegBackend
from ..tool.edges import EdgeConfig, extract_shaft_edges
from ..tool.uncertainty import shift_edges
from .data import load_annotations, load_manifest
from .metrics import compare_edges

FIELDS = ["frame", "instrument", "found", "reason", "n_instances", "latency_ms", "match_cost_px", "bias_px",
          "offset_a_px", "offset_b_px", "offset_sd_px", "max_abs_offset_px", "angle_err_deg", "width_gt_px",
          "width_err_px", "width_err_pct", "depth_err_pct", "depth_err_pct_bias_corrected"]


def run_benchmark(bench: str | Path, backend: SegBackend, camera: Camera, name: str | None = None,
                  edge_cfg: EdgeConfig | None = None, max_match_px: float = 15.0, overlays: bool = True,
                  log=print) -> dict:
    bench = Path(bench)
    name = name or backend.name
    out = bench / "results" / name
    (out / "overlays").mkdir(parents=True, exist_ok=True)
    frames = {f["id"]: f for f in load_manifest(bench)["frames"]}
    ann = load_annotations(bench)
    from .masks import load_frame_masks, load_index, mask_metrics
    edge_frames = {fid for fid, a in ann.items() if a.status == "ok" and a.instruments}
    mask_frames = {fid for fid in load_index(bench) if fid in frames}
    todo = sorted(edge_frames | mask_frames)
    if not todo:
        raise RuntimeError(f"no annotated frames in {bench}: run `lapscale segbench annotate` (edges) "
                           "or `annotate-masks` first")
    rows, latencies, kept, mrows = [], [], [], []   # kept: (row index, pred edges) for the bias correction
    for n_done, fid in enumerate(todo, 1):
        img = cv2.imread(str(bench / frames[fid]["file"]))
        cam = camera.for_size((img.shape[1], img.shape[0]))
        if hasattr(backend, "set_frame_id"):          # precomputed-masks backend
            backend.set_frame_id(fid)
        t0 = time.perf_counter()
        instances = backend.predict(img)
        lat = 1000 * (time.perf_counter() - t0)
        latencies.append(lat)
        if fid in mask_frames:
            mm = mask_metrics([i.prob > 0.5 for i in instances], load_frame_masks(bench, fid), img.shape[:2])
            mrows.append(dict(frame=fid, **{k: round(v, 4) for k, v in mm.items()}))
        if fid not in edge_frames:
            continue
        cands = []                               # (edges, reason) per instance
        for inst in instances:
            res = extract_shaft_edges(inst.prob, cam, edge_cfg)
            cands.append(res)
        used = set()
        vis = img.copy() if overlays else None
        for gi, g in enumerate(ann[fid].instruments):
            ga, gb = g.arrays()
            best = None
            for ci, res in enumerate(cands):
                if ci in used or res.edges is None:
                    continue
                try:
                    m = compare_edges(ga, gb, res.edges.a_raw, res.edges.b_raw, cam)
                except ValueError:
                    continue
                if np.isfinite(m.match_cost_px) and (best is None or m.match_cost_px < best[1].match_cost_px):
                    best = (ci, m)
            row = dict(frame=fid, instrument=gi, n_instances=len(instances), latency_ms=round(lat, 2))
            if best is None or best[1].match_cost_px > max_match_px:
                reasons = sorted({r.reason for r in cands}) or ["no instance"]
                row.update(found=0, reason="; ".join(reasons) if best is None else "edges too far from annotation")
            else:
                ci, m = best
                used.add(ci)
                res = cands[ci]
                row.update(found=1, reason=res.reason, **{k: round(v, 4) for k, v in m.__dict__.items()})
                kept.append((len(rows), res.edges, ga, gb, cam))
                if vis is not None:
                    _draw_edges(vis, res.edges.a_raw, (0, 0, 255))
                    _draw_edges(vis, res.edges.b_raw, (255, 0, 0))
            if vis is not None:
                _draw_edges(vis, ga, (0, 255, 0), lines=True)
                _draw_edges(vis, gb, (0, 255, 0), lines=True)
                txt = (f"#{gi}: depth err {row['depth_err_pct']:+.1f}%, bias {row['bias_px']:+.2f} px"
                       if row["found"] else f"#{gi}: not found ({row['reason']})")
                cv2.putText(vis, txt, (10, 25 + 22 * gi), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                            cv2.LINE_AA)
            rows.append(row)
        if vis is not None:
            cv2.imwrite(str(out / "overlays" / f"{fid}.png"), vis)
        if n_done % 10 == 0:
            log(f"  {n_done}/{len(todo)} frames")

    _bias_corrected(rows, kept)
    with open(out / "per_instrument.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, FIELDS)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in FIELDS})
    if mrows:
        with open(out / "per_frame_masks.csv", "w", newline="") as fh:
            wr = csv.DictWriter(fh, list(mrows[0].keys()))
            wr.writeheader()
            wr.writerows(mrows)
    summary = summarise(rows, latencies) if rows else dict(n_annotated=0, n_found=0, found_rate=None,
                                                             latency_ms_median=round(float(np.median(latencies)), 2))
    if mrows:
        summary.update(n_mask_frames=len(mrows),
                       mask_iou_median=round(float(np.median([r["mask_iou"] for r in mrows])), 4),
                       boundary_err_px_median=round(float(np.nanmedian([r["boundary_err_px_mean"] for r in mrows])), 3))
    summary.update(backend=backend.name, name=name, camera=camera.source)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    log(format_summary(summary))
    log(f"results in {out}")
    return summary


def _bias_corrected(rows: list[dict], kept: list) -> None:
    """Depth error after removing a constant edge bias estimated on the OTHER instruments (leave one
    out): what is left if the model's bias is stable and calibrated away."""
    biases = np.array([rows[i]["bias_px"] for i, *_ in kept])
    for k, (i, edges, ga, gb, cam) in enumerate(kept):
        others = np.delete(biases, k)
        if len(others) < 3:
            continue
        b = float(np.median(others))
        a, bb = shift_edges(edges.a_raw, edges.b_raw, -b)
        rows[i]["depth_err_pct_bias_corrected"] = round(compare_edges(ga, gb, a, bb, cam).depth_err_pct, 4)


def summarise(rows: list[dict], latencies: list[float]) -> dict:
    found = [r for r in rows if r.get("found")]
    s = dict(n_annotated=len(rows), n_found=len(found),
             found_rate=round(len(found) / len(rows), 4) if rows else 0.0,
             latency_ms_median=round(float(np.median(latencies)), 2) if latencies else None)
    if found:
        de = np.array([r["depth_err_pct"] for r in found])
        bias = np.array([r["bias_px"] for r in found])
        s.update(
            depth_err_pct_median_abs=round(float(np.median(np.abs(de))), 3),
            depth_err_pct_p90_abs=round(float(np.percentile(np.abs(de), 90)), 3),
            depth_err_pct_median_signed=round(float(np.median(de)), 3),
            bias_px_median=round(float(np.median(bias)), 3),
            bias_px_spread_mad=round(float(1.4826 * np.median(np.abs(bias - np.median(bias)))), 3),
            offset_sd_px_median=round(float(np.median([r["offset_sd_px"] for r in found])), 3),
            angle_err_deg_median=round(float(np.median([r["angle_err_deg"] for r in found])), 3),
        )
        bc = [r["depth_err_pct_bias_corrected"] for r in found if r.get("depth_err_pct_bias_corrected") != ""
              and r.get("depth_err_pct_bias_corrected") is not None]
        if bc:
            s["depth_err_pct_bias_corrected_median_abs"] = round(float(np.median(np.abs(bc))), 3)
            s["depth_err_pct_bias_corrected_p90_abs"] = round(float(np.percentile(np.abs(bc), 90)), 3)
    reasons: dict[str, int] = {}
    for r in rows:
        if not r.get("found"):
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    s["not_found_reasons"] = reasons
    return s


def format_summary(s: dict) -> str:
    lines = [f"{s.get('name', '?')}: found {s['n_found']}/{s['n_annotated']} shafts, "
             f"latency {s.get('latency_ms_median')} ms/frame"]
    if s.get("n_mask_frames"):
        lines.append(f"  masks ({s['n_mask_frames']} frames): IoU {s['mask_iou_median']}, "
                     f"boundary error {s['boundary_err_px_median']} px")
    if s.get("n_found"):
        lines.append(f"  depth error |median| {s['depth_err_pct_median_abs']}%  p90 {s['depth_err_pct_p90_abs']}%  "
                     f"(signed median {s['depth_err_pct_median_signed']}%)")
        lines.append(f"  edge bias {s['bias_px_median']:+} px (spread {s['bias_px_spread_mad']} px), "
                     f"offset sd along edges {s['offset_sd_px_median']} px")
        if "depth_err_pct_bias_corrected_median_abs" in s:
            lines.append(f"  if the bias is calibrated away: |median| {s['depth_err_pct_bias_corrected_median_abs']}%"
                         f"  p90 {s['depth_err_pct_bias_corrected_p90_abs']}%")
    if s.get("not_found_reasons"):
        lines.append(f"  not found: {s['not_found_reasons']}")
    return "\n".join(lines)


def compare(bench: str | Path, log=print) -> list[dict]:
    """Table of all results/<name>/summary.json, best depth error first."""
    rows = []
    for p in sorted((Path(bench) / "results").glob("*/summary.json")):
        rows.append(json.loads(p.read_text()))
    rows.sort(key=lambda s: (s.get("depth_err_pct_p90_abs") is None, s.get("depth_err_pct_p90_abs", 0)))
    cols = [("name", "backend"), ("found_rate", "found"), ("depth_err_pct_median_abs", "|depth err| med %"),
            ("depth_err_pct_p90_abs", "p90 %"), ("bias_px_median", "bias px"), ("bias_px_spread_mad", "bias spread"),
            ("depth_err_pct_bias_corrected_p90_abs", "p90 % bias-corr."), ("mask_iou_median", "mask IoU"),
            ("boundary_err_px_median", "boundary px"), ("latency_ms_median", "ms/frame")]
    table = ["| " + " | ".join(c[1] for c in cols) + " |", "|" + "---|" * len(cols)]
    for s in rows:
        table.append("| " + " | ".join(str(s.get(c[0], "")) for c in cols) + " |")
    text = "\n".join(table)
    out = Path(bench) / "results" / "comparison.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    log(text)
    return rows


def _draw_edges(img, pts, col, lines=False):
    pts = np.asarray(pts)
    for p in pts:
        cv2.circle(img, (int(round(p[0])), int(round(p[1]))), 1, col, -1)
    if lines and len(pts) >= 2:
        cv2.polylines(img, [np.round(pts).astype(np.int32)], False, col, 1, cv2.LINE_AA)
