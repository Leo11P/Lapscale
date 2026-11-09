"""Run a depth backend on a depth benchmark folder and score it; compare backends."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from ..camera import Camera
from ..depth.base import DepthBackend
from .data import frame_ids, load_gt
from .metrics import ALIGNMENTS, align, band_metrics, depth_metrics


def run_depthbench(bench: str | Path, backend: DepthBackend, name: str | None = None, save_pred: bool = True,
                   log=print) -> dict:
    bench = Path(bench)
    name = name or backend.name.replace(":", "_").replace("/", "_")
    out = bench / "results" / name
    (out / "pred").mkdir(parents=True, exist_ok=True)
    cam = Camera.load(bench / "camera.yaml")
    fov = cam.fov_x_deg()
    kind = getattr(backend, "output_kind", "depth")
    native = getattr(backend, "native_unit_mm", None)
    rows, lat = [], []
    for k, fid in enumerate(frame_ids(bench)):
        img = cv2.imread(str(bench / "images" / f"{fid}.png"))
        gt, gvalid = load_gt(bench, fid)
        if hasattr(backend, "set_frame_id"):
            backend.set_frame_id(fid)
        t0 = time.perf_counter()
        pred = backend.predict(img, fov_x_deg=fov)
        lat.append(1000 * (time.perf_counter() - t0))
        mask = gvalid & pred.valid
        if save_pred:
            np.savez_compressed(out / "pred" / f"{fid}.npz", depth=pred.depth.astype(np.float16), valid=pred.valid)
        for mode in ALIGNMENTS:
            pm, params = align(pred.depth, gt, mask, mode, kind, native)
            if pm is None:
                continue
            m = depth_metrics(pm, gt, mask)
            if not m:
                continue
            rows.append(dict(frame=fid, align=mode, **{k2: round(v, 6) for k2, v in m.items()},
                             **{k2: round(v, 6) for k2, v in band_metrics(pm, gt, mask).items() if v is not None},
                             **{f"param_{k2}": round(v, 8) for k2, v in params.items()}))
        if (k + 1) % 10 == 0:
            log(f"  {k + 1} frames")
    if not rows:
        raise RuntimeError("no frame could be scored (empty ground truth?)")
    keys = sorted({k for r in rows for k in r}, key=lambda x: (x not in ("frame", "align"), x))
    with open(out / "per_frame.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, keys)
        wr.writeheader()
        wr.writerows(rows)
    summary = dict(name=name, backend=backend.name, output_kind=kind, native_unit_mm=native,
                   n_frames=len(frame_ids(bench)), latency_ms_median=round(float(np.median(lat)), 2))
    for mode in ALIGNMENTS:
        rs = [r for r in rows if r["align"] == mode]
        if not rs:
            continue
        agg = {}
        for key in rs[0]:
            if key in ("frame", "align"):
                continue
            vals = [r[key] for r in rs if key in r and r[key] != ""]
            if vals:
                agg[key] = round(float(np.median(vals)), 6)
        if mode == "native":
            ratios = np.array([r["median_ratio"] for r in rs])
            agg["scale_ratio_spread_pct"] = round(float(100 * np.std(np.log(ratios))), 3)
        summary[mode] = agg
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    log(format_summary(summary))
    return summary


def format_summary(s: dict) -> str:
    lines = [f"{s['name']} ({s['n_frames']} frames, {s['latency_ms_median']} ms/frame)"]
    for mode in ALIGNMENTS:
        a = s.get(mode)
        if a:
            lines.append(f"  {mode:7s} AbsRel {a.get('abs_rel'):.4f}  RMSE {a.get('rmse_mm'):.2f} mm  "
                         f"d<1.05 {a.get('delta_1_05'):.3f}  d<1.25 {a.get('delta_1_25'):.3f}"
                         + (f"  GT/pred scale {a.get('median_ratio'):.3f} (spread {a.get('scale_ratio_spread_pct')}%)"
                            if mode == "native" else ""))
    return "\n".join(lines)


def compare_depth(bench: str | Path, log=print) -> list[dict]:
    res_dir = Path(bench) / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(p.read_text()) for p in sorted(res_dir.glob("*/summary.json"))]
    if not rows:
        log(f"no results in {res_dir} yet")
        return []
    rows.sort(key=lambda s: s.get("scale", {}).get("abs_rel", np.inf))
    cols = ["backend", "AbsRel scale", "d<1.05 scale", "AbsRel affine", "AbsRel native", "native GT/pred", "ms/frame"]
    table = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for s in rows:
        sc, af, na = s.get("scale", {}), s.get("affine", {}), s.get("native", {})
        table.append("| " + " | ".join(str(x) for x in [
            s["name"], sc.get("abs_rel", ""), sc.get("delta_1_05", ""), af.get("abs_rel", ""), na.get("abs_rel", ""),
            na.get("median_ratio", ""), s.get("latency_ms_median", "")]) + " |")
    text = "\n".join(table)
    (res_dir / "comparison.md").write_text(text + "\n")
    log(text)
    return rows
