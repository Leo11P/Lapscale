"""``lapscale eval``: the WHOLE pipeline (instrument-derived scale) against depth ground truth.

Run the pipeline on the benchmark images as a video (``lapscale run --video <bench>/images --camera
<bench>/camera.yaml ...``), then score the metric depth of each frame. Besides the usual metrics, errors are
binned by the distance in depth from the instrument (|Z_gt - Z_shaft|), which is where the instrument-based
scale has to extrapolate. Two references are reported on the same frames: the backend's own metric scale
(``native``) and a per-frame ground-truth median scale (``oracle``, the best any single scale can do).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ..depthbench.data import frame_ids, load_gt
from ..depthbench.metrics import band_metrics, depth_metrics
from ..io.cache import DepthStore, ToolStore
from ..scale.fit import apply_scale
from ..scale.run import load_scale

SHAFT_BANDS_MM = ((0, 20), (20, 50), (50, np.inf))


def evaluate(bench: str | Path, run_dir: str | Path, depth_tag: str, tool_tag: str, variant: str = "scale_only",
             name: str | None = None, log=print) -> dict:
    bench, run_dir = Path(bench), Path(run_dir)
    ids = frame_ids(bench)
    depth = DepthStore(run_dir, depth_tag)
    tool = ToolStore(run_dir, tool_tag)
    meta = depth.meta()
    kind, native = meta.get("output_kind", "depth"), meta.get("native_unit_mm")
    scales = load_scale(run_dir, depth_tag, tool_tag, variant)
    rows = []
    for k, fid in enumerate(ids):
        if k not in scales or not depth.has(k):
            continue
        a, b, st = scales[k]
        gt, gv = load_gt(bench, fid)
        df = depth.load(k)
        mask = gv & df.valid
        if gt.shape != df.depth.shape:
            raise ValueError(f"frame {fid}: ground truth {gt.shape} vs depth {df.depth.shape}")
        shaft = [s.depth_mid_mm for s in tool.load(k)] if tool.has(k) else []
        z_ref = float(np.median(shaft)) if shaft else np.nan
        refs = {}
        if st != "none" and np.isfinite(a):
            refs["instrument"] = apply_scale(df.depth, a, b, kind)
        if native and kind == "depth":
            refs["native"] = df.depth * native
        p = df.depth[mask]
        if p.size:
            t = gt[mask] if kind == "depth" else 1 / gt[mask]
            refs["oracle"] = apply_scale(df.depth, float(np.median(t / p)), 0.0, kind)
        for ref, mm in refs.items():
            m = depth_metrics(mm, gt, mask)
            if not m:
                continue
            row = dict(frame=fid, ref=ref, status=st, shaft_depth_mm=round(z_ref, 2) if np.isfinite(z_ref) else "",
                       **{x: round(v, 6) for x, v in m.items()},
                       **{x: round(v, 6) for x, v in band_metrics(mm, gt, mask).items() if v is not None})
            if np.isfinite(z_ref):
                for lo, hi in SHAFT_BANDS_MM:
                    sel = mask & (np.abs(gt - z_ref) >= lo) & (np.abs(gt - z_ref) < hi)
                    if sel.sum() >= 50:
                        d = depth_metrics(mm, gt, sel)
                        nm = f"{lo:g}-{hi:g}mm" if np.isfinite(hi) else f">{lo:g}mm"
                        row[f"abs_rel_from_shaft_{nm}"] = round(d["abs_rel"], 6)
            rows.append(row)
    if not rows:
        raise RuntimeError("no frame with both a scale and ground truth")
    name = name or f"{depth_tag}__{tool_tag}__{variant}"
    out = bench / "eval" / name
    out.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r}, key=lambda x: (x not in ("frame", "ref", "status"), x))
    with open(out / "per_frame.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, keys)
        wr.writeheader()
        wr.writerows(rows)
    summary = dict(name=name, variant=variant, n_frames=len(ids),
                   n_scaled=sum(1 for r in rows if r["ref"] == "instrument"))
    for ref in ("instrument", "native", "oracle"):
        rs = [r for r in rows if r["ref"] == ref]
        if rs:
            summary[ref] = {k: round(float(np.median([r[k] for r in rs if k in r])), 6)
                            for k in rs[0] if k not in ("frame", "ref", "status", "shaft_depth_mm")
                            and any(k in r for r in rs)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    for ref in ("instrument", "native", "oracle"):
        if ref in summary:
            s = summary[ref]
            nan = float("nan")
            log(f"{ref:10s} AbsRel {s.get('abs_rel', nan):.4f}  RMSE {s.get('rmse_mm', nan):.2f} mm  "
                f"d<1.05 {s.get('delta_1_05', nan):.3f}  scale GT/pred {s.get('median_ratio', nan):.3f}")
    log(f"results in {out}")
    return summary
