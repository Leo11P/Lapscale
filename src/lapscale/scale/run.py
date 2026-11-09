"""Stage 4 over a video: join the depth cache (stage 2) and the shaft samples (stage 3)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ..io.cache import DepthFrame, DepthStore, ToolStore
from .fit import VARIANTS, FrameFit, ScaleConfig, apply_scale, fit_frame, smooth_offline

FIELDS = ["frame", "n_instruments", "n_samples", "a_raw", "b_raw", "rel_residual_pct", "span_ratio", "fit_ok",
          "reason", "a", "b", "status", "native_ratio"]


def frame_pairs(depth: DepthFrame, samples, cfg: ScaleConfig) -> tuple[np.ndarray, np.ndarray, int]:
    """(d_net, z_mm) over the instruments that pass the uncertainty gate."""
    ds, zs, used = [], [], 0
    for s in samples:
        if np.isfinite(s.rel_sd_pct) and s.rel_sd_pct > cfg.max_instrument_rel_sd_pct:
            continue
        d = depth.sample(s.rays)
        ds.append(d)
        zs.append(s.z_mm)
        used += 1
    if not ds:
        return np.zeros(0), np.zeros(0), 0
    return np.concatenate(ds), np.concatenate(zs), used


def run_scale(run_dir: str | Path, depth_tag: str, tool_tag: str, cfg: ScaleConfig | None = None,
              variants=VARIANTS, log=print) -> dict:
    cfg = cfg or ScaleConfig()
    depth = DepthStore(run_dir, depth_tag)
    tool = ToolStore(run_dir, tool_tag)
    dmeta = depth.meta()
    cfg.output_kind = dmeta.get("output_kind", cfg.output_kind)
    frames = sorted(set(depth.frames()) & set(tool.frames()))
    if not frames:
        raise RuntimeError(f"no frame processed by both depth '{depth_tag}' and tool '{tool_tag}'")
    native_mm = dmeta.get("native_unit_mm")          # e.g. 1000 if the network claims metres
    per_variant = {}
    for variant in variants:
        fits, extra = [], []
        for f in frames:
            df = depth.load(f)
            d, z, used = frame_pairs(df, tool.load(f), cfg)
            fit = fit_frame(d, z, variant, cfg) if used else FrameFit(np.nan, np.nan, 0, np.nan, np.nan, False,
                                                                      "no usable instrument")
            fits.append(fit)
            extra.append(used)
        smooth = smooth_offline(frames, fits, cfg.smooth_window, cfg.max_hold)
        out = Path(run_dir) / "scale" / f"{depth_tag}__{tool_tag}" / variant
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "frames.csv", "w", newline="") as fh:
            wr = csv.DictWriter(fh, FIELDS)
            wr.writeheader()
            for f, fit, used, (a, b, st) in zip(frames, fits, extra, smooth, strict=True):
                wr.writerow(dict(frame=f, n_instruments=used, n_samples=fit.n, a_raw=_r(fit.a), b_raw=_r(fit.b),
                                 rel_residual_pct=_r(fit.rel_residual_pct), span_ratio=_r(fit.span_ratio),
                                 fit_ok=int(fit.ok), reason=fit.reason, a=_r(a), b=_r(b), status=st,
                                 native_ratio=_r(a / native_mm) if native_mm and np.isfinite(a) else ""))
        ok = [x for x in fits if x.ok]
        s = dict(variant=variant, n_frames=len(frames), n_fit=len(ok),
                 n_scaled=int(sum(st != "none" for *_, st in smooth)),
                 a_median=_r(float(np.median([x.a for x in ok]))) if ok else None,
                 a_frame_to_frame_spread_pct=_r(100 * 1.4826 * float(np.median(np.abs(np.log([x.a for x in ok])
                                                - np.median(np.log([x.a for x in ok])))))) if ok else None,
                 residual_pct_median=_r(float(np.median([x.rel_residual_pct for x in ok]))) if ok else None)
        if native_mm and ok:
            s["native_scale_ratio"] = _r(s["a_median"] / native_mm)    # 1 = network's own metric scale right
        (out / "summary.json").write_text(json.dumps(dict(s, config=cfg.__dict__), indent=2))
        per_variant[variant] = s
        log(f"scale [{variant}]: fitted {s['n_fit']}/{s['n_frames']} frames, scaled {s['n_scaled']}, "
            f"a {s['a_median']} (spread {s['a_frame_to_frame_spread_pct']}%)"
            + (f", network's own metric scale off by x{s.get('native_scale_ratio')}" if native_mm and ok else ""))
    return per_variant


def load_scale(run_dir: str | Path, depth_tag: str, tool_tag: str, variant: str) -> dict[int, tuple[float, float, str]]:
    p = Path(run_dir) / "scale" / f"{depth_tag}__{tool_tag}" / variant / "frames.csv"
    out = {}
    with open(p) as fh:
        for r in csv.DictReader(fh):
            a = float(r["a"]) if r["a"] else np.nan
            b = float(r["b"]) if r["b"] else np.nan
            out[int(r["frame"])] = (a, b, r["status"])
    return out


def metric_depth(run_dir: str | Path, depth_tag: str, tool_tag: str, variant: str, frame: int,
                 _cache: dict = {}) -> DepthFrame | None:  # noqa: B006 - small per-process memo
    """Depth of a frame in millimetres (None if the frame has no scale)."""
    key = (str(run_dir), depth_tag, tool_tag, variant)
    if key not in _cache:
        _cache[key] = load_scale(run_dir, depth_tag, tool_tag, variant)
    a, b, st = _cache[key].get(frame, (np.nan, np.nan, "none"))
    if st == "none" or not np.isfinite(a):
        return None
    store = DepthStore(run_dir, depth_tag)
    df = store.load(frame)
    kind = store.meta().get("output_kind", "depth")
    mm = apply_scale(df.depth, a, b, kind)
    valid = df.valid & np.isfinite(mm) & (mm > 0)
    return DepthFrame(frame, np.where(valid, mm, 0).astype(np.float32), valid, df.K)


def _r(x, nd: int = 6):
    return round(float(x), nd) if x is not None and np.isfinite(x) else ""
