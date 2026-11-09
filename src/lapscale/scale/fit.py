"""Stage 4: from the depth network's units to millimetres, using the instrument.

Per frame, on the shaft pixels: pairs (d_net, Z_mm), d_net from the network (sampled on the same rays,
``DepthFrame.sample``), Z_mm from the cylinder geometry. Two models, both always computed and compared:

* ``scale_only``: Z = a * d. a = median(Z / d). Robust, one parameter; right if the network's depth is
  correct up to scale.
* ``affine``: Z = a * d + b, Tukey-reweighted least squares. Absorbs an offset of the network, but the
  shaft covers a narrow depth range, so b is poorly determined and errors grow far from the shaft: the
  depth span of the samples is reported and the fit is refused below ``min_span_ratio``.

Networks that predict INVERSE depth (affine-invariant disparity, e.g. Depth Anything V2) are fitted in
inverse space: 1/Z = a * d (+ b).

Over time: the per-frame parameters are gated (instrument uncertainty, fit residual, number of samples)
and smoothed with a running median (offline: centred; real time: causal); frames without a valid fit
hold the nearest smoothed value within ``max_hold`` frames.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

VARIANTS = ("scale_only", "affine")


@dataclass
class ScaleConfig:
    min_samples: int = 50
    max_rel_residual_pct: float = 5.0      # robust spread of (fit - Z) / Z on the shaft
    max_instrument_rel_sd_pct: float = 5.0  # stage-3 uncertainty of the shaft depth (total)
    min_span_ratio: float = 1.15           # affine only: max(Z) / min(Z) of the samples
    smooth_window: int = 9                 # frames with a valid fit
    max_hold: int = 50                     # frames a smoothed value may be carried to
    output_kind: str = "depth"             # "depth" or "inverse_depth" (of the network)


@dataclass
class FrameFit:
    a: float
    b: float
    n: int
    rel_residual_pct: float
    span_ratio: float
    ok: bool
    reason: str = "ok"


def _to_target(z_mm: np.ndarray, kind: str) -> np.ndarray:
    return z_mm if kind == "depth" else 1.0 / z_mm


def fit_frame(d_net: np.ndarray, z_mm: np.ndarray, variant: str, cfg: ScaleConfig) -> FrameFit:
    d = np.asarray(d_net, np.float64)
    z = np.asarray(z_mm, np.float64)
    ok = np.isfinite(d) & np.isfinite(z) & (d > 0) & (z > 0)
    d, z = d[ok], z[ok]
    n = len(d)
    if n < cfg.min_samples:
        return FrameFit(np.nan, np.nan, n, np.nan, np.nan, False, f"{n} samples < {cfg.min_samples}")
    span = float(np.percentile(z, 95) / np.percentile(z, 5))
    t = _to_target(z, cfg.output_kind)
    if variant == "scale_only":
        a, b = float(np.median(t / d)), 0.0
    elif variant == "affine":
        if span < cfg.min_span_ratio:
            return FrameFit(np.nan, np.nan, n, np.nan, span, False,
                            f"depth span {span:.2f} < {cfg.min_span_ratio}: offset not identifiable")
        a, b = _tukey_line(d, t)
    else:
        raise ValueError(f"variant must be one of {VARIANTS}")
    pred = a * d + b
    z_fit = pred if cfg.output_kind == "depth" else 1.0 / np.where(pred > 0, pred, np.nan)
    rel = (z_fit - z) / z
    rr = float(100 * 1.4826 * np.nanmedian(np.abs(rel - np.nanmedian(rel))))
    if not np.isfinite(a) or a <= 0:
        return FrameFit(a, b, n, rr, span, False, "non-positive scale")
    if rr > cfg.max_rel_residual_pct:
        return FrameFit(a, b, n, rr, span, False, f"residual {rr:.1f}% > {cfg.max_rel_residual_pct}%")
    return FrameFit(a, b, n, rr, span, True)


def _tukey_line(x: np.ndarray, y: np.ndarray, iters: int = 10, c: float = 4.685) -> tuple[float, float]:
    a = float(np.median(y / x))
    b = 0.0
    w = np.ones_like(x)
    for _ in range(iters):
        A = np.column_stack([x, np.ones_like(x)]) * np.sqrt(w)[:, None]
        sol, *_ = np.linalg.lstsq(A, y * np.sqrt(w), rcond=None)
        a, b = float(sol[0]), float(sol[1])
        r = y - (a * x + b)
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-12
        u = r / (c * s)
        w = np.where(np.abs(u) < 1, (1 - u**2) ** 2, 0.0)
        if w.sum() < 3:
            break
    return a, b


# ---------------------------------------------------------------------------------------- over time
@dataclass
class TrackState:
    """Causal smoother for real time: running median of the last valid fits."""

    window: int = 9
    max_hold: int = 50
    hist: list[tuple[int, float, float]] = field(default_factory=list)   # (frame, a, b)

    def update(self, frame: int, fit: FrameFit | None) -> tuple[float, float, str]:
        if fit is not None and fit.ok:
            self.hist.append((frame, fit.a, fit.b))
            self.hist = self.hist[-self.window:]
        if not self.hist or frame - self.hist[-1][0] > self.max_hold:
            return np.nan, np.nan, "none"
        a = float(np.exp(np.median([np.log(h[1]) for h in self.hist])))
        b = float(np.median([h[2] for h in self.hist]))
        return a, b, "fit" if fit is not None and fit.ok else "held"


def smooth_offline(frames: list[int], fits: list[FrameFit], window: int, max_hold: int
                   ) -> list[tuple[float, float, str]]:
    """Centred running median over the valid fits (log a, b); invalid frames take the nearest valid
    smoothed value within ``max_hold`` frames."""
    valid = [(f, x) for f, x in zip(frames, fits, strict=True) if x.ok]
    if not valid:
        return [(np.nan, np.nan, "none")] * len(frames)
    vf = np.array([f for f, _ in valid])
    la = np.log([x.a for _, x in valid])
    bb = np.array([x.b for _, x in valid])
    h = window // 2
    sm = [(float(np.exp(np.median(la[max(0, k - h):k + h + 1]))), float(np.median(bb[max(0, k - h):k + h + 1])))
          for k in range(len(vf))]
    out = []
    for f, x in zip(frames, fits, strict=True):
        k = int(np.argmin(np.abs(vf - f)))
        if abs(vf[k] - f) > max_hold:
            out.append((np.nan, np.nan, "none"))
        else:
            out.append((sm[k][0], sm[k][1], "fit" if x.ok else "held"))
    return out


def apply_scale(depth_net: np.ndarray, a: float, b: float, kind: str = "depth") -> np.ndarray:
    """Network depth map -> millimetres."""
    v = a * depth_net + b
    if kind == "depth":
        return v
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(v > 0, 1.0 / v, np.nan)
