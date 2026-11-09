import numpy as np
import pytest

from lapscale.scale.fit import FrameFit, ScaleConfig, TrackState, apply_scale, fit_frame, smooth_offline


def _pairs(n=400, span=(60, 90), scale=0.0123, offset=0.0, noise=0.002, kind="depth", seed=0):
    rng = np.random.default_rng(seed)
    z = rng.uniform(*span, n)
    d = (scale * z + offset) if kind == "depth" else (scale / z + offset)
    return d * (1 + rng.normal(0, noise, n)), z


def test_scale_only_recovers_the_scale():
    d, z = _pairs()
    f = fit_frame(d, z, "scale_only", ScaleConfig())
    assert f.ok and f.a == pytest.approx(1 / 0.0123, rel=0.002) and f.b == 0


def test_affine_recovers_offset_on_a_wide_span():
    d, z = _pairs(span=(40, 160), offset=0.3)
    f = fit_frame(d, z, "affine", ScaleConfig())
    assert f.ok and np.allclose(apply_scale(np.array([0.0123 * 100 + 0.3]), f.a, f.b), 100, rtol=0.01)


def test_affine_refused_on_a_narrow_span():
    d, z = _pairs(span=(70, 75))
    f = fit_frame(d, z, "affine", ScaleConfig())
    assert not f.ok and "span" in f.reason


def test_inverse_depth_networks():
    d, z = _pairs(kind="inverse")
    cfg = ScaleConfig(output_kind="inverse_depth")
    f = fit_frame(d, z, "scale_only", cfg)
    assert f.ok
    assert apply_scale(np.array([0.0123 / 80]), f.a, f.b, "inverse_depth")[0] == pytest.approx(80, rel=0.005)


def test_gates():
    d, z = _pairs(n=20)
    assert not fit_frame(d, z, "scale_only", ScaleConfig()).ok
    d, z = _pairs(noise=0.2)
    f = fit_frame(d, z, "scale_only", ScaleConfig())
    assert not f.ok and "residual" in f.reason


def test_temporal_smoothing_and_hold():
    good = FrameFit(10.0, 0.0, 100, 1.0, 1.2, True)
    out_lier = FrameFit(13.0, 0.0, 100, 1.0, 1.2, True)
    bad = FrameFit(np.nan, np.nan, 0, np.nan, np.nan, False, "x")
    frames = list(range(12)) + [200]
    fits = [good] * 5 + [out_lier] + [good] * 3 + [bad] * 3 + [bad]
    sm = smooth_offline(frames, fits, window=5, max_hold=10)
    assert sm[5][0] == pytest.approx(10.0)            # isolated outlier removed by the median
    assert sm[10][2] == "held" and sm[10][0] == pytest.approx(10.0)
    assert sm[-1][2] == "none"                         # too far from any valid fit
    tr = TrackState(window=3, max_hold=5)
    assert tr.update(0, good)[2] == "fit"
    assert tr.update(3, None)[2] == "held"
    assert tr.update(20, None)[2] == "none"
