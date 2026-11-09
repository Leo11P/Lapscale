import numpy as np
import pytest

from lapscale.calib.optical_mask import detect_optical_circle


def frames(cx, cy, r, size=(960, 540), n=10, seed=0):
    rng = np.random.default_rng(seed)
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
    out = []
    for _ in range(n):
        img = rng.uniform(40, 200, (h, w)).astype(np.float32)
        img[~inside] = rng.uniform(0, 6, (~inside).sum())
        out.append(np.repeat(img[..., None], 3, 2).astype(np.uint8))
    return out


def test_clipped_circle():
    c, diag = detect_optical_circle(frames(481, 270, 330), margin_px=0)
    assert c is not None and diag["result"] == "circle"
    assert c.cx == pytest.approx(481, abs=1.5) and c.cy == pytest.approx(270, abs=1.5)
    assert c.r == pytest.approx(330, abs=1.5)


def test_no_rim_means_full_frame():
    c, diag = detect_optical_circle(frames(480, 270, 2000))
    assert c is None and "full frame" in diag["result"]
