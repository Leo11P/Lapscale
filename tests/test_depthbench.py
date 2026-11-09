import json

import cv2
import numpy as np
import pytest
from synthetic_stereo import D1, D2, K1, K2, SIZE, R, T, plane_depth_pinhole, write_pair

from lapscale.camera import Camera
from lapscale.cli import main
from lapscale.depth.base import DepthPrediction
from lapscale.depthbench import compare_depth, load_gt, prepare_folder, prepare_stereo, run_depthbench
from lapscale.depthbench.metrics import align, depth_metrics


@pytest.fixture(scope="module")
def stereo_bench(tmp_path_factory):
    d = tmp_path_factory.mktemp("stereo")
    left, right, calib = write_pair(d)
    prepare_stereo(left, right, calib, d / "db", n=2, log=lambda *_: None)
    return d / "db"


def _rect():
    R1, _, P1, _, _, _, _ = cv2.stereoRectify(K1, D1, K2, D2, SIZE, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    return R1, P1[:3, :3]


def test_stereo_ground_truth_matches_the_plane(stereo_bench):
    R1, P = _rect()
    gt, v = load_gt(stereo_bench, "000000")
    true = plane_depth_pinhole(P, SIZE, R1)
    e = gt[v] / true[v] - 1
    assert v.mean() > 0.4
    assert abs(np.median(e)) < 0.003 and np.percentile(np.abs(e), 95) < 0.02
    cam = Camera.load(stereo_bench / "camera.yaml")
    assert not cam.has_distortion() and np.allclose(cam.K, P)


class PlaneNet:
    """Exact plane depth in the rectified view, in 'metres' that are off by 20%, plus 1% noise."""

    name = "fake:plane"
    output_kind = "depth"
    native_unit_mm = 1000.0

    def predict(self, img, fov_x_deg=None):
        R1, P = _rect()
        z = plane_depth_pinhole(P, SIZE, R1) / 1000 * 1.2
        z *= 1 + np.random.default_rng(0).normal(0, 0.01, z.shape)
        return DepthPrediction(z.astype(np.float32), np.ones(z.shape, bool))


def test_run_and_compare(stereo_bench):
    s = run_depthbench(stereo_bench, PlaneNet(), log=lambda *_: None)
    assert s["scale"]["abs_rel"] < 0.02 and s["affine"]["abs_rel"] < 0.02
    assert s["native"]["median_ratio"] == pytest.approx(1 / 1.2, rel=0.01)       # 20% too far
    assert s["native"]["abs_rel"] > 0.15
    rows = compare_depth(stereo_bench, log=lambda *_: None)
    assert rows[0]["name"] == "fake_plane" and (stereo_bench / "results" / "comparison.md").exists()


def test_alignment_in_inverse_space():
    rng = np.random.default_rng(0)
    gt = rng.uniform(40, 150, (50, 60)).astype(np.float32)
    pred = 3.0 / gt + 0.01                                   # affine-invariant disparity
    mask = np.ones_like(gt, bool)
    pm, p = align(pred, gt, mask, "affine", "inverse_depth")
    assert depth_metrics(pm, gt, mask)["abs_rel"] < 1e-4
    assert align(pred, gt, mask, "native", "inverse_depth")[0] is None


def test_prepare_folder_and_cli(tmp_path, stereo_bench):
    imgs, gts = tmp_path / "imgs", tmp_path / "gts"
    imgs.mkdir()
    gts.mkdir()
    for k in range(2):
        cv2.imwrite(str(imgs / f"f{k}.png"), cv2.imread(str(stereo_bench / "images" / f"{k:06d}.png")))
        np.save(gts / f"f{k}.npy", load_gt(stereo_bench, f"{k:06d}")[0] / 1000)    # metres
    m = prepare_folder(imgs, gts, stereo_bench / "camera.yaml", tmp_path / "db2", gt_scale_mm=1000,
                       log=lambda *_: None)
    assert len(m["frames"]) == 2
    preds = tmp_path / "preds"
    preds.mkdir()
    for k in range(2):
        np.save(preds / f"{k:06d}.npy", load_gt(stereo_bench, f"{k:06d}")[0] * 0.5)
    assert main(["depthbench", "run", "--bench", str(tmp_path / "db2"), "--depth-backend", "folder",
                 "--depth-dir", str(preds), "--name", "half"]) == 0
    s = json.loads((tmp_path / "db2" / "results" / "half" / "summary.json").read_text())
    assert s["scale"]["abs_rel"] < 1e-3 and "native" not in s
    assert main(["depthbench", "compare", "--bench", str(tmp_path / "db2")]) == 0
