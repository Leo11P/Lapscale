"""End to end on a synthetic laparoscopic video: segmentation (oracle) -> edges -> geometry -> depth network
(exact depth in unknown units) -> scale -> metric depth of the whole tissue."""
import json
import time

import cv2
import numpy as np
import pytest
from synthetic_lap import (
    CAMERA,
    NET_SCALE,
    OracleSeg,
    TrueDepthNet,
    frame_image,
    instrument_mask,
    true_depth,
    write_video,
)

from lapscale.cli import main
from lapscale.depth.view import PinholeView
from lapscale.pipeline import run_offline, run_stream
from lapscale.scale.run import metric_depth
from lapscale.tool.run import ToolConfig

N = 4


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    d = tmp_path_factory.mktemp("lap")
    return write_video(d / "lap.avi", N)


def _tissue_error(md):
    yy, xx = np.nonzero(md.valid)
    K = md.K
    xy = np.column_stack([(xx - K[0, 2]) / K[0, 0], (yy - K[1, 2]) / K[1, 1]])
    return np.abs(md.depth[yy, xx] / true_depth(xy, md.index) - 1)


@pytest.fixture(scope="module")
def offline(video, tmp_path_factory):
    run = tmp_path_factory.mktemp("run")
    view = PinholeView(CAMERA, CAMERA.image_size)
    res = run_offline(video, CAMERA, OracleSeg(), TrueDepthNet(view.K_view), run,
                      tool_cfg=ToolConfig(diameter_mm=5.0, mc_draws=30), log=lambda *_: None)
    return run, res


def test_offline_metric_depth_of_the_whole_frame(offline):
    run, res = offline
    dt, tt = res["depth"]["tag"], res["tool"]["tag"]
    for f in range(N):
        md = metric_depth(run, dt, tt, "scale_only", f)
        assert md is not None
        e = _tissue_error(md)
        assert np.median(e) < 0.003 and np.percentile(e, 95) < 0.006
    s = res["scale"]["scale_only"]
    assert s["native_scale_ratio"] == pytest.approx(1 / NET_SCALE / 1000, rel=0.01)   # network "metres" were wrong


def test_viz_writes_a_video(offline, video):
    from lapscale.viz import render_run
    run, res = offline
    out = render_run(run, video, CAMERA, res["depth"]["tag"], res["tool"]["tag"], log=lambda *_: None)
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == N


def test_stream_mode(video, tmp_path):
    view = PinholeView(CAMERA, CAMERA.image_size)

    class SlowNet(TrueDepthNet):            # slower than segmentation: depth must run asynchronously
        def predict(self, img, fov_x_deg=None):
            time.sleep(0.05)
            return super().predict(img, fov_x_deg)

    s = run_stream(video, CAMERA, OracleSeg(), SlowNet(view.K_view), out_video=tmp_path / "s.mp4",
                   log_csv=tmp_path / "log.csv", log=lambda *_: None)
    assert s["frames"] == N and s["frames_with_scale"] >= 1
    assert s["final_a"] == pytest.approx(1 / NET_SCALE, rel=0.005)
    assert (tmp_path / "s.mp4").exists() and (tmp_path / "log.csv").read_text().count("\n") == N + 1


def test_cli_run_offline_with_folder_backends(video, tmp_path):
    """The same pipeline driven from the command line, with precomputed masks and depth maps."""
    masks, depths = tmp_path / "masks", tmp_path / "depth"
    masks.mkdir()
    depths.mkdir()
    view = PinholeView(CAMERA, CAMERA.image_size)
    net = TrueDepthNet(view.K_view)
    for k in range(N):
        cv2.imwrite(str(masks / f"lap_{k:06d}.png"), np.round(instrument_mask(k) * 255).astype(np.uint8))
        net.k = k
        np.save(depths / f"lap_{k:06d}.npy", net.predict(np.zeros((540, 960, 3), np.uint8)).depth)
    cam = CAMERA.save(tmp_path / "cam.yaml")
    run = tmp_path / "run"
    assert main(["run", "--video", str(video), "--camera", str(cam), "--run-dir", str(run), "--diameter-mm", "5",
                 "--mc", "0", "--seg-backend", "masks", "--masks-dir", str(masks), "--depth-backend", "folder",
                 "--depth-dir", str(depths)]) == 0
    s = json.loads(next((run / "scale").glob("*/scale_only/summary.json")).read_text())
    assert s["n_fit"] == N and s["a_median"] == pytest.approx(1 / NET_SCALE, rel=0.005)


def test_eval_against_ground_truth(offline, video, tmp_path):
    """Benchmark folder whose ground truth lives on the grid of the depth maps (undistorted view)."""
    from lapscale.eval import evaluate
    run, res = offline
    bench = tmp_path / "db"
    (bench / "gt").mkdir(parents=True)
    view = PinholeView(CAMERA, CAMERA.image_size)
    K = view.K_view
    h, w = 540, 960
    yy, xx = np.mgrid[0:h, 0:w]
    xy = np.column_stack([(xx.ravel() - K[0, 2]) / K[0, 0], (yy.ravel() - K[1, 2]) / K[1, 1]])
    frames = []
    for k in range(N):
        np.savez_compressed(bench / "gt" / f"{k:06d}.npz", depth=true_depth(xy, k).reshape(h, w).astype(np.float32),
                            valid=view.mask)
        frames.append(dict(id=f"{k:06d}"))
    (bench / "manifest.json").write_text(json.dumps(dict(frames=frames)))
    s = evaluate(bench, run, res["depth"]["tag"], res["tool"]["tag"], log=lambda *_: None)
    assert s["n_scaled"] == N
    assert s["instrument"]["abs_rel"] < 0.003 and s["oracle"]["abs_rel"] < 0.002
    assert s["native"]["median_ratio"] == pytest.approx(1 / (NET_SCALE * 1000), rel=0.01)   # GT / pred


def test_frame_image_is_valid():
    img = frame_image(0, np.random.default_rng(0))
    assert img.shape == (540, 960, 3) and img[270, 480].sum() > 0 and img[5, 5].sum() == 0
