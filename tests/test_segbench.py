"""Segmentation benchmark on a synthetic benchmark folder: the pipeline must report ~0 error for the
true masks and the right bias / depth error for masks that are 1 px too wide."""
import json

import cv2
import numpy as np
import pytest
from synthetic_scene import render_instrument
from synthetic_shaft import CAMERA, project, silhouette_points

from lapscale.segbench import compare, load_annotations, run_benchmark, sample_frames, save_annotations
from lapscale.segbench.annotate import AnnotationState, Magnifier, display_to_image
from lapscale.segbench.data import FrameAnnotation, GTInstrument
from lapscale.tool.backends import make_seg_backend

SCENES = [  # (axis point, direction, radius)
    (np.array([8.0, 4.0, 80.0]), np.array([0.8, 0.35, 0.35]), 2.5),
    (np.array([-5.0, 2.0, 60.0]), np.array([-0.9, 0.1, 0.4]), 5.0),
    (np.array([0.0, -3.0, 110.0]), np.array([0.3, -0.9, 0.3]), 2.5),
    (np.array([-10.0, 6.0, 90.0]), np.array([-0.5, 0.6, 0.6]), 2.5),
    (np.array([12.0, -4.0, 70.0]), np.array([0.7, -0.6, 0.4]), 5.0),
]


def _clicks(a, u, r, rng, n=4, noise=0.0):
    """Points an annotator would click: on both silhouette edges along the straight part (inside image)."""
    edges = []
    for g in silhouette_points(a, u, r, np.linspace(-40, 0, 60)):
        px = project(g)
        w, h = CAMERA.image_size
        px = px[(px[:, 0] > 8) & (px[:, 0] < w - 8) & (px[:, 1] > 8) & (px[:, 1] < h - 8)]
        idx = np.linspace(0, len(px) - 1, n).round().astype(int)
        edges.append((px[idx] + rng.normal(0, noise, (n, 2))).round(2).tolist())
    return edges


@pytest.fixture(scope="module")
def bench(tmp_path_factory):
    root = tmp_path_factory.mktemp("bench")
    (root / "frames").mkdir()
    oracle, wide = root / "masks_true", root / "masks_wide"
    oracle.mkdir()
    wide.mkdir()
    rng = np.random.default_rng(0)
    frames, ann = [], {}
    for k, (a, u, r) in enumerate(SCENES):
        u = u / np.linalg.norm(u)
        soft = render_instrument(a, u, r)
        bg = rng.normal(120, 15, soft.shape)
        img = np.clip(bg * (1 - soft) + 200 * soft, 0, 255).astype(np.uint8)
        fid = f"synthetic_{k:06d}"
        cv2.imwrite(str(root / "frames" / f"{fid}.png"), cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        frames.append(dict(id=fid, video="synthetic", index=k, file=f"frames/{fid}.png", size=[960, 540]))
        ea, eb = _clicks(a, u, r, rng, noise=0.2)
        ann[fid] = FrameAnnotation("ok", [GTInstrument(ea, eb, 2 * r)])
        cv2.imwrite(str(oracle / f"{fid}_0.png"), np.round(soft * 255).astype(np.uint8))
        dil = cv2.dilate((soft > 0.5).astype(np.uint8), np.ones((3, 3), np.uint8)) * 255
        if k != 4:                                         # one frame without prediction
            cv2.imwrite(str(wide / f"{fid}.png"), dil)
    (root / "manifest.json").write_text(json.dumps(dict(frames=frames)))
    save_annotations(root, ann)
    return root


def test_oracle_masks_give_no_error(bench):
    s = run_benchmark(bench, make_seg_backend("masks", masks_dir=bench / "masks_true"), CAMERA, "oracle",
                      log=lambda *_: None)
    assert s["found_rate"] == 1.0
    assert s["depth_err_pct_p90_abs"] < 1.0               # 0.2 px click noise on 4 points per edge
    assert abs(s["bias_px_median"]) < 0.15
    assert (bench / "results" / "oracle" / "overlays").glob("*.png")


def test_wide_masks_show_bias_and_depth_error(bench):
    s = run_benchmark(bench, make_seg_backend("masks", masks_dir=bench / "masks_wide"), CAMERA, "wide",
                      log=lambda *_: None)
    assert s["n_found"] == 4 and s["n_annotated"] == 5
    assert 0.9 < s["bias_px_median"] < 1.5
    assert s["depth_err_pct_median_signed"] < -1.0       # too wide -> too close
    assert s["depth_err_pct_bias_corrected_median_abs"] < 0.5 * s["depth_err_pct_median_abs"]
    assert "no instance" in s["not_found_reasons"]


def test_compare_table(bench):
    rows = compare(bench, log=lambda *_: None)
    assert {r["name"] for r in rows} >= {"oracle", "wide"}
    assert rows[0]["name"] == "oracle"
    assert (bench / "results" / "comparison.md").read_text().count("|") > 10


def test_annotation_state_machine():
    st = AnnotationState(default_diameter=5.0)
    st.click(10, 10)
    assert "at least 2" in st.close_edge()
    st.click(50, 12)
    assert "edge B" in st.close_edge()
    st.click(10, 30)
    st.click(50, 31)
    st.diameter = 10.0
    assert "instrument 1 done" in st.close_edge()
    assert st.complete and st.instruments[0].diameter_mm == 10.0
    st.undo()                                    # reopens the last edge of the instrument
    assert not st.complete and len(st.points) == 2 and len(st.current) == 1
    st.reset()
    assert st.to_annotation().status == "skip"


def test_click_mapping():
    assert display_to_image(0, 0, 1.0) == (0.0, 0.0)
    assert display_to_image(1, 1, 2.0) == (0.25, 0.25)        # 2nd display px of image px 0
    mag = Magnifier(zoom=6, half=20)
    mag.centre = (100.2, 50.0)
    x0, y0, w, _ = mag.box(1000)
    # centre of the central zoomed pixel -> the (rounded) centre pixel itself
    cx, cy = mag.to_image(x0 + 20 * 6 + 2, y0 + 20 * 6 + 2, 1000)
    assert cx == pytest.approx(100.0, abs=1 / 6) and cy == pytest.approx(50.0, abs=1 / 6)


def test_annotations_roundtrip(tmp_path):
    ann = {"f1": FrameAnnotation("ok", [GTInstrument([[0, 0], [1, 1]], [[0, 2], [1, 3]], 5.0)]),
           "f2": FrameAnnotation("skip", [])}
    save_annotations(tmp_path, ann)
    back = load_annotations(tmp_path)
    assert back["f1"].instruments[0].diameter_mm == 5.0 and back["f2"].status == "skip"


def test_sample_frames_skips_dark(tmp_path):
    path = tmp_path / "v.avi"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (160, 120))
    for k in range(100):
        level = 5 if 40 <= k < 60 else 120                 # scope out of the body in the middle
        vw.write(np.full((120, 160, 3), level, np.uint8))
    vw.release()
    m = sample_frames([path], tmp_path / "bench", n_per_video=5, log=lambda *_: None)
    idx = [f["index"] for f in m["frames"]]
    assert len(idx) >= 4 and not any(40 <= i < 60 for i in idx)


def test_cli_run_with_config(bench, tmp_path):
    from lapscale.cli import main
    cam = CAMERA.save(tmp_path / "cam.yaml")
    cfg = tmp_path / "c.yaml"
    cfg.write_text("tool:\n  edges:\n    ransac-tol-px: 1.0\n    min-span-px: 40\n")
    assert main(["--config", str(cfg), "segbench", "run", "--bench", str(bench), "--camera", str(cam),
                 "--backend", "masks", "--masks-dir", str(bench / "masks_true"), "--name", "cli",
                 "--no-overlays"]) == 0
    assert json.loads((bench / "results" / "cli" / "summary.json").read_text())["found_rate"] == 1.0
