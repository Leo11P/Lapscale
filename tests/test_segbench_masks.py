import json

import cv2
import numpy as np
from synthetic_scene import render_instrument
from synthetic_shaft import CAMERA

from lapscale.segbench import run_benchmark, save_annotations
from lapscale.segbench.masks import (
    MaskEditState,
    export_yolo,
    load_frame_masks,
    load_index,
    mask_metrics,
    save_frame_masks,
)
from lapscale.segtrain.data import augment, collect, load_pair, split_by_video
from lapscale.tool.backends import make_seg_backend

POSES = [(np.array([8.0, 4.0, 80.0]), np.array([0.8, 0.35, 0.35]), 2.5),
         (np.array([-5.0, 2.0, 60.0]), np.array([-0.9, 0.1, 0.4]), 5.0),
         (np.array([0.0, -3.0, 110.0]), np.array([0.3, -0.9, 0.3]), 2.5)]


def _bench(tmp_path):
    (tmp_path / "frames").mkdir(parents=True)
    frames = []
    for k, (a, u, r) in enumerate(POSES):
        m = render_instrument(a, u / np.linalg.norm(u), r)
        img = (np.full((540, 960, 3), 110) * (1 - m[..., None]) + 200 * m[..., None]).astype(np.uint8)
        fid = f"v{k % 2}_{k:06d}"
        cv2.imwrite(str(tmp_path / "frames" / f"{fid}.png"), img)
        frames.append(dict(id=fid, video=f"v{k % 2}.mp4", index=k, file=f"frames/{fid}.png", size=[960, 540]))
        save_frame_masks(tmp_path, fid, [m > 0.5])
    (tmp_path / "manifest.json").write_text(json.dumps(dict(frames=frames)))
    save_annotations(tmp_path, {})
    return tmp_path


def test_edit_state():
    st = MaskEditState((100, 100))
    st.paint(50, 50, True)
    assert st.instances[0][50, 50] and st.current == 0
    st.polygon = [(10, 10), (30, 10), (30, 30), (10, 30)]
    assert "added" in st.fill_polygon(True)
    assert st.instances[0][20, 20]
    st.polygon = [(12, 12), (28, 12), (28, 28)]
    st.fill_polygon(False)
    assert not st.instances[0][15, 25] and st.instances[0][25, 15]      # [row, col]: inside / outside the triangle
    st.add_instance()
    st.paint(80, 80, True)
    assert len(st.cleaned(min_area=10)) == 2
    st.delete_current()
    assert len(st.instances) == 1


def test_mask_files_metrics_and_benchmark(tmp_path):
    b = _bench(tmp_path)
    assert set(load_index(b).values()) == {"ok"}
    gt = load_frame_masks(b, "v0_000000")
    assert len(gt) == 1
    mm = mask_metrics(gt, gt, gt[0].shape)
    assert mm["mask_iou"] == 1.0 and mm["boundary_err_px_mean"] == 0.0
    wide = [cv2.dilate(gt[0].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)]
    mm = mask_metrics(wide, gt, gt[0].shape)
    assert mm["mask_iou"] < 1.0 and 1.5 < mm["boundary_err_px_mean"] < 2.5
    s = run_benchmark(b, make_seg_backend("masks", masks_dir=b / "masks_gt"), CAMERA, "gt", log=lambda *_: None)
    assert s["n_mask_frames"] == 3 and s["mask_iou_median"] == 1.0


def test_export_yolo_splits_by_video(tmp_path):
    b = _bench(tmp_path / "b")
    y = export_yolo(b, tmp_path / "yolo", val_fraction=0.5, log=lambda *_: None)
    assert "names" in y.read_text()
    labels = list((tmp_path / "yolo" / "labels").rglob("*.txt"))
    assert len(labels) == 3
    first = labels[0].read_text().split()
    assert first[0] == "0" and len(first) >= 7 and all(0 <= float(v) <= 1 for v in first[1:])
    val_imgs = {p.stem for p in (tmp_path / "yolo" / "images" / "val").glob("*.png")}
    assert {s[:2] for s in val_imgs} in ({"v0"}, {"v1"})            # one whole video in validation


def test_training_data_pipeline(tmp_path):
    b = _bench(tmp_path)
    items = collect([b])
    tr, va = split_by_video(items, 0.5)
    assert {i["video"] for i in tr}.isdisjoint({i["video"] for i in va})
    img, m = load_pair(items[0])
    rng = np.random.default_rng(0)
    for _ in range(5):
        ai, am = augment(img, m, rng, (480, 640))
        assert ai.shape == (480, 640, 3) and am.shape == (480, 640) and am.dtype == bool
        assert 0 < am.mean() < 0.5
