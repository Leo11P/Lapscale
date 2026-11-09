"""Depth benchmark folder.

    images/<NNNNNN>.png   pinhole images given to the networks (rectified left for stereo sources)
    gt/<NNNNNN>.npz       depth (float32, mm), valid (bool)
    camera.yaml           pinhole camera of the images (no distortion)
    manifest.json         source and per-frame provenance
    results/<name>/       one folder per evaluated backend

File names are zero-padded indices, so the image folder can also be fed to ``lapscale run`` as a "video"
(frame k = k-th image) and evaluated with ``lapscale eval``.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from ..camera import Camera
from ..io.video import FrameSource
from .stereo import Rectifier, StereoConfig, load_stereo_calib, stereo_depth


def _crop(img, crop):
    if crop is None:
        return img
    y0, y1, x0, x1 = crop
    return img[y0:y1, x0:x1]


def prepare_stereo(left: str | Path, right: str | Path, calib: str | Path, out: str | Path, n: int = 50,
                   offset: int = 0, crop: tuple[int, int, int, int] | None = None, calib_frame: str = "full",
                   start: int = 0, stop: int | None = None, stereo_cfg: StereoConfig | None = None,
                   log=print) -> dict:
    """``crop`` = (y0, y1, x0, x1) applied to both videos; ``calib_frame`` 'full' if the intrinsics refer to
    the uncropped frame (they are then shifted by the crop), 'crop' if they already refer to the crop."""
    out = Path(out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "gt").mkdir(parents=True, exist_ok=True)
    sl, sr = FrameSource(left), FrameSource(right)
    w, h = sl.size if crop is None else (crop[3] - crop[2], crop[1] - crop[0])
    off = (crop[2], crop[0]) if (crop is not None and calib_frame == "full") else (0, 0)
    rect = Rectifier.from_calib(load_stereo_calib(calib, (w, h), off))
    idx = sl.sample_indices(n=n, start=start, stop=stop)
    idx = [i for i in idx if 0 <= i + offset < len(sr)]
    lf = dict(sl.iter(idx))
    rf = dict(sr.iter([i + offset for i in idx]))
    frames = []
    for k, i in enumerate(idx):
        if i not in lf or i + offset not in rf:
            continue
        L, R = rect.rectify(_crop(lf[i], crop), _crop(rf[i + offset], crop))
        depth, valid, diag = stereo_depth(L, R, rect.f, rect.B, stereo_cfg)
        fid = f"{k:06d}"
        cv2.imwrite(str(out / "images" / f"{fid}.png"), L)
        np.savez_compressed(out / "gt" / f"{fid}.npz", depth=depth, valid=valid)
        frames.append(dict(id=fid, left=str(left), right=str(right), left_index=i, right_index=i + offset, **diag))
        log(f"  frame {i}: GT valid {100 * diag['valid_fraction']:.0f}%, median depth {diag['depth_median_mm']} mm")
    cam = Camera(rect.K, [], (w, h), source="import",
                 notes=f"rectified left of {Path(calib).name} (baseline {rect.B:.3f} mm)")
    cam.save(out / "camera.yaml")
    manifest = dict(source="stereo", calib=str(calib), crop=crop, calib_frame=calib_frame, offset=offset,
                    f_px=rect.f, baseline_mm=rect.B, stereo=(stereo_cfg or StereoConfig()).__dict__, frames=frames,
                    created=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"depth benchmark: {len(frames)} frames -> {out}")
    return manifest


def prepare_folder(images: str | Path, gt: str | Path, camera: str | Path, out: str | Path, gt_scale_mm: float = 1.0,
                   log=print) -> dict:
    """Any other source with images and depth maps (``<stem>.npy``/``.npz``/16-bit ``.png``, times
    ``gt_scale_mm`` to get mm; 0 or non-finite = invalid). Images must already be pinhole (undistorted)."""
    out = Path(out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "gt").mkdir(parents=True, exist_ok=True)
    cam = Camera.load(camera)
    if cam.has_distortion():
        raise ValueError("the benchmark images must be undistorted (camera without distortion)")
    frames = []
    for k, p in enumerate(sorted(q for q in Path(images).iterdir() if q.suffix.lower() in (".png", ".jpg", ".jpeg"))):
        g = None
        for ext in (".npy", ".npz", ".png", ".tiff", ".tif"):
            q = Path(gt) / f"{p.stem}{ext}"
            if q.exists():
                g = (np.load(q) if ext == ".npy" else np.load(q)["depth"] if ext == ".npz"
                     else cv2.imread(str(q), cv2.IMREAD_UNCHANGED)).astype(np.float32) * gt_scale_mm
                break
        if g is None:
            log(f"  {p.name}: no ground truth, skipped")
            continue
        fid = f"{k:06d}"
        cv2.imwrite(str(out / "images" / f"{fid}.png"), cv2.imread(str(p)))
        valid = np.isfinite(g) & (g > 0)
        np.savez_compressed(out / "gt" / f"{fid}.npz", depth=np.where(valid, g, 0).astype(np.float32), valid=valid)
        frames.append(dict(id=fid, image=str(p)))
    cam.save(out / "camera.yaml")
    manifest = dict(source="folder", images=str(images), gt=str(gt), frames=frames,
                    created=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"depth benchmark: {len(frames)} frames -> {out}")
    return manifest


def load_gt(bench: str | Path, fid: str) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(Path(bench) / "gt" / f"{fid}.npz")
    return z["depth"], z["valid"]


def frame_ids(bench: str | Path) -> list[str]:
    return [f["id"] for f in json.loads((Path(bench) / "manifest.json").read_text())["frames"]]
