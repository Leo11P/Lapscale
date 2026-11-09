"""Training frames from one or more benchmark folders: image + union of the instrument masks."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..segbench.data import load_manifest
from ..segbench.masks import load_frame_masks, load_index


def collect(benches: list[str | Path]) -> list[dict]:
    items = []
    for b in benches:
        b = Path(b)
        frames = {f["id"]: f for f in load_manifest(b)["frames"]}
        for fid, status in load_index(b).items():
            if fid in frames:
                items.append(dict(bench=b, id=fid, video=frames[fid]["video"], file=b / frames[fid]["file"],
                                  empty=status == "empty"))
    if not items:
        raise RuntimeError("no ground-truth masks found: run `lapscale segbench annotate-masks`")
    return items


def split_by_video(items: list[dict], val_fraction: float, seed: int = 0) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(seed)
    videos = sorted({i["video"] for i in items})
    if len(videos) > 1:
        n = max(1, int(round(val_fraction * len(videos))))
        val_v = set(rng.choice(videos, n, replace=False))
        return [i for i in items if i["video"] not in val_v], [i for i in items if i["video"] in val_v]
    perm = rng.permutation(len(items))
    n = max(1, int(round(val_fraction * len(items))))
    val = {int(k) for k in perm[:n]}
    return [i for k, i in enumerate(items) if k not in val], [i for k, i in enumerate(items) if k in val]


def load_pair(item: dict) -> tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(str(item["file"]))
    m = np.zeros(img.shape[:2], bool)
    for x in load_frame_masks(item["bench"], item["id"]):
        m |= x
    return img, m


def augment(img: np.ndarray, m: np.ndarray, rng, size_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Random scale/crop, flips, small rotation, brightness/contrast/hue jitter. Geometric transforms are
    applied identically to the mask (nearest) so the boundary is not shifted."""
    h, w = img.shape[:2]
    s = rng.uniform(0.7, 1.0)
    ch, cw = int(h * s), int(w * s)
    y0, x0 = rng.integers(0, h - ch + 1), rng.integers(0, w - cw + 1)
    img, m = img[y0:y0 + ch, x0:x0 + cw], m[y0:y0 + ch, x0:x0 + cw]
    if rng.random() < 0.5:
        img, m = img[:, ::-1], m[:, ::-1]
    if rng.random() < 0.3:
        img, m = img[::-1], m[::-1]
    ang = rng.uniform(-15, 15)
    R = cv2.getRotationMatrix2D((cw / 2, ch / 2), ang, 1.0)
    img = cv2.warpAffine(np.ascontiguousarray(img), R, (cw, ch), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    m = cv2.warpAffine(np.ascontiguousarray(m).astype(np.uint8), R, (cw, ch), flags=cv2.INTER_NEAREST,
                       borderMode=cv2.BORDER_REFLECT).astype(bool)
    img = img.astype(np.float32) * rng.uniform(0.8, 1.2) + rng.uniform(-20, 20)
    hsv = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + rng.integers(-5, 6)) % 180
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    img = cv2.resize(img, (size_hw[1], size_hw[0]), interpolation=cv2.INTER_AREA)
    m = cv2.resize(m.astype(np.uint8), (size_hw[1], size_hw[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    return img, m
