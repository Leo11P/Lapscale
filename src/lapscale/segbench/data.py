"""Benchmark data: sampled frames, manifest, hand-annotated shaft edges.

Layout of a benchmark folder::

    frames/<video>_<index>.png     raw frames (as recorded, distortion included)
    manifest.json                  where each frame comes from
    annotations.json               ground-truth shaft edges, see below
    results/<backend>/             one folder per evaluated backend

Ground truth is the two EDGES of each measurable shaft (points clicked on the instrument/background
boundary along the straight part, >= 2 per edge, raw pixels), not a mask: the edges are exactly what
the metric scale depends on, and clicking them is much faster than painting masks.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from ..io.video import FrameSource

ANNOTATION_VERSION = 1


@dataclass
class GTInstrument:
    edge_a: list[list[float]]           # raw pixels
    edge_b: list[list[float]]
    diameter_mm: float | None = None
    note: str = ""

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(self.edge_a, np.float64), np.asarray(self.edge_b, np.float64)


@dataclass
class FrameAnnotation:
    status: str = "todo"                # todo | ok | skip (no measurable shaft)
    instruments: list[GTInstrument] = field(default_factory=list)


# ------------------------------------------------------------------------------------------ frames
def sample_frames(videos: list[str | Path], out_dir: str | Path, n_per_video: int = 25,
                  min_brightness: float = 20.0, log=print) -> dict:
    """Evenly spaced frames from each video; frames darker than ``min_brightness`` (mean grey level,
    e.g. scope out of the body) are replaced by the next usable one."""
    out = Path(out_dir)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    mpath = out / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {"frames": []}
    have = {f["id"] for f in manifest["frames"]}
    for v in videos:
        src = FrameSource(v)
        stem = Path(v).stem
        targets = src.sample_indices(n=n_per_video)
        step = max(1, len(src) // max(1, 8 * n_per_video))
        tries = {t: [t + k * step for k in range(4) if t + k * step < len(src)] for t in targets}
        frames = dict(src.iter(sorted({i for ts in tries.values() for i in ts})))
        taken = 0
        for t in targets:
            pick = next((i for i in tries[t] if i in frames and float(frames[i].mean()) >= min_brightness), None)
            if pick is None:
                continue
            fid = f"{stem}_{pick:06d}"
            if fid not in have:
                cv2.imwrite(str(out / "frames" / f"{fid}.png"), frames[pick])
                manifest["frames"].append(dict(id=fid, video=str(v), index=pick, file=f"frames/{fid}.png",
                                               size=list(src.size)))
                have.add(fid)
            taken += 1
        log(f"{stem}: {taken} frames")
    manifest["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    mpath.write_text(json.dumps(manifest, indent=2))
    return manifest


def load_manifest(bench: str | Path) -> dict:
    p = Path(bench) / "manifest.json"
    if not p.exists():
        raise FileNotFoundError(f"{p}: run `lapscale segbench sample` first")
    return json.loads(p.read_text())


# ------------------------------------------------------------------------------------------ annotations
def load_annotations(bench: str | Path) -> dict[str, FrameAnnotation]:
    p = Path(bench) / "annotations.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    out = {}
    for fid, fa in d.get("frames", {}).items():
        out[fid] = FrameAnnotation(fa.get("status", "todo"),
                                   [GTInstrument(**g) for g in fa.get("instruments", [])])
    return out


def save_annotations(bench: str | Path, ann: dict[str, FrameAnnotation]) -> Path:
    p = Path(bench) / "annotations.json"
    d = dict(version=ANNOTATION_VERSION, updated=datetime.now(timezone.utc).isoformat(timespec="seconds"),
             frames={fid: dict(status=fa.status, instruments=[asdict(g) for g in fa.instruments])
                     for fid, fa in sorted(ann.items())})
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, indent=1))
    tmp.replace(p)                       # atomic: an interrupted save never corrupts the file
    return p
