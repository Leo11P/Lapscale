"""Ground-truth MASKS of the instruments (optional, next to the edge annotations).

Uses: training data for fine-tuning (YOLO-seg, a SurgeNetXL-based network), mask metrics (IoU, boundary
error) in ``segbench run``, and the ``masks`` backend as an oracle.

Files: ``<bench>/masks_gt/<frame>_<k>.png`` (one 0/255 PNG per instance) and ``masks_gt/index.json``
(frame -> "ok" | "empty"). Same naming as the ``masks`` backend, so ``--masks-dir <bench>/masks_gt``
benchmarks the ground truth itself.

Editor (``lapscale segbench annotate-masks``): proposals from a backend (e.g. SAM 3) are loaded as
instances, then corrected.
  left drag   paint the current instance      right drag  erase
  p           polygon mode: clicks add vertices, SPACE fills (adds), e + SPACE subtracts
  [ / ]       brush size                       TAB         next instance
  n           new empty instance               d           delete current instance
  ENTER       save and next                    x           frame without instruments
  b           back                             q           save and quit
  (with a SAM 3 proposer) drag with the middle button: box -> SAM 3 mask as a new instance
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .data import load_manifest

MASK_DIR = "masks_gt"


# ---------------------------------------------------------------------------------------- state
@dataclass
class MaskEditState:
    shape: tuple[int, int]                              # (h, w)
    instances: list[np.ndarray] = field(default_factory=list)
    current: int = -1
    brush: int = 6
    polygon: list[tuple[float, float]] = field(default_factory=list)

    def add_instance(self, mask: np.ndarray | None = None) -> int:
        m = np.zeros(self.shape, bool) if mask is None else mask.astype(bool).copy()
        self.instances.append(m)
        self.current = len(self.instances) - 1
        return self.current

    def delete_current(self) -> None:
        if 0 <= self.current < len(self.instances):
            del self.instances[self.current]
            self.current = min(self.current, len(self.instances) - 1)

    def next_instance(self) -> None:
        if self.instances:
            self.current = (self.current + 1) % len(self.instances)

    def _cur(self) -> np.ndarray:
        if self.current < 0:
            self.add_instance()
        return self.instances[self.current]

    def paint(self, x: float, y: float, value: bool) -> None:
        m = self._cur().astype(np.uint8)
        cv2.circle(m, (int(round(x)), int(round(y))), self.brush, int(value), -1)
        self.instances[self.current] = m.astype(bool)

    def fill_polygon(self, value: bool = True) -> str:
        if len(self.polygon) < 3:
            return "a polygon needs at least 3 vertices"
        m = self._cur().astype(np.uint8)
        cv2.fillPoly(m, [np.round(np.array(self.polygon)).astype(np.int32)], int(value))
        self.instances[self.current] = m.astype(bool)
        self.polygon = []
        return "polygon " + ("added" if value else "subtracted")

    def cleaned(self, min_area: int = 50) -> list[np.ndarray]:
        """Non-empty instances, each restricted to its largest connected component pieces >= min_area."""
        out = []
        for m in self.instances:
            n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
            keep = np.zeros_like(m)
            for k in range(1, n):
                if stats[k, cv2.CC_STAT_AREA] >= min_area:
                    keep |= lab == k
            if keep.any():
                out.append(keep)
        return out


# ---------------------------------------------------------------------------------------- files
def mask_dir(bench: str | Path) -> Path:
    return Path(bench) / MASK_DIR


def load_index(bench) -> dict[str, str]:
    p = mask_dir(bench) / "index.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_frame_masks(bench, frame_id: str, masks: list[np.ndarray]) -> None:
    d = mask_dir(bench)
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob(f"{frame_id}_*.png"):
        old.unlink()
    for k, m in enumerate(masks):
        cv2.imwrite(str(d / f"{frame_id}_{k}.png"), m.astype(np.uint8) * 255)
    idx = load_index(bench)
    idx[frame_id] = "ok" if masks else "empty"
    (d / "index.json").write_text(json.dumps(dict(sorted(idx.items())), indent=1))


def load_frame_masks(bench, frame_id: str) -> list[np.ndarray]:
    return [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 127
            for p in sorted(mask_dir(bench).glob(f"{frame_id}_*.png"))]


# ---------------------------------------------------------------------------------------- metrics
def mask_metrics(pred: list[np.ndarray], gt: list[np.ndarray], shape: tuple[int, int]) -> dict:
    """Frame-level (all instruments together): IoU, and mean / 95th percentile distance between the
    predicted and the ground-truth instrument boundaries (pixels)."""
    p = np.zeros(shape, bool)
    g = np.zeros(shape, bool)
    for m in pred:
        p |= m
    for m in gt:
        g |= m
    union = np.logical_or(p, g).sum()
    iou = float(np.logical_and(p, g).sum() / union) if union else 1.0
    if not p.any() or not g.any():
        return dict(mask_iou=iou, boundary_err_px_mean=np.nan, boundary_err_px_p95=np.nan)
    bp = _boundary(p)
    bg = _boundary(g)
    dg = cv2.distanceTransform((~bg).astype(np.uint8), cv2.DIST_L2, 5)
    dp = cv2.distanceTransform((~bp).astype(np.uint8), cv2.DIST_L2, 5)
    d = np.concatenate([dg[bp], dp[bg]])
    return dict(mask_iou=iou, boundary_err_px_mean=float(d.mean()), boundary_err_px_p95=float(np.percentile(d, 95)))


def _boundary(m: np.ndarray) -> np.ndarray:
    er = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8))
    return m & ~er.astype(bool)


# ---------------------------------------------------------------------------------------- export
def export_yolo(bench: str | Path, out: str | Path, val_fraction: float = 0.2, seed: int = 0, log=print) -> Path:
    """YOLO-seg dataset (one class, "instrument") from the ground-truth masks. Frames are split BY VIDEO
    so that near-identical frames do not end up on both sides; with a single video the split is random
    (and the validation score is optimistic)."""
    bench, out = Path(bench), Path(out)
    frames = {f["id"]: f for f in load_manifest(bench)["frames"]}
    idx = load_index(bench)
    ids = [fid for fid in sorted(idx) if fid in frames]
    if not ids:
        raise RuntimeError("no ground-truth masks: run `lapscale segbench annotate-masks` first")
    videos = sorted({frames[fid]["video"] for fid in ids})
    rng = np.random.default_rng(seed)
    if len(videos) > 1:
        n_val = max(1, int(round(val_fraction * len(videos))))
        val_v = set(rng.choice(videos, n_val, replace=False))
        split = {fid: "val" if frames[fid]["video"] in val_v else "train" for fid in ids}
    else:
        log("WARNING: one video only: random frame split, validation will be optimistic")
        perm = rng.permutation(len(ids))
        n_val = max(1, int(round(val_fraction * len(ids))))
        split = {fid: "val" if k in set(perm[:n_val]) else "train" for k, fid in enumerate(ids)}
    for sp in ("train", "val"):
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)
    for fid in ids:
        img = cv2.imread(str(bench / frames[fid]["file"]))
        h, w = img.shape[:2]
        sp = split[fid]
        cv2.imwrite(str(out / "images" / sp / f"{fid}.png"), img)
        lines = []
        for m in load_frame_masks(bench, fid):
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cs:
                if cv2.contourArea(c) < 20 or len(c) < 3:
                    continue
                c = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2)
                xy = (c / [w, h]).clip(0, 1).ravel()
                lines.append("0 " + " ".join(f"{v:.6f}" for v in xy))
        (out / "labels" / sp / f"{fid}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    yaml_path = out / "data.yaml"
    yaml_path.write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: instrument\n")
    n_tr = sum(v == "train" for v in split.values())
    log(f"YOLO dataset: {n_tr} train / {len(split) - n_tr} val frames -> {yaml_path}")
    return yaml_path


# ---------------------------------------------------------------------------------------- GUI
def annotate_masks(bench: str | Path, proposer=None, display_scale: float = 1.0,
                   only_todo: bool = True) -> None:  # pragma: no cover - interactive
    bench = Path(bench)
    frames = load_manifest(bench)["frames"]
    idx = load_index(bench)
    order = [f for f in frames if not only_todo or f["id"] not in idx]
    if not order:
        print("nothing to annotate (use --all to revisit)")
        return
    win = "lapscale segbench annotate-masks"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    k = 0
    while 0 <= k < len(order):
        f = order[k]
        img = cv2.imread(str(bench / f["file"]))
        st = MaskEditState(img.shape[:2])
        existing = load_frame_masks(bench, f["id"])
        if existing:
            for m in existing:
                st.add_instance(m)
        elif proposer is not None:
            if hasattr(proposer, "set_frame_id"):
                proposer.set_frame_id(f["id"])
            for inst in proposer.predict(img):
                st.add_instance(inst.prob > 0.5)
        mode = {"poly": False, "erase_poly": False, "drag": None, "box0": None, "msg": ""}

        def to_img(xd, yd):
            return (xd + 0.5) / display_scale - 0.5, (yd + 0.5) / display_scale - 0.5

        def on_mouse(ev, xd, yd, flags, *_, st=st, img=img, mode=mode):
            x, y = to_img(xd, yd)
            if mode["poly"] and ev == cv2.EVENT_LBUTTONDOWN:
                st.polygon.append((x, y))
            elif ev in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
                mode["drag"] = ev == cv2.EVENT_LBUTTONDOWN
                st.paint(x, y, mode["drag"])
            elif ev == cv2.EVENT_MOUSEMOVE and mode["drag"] is not None:
                st.paint(x, y, mode["drag"])
            elif ev in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
                mode["drag"] = None
            elif ev == cv2.EVENT_MBUTTONDOWN:
                mode["box0"] = (x, y)
            elif ev == cv2.EVENT_MBUTTONUP and mode["box0"] and hasattr(proposer, "predict_box"):
                x0, y0 = mode["box0"]
                inst = proposer.predict_box(img, (min(x0, x), min(y0, y), max(x0, x), max(y0, y)))
                if inst is not None:
                    st.add_instance(inst.prob > 0.5)
                    mode["msg"] = "SAM 3 box -> new instance"
                mode["box0"] = None

        cv2.setMouseCallback(win, on_mouse)
        while True:
            vis = img.copy()
            for i, m in enumerate(st.instances):
                col = np.array((0, 255, 0) if i == st.current else (255, 160, 0), np.uint8)
                vis[m] = (0.5 * vis[m] + 0.5 * col).astype(np.uint8)
            for p in st.polygon:
                cv2.circle(vis, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1)
            vis = cv2.resize(vis, None, fx=display_scale, fy=display_scale)
            txt = (f"{k + 1}/{len(order)} {f['id']} | instance {st.current + 1}/{len(st.instances)} | brush {st.brush}"
                   f" | {'POLYGON' + (' (erase)' if mode['erase_poly'] else '') if mode['poly'] else 'brush'}"
                   f" | {mode['msg']}")
            cv2.putText(vis, txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(win, vis)
            key = cv2.waitKey(20) & 0xFF
            if key == ord("p"):
                mode["poly"] = not mode["poly"]
                st.polygon = []
            elif key == ord("e"):
                mode["erase_poly"] = not mode["erase_poly"]
            elif key == 32 and mode["poly"]:
                mode["msg"] = st.fill_polygon(not mode["erase_poly"])
            elif key == ord("["):
                st.brush = max(1, st.brush - 1)
            elif key == ord("]"):
                st.brush += 1
            elif key == 9:
                st.next_instance()
            elif key == ord("n"):
                st.add_instance()
            elif key == ord("d"):
                st.delete_current()
            elif key in (10, 13):
                save_frame_masks(bench, f["id"], st.cleaned())
                k += 1
                break
            elif key == ord("x"):
                save_frame_masks(bench, f["id"], [])
                k += 1
                break
            elif key == ord("b"):
                k = max(0, k - 1)
                break
            elif key == ord("q"):
                cv2.destroyAllWindows()
                return
    cv2.destroyAllWindows()
