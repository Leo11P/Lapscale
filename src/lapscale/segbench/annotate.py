"""Click annotation of shaft edges.

For every measurable shaft (both edges visible over a good length, straight part only, NOT the jaws):
click >= 2 points on edge A (3-5 spread along the visible length is better, edges are slightly curved
by the lens distortion), press SPACE, the same for edge B, press SPACE. Points go ON the boundary
between instrument and background. A magnifier follows the cursor; clicking inside the magnifier gives
sub-pixel positions.

Keys:  SPACE close edge | u undo | r reset frame | 5 / 8 / 0 diameter 5 / 8 / 10 mm for the current
       instrument | ENTER save and next | x no measurable shaft, next | b back | q save and quit
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..io.gui import require_gui
from .data import FrameAnnotation, GTInstrument, load_annotations, load_manifest, save_annotations

DIAMETER_KEYS = {ord("5"): 5.0, ord("8"): 8.0, ord("0"): 10.0}


@dataclass
class AnnotationState:
    """Pure state machine of the annotation of one frame (tested without a GUI)."""

    default_diameter: float | None = None
    instruments: list[GTInstrument] = field(default_factory=list)
    current: list[list[list[float]]] = field(default_factory=list)   # edges of the instrument in progress
    points: list[list[float]] = field(default_factory=list)          # points of the edge in progress
    diameter: float | None = None

    def __post_init__(self):
        self.diameter = self.default_diameter

    def click(self, x: float, y: float) -> None:
        self.points.append([round(float(x), 2), round(float(y), 2)])

    def close_edge(self) -> str:
        if len(self.points) < 2:
            return "an edge needs at least 2 points"
        self.current.append(self.points)
        self.points = []
        if len(self.current) == 2:
            self.instruments.append(GTInstrument(self.current[0], self.current[1], self.diameter))
            self.current = []
            self.diameter = self.default_diameter
            return f"instrument {len(self.instruments)} done"
        return "edge A done, now edge B"

    def undo(self) -> None:
        if self.points:
            self.points.pop()
        elif self.current:
            self.points = self.current.pop()
        elif self.instruments:
            last = self.instruments.pop()
            self.current, self.points = [last.edge_a], last.edge_b
            self.diameter = last.diameter_mm

    def reset(self) -> None:
        self.instruments, self.current, self.points = [], [], []
        self.diameter = self.default_diameter

    @property
    def complete(self) -> bool:
        return not self.points and not self.current

    def to_annotation(self) -> FrameAnnotation:
        return FrameAnnotation("ok" if self.instruments else "skip", list(self.instruments))


class Magnifier:
    """Zoomed view of the area around the cursor, drawn in the top-right corner of the display."""

    def __init__(self, zoom: int = 6, half: int = 20):
        self.zoom, self.half = zoom, half
        self.centre = (0.0, 0.0)            # image pixels

    def box(self, disp_w: int) -> tuple[int, int, int, int]:
        size = (2 * self.half + 1) * self.zoom
        return disp_w - size - 10, 10, size, size     # x, y, w, h in display pixels

    def inside(self, xd: int, yd: int, disp_w: int) -> bool:
        x, y, w, h = self.box(disp_w)
        return x <= xd < x + w and y <= yd < y + h

    def to_image(self, xd: int, yd: int, disp_w: int) -> tuple[float, float]:
        """Display click inside the magnifier -> image pixel (sub-pixel: 1 / zoom)."""
        x, y, _, _ = self.box(disp_w)
        cx, cy = round(self.centre[0]), round(self.centre[1])          # same rounding as render()
        return (cx - self.half + (xd - x + 0.5) / self.zoom - 0.5,
                cy - self.half + (yd - y + 0.5) / self.zoom - 0.5)

    def render(self, img: np.ndarray) -> np.ndarray:
        cx, cy = round(self.centre[0]), round(self.centre[1])
        h = self.half
        pad = cv2.copyMakeBorder(img, h, h, h, h, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        crop = pad[cy:cy + 2 * h + 1, cx:cx + 2 * h + 1]
        return cv2.resize(crop, None, fx=self.zoom, fy=self.zoom, interpolation=cv2.INTER_NEAREST)


def display_to_image(xd: int, yd: int, scale: float) -> tuple[float, float]:
    """Display pixel -> image coordinates (pixel centres at integers)."""
    return (xd + 0.5) / scale - 0.5, (yd + 0.5) / scale - 0.5


def _draw(img, st: AnnotationState, scale: float, mag: Magnifier, msg: str, title: str) -> np.ndarray:
    base = img.copy()
    cols = [(0, 255, 0), (255, 0, 255)]

    def poly(pts, col):
        for p in pts:
            cv2.circle(base, (int(round(p[0])), int(round(p[1]))), 2, col, -1, cv2.LINE_AA)
        for p, q in zip(pts[:-1], pts[1:], strict=False):
            cv2.line(base, (int(round(p[0])), int(round(p[1]))), (int(round(q[0])), int(round(q[1]))), col, 1,
                     cv2.LINE_AA)

    for g in st.instruments:
        poly(g.edge_a, (0, 200, 255))
        poly(g.edge_b, (0, 200, 255))
    for k, e in enumerate(st.current):
        poly(e, cols[k])
    poly(st.points, cols[len(st.current)])
    disp = cv2.resize(base, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    zoomed = mag.render(base)
    x, y, w, h = mag.box(disp.shape[1])
    if x > 0 and y + h < disp.shape[0]:
        disp[y:y + h, x:x + w] = zoomed
        cv2.rectangle(disp, (x, y), (x + w, y + h), (255, 255, 255), 1)
        c = x + w // 2, y + h // 2
        cv2.line(disp, (c[0] - 6, c[1]), (c[0] + 6, c[1]), (0, 0, 255), 1)
        cv2.line(disp, (c[0], c[1] - 6), (c[0], c[1] + 6), (0, 0, 255), 1)
    edge = "A" if not st.current else "B"
    d = f"{st.diameter:g} mm" if st.diameter else "diameter ?"
    lines = [f"{title} | instrument {len(st.instruments) + 1}, edge {edge}, {len(st.points)} pts | {d}",
             "SPACE close edge  u undo  r reset  5/8/0 diameter  ENTER save+next  x skip  b back  q quit",
             msg]
    for k, t in enumerate(lines):
        cv2.putText(disp, t, (10, 22 + 20 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return disp


def annotate(bench: str | Path, display_scale: float = 1.0, default_diameter: float | None = None,
             only_todo: bool = True) -> None:  # pragma: no cover - interactive
    bench = Path(bench)
    frames = load_manifest(bench)["frames"]
    ann = load_annotations(bench)
    order = [f for f in frames if not only_todo or ann.get(f["id"], FrameAnnotation()).status == "todo"]
    if not order:
        print("nothing to annotate (use --all to revisit annotated frames)")
        return
    win = "lapscale segbench annotate"
    require_gui()
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    mag = Magnifier()
    k = 0
    while 0 <= k < len(order):
        f = order[k]
        img = cv2.imread(str(bench / f["file"]))
        prev = ann.get(f["id"])
        st = AnnotationState(default_diameter)
        if prev and prev.status == "ok":
            st.instruments = list(prev.instruments)
        msg = ""

        def on_mouse(ev, xd, yd, *_, img=img, st=st):
            nonlocal msg
            disp_w = int(img.shape[1] * display_scale)
            if ev == cv2.EVENT_MOUSEMOVE and not mag.inside(xd, yd, disp_w):
                mag.centre = display_to_image(xd, yd, display_scale)
            elif ev == cv2.EVENT_LBUTTONDOWN:
                x, y = (mag.to_image(xd, yd, disp_w) if mag.inside(xd, yd, disp_w)
                        else display_to_image(xd, yd, display_scale))
                st.click(x, y)
                msg = f"point ({x:.2f}, {y:.2f})"

        cv2.setMouseCallback(win, on_mouse)
        while True:
            cv2.imshow(win, _draw(img, st, display_scale, mag, msg, f"{k + 1}/{len(order)} {f['id']}"))
            key = cv2.waitKey(20) & 0xFF
            if key == 32:
                msg = st.close_edge()
            elif key == ord("u"):
                st.undo()
            elif key == ord("r"):
                st.reset()
            elif key in DIAMETER_KEYS:
                st.diameter = DIAMETER_KEYS[key]
            elif key in (10, 13):
                if not st.complete:
                    msg = "finish the instrument in progress (SPACE) or undo it"
                    continue
                ann[f["id"]] = st.to_annotation()
                save_annotations(bench, ann)
                k += 1
                break
            elif key == ord("x"):
                ann[f["id"]] = FrameAnnotation("skip", [])
                save_annotations(bench, ann)
                k += 1
                break
            elif key == ord("b"):
                k = max(0, k - 1)
                break
            elif key == ord("q"):
                if st.complete and st.instruments:
                    ann[f["id"]] = st.to_annotation()
                save_annotations(bench, ann)
                cv2.destroyAllWindows()
                return
    cv2.destroyAllWindows()
    print(f"saved {bench / 'annotations.json'}")
