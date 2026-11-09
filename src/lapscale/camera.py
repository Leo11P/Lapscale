"""Camera model shared by every stage.

A camera file (YAML, schema ``lapscale.camera/1``) is the only thing the downstream stages need
from calibration. It can come from three places, in decreasing order of trust:

* ``checkerboard`` / ``charuco``: ``lapscale calib board`` on images of a calibration target;
* ``import``: intrinsics you already have (OpenCV XML/YAML, e.g. one side of a stereo calibration),
  converted with ``lapscale calib import``, or a camera YAML written by hand (the bypass);
* ``moge``: focal length estimated by MoGe from the surgical frames, no distortion. Last resort:
  in monocular metric reconstruction an error on f is an error on the scale, one to one.

Conventions: pixel coordinates are those of the full video frame (no crop), (0, 0) at the centre of
the top-left pixel, OpenCV distortion model ``[k1, k2, p1, p2, k3(, k4, k5, k6)]``.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

SCHEMA = "lapscale.camera/1"
SOURCES = ("checkerboard", "charuco", "import", "manual", "moge")


@dataclass
class OpticalCircle:
    """Visible disc of the laparoscope optics, in pixels. Outside it the image is black."""

    cx: float
    cy: float
    r: float

    def mask(self, size: tuple[int, int], margin_px: float = 0.0) -> np.ndarray:
        w, h = size
        yy, xx = np.mgrid[0:h, 0:w]
        return ((xx - self.cx) ** 2 + (yy - self.cy) ** 2) <= (self.r - margin_px) ** 2


@dataclass
class Camera:
    K: np.ndarray                       # 3x3
    dist: np.ndarray                    # (n,) OpenCV pinhole distortion, n in {0, 4, 5, 8}
    image_size: tuple[int, int]         # (width, height) the intrinsics refer to
    source: str = "manual"
    optical_circle: OpticalCircle | None = None
    quality: dict = field(default_factory=dict)
    notes: str = ""
    created: str = ""

    def __post_init__(self):
        self.K = np.asarray(self.K, dtype=np.float64).reshape(3, 3)
        self.dist = np.asarray(self.dist, dtype=np.float64).ravel()
        self.image_size = (int(self.image_size[0]), int(self.image_size[1]))
        if self.dist.size not in (0, 4, 5, 8):
            raise ValueError(f"distortion must have 0, 4, 5 or 8 coefficients, got {self.dist.size}")
        if self.source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}, got {self.source!r}")
        if not self.created:
            self.created = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # ------------------------------------------------------------------ derived quantities
    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K[1, 1])

    @property
    def cx(self) -> float:
        return float(self.K[0, 2])

    @property
    def cy(self) -> float:
        return float(self.K[1, 2])

    def fov_x_deg(self) -> float:
        """Horizontal field of view of the pinhole model over the full image width (what MoGe expects).
        Image edges are at x = -0.5 and x = w - 0.5 (pixel-centre convention)."""
        w = self.image_size[0]
        return math.degrees(math.atan2(w - 0.5 - self.cx, self.fx) + math.atan2(self.cx + 0.5, self.fx))

    def has_distortion(self) -> bool:
        return bool(self.dist.size and np.any(np.abs(self.dist) > 0))

    # ------------------------------------------------------------------ resolution handling
    def for_size(self, size: tuple[int, int], tol: float = 1e-3) -> Camera:
        """Camera for a video of another resolution with the SAME framing (a pure resize).

        A different aspect ratio means a crop or a different capture setting, which cannot be
        handled by scaling: that is refused instead of silently producing wrong intrinsics.
        """
        w0, h0 = self.image_size
        w1, h1 = int(size[0]), int(size[1])
        if (w0, h0) == (w1, h1):
            return self
        sx, sy = w1 / w0, h1 / h0
        if abs(sx - sy) > tol * max(sx, sy):
            raise ValueError(
                f"camera calibrated at {w0}x{h0}, video is {w1}x{h1}: different aspect ratio "
                "(crop?). Calibrate at this resolution or import with the right crop offset."
            )
        # pixel centres: x_new + 0.5 = s (x_old + 0.5)
        K = self.K.copy()
        K[0, 0] *= sx
        K[1, 1] *= sy
        K[0, 2] = sx * (K[0, 2] + 0.5) - 0.5
        K[1, 2] = sy * (K[1, 2] + 0.5) - 0.5
        circ = None
        if self.optical_circle is not None:
            c = self.optical_circle
            circ = OpticalCircle(sx * (c.cx + 0.5) - 0.5, sy * (c.cy + 0.5) - 0.5, c.r * sx)
        return Camera(K, self.dist.copy(), (w1, h1), self.source, circ, dict(self.quality),
                      (self.notes + f" | rescaled from {w0}x{h0}").strip(" |"), self.created)

    # ------------------------------------------------------------------ undistortion
    def undistort_maps(self, alpha: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Maps to an undistorted pinhole image of the same size, and its new K.

        Downstream geometry (MoGe, the cylinder model) assumes a pinhole camera, so frames are
        undistorted once, here, with this new K.
        """
        newK, _ = cv2.getOptimalNewCameraMatrix(self.K, self.dist, self.image_size, alpha, self.image_size)
        m1, m2 = cv2.initUndistortRectifyMap(self.K, self.dist, None, newK, self.image_size, cv2.CV_32FC1)
        return m1, m2, newK

    def mask(self, margin_px: float = 0.0) -> np.ndarray:
        """Boolean mask of valid pixels (inside the optical circle, if known)."""
        w, h = self.image_size
        if self.optical_circle is None:
            return np.ones((h, w), bool)
        return self.optical_circle.mask(self.image_size, margin_px)

    # ------------------------------------------------------------------ I/O
    def to_dict(self) -> dict:
        d = dict(
            schema=SCHEMA,
            source=self.source,
            created=self.created,
            image_size=list(self.image_size),
            K=[[float(v) for v in row] for row in self.K],
            dist=[float(v) for v in self.dist],
            fov_x_deg=round(self.fov_x_deg(), 4),
            optical_circle=None if self.optical_circle is None
            else {k: round(float(v), 3) for k, v in asdict(self.optical_circle).items()},
            quality=_plain(self.quality),
            notes=self.notes,
        )
        return d

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True))
        return path

    @classmethod
    def from_dict(cls, d: dict) -> Camera:
        if d.get("schema") != SCHEMA:
            raise ValueError(f"not a {SCHEMA} file (schema={d.get('schema')!r})")
        circ = d.get("optical_circle")
        return cls(K=d["K"], dist=d.get("dist", []), image_size=tuple(d["image_size"]),
                   source=d.get("source", "manual"),
                   optical_circle=OpticalCircle(**circ) if circ else None,
                   quality=d.get("quality") or {}, notes=d.get("notes", ""), created=d.get("created", ""))

    @classmethod
    def load(cls, path: str | Path) -> Camera:
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))

    def summary(self) -> str:
        lines = [
            f"source      : {self.source}   ({self.created})",
            f"image size  : {self.image_size[0]} x {self.image_size[1]}",
            f"focal       : fx {self.fx:.2f}  fy {self.fy:.2f} px   (fov_x {self.fov_x_deg():.2f} deg)",
            f"principal   : cx {self.cx:.2f}  cy {self.cy:.2f} px",
            f"distortion  : {np.array2string(self.dist, precision=5) if self.dist.size else 'none'}",
        ]
        if self.optical_circle:
            c = self.optical_circle
            lines.append(f"optical disc: centre ({c.cx:.1f}, {c.cy:.1f}) r {c.r:.1f} px")
        for k, v in self.quality.items():
            if not isinstance(v, (dict, list)):
                lines.append(f"{k:12s}: {v}")
        if self.source == "moge":
            lines.append("WARNING: focal length estimated by MoGe, no distortion model; the metric scale "
                         "inherits its error one to one. Use only when no calibration is possible.")
        return "\n".join(lines)


def _plain(x):
    """numpy -> plain python, for YAML."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return _plain(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, float):
        return round(x, 6)
    return x


def resolve_camera(spec: str | Path, video_size: tuple[int, int] | None = None) -> Camera | None:
    """What every stage calls with its ``--camera`` argument.

    ``spec`` is a camera YAML path, or the literal ``moge`` meaning "no calibration: let the depth
    stage estimate the focal length" (returns None). With ``video_size`` the camera is adapted to the
    video resolution (pure resize only).
    """
    if str(spec).lower() == "moge":
        return None
    cam = Camera.load(spec)
    return cam.for_size(video_size) if video_size is not None else cam
