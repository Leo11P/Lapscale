"""How the depth network sees a frame.

MoGe (like every monocular geometry network) assumes a pinhole camera, so the raw frame is undistorted
first. The undistorted image keeps the size of the frame and, by default, the calibrated K (same
resolution at the centre). Its focal length gives the horizontal FOV passed to MoGe, so the network does
not have to guess it. The optical-disc mask is undistorted with the same maps.

Coordinates: everything downstream addresses the depth map through NORMALISED undistorted coordinates
(x, y, 1), the same ones used by the shaft geometry (``tool.geometry.to_normalized``). ``K_view`` maps them
to pixels of the depth map.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..camera import Camera


class PinholeView:
    def __init__(self, camera: Camera | None, size: tuple[int, int], alpha: float | None = None):
        self.size = (int(size[0]), int(size[1]))
        w, h = self.size
        if camera is None:                      # no calibration: the backend estimates the focal length
            self.camera, self.maps, self.K_view = None, None, None
            self.mask = np.ones((h, w), bool)
            return
        cam = camera.for_size(self.size)
        self.camera = cam
        if cam.has_distortion():
            m1, m2, newK = cam.undistort_maps(alpha)
            self.maps = (m1, m2)
            self.K_view = newK
            self.mask = cv2.remap(cam.mask().astype(np.uint8), m1, m2, cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        else:
            self.maps, self.K_view, self.mask = None, cam.K.copy(), cam.mask()

    @property
    def fov_x_deg(self) -> float | None:
        if self.K_view is None:
            return None
        return Camera(self.K_view, [], self.size).fov_x_deg()

    def apply(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Undistorted frame, black outside the optical disc."""
        img = frame_bgr if self.maps is None else cv2.remap(frame_bgr, *self.maps, cv2.INTER_LINEAR,
                                                             borderMode=cv2.BORDER_CONSTANT)
        return img * self.mask[..., None].astype(img.dtype)
