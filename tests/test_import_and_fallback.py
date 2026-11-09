import cv2
import numpy as np
import pytest

from lapscale.calib.importers import import_opencv
from lapscale.calib.moge_fov import estimate_focal
from lapscale.depth.base import DepthPrediction
from lapscale.io.video import FrameSource


def _stereo_xml(path):
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    fs.write("M_l", np.array([[1103.4, 0, 601.7], [0, 1101.5, 545.8], [0, 0, 1]]))
    fs.write("D_l", np.array([[-0.2, 0.1, 0, 0, 0]]))
    fs.write("M_r", np.array([[1102.8, 0, 698.5], [0, 1100.5, 544.7], [0, 0, 1]]))
    fs.write("D_r", np.array([[-0.2, 0.1, 0, 0, 0]]))
    fs.release()


def test_import_stereo_side_with_crop(tmp_path):
    p = tmp_path / "stereo.xml"
    _stereo_xml(p)
    cam = import_opencv(p, side="left", image_size=(1920, 1080), crop_offset=(320, 0))
    assert cam.cx == pytest.approx(921.7) and cam.fx == pytest.approx(1103.4)
    assert cam.source == "import" and cam.dist.size == 5
    right = import_opencv(p, side="right", image_size=(1920, 1080), crop_offset=(320, 0))
    assert right.cx == pytest.approx(1018.5)


def test_import_needs_size(tmp_path):
    p = tmp_path / "stereo.xml"
    _stereo_xml(p)
    with pytest.raises(ValueError, match="image size"):
        import_opencv(p, side="left")


def test_import_warns_on_missing_crop(tmp_path):
    p = tmp_path / "stereo.xml"
    _stereo_xml(p)
    cam = import_opencv(p, side="left", image_size=(1920, 1080))     # crop offset forgotten
    assert any("crop offset" in w for w in cam.quality.get("warnings", []))


class FakeBackend:
    name = "fake"

    def __init__(self, f):
        self.f, self.k = f, 0

    def predict(self, img, fov_x_deg=None):
        self.k += 1
        f = self.f * (1 + 0.01 * ((self.k % 3) - 1))      # +-1% frame to frame
        h, w = img.shape[:2]
        return DepthPrediction(np.ones((h, w), np.float32), np.ones((h, w), bool), f, f)


def test_moge_fallback_with_fake_backend(tmp_path):
    folder = tmp_path / "f"
    folder.mkdir()
    for k in range(9):
        cv2.imwrite(str(folder / f"{k}.png"), np.full((540, 960, 3), 100, np.uint8))
    src = FrameSource(folder)
    cam = estimate_focal(src, FakeBackend(800.0), src.sample_indices(n=9), log=lambda *_: None)
    assert cam.source == "moge" and cam.fx == pytest.approx(800, rel=0.011)
    assert cam.dist.size == 0 and cam.quality["focal_spread_pct"] > 0
    assert any("one to one" in w for w in cam.quality["warnings"])
