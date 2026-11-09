"""The annotation tools need an OpenCV build with a GUI backend; the cluster uses the headless build."""
from __future__ import annotations

import cv2


def has_gui() -> bool:
    info = cv2.getBuildInformation()
    gui = [ln for ln in info.splitlines() if ln.strip().startswith("GUI:")]
    return not gui or "NONE" not in gui[0].upper()


def require_gui() -> None:
    if not has_gui():
        raise RuntimeError("this OpenCV build has no GUI (opencv-python-headless). Annotation is interactive: run "
                           "it on your PC, after `uv pip uninstall opencv-python-headless && uv pip install "
                           "opencv-python`; copy the benchmark folder to/from the cluster")
