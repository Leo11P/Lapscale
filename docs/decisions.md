# Decisions

Each entry: what, why, what would change it.

## Repository
* **Stages are separate commands sharing files on disk** (camera YAML, per-run cache), plus `run` to chain
  them. A stage can be rerun alone (new segmentation model, same depth) and inspected in isolation.
* **Backends behind small protocols** (`DepthBackend`, later `SegBackend`): MoGe v2/v3 and segmentation
  models are chosen by benchmark, not hard-wired.
* **Heavy dependencies are optional and imported lazily** (`.[moge]`): calibration and tests need only
  numpy, OpenCV and PyYAML, and run on Windows.
* **Synthetic ground truth in the tests**: rendered targets seen through known intrinsics; the same idea
  will test the cylinder geometry (as `phase0_check selftest` did).

## Stage 1: camera
* **One camera YAML (`lapscale.camera/1`) is the only contract with the rest.** Calibration, import and
  hand-written files all produce it; downstream code never knows which.
* **Pixel coordinates refer to the full recorded frame.** Intrinsics computed on a crop are moved back with
  `--crop-offset` at import. A video with a different aspect ratio is refused (a resize is handled, a crop
  is not silently guessed).
* **ChArUco by default**: partial views are usable, so corners reach the rim of the optical disc.
* **Pinhole + OpenCV radial-tangential model** (`k1k2` / `standard` / `rational`). Fisheye model not
  implemented: add it only if `undistort_check.png` shows the standard models failing at the rim.
* **View selection**: sharpness filter, de-duplication by corner motion (consecutive video frames are
  nearly identical and would overweight one pose), even spread over time, max 60 views.
* **Held-out views (20%)** evaluated with the calibrated intrinsics: the training RMS alone does not show
  overfitting of the distortion model.
* **Focal-length std reported as a percentage = scale uncertainty**, labelled as a lower bound (OpenCV
  covariance; synthetic tests show the true error up to ~2x it).
* **Optical disc** detected from the brightness envelope of many frames, circle fitted only on boundary
  points not on the image border (the disc is usually clipped top and bottom). Stored in the distorted
  frame; stages working on undistorted images must remap it.
* **MoGe fallback** kept but marked: no distortion, principal point at the centre, frame-to-frame spread
  of f reported as a lower bound of the scale error.
