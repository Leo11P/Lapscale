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

## Stage 3: shaft geometry
* **Plane formulation of the cylinder** (equivalent to the Pluecker form of Wei et al. / Doignon & de Mathelin):
  each edge back-projects to a plane through the camera centre; the axis direction is the cross product of
  the two plane normals; the axis lies at distance r from both planes inside the wedge. Closed form, no
  optimisation, exact on noise-free data (tests: relative error < 1e-6).
* **Edges are fitted as straight lines AFTER undistortion** (points undistorted first, then total least
  squares). In the raw image the edges are curves. Undistortion uses 50 iterations: OpenCV's default leaves
  ~1e-5 normalised error with strong barrel distortion.
* **The quantity used downstream is the depth of the VISIBLE shaft** (surface depth along the rays of shaft
  pixels; summary value `depth_mid` at the middle of the edges), not the distance to the infinite axis.
  Measured on synthetic shafts tilted > 45 deg towards the camera: the axis distance is > 3x less stable,
  because its closest point is an extrapolation outside the visible segment. The paper uses the axis
  pose; we keep it only as a diagnostic.
* **Per-pixel depth by ray-cylinder intersection** (front surface) over the whole shaft mask, instead of
  sampling only along the projected axis as in the paper: more points for the scale fit, same model.
* **Uncertainty split into random and systematic parts.** Random edge noise averages out over frames and
  contour points; an edge bias of the segmentation (mask too wide/narrow) and the focal-length error do
  not. Reported separately, plus the deterministic sensitivity to a 1 px outward shift of both edges
  (~ -2/width: -2% for a 100 px wide shaft, -7% for a 28 px wide one). This sensitivity is the
  acceptance criterion for the segmentation network.

## Stage 3: segmentation and edges
* **Benchmark on shaft edges, not masks.** Ground truth = points clicked on the two edges of each measurable
  shaft (faster than painting masks, and exactly what the scale depends on). Metrics in undistorted pixels:
  outward offset of the predicted edges along the annotated ones, width error, and the depth error through
  the full cylinder geometry (independent of the diameter).
* **Edges from soft masks with sub-pixel refinement** (0.5 crossing along the local normal). Measured on
  rendered instruments: offset ~0.01 px on soft masks, ~0.04 px on binary masks; without it OpenCV
  contours bias each edge inward by 0.5 px.
* **RANSAC in undistorted space, edges as contiguous runs**: jaws, tip and the rim of the field drop out
  without part labels; an edge must enclose the instrument and run side by side with the other.
* **Bias vs spread reported separately**, with a leave-one-out bias-corrected depth error: a stable bias
  of a model can be calibrated, a spread cannot.
* **SAM 3 zero-shot first** (no training data needed); `masks` backend so any external model can be
  benchmarked; `yolo` wrapper for a fine-tuned YOLO-seg. A SurgeNetXL-based network is implemented only
  if the benchmark shows that zero-shot is not enough.

## Stage 2: depth
* **Frames are undistorted before the network**, into a pinhole view of the same size that keeps the
  calibrated K (same resolution at the centre). Its FOV is passed to MoGe (`fov_x`), so the network does
  not estimate it. The optical-disc mask is undistorted with the same maps; pixels outside it are black.
* **The depth map is addressed by normalised rays**, the same coordinates as the shaft geometry
  (`DepthFrame.sample`). Bilinear sampling, NaN if any neighbour is invalid: no blending across the
  instrument boundary or the rim. Tested end to end with a fake backend on a tilted plane: rays from raw
  pixels recover the plane depth within 0.2% (float16 storage + interpolation).
* **Cache per frame** (`runs/<video>/depth/<tag>/`, float16 compressed, ~0.05% relative precision),
  resumable; a different configuration under the same tag is refused instead of mixing results.
* **No calibration (`--camera moge`)**: no undistortion, focal length estimated per frame by the backend
  and stored with each map (principal point at the centre).
* The network output is treated as being in unknown units even when MoGe claims metres: the scale comes
  from the instrument (stage 4). MoGe's own metric scale is kept as a baseline to compare against.
