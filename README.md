# lapscale

Metric depth for **monocular laparoscopy**, image only: the instrument shaft (a cylinder of known
diameter) is the ruler that turns a monocular depth network (MoGe) into millimetres.
Method after Wei et al., *Enhanced Scale-aware Depth Estimation for Monocular Endoscopic Scenes with
Geometric Modeling* (MonoEndoDepth), with MoGe in place of Monodepth2.

| Stage | Command | Status |
|---|---|---|
| 1 camera intrinsics | `lapscale calib …` | **done** |
| 2 depth (MoGe) | `lapscale depth` | backend wrapper only |
| 3 instrument geometry | `lapscale tool` | geometry + uncertainty done (Python API); segmentation planned |
| 4 scale recovery | `lapscale scale` | planned |
| all, offline / real time | `lapscale run` | planned |
| evaluation vs stereo GT | `lapscale eval` | planned |

## Install

```bash
uv venv && uv pip install -e ".[dev]"          # calibration + tests (numpy, opencv, pyyaml)
uv pip install -e ".[moge]"                     # + MoGe (torch). MoGe-3 needs Linux or WSL2 (Triton)
pytest                                          # synthetic tests, no data needed
```

## Stage 1: camera intrinsics

In monocular metric reconstruction **an error on the focal length is the same error on the scale**,
so this stage reports the focal-length uncertainty as a percentage, i.e. directly as a scale error.

**A. Calibrate** (recommended). Print a ChArUco board, *measure* a square with a calliper, record a
video of it with the laparoscope (same optics, camera head and zoom as in surgery; tilt the board
20–45°, cover the whole field including the edge of the optical disc):

```bash
lapscale calib target --target charuco --cols 11 --rows 8 --square-mm 5 --marker-mm 3.7 --out board.png
lapscale calib board  --input calib_video.mp4 --target charuco --cols 11 --rows 8 \
                      --square-mm 4.98 --marker-mm 3.68 --out configs/camera/lap_ieo.yaml
```

Output: `lap_ieo.yaml` plus `lap_ieo_report/` with `calib_report.json`, `coverage.png` (all corners
on one frame: green train, orange held-out, red rejected) and `undistort_check.png` (straight edges
must look straight on the right). Read the warnings: they say what to fix.

**B. Bypass** — you already have intrinsics:

```bash
# from an OpenCV file (e.g. one side of a stereo calibration, intrinsics computed on a crop)
lapscale calib import --file stereo_calib.xml --side left --image-size 1280x960 --crop-offset 320,0 \
                      --video surgery.mp4 --out configs/camera/x.yaml
# or write a camera YAML by hand: see configs/camera/example_manual.yaml, then
lapscale calib mask --camera configs/camera/x.yaml --video surgery.mp4    # adds the optical disc
```

**C. Last resort** — no calibration possible: focal length from MoGe, no distortion model.

```bash
lapscale calib moge --video surgery.mp4 --out configs/camera/x_moge.yaml
```

Every later stage takes `--camera PATH` or `--camera moge`.

## Stage 3 (geometry part): metric depth of the shaft

```python
from lapscale.camera import Camera
from lapscale.tool import shaft_from_edges, shaft_uncertainty

cam = Camera.load("configs/camera/lap_ieo.yaml")
est = shaft_from_edges(edge_a_px, edge_b_px, cam, diameter_mm=5.0)   # points (>= 2 each), raw pixels
est.depth_mid                    # mm, shaft surface at the middle of the edges
est.depth_at(xy_normalised)      # mm, any shaft pixels (NaN where the ray misses)
unc = shaft_uncertainty(edge_a_px, edge_b_px, cam, 5.0, point_sd_px=0.5, bias_sd_px=1.0,
                        focal_rel_sd=0.003)
unc.rel_sd_total_pct, unc.bias_sensitivity_pct_per_px
```

## Layout

```
configs/            default.yaml (per-stage defaults), instruments.yaml (shaft diameters), camera/
src/lapscale/
  camera.py         Camera model + YAML I/O (the contract between stage 1 and the rest)
  io/               frame sources (video / image folder), run cache
  calib/            targets, optical disc, calibration, import, MoGe fallback, report
  depth/            DepthBackend protocol, MoGe v2/v3
  tool/             shaft geometry (two edges + diameter -> metric depth) and its uncertainty
  scale/ pipeline/ eval/ viz/     next stages
tests/              synthetic tests (rendered boards with known intrinsics)
docs/               decisions, outputs, roadmap, experiment log
scripts/slurm/      cluster jobs (MoGe-3 offline)
```
