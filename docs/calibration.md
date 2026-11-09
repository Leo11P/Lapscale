# Calibrating the laparoscope

The focal length sets the metric scale of everything downstream: **1% error on f = 1% error on every
depth**. Distortion matters as much near the edge of the field, where instruments often are.

## What must match the surgery

Intrinsics belong to the *optical chain*, not to the camera alone. Calibrate with:

* the same **scope** (diameter, 0° or 30°) and the same **camera head**;
* the same **zoom** and **focus** setting as in surgery (camera-head zoom changes f directly; focus
  changes it slightly). If they vary during surgery, note it: one calibration per setting;
* the same **video output resolution** (the one you record);
* for a 30° scope, the scope **rotated as usual** relative to the camera head: rotating it moves the
  image on the sensor (principal point, optical disc).

Re-calibrate after any change of scope, camera head or video settings.

## Target

* **ChArUco** (recommended): partial views count, so the board can reach the rim of the optical disc
  where distortion is strongest. `lapscale calib target --target charuco ...` renders it.
* Print on matte paper, glue to a **rigid flat** support (glass, aluminium, foam board).
* **Measure** a square (and a marker) with a calliper over several squares; use the measured values.
  A 2% printing scale error is a 2% scale error in every depth.
* Small squares (3–5 mm): working distances in the abdomen are a few centimetres.

## Recording

* 30–60 s of video, slow movements (blur is filtered but wastes views).
* Distances similar to the surgical working range (about 3–15 cm).
* **Tilt** the board 20–45° in different directions: fronto-parallel views do not determine f.
* **Cover the whole disc**, including its rim. `coverage.png` shows what you covered.
* Even, diffuse light; avoid specular glare on the board.

## Reading the result

| Field | Good | Meaning if bad |
|---|---|---|
| `reprojection_rms_px` | < 0.5 px | blur, wrong square size, model too simple |
| `holdout_rms_px` | close to the training RMS | overfitting: fewer coefficients (`--dist-model k1k2`) or more varied views |
| `focal_std_pct` | < 0.3 % | scale uncertainty from calibration (a lower bound) |
| `tilt_deg_median` | > 15° | f poorly determined |
| `coverage.grid_cells_covered` | > 0.7 | distortion extrapolated where there are no corners |
| `coverage.outer_ring_corner_fraction` | > 0.05 | rim distortion is a guess |

Then look at `undistort_check.png`: straight edges must be straight on the right.
