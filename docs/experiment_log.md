# Experiment log

| # | Date | Stage | Data | Config / change | Result | Notes |
|---|---|---|---|---|---|---|
| 1 | 2026-11-09 | calib | synthetic ChArUco, 960x540, f = 700 px | default | f error 0.07-0.27%, holdout RMS ~0.2 px | reported f std 0.14-0.23%: same order, can be optimistic |
| 2 | 2026-11-09 | tool/geometry | synthetic shafts, D 5 and 10 mm, 30-150 mm, tilt 0-75 deg, distorted camera | noise-free | depth, axis direction and axis position exact (rel. < 1e-6, < 1e-5 mm) | edges fitted after undistortion |
| 3 | 2026-11-09 | tool/geometry | same, 0.5 px noise, 60 points per edge, widths 23-113 px | - | depth error median 0.15%, p90 0.41% | depth of visible shaft; axis distance > 3x less stable when tilted > 45 deg |
| 4 | 2026-11-09 | tool/uncertainty | 80 shafts, 1 px noise, 20 points per edge | Monte Carlo n = 150 | coverage 72% at 1 sd, 91% at 2 sd (ideal 68/95); median rel. sd 0.96% | slightly heavy tails |
| 5 | 2026-11-09 | tool/uncertainty | one shaft, 28 px wide at 130 mm | +1 px outward edge shift | depth -7.2% | thin/distant shafts need near-pixel-exact edges |
