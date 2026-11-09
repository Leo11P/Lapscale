# Experiment log

| # | Date | Stage | Data | Config / change | Result | Notes |
|---|---|---|---|---|---|---|
| 1 | 2026-11-09 | calib | synthetic ChArUco, 960x540, f = 700 px | default | f error 0.07-0.27%, holdout RMS ~0.2 px | reported f std 0.14-0.23%: same order, can be optimistic |
| 2 | 2026-11-09 | tool/geometry | synthetic shafts, D 5 and 10 mm, 30-150 mm, tilt 0-75 deg, distorted camera | noise-free | depth, axis direction and axis position exact (rel. < 1e-6, < 1e-5 mm) | edges fitted after undistortion |
| 3 | 2026-11-09 | tool/geometry | same, 0.5 px noise, 60 points per edge, widths 23-113 px | - | depth error median 0.15%, p90 0.41% | depth of visible shaft; axis distance > 3x less stable when tilted > 45 deg |
| 4 | 2026-11-09 | tool/uncertainty | 80 shafts, 1 px noise, 20 points per edge | Monte Carlo n = 150 | coverage 72% at 1 sd, 91% at 2 sd (ideal 68/95); median rel. sd 0.96% | slightly heavy tails |
| 5 | 2026-11-09 | tool/uncertainty | one shaft, 28 px wide at 130 mm | +1 px outward edge shift | depth -7.2% | thin/distant shafts need near-pixel-exact edges |
| 6 | 2026-11-09 | tool/edges | rendered instruments (shaft + jaw), distorted frame, widths 34-132 px | soft / binary / dilated 1 px | edge offset 0.01 / 0.04 / 1.2-1.4 px; depth error <= 0.17% for soft and binary | the 1 px dilation is measured, not absorbed |
| 7 | 2026-11-09 | segbench | 5 synthetic frames, annotation noise 0.2 px, 4 clicks per edge | oracle masks | found 5/5, depth error median 0.17%, p90 0.53%, bias +0.01 px | floor set by the annotation noise |
| 8 | 2026-11-09 | segbench | same | masks dilated 1 px | found 4/4 predicted, bias +1.38 px, depth error -6.1%; bias-corrected 0.45% | stable bias is correctable |
| 9 | 2026-11-09 | depth | fake backend = exact depth of a tilted plane in the undistorted view | ray sampling from raw pixels | plane depth recovered within 0.2% | checks undistortion, FOV, cache and the bridge to the geometry; MoGe itself not run here |
