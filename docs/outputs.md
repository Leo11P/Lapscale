# Outputs

## Camera YAML (`lapscale.camera/1`)
| Key | Content |
|---|---|
| `source` | `charuco`, `checkerboard`, `import`, `manual`, `moge` |
| `image_size` | `[width, height]` the intrinsics refer to |
| `K` | 3x3 intrinsic matrix, pixels of the full frame |
| `dist` | OpenCV `[k1, k2, p1, p2, k3(, k4, k5, k6)]`, or `[]` |
| `fov_x_deg` | horizontal FOV of the pinhole model (informative; what MoGe receives) |
| `optical_circle` | `{cx, cy, r}` of the visible disc (distorted frame), or `null` = whole frame valid |
| `quality` | see below |
| `notes` | target, input file, import keys, rescaling |

`quality` after `calib board`: `reprojection_rms_px`, `holdout_rms_px`, `n_views_train`,
`n_views_holdout`, `n_views_rejected`, `n_corners`, `fx_std_px`, `fy_std_px`, `focal_std_pct`,
`focal_std_note`, `tilt_deg_median`, `tilt_deg_max`, `dist_model`, `coverage` (`grid_cells_covered`,
`outer_ring_corner_fraction`, `max_radius_fraction`, `grid`), `warnings`.

## Calibration report (`<camera>_report/`)
* `calib_report.json`: the camera, frames used for training / held out / rejected, per-view RMS,
  optical-disc detection diagnostics, target and configuration.
* `coverage.png`: all corners on one frame (green training, orange held-out, red rejected) and the disc.
* `undistort_check.png`: original | undistorted with a straight grid.
