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

## Segmentation benchmark (`<bench>/`)
* `manifest.json`: `frames[]` with `id`, `video`, `index`, `file`, `size`.
* `annotations.json`: `frames.<id>.status` (`todo` / `ok` / `skip`), `frames.<id>.instruments[]` with
  `edge_a`, `edge_b` (lists of [x, y] raw pixels, >= 2 each), `diameter_mm`, `note`.
* `results/<name>/per_instrument.csv`: `frame`, `instrument`, `found`, `reason`, `n_instances`,
  `latency_ms`, `match_cost_px`, `bias_px`, `offset_a_px`, `offset_b_px`, `offset_sd_px`,
  `max_abs_offset_px`, `angle_err_deg`, `width_gt_px`, `width_err_px`, `width_err_pct`, `depth_err_pct`,
  `depth_err_pct_bias_corrected`. Offsets in undistorted pixels, positive outward.
* `results/<name>/summary.json`: `found_rate`, `depth_err_pct_median_abs`, `depth_err_pct_p90_abs`,
  `depth_err_pct_median_signed`, `bias_px_median`, `bias_px_spread_mad`, `offset_sd_px_median`,
  `angle_err_deg_median`, `depth_err_pct_bias_corrected_*`, `latency_ms_median`, `not_found_reasons`.
* `results/<name>/overlays/<frame>.png`; `results/comparison.md`.

## Depth cache (`runs/<video>/depth/<tag>/`)
* `meta.json`: `backend`, `video`, `size`, `camera` (the camera file content), `K_view` (intrinsics of the
  undistorted view; null without calibration), `fov_x_deg`, `undistorted`, `config`.
* `frames.csv`: `frame`, `latency_ms`, `fx_px`, `fy_px` (estimated focal, empty when given),
  `valid_fraction`, `depth_median`.
* `<frame:06d>.npz`: `depth` (float16, network units, 0 where invalid), `valid` (bool), `K` (3x3 of the map).

## Tool cache (`runs/<video>/tool/<tag>/`)
* `frames.csv`: one row per instance: `status` (ok / rejected / none), `reason`, `depth_mid_mm`, `width_px`,
  `rel_sd_random_pct`, `rel_sd_total_pct`, `bias_sensitivity_pct_per_px`, `n_samples`, `latency_ms`.
* `<frame>.npz`: per valid instrument k: `rays_k` (normalised), `z_k` (mm), `edge_a_k`, `edge_b_k` (raw px),
  `info_k` = [depth_mid_mm, rel_sd_total_pct].

## Scale (`runs/<video>/scale/<depth tag>__<tool tag>/<variant>/`)
* `frames.csv`: `a_raw`, `b_raw` (per-frame fit), `rel_residual_pct`, `span_ratio`, `fit_ok`, `reason`, `a`, `b`
  (smoothed / held), `status` (fit / held / none), `native_ratio` (a / the network's claimed mm per unit).
* `summary.json`: fitted / scaled frames, median a, frame-to-frame spread of a, native scale ratio.
* Metric depth of a frame: `scale.run.metric_depth(run_dir, depth_tag, tool_tag, variant, frame)`.

## Stream (`runs/<video>/stream_log.csv`)
`frame`, `seg_ms`, `loop_ms`, `depth_frame` (frame of the depth map in use), `n_shafts`, `a`, `b`, `status`.

## Depth benchmark (`<dbench>/`)
* `images/`, `gt/<id>.npz` (`depth` mm, `valid`), `camera.yaml`, `manifest.json` (source, per-frame GT
  valid fraction and median depth).
* `results/<name>/per_frame.csv` (one row per frame and alignment), `summary.json` (medians per alignment),
  `pred/<id>.npz`; `results/comparison.md`.
* `eval/<name>/per_frame.csv` and `summary.json`: references `instrument`, `native`, `oracle`; errors per GT
  distance band and per depth distance from the shaft.

## Segmentation masks (`<bench>/masks_gt/`)
`<frame>_<k>.png` (0/255 per instance), `index.json` (frame -> ok / empty). `segbench run` adds
`per_frame_masks.csv` (IoU, boundary error) when masks exist. `export-yolo` writes `images/`, `labels/`,
`data.yaml`.
