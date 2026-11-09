# Roadmap

Done (verified on synthetic data, see experiment_log.md):
1. Skeleton, camera model, stage 1 `calib` (ChArUco / checkerboard, import, MoGe fallback, optical disc).
2. Stage 3 geometry: cylinder from two edges + diameter, per-pixel depth, Monte Carlo uncertainty.
3. Stage 3 segmentation: backends sam3 / yolo / surgenet / masks; mask -> sub-pixel shaft edges;
   `segbench` (sample, annotate edges, annotate masks, run, compare, export-yolo); `segtrain` (SurgeNetXL, YOLO).
4. Stage 2 depth: backends moge (v2/v3) / da3 / folder; undistorted view; per-frame cache; ray sampling.
5. `depthbench`: stereo ground truth (SGBM, left-right check), metrics under native / scale / affine alignment.
6. Stage 4 scale (scale-only and affine, gating, smoothing); `lapscale tool | scale | run (offline, stream) |
   viz | eval`; `lapscale check`.

To do (needs real data / GPU, see verification.md):
7. `lapscale check` with all weights; timings on the target GPU.
8. Depth benchmark on the colleague's stereo videos -> choice of the depth network.
9. Segmentation benchmark on 30-50 annotated IEO frames -> choice (or fine-tuning) of the segmentation.
10. Whole pipeline on the stereo data (`eval`) and on the IEO monocular videos; real-time profiling.

Open questions: GPU; diameters of the IEO instruments; calibration of the IEO laparoscope; whether the
affine variant is ever needed (decided by `eval`).
