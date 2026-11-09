# Roadmap

1. ~~Skeleton, camera model, stage 1 (calibration / import / MoGe fallback), synthetic tests~~
2. ~~`tool/geometry.py` + `tool/uncertainty.py`: cylinder from two edges + known diameter, per-pixel depth,
   Monte Carlo uncertainty (random / systematic), synthetic tests~~
3. Stage 2 `depth`:
   * ~~`lapscale depth`: undistorted pinhole view with the calibrated FOV, disc mask, MoGe v2/v3 backend,
     per-frame cache (resumable, timed), ray sampling `DepthFrame.sample` (the bridge to the geometry)~~
   * run MoGe-2 and MoGe-3 on the IEO videos (MoGe-3 on Linux/WSL2 or Mufasa); timings on the target GPU
4. Stage 3 segmentation:
   * ~~SegBackend protocol; backends sam3 / yolo / masks; mask -> shaft edges (sub-pixel, RANSAC, gate);
     `segbench` (sample, annotate, run, compare) with synthetic tests~~
   * annotate 30-50 IEO frames; run SAM 3 (several prompts); decide with `docs/segbench.md`
   * only if needed: fine-tuning (SurgeNetXL backbone or YOLO-seg), training masks pre-annotated with SAM 3
5. Stage 4 `scale`: scale-only and affine fits on the shaft region, gating by the stage-3 uncertainty,
   temporal smoothing. `lapscale tool` over a video (cached). `run --mode offline`; `viz`.
   This is what joins everything: see `architecture.md`.
6. `eval` on the colleague's stereo data (left video as monocular input, stereo as ground truth): errors
   per distance band from the instrument; MoGe-3 vs MoGe-2.
7. `run --mode stream`: segmentation every frame, depth asynchronous; measured latencies.
