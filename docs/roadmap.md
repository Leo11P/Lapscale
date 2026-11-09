# Roadmap

1. ~~Skeleton, camera model, stage 1 (calibration / import / MoGe fallback), synthetic tests~~
2. ~~`tool/geometry.py` + `tool/uncertainty.py`: cylinder from two edges + known diameter, per-pixel depth,
   Monte Carlo uncertainty (random / systematic), synthetic tests~~
3. Stage 2 `depth`: run cache, MoGe-2 (any OS) and MoGe-3 (Linux/WSL2, Mufasa job); undistortion and
   remapped optical-disc mask before inference; `lapscale bench` for timings on the target GPU.
4. Stage 3 segmentation:
   * ~~SegBackend protocol; backends sam3 / yolo / masks; mask -> shaft edges (sub-pixel, RANSAC, gate);
     `segbench` (sample, annotate, run, compare) with synthetic tests~~
   * annotate 30-50 IEO frames; run SAM 3 (several prompts); decide with `docs/segbench.md`
   * only if needed: fine-tuning (SurgeNetXL backbone or YOLO-seg), training masks pre-annotated with SAM 3
5. Stage 4 `scale`: scale-only and affine fits, gating, temporal smoothing; `run --mode offline`; `viz`.
6. `eval` on the colleague's stereo data (left video as monocular input, stereo as ground truth): errors
   per distance band from the instrument; MoGe-3 vs MoGe-2.
7. `run --mode stream`: segmentation every frame, depth asynchronous; measured latencies.
