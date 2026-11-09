# Roadmap

1. ~~Skeleton, camera model, stage 1 (calibration / import / MoGe fallback), synthetic tests~~
2. `tool/geometry.py`: cylinder from two edges + known diameter (port of `phase0_check.shaft_diameter`,
   inverted), Monte Carlo uncertainty, synthetic tests. No network needed.
3. Stage 2 `depth`: run cache, MoGe-2 (any OS) and MoGe-3 (Linux/WSL2, Mufasa job); undistortion and
   remapped optical-disc mask before inference; `lapscale bench` for timings on the target GPU.
4. Stage 3 `tool`: SegBackend + SAM 3 zero-shot; edges (RANSAC) and tip; quality gate; edge-accuracy
   benchmark on ~30-50 hand-annotated frames -> choose / fine-tune the segmentation network.
5. Stage 4 `scale`: scale-only and affine fits, gating, temporal smoothing; `run --mode offline`; `viz`.
6. `eval` on the colleague's stereo data (left video as monocular input, stereo as ground truth): errors
   per distance band from the instrument; MoGe-3 vs MoGe-2.
7. `run --mode stream`: segmentation every frame, depth asynchronous; measured latencies.
