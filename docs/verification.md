# Verification plan (on your machine)

Everything below was verified here on synthetic data with exact ground truth (see `experiment_log.md`). What
could NOT be run here is every real network (no GPU, no torch, no weights) and the two GUIs. This is the order
to verify them, from the cheapest to the most informative.

1. **Tests**: `pytest` (synthetic, ~2-3 min). Must be all green before anything else.
2. **Backends**: `lapscale check ...` (`install.md`). Every line OK; note the latencies on your GPU.
3. **Camera**: `lapscale calib import` (or `calib board` with a ChArUco video), then look at
   `undistort_check.png` and the warnings. `lapscale calib show`.
4. **Stereo data, depth**: `depthbench prepare-stereo` on the colleague's videos; open a few
   `dbench/images` and check the ground truth is plausible (`valid_fraction`, median depth ~100 mm).
   Then `depthbench run` for each depth network and `depthbench compare`.
5. **Stereo data, whole pipeline**: `lapscale run` on `dbench/images` (diameter 8 mm, da Vinci), then
   `lapscale eval`. The instrument-scaled AbsRel should approach the `oracle` line; compare with `native`.
6. **Segmentation**: `segbench sample` on the IEO laparoscopic videos, `segbench annotate` (edges, 30-50
   frames); optionally `segbench annotate-masks --seg-backend sam3` (SAM 3 proposals corrected by hand).
   `segbench run` for SAM 3 (several prompts) and the others; `segbench compare`; decide with `segbench.md`.
7. **Monocular IEO videos**: `lapscale run --mode offline --overlay` with the chosen networks; look at the
   overlay video and at `scale/*/summary.json` (fitted frames, frame-to-frame spread of the scale).
8. **Real time**: `lapscale run --mode stream --out-video ...`; `stream_log.csv` has the per-frame timings.

If a step fails, the error message of `check` or the stage's `reason` columns usually say why; the stage
caches make it cheap to rerun only what changed.
