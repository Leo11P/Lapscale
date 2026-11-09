# lapscale

Metric depth for **monocular laparoscopy**, image only: the instrument shaft (a cylinder of known diameter) is
the ruler that turns a monocular depth network (MoGe, Depth Anything 3, ...) into millimetres. Method after Wei
et al., *Enhanced Scale-aware Depth Estimation for Monocular Endoscopic Scenes with Geometric Modeling*
(MonoEndoDepth), with a modern depth network, per-pixel shaft depth, explicit uncertainty and benchmarks for
every choice.

| Part | Command | |
|---|---|---|
| 1 camera intrinsics | `lapscale calib board / import / moge / mask / show / target` | ChArUco or checkerboard; import = bypass |
| 2 depth | `lapscale depth` | backends `moge` (v2, v3), `da3`, `folder` |
| 3 instrument | `lapscale tool` | backends `sam3`, `yolo`, `surgenet`, `masks`; sub-pixel shaft edges; cylinder geometry |
| 4 scale | `lapscale scale` | scale-only and affine, gating, temporal smoothing |
| whole video | `lapscale run --mode offline / stream`, `lapscale viz` | cached stages / real time |
| segmentation benchmark | `lapscale segbench sample / annotate / annotate-masks / run / compare / export-yolo` | by shaft-edge accuracy |
| segmentation training | `lapscale segtrain surgenet / yolo` | only if zero-shot is not enough |
| depth benchmark | `lapscale depthbench prepare-stereo / prepare-folder / run / compare` | stereo ground truth |
| evaluation | `lapscale eval` | whole pipeline against ground truth |
| backends on this machine | `lapscale check` | loads each network, runs it, reports OK / FAIL |

Everything is verified on synthetic data with exact ground truth (`pytest`); the real networks run only where
they are installed. Start with `docs/install.md`, then follow `docs/verification.md`.

## Quick start

```bash
uv venv && uv pip install -e ".[dev]" && pytest

# camera (bypass with known intrinsics, or calibrate: docs/calibration.md)
lapscale calib import --file stereo_calib.xml --side left --image-size 1920x1080 --video surgery.mp4 \
                      --out configs/camera/lap.yaml

# whole video, offline, with overlay
lapscale run --video surgery.mp4 --camera configs/camera/lap.yaml --diameter-mm 5 \
             --seg-backend sam3 --depth-backend moge --moge-version v3 --stride 5 --overlay

# real time
lapscale run --mode stream --video surgery.mp4 --camera configs/camera/lap.yaml --diameter-mm 5 \
             --seg-backend sam3 --depth-backend moge --moge-version v2 --depth-checkpoint Ruicheng/moge-2-vits-normal \
             --out-video stream.mp4
```

Outputs go to `runs/<video>/` (`docs/outputs.md`). Choosing the networks: `docs/segbench.md` (segmentation)
and `docs/depthbench.md` (depth). How the stages connect: `docs/architecture.md`.

## Layout

```
configs/            default.yaml (per-stage defaults), instruments.yaml (shaft diameters), camera/
src/lapscale/
  camera.py         Camera model + YAML I/O (the contract between stage 1 and the rest)
  io/               frame sources (video / image folder), per-run caches (depth, tool)
  calib/            targets, optical disc, calibration, import, MoGe fallback, report
  depth/            DepthBackend protocol, MoGe, DA3, folder; undistorted view; per-video run
  tool/             segmentation backends, mask -> shaft edges, cylinder geometry, uncertainty, per-video run
  scale/            per-frame fit, gating, smoothing; metric depth of a frame
  pipeline/         offline and real-time runners
  viz/              overlays
  segbench/         segmentation benchmark (edges and masks), YOLO export
  segtrain/         SurgeNetXL and YOLO fine-tuning
  depthbench/       stereo ground truth, depth metrics, comparison
  eval/             whole pipeline against ground truth
  check.py          backends on this machine
tests/              synthetic scenes with exact ground truth (calibration, shafts, laparoscopic video, stereo)
docs/               install, verification, architecture, calibration, segbench, depthbench, decisions,
                    outputs, roadmap, experiment log
```
