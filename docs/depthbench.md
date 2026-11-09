# Depth benchmark

Which monocular depth network (MoGe-2, MoGe-3, Depth Anything 3 variants, any precomputed model) gives the
best depth SHAPE on laparoscopic images, and how wrong is its own metric scale?

## Ground truth

From the colleague's calibrated stereo videos (the same validated with `phase0_check`):

```bash
lapscale depthbench prepare-stereo --left VideoSX.mp4 --right VideoDX.mp4 \
    --calib out/stereo_calib_par_30_refined.xml --crop 0,960,320,1600 --calib-frame full \
    --offset 0 --n 50 --bench dbench/
```

The benchmark images are the RECTIFIED LEFT frames: an exact pinhole camera (`dbench/camera.yaml`, K = P1, no
distortion), so the networks see a normal camera and the ground truth is on the same pixel grid. Depth =
f B / disparity from SGBM, with a left-right consistency check, specular (saturated) pixels removed and a depth
range. On a synthetic pair the ground truth is within 0.1% (median) of the exact depth.

Any other source (simulation, a public dataset): `depthbench prepare-folder --images --gt --camera`.

## Run and compare

```bash
lapscale depthbench run --bench dbench/ --depth-backend moge --moge-version v3
lapscale depthbench run --bench dbench/ --depth-backend moge --moge-version v2 --depth-checkpoint Ruicheng/moge-2-vitl
lapscale depthbench run --bench dbench/ --depth-backend da3 --depth-checkpoint depth-anything/DA3MONO-LARGE
lapscale depthbench run --bench dbench/ --depth-backend da3 --depth-checkpoint depth-anything/DA3METRIC-LARGE
lapscale depthbench compare --bench dbench/
```

Metrics (mm): AbsRel, SqRel, RMSE, MAE, RMSE log, delta < 1.05 and < 1.25, per ground-truth distance band, under
three alignments:

* `scale`: one scale per frame. The SHAPE quality, i.e. the best the instrument-based scale could reach.
  This is the main criterion for choosing the network.
* `affine`: scale + shift per frame (usual protocol for affine-invariant models).
* `native`: the network's own metric output (MoGe-2/3, DA3METRIC, DA3NESTED). `median_ratio` = GT / prediction:
  how far its metric scale is off, and its frame-to-frame spread.

## The whole pipeline

The image folder of a depth benchmark is also a valid "video":

```bash
lapscale run --video dbench/images --camera dbench/camera.yaml --diameter-mm 8 \
    --seg-backend sam3 --depth-backend moge --moge-version v3 --run-dir runs/dbench
lapscale eval --bench dbench/ --run-dir runs/dbench --depth-tag <depth tag> --tool-tag <tool tag>
```

`eval` reports the depth error with the instrument-derived scale, next to the network's native scale and a
per-frame ground-truth scale (the upper bound), and the error as a function of the depth distance from the
instrument, where the scale is extrapolated.
