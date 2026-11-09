# How the stages connect

Per frame of a monocular laparoscopic video:

```
raw frame (distorted)
 |
 |-- stage 3 tool --------------------------------------------------------------
 |   segmentation backend  -> instrument instances (soft masks, raw frame)
 |   tool/edges.py         -> two shaft edges (sub-pixel, raw pixels) + shaft region
 |   tool/geometry.py      -> cylinder; shaft pixels -> normalised rays -> Z_mm   (diameter known)
 |
 |-- stage 2 depth ---------------------------------------------------------------
 |   depth/view.py         -> undistorted pinhole view, FOV from calibration, disc mask
 |   MoGe                  -> depth map in network units (cached per frame)
 |
 '-- bridge: DepthFrame.sample(normalised rays of the shaft pixels) -> d_net
     stage 4 scale (planned): fit Z_mm ~ a * d_net (+ b) on the shaft, gate by uncertainty,
     smooth over time -> metric depth of the whole frame = a * depth map (+ b)
```

The common currency is the **normalised undistorted ray** (x, y, 1): the geometry works on rays from raw
pixels (`tool.geometry.to_normalized`), the depth map is addressed by rays (`DepthFrame.sample`), so the
two never disagree about which pixel is which, whatever the distortion.

Still missing to run a whole video end to end (see `roadmap.md`):
* `scale/`: the fit and its gating / smoothing;
* `lapscale tool` over a video (segmentation + edges + geometry + shaft region, cached per frame);
* `lapscale run`: chains the stages offline (cache) and in real time (segmentation every frame, depth
  asynchronous); `eval` against stereo ground truth; `viz` overlays.
