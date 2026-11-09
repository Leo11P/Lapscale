# How the stages connect

Per frame of a monocular laparoscopic video:

```
raw frame (distorted)
 |
 |-- stage 3 tool ---------------------------------------------------------------- lapscale tool
 |   segmentation backend  -> instrument instances (soft masks, raw frame)        sam3 | yolo | surgenet | masks
 |   tool/edges.py         -> two shaft edges (sub-pixel, raw pixels) + shaft region
 |   tool/geometry.py      -> cylinder; shaft pixels -> normalised rays -> Z_mm    (diameter known)
 |   tool/uncertainty.py   -> uncertainty of the shaft depth (gate for stage 4)
 |
 |-- stage 2 depth --------------------------------------------------------------- lapscale depth
 |   depth/view.py         -> undistorted pinhole view, FOV from calibration, disc mask
 |   depth backend         -> depth map in network units (cached per frame)        moge | da3 | folder
 |
 '-- stage 4 scale ----------------------------------------------------------------- lapscale scale
     DepthFrame.sample(rays of the shaft pixels) -> d_net
     fit Z_mm ~ a d_net (+ b) per frame, gate, smooth over time -> metric depth = a * map (+ b)

lapscale run --mode offline   = depth + tool + scale (cached), --overlay -> lapscale viz
lapscale run --mode stream    = tool every frame, depth asynchronous on the latest frame, causal smoothing
```

The common currency is the **normalised undistorted ray** (x, y, 1): the geometry works on rays from raw
pixels (`tool.geometry.to_normalized`), the depth map is addressed by rays (`DepthFrame.sample`), so the two
never disagree about which pixel is which, whatever the distortion.

Benchmarks and evaluation around it: `segbench` (segmentation, shaft edges), `depthbench` (depth networks
against stereo ground truth), `eval` (whole pipeline against ground truth), `check` (backends on this machine).
