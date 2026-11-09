"""Per-run cache shared by the stages (to be implemented with stage 2).

Layout of a run directory (``runs/<video_stem>/``)::

    camera.yaml                 copy of the camera used (reproducibility)
    depth/<backend>/            one .npz per frame: depth (float16), valid
    tool/<backend>/frames.csv   one row per frame: edges, axis, distance, uncertainty, reject reason
    tool/<backend>/masks/       instrument masks (png)
    scale/<variant>/frames.csv  per-frame scale parameters, residuals, smoothing state
    manifest.json               stage -> config hash, inputs, code version

A stage recomputes only if its config hash or an upstream output changed.
"""
