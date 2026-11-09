"""Benchmark of monocular depth networks (MoGe, Depth Anything 3, precomputed maps) against depth ground truth,
typically from the colleague's calibrated stereo videos.

  lapscale depthbench prepare-stereo   rectified left images + SGBM ground truth
  lapscale depthbench prepare-folder   any images + depth maps
  lapscale depthbench run              one backend -> per-frame metrics (native / scale / affine alignment)
  lapscale depthbench compare          table of all backends
"""
from .data import frame_ids, load_gt, prepare_folder, prepare_stereo
from .metrics import ALIGNMENTS, align, band_metrics, depth_metrics
from .run import compare_depth, run_depthbench

__all__ = ["frame_ids", "load_gt", "prepare_folder", "prepare_stereo", "ALIGNMENTS", "align", "band_metrics",
           "depth_metrics", "compare_depth", "run_depthbench"]
