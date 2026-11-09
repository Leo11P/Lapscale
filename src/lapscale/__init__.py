"""lapscale: metric depth for monocular laparoscopy.

Stages (each runnable alone, all chained by ``lapscale run``):
  1 calib  camera intrinsics (calibration target, import = bypass, MoGe fallback)
  2 depth  per-frame depth with MoGe (backend units)
  3 tool   instrument segmentation -> shaft edges -> cylinder geometry -> metric depth on the shaft
  4 scale  fit backend depth to the shaft's metric depth -> metric depth of the whole frame
"""
__version__ = "0.1.0"
