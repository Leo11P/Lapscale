"""Benchmark of instrument segmentation by SHAFT EDGE ACCURACY (and the depth error it causes).

  lapscale segbench sample    frames from videos
  lapscale segbench annotate  click the two edges of each measurable shaft
  lapscale segbench run       one backend -> per-instrument metrics, summary, overlays
  lapscale segbench compare   table of all backends
"""
from .data import FrameAnnotation, GTInstrument, load_annotations, sample_frames, save_annotations
from .metrics import EdgePairMetrics, compare_edges
from .run import compare, run_benchmark, summarise

__all__ = ["FrameAnnotation", "GTInstrument", "load_annotations", "save_annotations", "sample_frames",
           "EdgePairMetrics", "compare_edges", "run_benchmark", "summarise", "compare"]
