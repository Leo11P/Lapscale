"""Stage 3: instrument -> metric geometry, image only.

Done:
  geometry.py     two straight edges + known diameter + intrinsics -> shaft axis, metric depth of shaft
                  pixels, axis projection, axis point under a pixel (tip)
  uncertainty.py  Monte Carlo on the depth of the visible shaft (random edge noise, systematic edge bias,
                  focal-length error)

Planned:
  base.py         SegBackend protocol: frame -> instrument mask(s)
  backends/       sam3, semantic net with surgical backbone, yolo-seg (chosen by an edge-accuracy benchmark)
  edges.py        mask contour -> two straight shaft edges (RANSAC), tip, quality gate
"""
from .geometry import Cylinder, LineFit, ShaftEstimate, cylinder_from_planes, fit_line, shaft_from_edges, to_normalized
from .uncertainty import ShaftUncertainty, shaft_uncertainty, shift_edges

__all__ = ["Cylinder", "LineFit", "ShaftEstimate", "cylinder_from_planes", "fit_line", "shaft_from_edges",
           "to_normalized", "ShaftUncertainty", "shaft_uncertainty", "shift_edges"]
