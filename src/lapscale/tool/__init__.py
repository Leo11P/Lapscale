"""Stage 3 (to be implemented): instrument -> metric geometry, image only.

Planned modules:
  base.py      SegBackend protocol: frame -> instrument mask(s)
  backends/    sam3, semantic net with surgical backbone, yolo-seg (chosen by an edge-accuracy benchmark)
  edges.py     mask contour -> two straight shaft edges (RANSAC), tip, quality gate
  geometry.py  two edges + known diameter + K -> axis (Pluecker), distance, metric depth of shaft pixels
               (port of phase0_check.shaft_diameter, inverted: diameter known -> range)
  uncertainty.py  Monte Carlo on the edges -> distance / scale uncertainty
"""
