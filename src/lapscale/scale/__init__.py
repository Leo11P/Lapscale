"""Stage 4 (to be implemented): backend depth -> millimetres.

Per frame, on shaft pixels: scale-only fit (median ratio) and affine fit (a*d + b), both kept for
comparison; frames gated by the stage-3 uncertainty; temporal smoothing of the parameters.
"""
