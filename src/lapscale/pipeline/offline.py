"""``lapscale run --mode offline``: depth, tool and scale on the same frames, each stage cached."""
from __future__ import annotations

from pathlib import Path

from ..camera import Camera
from ..depth.base import DepthBackend
from ..depth.run import run_depth
from ..scale.fit import VARIANTS, ScaleConfig
from ..scale.run import run_scale
from ..tool.base import SegBackend
from ..tool.edges import EdgeConfig
from ..tool.run import ToolConfig, run_tool


def run_offline(video: str | Path, camera: Camera, seg: SegBackend, depth: DepthBackend, run_dir: str | Path,
                indices: list[int] | None = None, tool_cfg: ToolConfig | None = None,
                edge_cfg: EdgeConfig | None = None, scale_cfg: ScaleConfig | None = None,
                depth_tag: str | None = None, tool_tag: str | None = None, depth_config: dict | None = None,
                log=print) -> dict:
    d = run_depth(video, camera, depth, run_dir, depth_tag, indices, config=depth_config, log=log)
    t = run_tool(video, camera, seg, run_dir, tool_cfg, edge_cfg, tool_tag, indices, log=log)
    s = run_scale(run_dir, d["tag"], t["tag"], scale_cfg, VARIANTS, log=log)
    return dict(depth=d, tool=t, scale=s, run_dir=str(run_dir))
