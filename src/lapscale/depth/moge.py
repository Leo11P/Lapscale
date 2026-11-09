"""MoGe backend (microsoft/MoGe, MIT). Versions: v2 (any platform with torch) and v3 (needs
FlexGEMM/Triton: Linux, possibly WSL2).

Checkpoints (Hugging Face):
  v3: Ruicheng/moge-3-vitl (370M), Ruicheng/moge-3-vitg (1.25B)           -> offline, best quality
  v2: Ruicheng/moge-2-vits-normal (35M), -vitb-normal (104M), -vitl (326M) -> real time
Imported lazily: the rest of lapscale does not need torch.
"""
from __future__ import annotations

import numpy as np

from .base import DepthPrediction

DEFAULT_CHECKPOINT = {"v2": "Ruicheng/moge-2-vitl", "v3": "Ruicheng/moge-3-vitl"}


class MoGeBackend:
    def __init__(self, version: str = "v3", checkpoint: str | None = None, device: str = "cuda",
                 fp16: bool = True, resolution_level: int = 9, num_tokens: int | None = None,
                 refine_steps: int = 3):
        if version not in DEFAULT_CHECKPOINT:
            raise ValueError("version must be v2 or v3")
        try:
            import torch
            if version == "v3":
                from moge.model.v3 import MoGeModel
            else:
                from moge.model.v2 import MoGeModel
        except ImportError as e:  # pragma: no cover - depends on the environment
            hint = " (v3 also needs Triton/FlexGEMM: Linux or WSL2)" if version == "v3" else ""
            raise ImportError(f"MoGe {version} not available ({e}). Install with: uv sync --extra moge{hint}") from e
        self.torch = torch
        self.version = version
        self.checkpoint = checkpoint or DEFAULT_CHECKPOINT[version]
        self.name = f"moge-{version}:{self.checkpoint}"
        self.device = torch.device(device)
        self.model = MoGeModel.from_pretrained(self.checkpoint).to(self.device).eval()
        self.kw = dict(resolution_level=resolution_level, num_tokens=num_tokens, use_fp16=fp16)
        if version == "v3":
            self.kw["refine_steps"] = refine_steps

    def predict(self, image_bgr: np.ndarray, fov_x_deg: float | None = None) -> DepthPrediction:
        torch = self.torch
        h, w = image_bgr.shape[:2]
        rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
        x = torch.tensor(rgb / 255.0, dtype=torch.float32, device=self.device).permute(2, 0, 1)
        with torch.inference_mode():
            out = self.model.infer(x, fov_x=fov_x_deg, **self.kw)
        depth = out["depth"].float().cpu().numpy().astype(np.float32)
        valid = out["mask"].cpu().numpy().astype(bool) if "mask" in out else np.isfinite(depth)
        valid &= np.isfinite(depth) & (depth > 0)
        Kn = out["intrinsics"].float().cpu().numpy()      # normalised: fx/W, fy/H
        fx, fy = (None, None) if fov_x_deg is not None else (float(Kn[0, 0] * w), float(Kn[1, 1] * h))
        return DepthPrediction(np.nan_to_num(depth, nan=0.0), valid, fx, fy, metric=True)
