"""Instrument segmentation with a SurgeNetXL-pretrained CaFormer-S18 + FPN (Jaspers et al.), fine-tuned
with ``lapscale segtrain surgenet``.

The architecture code is the authors' (Apache 2.0): clone https://github.com/TimJaspers0801/SurgeNet and
pass the folder as ``repo`` (or put it on PYTHONPATH). Needs torch and timm (the authors pin timm 1.0.13).
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from ..base import InstrumentInstance, split_instances

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def import_metaformer(repo: str | Path | None):
    if repo is not None:
        sys.path.insert(0, str(Path(repo).resolve()))
    try:
        import metaformer  # noqa: F401  (the authors' module)
    except ImportError as e:
        raise ImportError(f"SurgeNet code not found ({e}): clone github.com/TimJaspers0801/SurgeNet and pass "
                          "--surgenet-repo (needs torch and timm)") from e
    return sys.modules["metaformer"]


def build_model(repo, num_classes: int = 1, pretrained: str | Path | None = None, log=print):
    """MetaFormerFPN with the SurgeNet variant of CaFormer-S18; ``pretrained``: LOCAL teacher checkpoint
    (e.g. SurgeNetXL_checkpoint_epoch0050_teacher.pth). Missing / unexpected keys are reported, not hidden."""
    import torch
    mf = import_metaformer(repo)
    model = mf.MetaFormerFPN(num_classes=num_classes, pretrained="SurgeNet", pretrained_weights=None)
    if pretrained is not None:
        sd = torch.load(str(pretrained), map_location="cpu", weights_only=False)
        for key in ("teacher", "student", "state_dict", "model"):
            if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
                sd = sd[key]
                break
        clean = {}
        for k, v in sd.items():
            for pre in ("module.", "backbone."):
                if k.startswith(pre):
                    k = k[len(pre):]
            if not k.startswith("head."):
                clean[k] = v
        msg = model.metaformer.load_state_dict(clean, strict=False)
        n_model = len([k for k in model.metaformer.state_dict() if not k.startswith("head.")])
        missing = [k for k in msg.missing_keys if not k.startswith("head.")]
        log(f"SurgeNet backbone: loaded {n_model - len(missing)}/{n_model} tensors from {Path(pretrained).name}, "
            f"{len(msg.unexpected_keys)} unexpected")
        if len(missing) > 0.5 * n_model:
            raise RuntimeError(f"most backbone weights missing ({len(missing)}/{n_model}): wrong checkpoint?")
    return model


def to_tensor(img_bgr: np.ndarray, size_hw: tuple[int, int]):
    import torch
    x = cv2.resize(img_bgr, (size_hw[1], size_hw[0]), interpolation=cv2.INTER_AREA)[:, :, ::-1].astype(np.float32) / 255
    x = (x - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


class SurgeNetBackend:
    def __init__(self, checkpoint: str | Path, repo: str | Path | None = None, device: str = "cuda",
                 confidence: float = 0.5):
        import torch
        self.torch = torch
        ck = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        self.size = tuple(ck["input_size_hw"])
        self.model = build_model(repo, ck.get("num_classes", 1), None, log=lambda *_: None)
        self.model.load_state_dict(ck["state_dict"])
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.model.to(self.device).eval()
        self.threshold = confidence
        self.name = f"surgenet:{Path(checkpoint).stem}"

    def predict(self, frame_bgr: np.ndarray) -> list[InstrumentInstance]:
        h, w = frame_bgr.shape[:2]
        with self.torch.inference_mode():
            x = to_tensor(frame_bgr, self.size)[None].to(self.device)
            p = self.torch.sigmoid(self.model(x))[0, 0].float().cpu().numpy()
        prob = cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)
        if self.threshold != 0.5:                 # move the operating point onto the 0.5 level used by edges
            prob = np.clip(0.5 + (prob - self.threshold), 0, 1)
        return split_instances(prob.astype(np.float32))
