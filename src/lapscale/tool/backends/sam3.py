"""SAM 3 (facebookresearch/sam3, SAM License), zero-shot with text prompts.

Install separately (needs Python >= 3.12, PyTorch >= 2.7, CUDA >= 12.6, and approved access to the
gated checkpoints on Hugging Face: facebook/sam3, or the newer facebook/sam3.1):
    git clone https://github.com/facebookresearch/sam3 && pip install -e sam3

Note for edge accuracy: SAM 3 predicts masks at low resolution and upsamples them bilinearly to the
frame (see Sam3Processor._forward_grounding). The soft upsampled probability is kept here so that the
edge extractor can locate the 0.5 crossing; whether that is accurate enough is what the benchmark
measures.
"""
from __future__ import annotations

import numpy as np

from ..base import InstrumentInstance

DEFAULT_PROMPTS = ("surgical instrument",)


class Sam3Backend:
    def __init__(self, prompts: tuple[str, ...] | list[str] | str = DEFAULT_PROMPTS, confidence: float = 0.5,
                 checkpoint: str | None = None, device: str = "cuda", resolution: int = 1008,
                 dedup_iou: float = 0.7):
        try:
            import torch
            from PIL import Image
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise ImportError(f"SAM 3 not available ({e}); see the module docstring for installation") from e
        self.torch, self.Image = torch, Image
        self.prompts = [prompts] if isinstance(prompts, str) else list(prompts)
        model = build_sam3_image_model(device=device, checkpoint_path=checkpoint, load_from_HF=checkpoint is None)
        self.proc = Sam3Processor(model, resolution=resolution, device=device, confidence_threshold=confidence)
        self.dedup_iou = dedup_iou
        self.name = "sam3:" + "|".join(p.replace(" ", "_") for p in self.prompts)

    def predict(self, frame_bgr: np.ndarray) -> list[InstrumentInstance]:  # pragma: no cover - needs SAM 3
        img = self.Image.fromarray(np.ascontiguousarray(frame_bgr[:, :, ::-1]))
        out: list[InstrumentInstance] = []
        with self.torch.inference_mode():
            state = self.proc.set_image(img)
            for prompt in self.prompts:
                self.proc.reset_all_prompts(state)
                st = self.proc.set_text_prompt(prompt=prompt, state=state)
                probs = st["masks_logits"].float().cpu().numpy()[:, 0]      # already sigmoid, frame size
                scores = st["scores"].float().cpu().numpy()
                for p, s in zip(probs, scores, strict=True):
                    out.append(InstrumentInstance(p.astype(np.float32), float(s), prompt))
        return dedup(out, self.dedup_iou)


def dedup(instances: list[InstrumentInstance], iou: float) -> list[InstrumentInstance]:
    """Drop instances that overlap a higher-scoring one (several prompts can find the same tool)."""
    keep: list[InstrumentInstance] = []
    for inst in sorted(instances, key=lambda i: -i.score):
        m = inst.prob > 0.5
        if all(_iou(m, k.prob > 0.5) < iou for k in keep):
            keep.append(inst)
    return keep


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0
