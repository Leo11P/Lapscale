"""Fine-tune an Ultralytics YOLO segmentation model on the dataset written by ``segbench export-yolo``."""
from __future__ import annotations

from pathlib import Path


def train_yolo(data_yaml: str | Path, model: str = "yolo26n-seg.pt", epochs: int = 100, imgsz: int = 960,
               batch: int = 8, device: str | None = None, project: str | Path = "runs/segtrain",
               name: str = "yolo", log=print) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError as e:  # pragma: no cover
        raise ImportError(f"ultralytics not available ({e}): pip install ultralytics") from e
    m = YOLO(model)
    kw = dict(data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch, project=str(project), name=name,
              exist_ok=True, retina_masks=True)
    if device:
        kw["device"] = device
    res = m.train(**kw)
    best = Path(res.save_dir) / "weights" / "best.pt"
    log(f"YOLO weights: {best}")
    return best
