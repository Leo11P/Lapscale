"""Fine-tune SurgeNetXL (CaFormer-S18 + FPN) for binary instrument segmentation.

Loss: BCE + soft Dice. Validation by video (no near-duplicate frames on both sides), model selection on
the validation IoU. The checkpoint stores the state dict and the input size; load it with the
``surgenet`` segmentation backend.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..tool.backends.surgenet import build_model, to_tensor
from .data import augment, collect, load_pair, split_by_video


def train_surgenet(benches: list[str | Path], out: str | Path, repo: str | Path | None, pretrained: str | Path | None,
                   epochs: int = 30, batch: int = 8, size_hw: tuple[int, int] = (480, 640), lr: float = 1e-4,
                   val_fraction: float = 0.2, device: str = "cuda", workers_seed: int = 0, log=print) -> Path:
    import torch
    import torch.nn.functional as F
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    items = collect(benches)
    tr, va = split_by_video(items, val_fraction, workers_seed)
    log(f"SurgeNet fine-tuning: {len(tr)} train / {len(va)} val frames, input {size_hw[1]}x{size_hw[0]}, device {dev}")
    data_tr = [load_pair(i) for i in tr]
    data_va = [load_pair(i) for i in va]
    model = build_model(repo, 1, pretrained, log=log).to(dev)
    dec = [p for n, p in model.named_parameters() if not n.startswith("metaformer.")]
    enc = [p for n, p in model.named_parameters() if n.startswith("metaformer.")]
    opt = torch.optim.AdamW([{"params": enc, "lr": lr}, {"params": dec, "lr": 10 * lr}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    rng = np.random.default_rng(workers_seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    best, history = -1.0, []
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        perm = rng.permutation(len(data_tr))
        losses = []
        for s in range(0, len(perm), batch):
            xs, ys = [], []
            for k in perm[s:s + batch]:
                img, m = augment(*data_tr[k], rng, size_hw)
                xs.append(to_tensor(img, size_hw))
                ys.append(torch.from_numpy(m.astype(np.float32))[None])
            x, y = torch.stack(xs).to(dev), torch.stack(ys).to(dev)
            logits = model(x)
            if logits.shape[-2:] != y.shape[-2:]:
                logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
            p = torch.sigmoid(logits)
            dice = 1 - (2 * (p * y).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)
            loss = F.binary_cross_entropy_with_logits(logits, y) + dice.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        sched.step()
        iou = _validate(model, data_va, size_hw, dev)
        history.append(dict(epoch=ep, loss=float(np.mean(losses)), val_iou=iou, seconds=round(time.time() - t0, 1)))
        log(f"  epoch {ep}/{epochs}: loss {np.mean(losses):.4f}, val IoU {iou:.4f}")
        if iou > best:
            best = iou
            torch.save(dict(state_dict=model.state_dict(), input_size_hw=list(size_hw), num_classes=1,
                            arch="surgenet_caformer_s18_fpn", pretrained=str(pretrained), val_iou=iou, epoch=ep),
                       out / "best.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2))
    log(f"best val IoU {best:.4f} -> {out / 'best.pt'}")
    return out / "best.pt"


def _validate(model, data, size_hw, dev) -> float:
    import torch
    import torch.nn.functional as F
    if not data:
        return float("nan")
    model.eval()
    inter = union = 0
    with torch.inference_mode():
        for img, m in data:
            x = to_tensor(img, size_hw)[None].to(dev)
            logits = model(x)
            logits = F.interpolate(logits, size=m.shape, mode="bilinear", align_corners=False)
            p = (torch.sigmoid(logits)[0, 0] > 0.5).cpu().numpy()
            inter += np.logical_and(p, m).sum()
            union += np.logical_or(p, m).sum()
    return float(inter / union) if union else 1.0
