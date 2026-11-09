"""``lapscale check``: does each installed backend load its weights and produce a sane output on this
machine? Run it after installing / downloading weights; every check reports OK or the exact error."""
from __future__ import annotations

import platform
import time
import traceback

import numpy as np


def _test_image(path: str | None) -> np.ndarray:
    import cv2
    if path:
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"cannot read {path}")
        return img
    rng = np.random.default_rng(0)
    img = np.full((540, 960, 3), (90, 70, 160), np.uint8) + rng.integers(0, 30, (540, 960, 3), dtype=np.uint8)
    cv2.line(img, (100, 500), (600, 250), (190, 190, 195), 40)          # a "shaft"
    return img


def _env() -> list[tuple[str, bool, str]]:
    out = [("python", True, platform.python_version())]
    import cv2
    out.append(("opencv", True, cv2.__version__))
    try:
        import torch
        msg = f"{torch.__version__}, CUDA {'yes' if torch.cuda.is_available() else 'NO'}"
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            msg += f", {p.name}, {p.total_memory / 2**30:.1f} GB"
        out.append(("torch", True, msg))
    except ImportError as e:
        out.append(("torch", False, str(e)))
    return out


def _time_call(fn, n=3):
    fn()                                   # warm-up (compilation, caches)
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        r = fn()
        t.append(1000 * (time.perf_counter() - t0))
    return r, float(np.median(t))


def run_checks(image: str | None = None, device: str = "cuda", moge: list[str] | None = None,
               da3: list[str] | None = None, sam3: str | bool | None = None, yolo: str | None = None,
               surgenet_ckpt: str | None = None, surgenet_repo: str | None = None,
               surgenet_pretrained: str | None = None, log=print) -> list[tuple[str, bool, str]]:
    img = _test_image(image)
    h, w = img.shape[:2]
    results = _env()

    def attempt(label, fn):
        try:
            results.append((label, True, fn()))
        except Exception as e:                                   # report, never crash the whole check
            results.append((label, False, f"{type(e).__name__}: {e}"))
            log(traceback.format_exc(limit=2))

    for spec in moge or []:
        version, _, ckpt = spec.partition("=")
        def f(version=version, ckpt=ckpt or None):
            from .depth.moge import MoGeBackend
            b = MoGeBackend(version=version, checkpoint=ckpt, device=device)
            p, ms = _time_call(lambda: b.predict(img, fov_x_deg=70.0))
            assert p.depth.shape == (h, w), p.depth.shape
            v = p.depth[p.valid]
            return f"{b.name}: {ms:.0f} ms, valid {100 * p.valid.mean():.0f}%, depth {np.median(v):.3g} (median)"
        attempt(f"depth moge {version}", f)
    for ckpt in da3 or []:
        def f(ckpt=ckpt):
            from .depth.da3 import DA3Backend
            b = DA3Backend(checkpoint=ckpt, device=device)
            p, ms = _time_call(lambda: b.predict(img, fov_x_deg=70.0))
            assert p.depth.shape == (h, w)
            return f"{b.name}: {ms:.0f} ms, valid {100 * p.valid.mean():.0f}%, depth {np.median(p.depth[p.valid]):.3g}"
        attempt(f"depth da3 {ckpt.split('/')[-1]}", f)
    if sam3:
        def f():
            from .tool.backends.sam3 import Sam3Backend
            b = Sam3Backend(checkpoint=None if sam3 is True else sam3, device=device)
            inst, ms = _time_call(lambda: b.predict(img), n=2)
            return f"{b.name}: {ms:.0f} ms, {len(inst)} instance(s)"
        attempt("seg sam3", f)
    if yolo:
        def f():
            from .tool.backends.yolo import YoloBackend
            b = YoloBackend(weights=yolo, device=device)
            inst, ms = _time_call(lambda: b.predict(img))
            names = sorted({i.label for i in inst})
            return f"{b.name}: {ms:.0f} ms, {len(inst)} instance(s) {names} (COCO weights do not know instruments)"
        attempt("seg yolo", f)
    if surgenet_pretrained:
        def f():
            from .tool.backends.surgenet import build_model
            msgs = []
            build_model(surgenet_repo, 1, surgenet_pretrained, log=msgs.append)
            return msgs[-1] if msgs else "built"
        attempt("surgenet backbone", f)
    if surgenet_ckpt:
        def f():
            from .tool.backends.surgenet import SurgeNetBackend
            b = SurgeNetBackend(surgenet_ckpt, surgenet_repo, device)
            inst, ms = _time_call(lambda: b.predict(img))
            return f"{b.name}: {ms:.0f} ms, {len(inst)} instance(s)"
        attempt("seg surgenet", f)
    width = max(len(r[0]) for r in results)
    for name, ok, msg in results:
        log(f"{'OK  ' if ok else 'FAIL'} {name:{width}s}  {msg}")
    return results
