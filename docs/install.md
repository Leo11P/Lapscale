# Installation and first check

lapscale itself needs only numpy, OpenCV and PyYAML. Each network is an optional backend with its own
requirements; install what you use, then run `lapscale check` to see what actually works on this machine.

```bash
uv venv && uv pip install -e ".[dev]"
pytest                                   # synthetic tests, no network, ~2-3 min
```

| Backend | Install | Weights | Notes |
|---|---|---|---|
| MoGe v2 / v3 | `uv pip install -e ".[moge]"` (or `git clone https://github.com/microsoft/MoGe && uv pip install -e MoGe`) | `Ruicheng/moge-2-vitl`, `Ruicheng/moge-3-vitl`, `-vitg` (HF, downloaded on first use) or a local path | v3 needs Triton / FlexGEMM: Linux or WSL2 |
| Depth Anything 3 | `git clone https://github.com/ByteDance-Seed/Depth-Anything-3 && uv pip install -e Depth-Anything-3` | `depth-anything/DA3MONO-LARGE`, `DA3METRIC-LARGE`, `DA3NESTED-GIANT-LARGE-1.1`, ... | licences differ per checkpoint (Apache 2.0 vs CC BY-NC 4.0) |
| SAM 3 | `git clone https://github.com/facebookresearch/sam3 && uv pip install -e sam3` | gated: request access on HF (`facebook/sam3`, `facebook/sam3.1`), `huggingface-cli login` | Python >= 3.12, PyTorch >= 2.7, CUDA >= 12.6 |
| YOLO-seg | `uv pip install -e ".[yolo]"` | `yolo26n-seg.pt` ... (COCO: does NOT know instruments) | fine-tune: `segbench export-yolo` + `segtrain yolo` |
| SurgeNetXL | `uv pip install -e ".[surgenet]"` + `git clone https://github.com/TimJaspers0801/SurgeNet` | `SurgeNetXL_checkpoint_epoch0050_teacher.pth` (backbone only) | needs fine-tuning on your masks: `segtrain surgenet` |

## `lapscale check`

Loads every backend you name, runs it 3 times on a test frame (yours with `--image`, else synthetic) and
prints OK / FAIL with the exact error, the latency and a sanity value:

```bash
lapscale check --image some_frame.png \
    --moge v2 v3 --da3 depth-anything/DA3MONO-LARGE depth-anything/DA3METRIC-LARGE \
    --sam3 --yolo yolo26n-seg.pt \
    --surgenet-repo ../SurgeNet --surgenet-pretrained weights/SurgeNetXL_checkpoint_epoch0050_teacher.pth
```

The SurgeNet line checks that the backbone weights actually load (it reports how many tensors were
matched): a wrong checkpoint would otherwise be silently ignored.
