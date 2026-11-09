# Segmentation benchmark

Goal: choose the instrument segmentation network by what the metric scale depends on, the position of
the two shaft edges, not by Dice. One pixel of edge error moves the depth by about 2/width
(2% on a 100 px wide shaft, 7% on a 28 px wide one).

## Candidates

| Backend | Status | Notes |
|---|---|---|
| `sam3` | wrapper ready, zero-shot | text prompts; gated checkpoints (facebook/sam3, facebook/sam3.1); masks upsampled from low resolution, so edges must be measured |
| `masks` | ready | precomputed masks from any external model (e.g. a SurgeNetXL-based network trained elsewhere) |
| `yolo` | wrapper ready | needs weights trained on surgical instruments (COCO weights do not know them): `segbench export-yolo` + `segtrain yolo` |
| `surgenet` | backend + training ready | SurgeNetXL CaFormer-S18 backbone + FPN (authors' code); the decoder must be trained on your masks: `segtrain surgenet` |

## Protocol

1. **Frames**: `lapscale segbench sample --video v1.mp4 v2.mp4 ... --bench bench/ --n-per-video 25`.
   Use several videos and phases; 30-50 frames with a measurable shaft in the end.
2. **Annotation**: `lapscale segbench annotate --bench bench/ --diameter-mm 5 --display-scale 1.5`.
   For each shaft whose two edges are visible over a good length: 3-5 points on edge A along the
   straight part (never on the jaws), SPACE, the same on edge B, SPACE. Click ON the boundary, using
   the magnifier (top right, clicks inside it are sub-pixel). Frames without a measurable shaft: `x`.
   Annotate some frames twice on different days: the difference is the annotation noise floor.
3. **Run** each backend, e.g.
   `lapscale segbench run --bench bench/ --camera configs/camera/lap.yaml --seg-backend sam3 --prompt "surgical instrument"`
   Try several prompts (each is a separate run with `--name`).
4. **Compare**: `lapscale segbench compare --bench bench/` writes `results/comparison.md`.

## Ground-truth masks (optional)

Needed only to fine-tune (YOLO-seg, SurgeNetXL) or to report mask IoU / boundary error next to the edge metrics.

```bash
lapscale segbench annotate-masks --bench bench/ --seg-backend sam3     # SAM 3 proposals, corrected by hand
lapscale segbench export-yolo --bench bench/ --out yolo_ds/              # split by video
lapscale segtrain yolo --data yolo_ds/data.yaml --model yolo26n-seg.pt
lapscale segtrain surgenet --bench bench/ --out models/surgenet --surgenet-repo ../SurgeNet \
    --pretrained weights/SurgeNetXL_checkpoint_epoch0050_teacher.pth
lapscale segbench run --bench bench/ --camera cam.yaml --seg-backend surgenet --surgenet-checkpoint models/surgenet/best.pt \
    --surgenet-repo ../SurgeNet
```

Editor: left drag paints the current instance, right drag erases, `p` polygon mode (SPACE fills, `e` toggles
subtract), `[` `]` brush size, TAB next instance, `n` new, `d` delete, ENTER save, `x` no instrument, `b` back,
`q` quit; with SAM 3 as proposer, a middle-button box adds the SAM 3 mask of that box. Use frames from
different videos than the benchmark frames if the masks are for training, or the benchmark is no longer
independent.

## Reading the results

* `found_rate`: annotated shafts for which the backend produced two valid edges.
* `depth_err_pct` (median / p90 of the absolute value): the error the edges cause on the depth at the
  middle of the shaft. The headline number.
* `bias_px`: mean outward offset of the predicted edges (positive = mask too wide). A STABLE bias (small
  `bias_px_spread_mad`) can be calibrated away: `depth_err_pct_bias_corrected_*` shows what would remain
  (leave-one-out estimate, not optimistic).
* `offset_sd_px`: noise along the edges; `angle_err_deg`: wrong edge direction.
* `not_found_reasons`: why the edge extractor rejected the instances (mask, not network, problems).
* Overlays in `results/<name>/overlays/`: annotated edges green, predicted red / blue.

## Proposed decision rule (to be agreed)

Clinical tolerance is a few centimetres; at 100 mm a 5% scale error is 5 mm. Proposal:
p90 |depth error| <= 5% (after bias correction if the bias is stable) and found rate >= 70%.
If SAM 3 zero-shot meets it, no training. If not, the failure mode decides: low found rate -> better
prompts or fine-tuning; good detection but biased or noisy edges -> a network trained for boundary
accuracy at full resolution.
