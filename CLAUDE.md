# CLAUDE.md — lapscale

Project context for any assistant session. Read it fully first, then `docs/roadmap.md` and
`docs/decisions.md`.

## What this repository is

Metric depth for **monocular laparoscopy**, from the image only. The instrument shaft (a cylinder of known
diameter) is the ruler that turns the depth of a monocular network (MoGe) into millimetres.
Method: Wei et al., MICCAI 2024, *Enhanced Scale-aware Depth Estimation for Monocular Endoscopic Scenes with
Geometric Modeling* (MonoEndoDepth, github med-air/MonoEndoDepth), with MoGe in place of Monodepth2.

## Wider context (not code in this repository)

PhD project of Leonardo Passoni (Politecnico di Milano, NEARLab, supervisor Prof. De Momi): augmented
reality for lymphadenectomy during Ivor Lewis esophagectomy, with IEO. Lymph nodes marked on the
preoperative CT are projected onto the endoscopic view. The work adapts LyNAR (Cruciani et al., RA-L 2026,
robotic prostatectomy).

- Target phase: **abdominal, laparoscopic, MONOCULAR endoscope**; anatomical anchor = coeliac trunk
  (splenic, common hepatic, left gastric arteries) + pancreas.
- This repository covers the intraoperative side: metric geometry from monocular video, later used to
  register the CT model. CT segmentation, annotation and registration are NOT here.
- The laparoscope is held by a resident: motion, tremor, removals for cleaning, smoke, blood. Real frames
  are not clean.

## How the method works (do not reinvent)

1. **Shaft geometry** (stage `tool`, done): each edge back-projects to a plane through the camera centre;
   axis direction = cross product of the plane normals; axis at distance r from both planes. Shaft pixels
   get metric depth by ray-cylinder intersection.
2. **Scale** (stage `scale`, planned): on shaft pixels, MoGe depth vs metric depth; fit `mm = a*d + b`.
   Scale-only (`a` = median ratio) and affine are **both kept and compared**. Applied to the whole frame.
3. Read MoGe depth **inside** the shaft mask, never on its boundary (depth discontinuity).

## Status

- Done: skeleton; `camera.py`; stage 1 `calib` (board / import / MoGe fallback); MoGe wrapper in
  `depth/` (not run here: no torch/weights); `tool/geometry.py` and `tool/uncertainty.py` (Python API).
  31 synthetic tests.
- Not done: commands `lapscale depth|tool|scale|run|eval|viz`; segmentation; edges from masks.
- Next: see `docs/roadmap.md` (stage 2 depth with cache, then segmentation and edges).

## Decisions already taken (do not reopen without a reason)

- Separate stages communicating through files, plus `run` to chain them. Backends behind protocols.
- MoGe: latest = **MoGe-3** (released 2026-08-18; `Ruicheng/moge-3-vitl` 370M, `-vitg` 1.25B; MIT).
  Offline: MoGe-3. Real time: MoGe-2 vits/vitb in fp16. **MoGe-3 needs Triton/FlexGEMM: Linux or WSL2, not
  native Windows.** MoGe-2 runs anywhere torch does.
- A known `fov_x` (from calibration) is passed to MoGe; otherwise MoGe estimates it (weaker fallback).
- In monocular metric reconstruction **a focal-length error is a scale error, one to one**: calibration is
  a prerequisite. Same for the instrument diameter: measure it, do not assume it.
- **Downstream quantity = depth of the VISIBLE shaft**, not the distance to the infinite axis (the latter is
  > 3x less stable for shafts tilted towards the camera).
- **Edge bias is the critical error**: +1 px outward on both edges changes the depth by ~ -2/width
  (-2% at 100 px width, -7% at 28 px). It is systematic for a given segmentation model.
- Instrument segmentation: NOT chosen yet. Candidates: SAM 3, semantic network with a surgical backbone
  (e.g. SurgeNetXL) + fine-tuning, YOLO-seg. Criterion = **shaft edge accuracy** (bias and spread in px on
  ~30-50 hand-annotated frames), NOT Dice. Shaft and jaws separated **geometrically** (RANSAC on the
  contour), not with annotated part classes.
- The near-horizontal-edge problem was **stereo only** (baseline); it does not exist in monocular.
- No MoGe fine-tuning until the evaluation on stereo data shows it is needed.
- Depth is not needed every frame: segmentation every frame, depth at 5-10 Hz in parallel (real time).
- Hardware: **GPU not confirmed yet** (ask Leonardo). Earlier timing estimates were not measured: the
  first real benchmark is `lapscale bench` (to be written).

## Validation with the colleague's stereo data

Stereo da Vinci videos (left/right, 1920x1080, 25 fps) with calibration `stereo_calib_par_30.xml`. The
script `phase0_check.py` (sync, check, refine, measure, selftest; outside this repository) validated them:
offset 0; calibration par_30 correct (par_0 wrong); refined with `refine` (median vertical disparity
1.00 -> 0.48 px). Shaft measurement: 8.01 ± 0.30 mm vs 8 mm nominal, **at 42 mm** (one usable frame, edges
at 10 deg).
- Known limit: validated only at close range; tissue is at ~100 mm, where a systematic error weighs more.
- Planned use in `eval`: left video as monocular input, stereo as ground truth, errors per distance band
  from the instrument.
- Radius: the paper states 4.5 mm for da Vinci instruments; the measurement here confirms an 8 mm diameter
  (radius 4). Always check the diameter of the instrument in use.

## Working rules

- **Before saying "done": `pytest` and `ruff check src tests` pass.** Report the measured numbers, not the
  expected ones. If a synthetic test does not match, investigate and explain (example: OpenCV's focal std
  can be ~2x optimistic; now documented).
- Every geometric module has a **test on synthetic data with known ground truth**.
- Method choices go in `docs/decisions.md`; experiments with results in `docs/experiment_log.md`; file
  formats in `docs/outputs.md`.
- User preferences: direct, technically precise communication; **minimal, honest solutions** over polished
  ones that overpromise; **no scope creep**: do what is asked, flag the rest instead of doing it.
- Do not invent references or numbers. If unsure, say so. If a library or model version may have changed,
  check the code or the official README.
- Heavy dependencies (torch, MoGe, SAM) only as optional extras, imported lazily.
- Never commit videos, weights or data (`.gitignore` excludes them). Clinical data: nothing identifiable.
- Language: everything in the repository (code, comments, docstrings, docs) in English.

## Commands

```bash
uv venv && uv pip install -e ".[dev]"     # base (numpy, opencv, pyyaml)
uv pip install -e ".[moge]"               # + torch and MoGe (v3: Linux/WSL2)
pytest                                    # ~30 s
ruff check src tests
lapscale calib --help
```

## Open questions (ask Leonardo, do not decide alone)

- Which GPU/machine for development and for real time?
- Actual diameters of the IEO laparoscopic instruments (5 or 10 mm?) and optics used (0 or 30 deg)?
- Is there a calibration of the IEO laparoscope? Otherwise a ChArUco video is needed (`docs/calibration.md`).
- Where are the monocular laparoscopic videos, and at which resolution/fps?
