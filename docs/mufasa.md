# Running lapscale on Mufasa

Same approach as the other projects (uv environments, no containers): one `.venv` per conflicting stack in the
repo, built on the login node by `setup_env.sh`, used by SLURM jobs through their interpreter. Caches, data and
results live on the shared group storage. Cluster rules and QoS are the ones of the Mufasa guide; the wiki
(biohpc.deib.polimi.it) is authoritative if they change.

## 1. Environments

| Env | Folder | Contains | Why separate |
|---|---|---|---|
| `main` | `.venv` | lapscale + dev tools, torch cu130, MoGe v2/v3, Ultralytics, timm 1.0.13 (SurgeNetXL) | MoGe needs numpy >= 2 |
| `sam3` | `.venv-sam3` | lapscale, torch cu130, SAM 3 | SAM 3 needs numpy < 2 and timm >= 1.0.17 |
| `da3` | `.venv-da3` | lapscale, torch cu130, xformers, Depth Anything 3 | numpy < 2, xformers tied to the torch version |

```bash
cd ~ && git clone <lapscale repo> && cd lapscale
bash setup_env.sh                    # main only (~7 GB in $HOME)
bash setup_env.sh main sam3 da3      # all three (~21 GB): check `df -h ~` first
```

* Python 3.12 (SAM 3 needs >= 3.12), torch from the cu130 index (`+cu130`), OpenCV **headless** everywhere
  (the nodes have no libGL; packages that pull `opencv-python` are switched to headless).
* The SurgeNet code (authors' repo, not a package) is cloned in `third_party/SurgeNet`; jobs point to it with
  `LAPSCALE_SURGENET_REPO`.
* **Pins**: the first build of each environment writes `requirements/<env>.txt` (`uv pip freeze`, git
  dependencies included with their commit). Commit them: the next `setup_env.sh` installs exactly those
  versions. `REPIN=1 bash setup_env.sh <env>` resolves again and rewrites the pins.
* `setup_env.sh` never imports torch and submits `slurm/verify_env.sbatch` for the environments it built.
* If `$HOME` is short of space, clone the repo under `/home/shared/nearmrs/$USER/` instead: everything (venvs
  included) follows the repo, and jobs are submitted from there.

## 2. Weights (login node, once)

```bash
bash slurm/download_weights.sh                         # MoGe-3 vitl, MoGe-2 vitl/vits/vitb, DA3 MONO/METRIC,
                                                       # SAM 3 (gated), SurgeNetXL teacher
bash slurm/download_weights.sh Ruicheng/moge-3-vitg depth-anything/DA3NESTED-GIANT-LARGE-1.1    # extras
```

SAM 3 is gated: request access on huggingface.co/facebook/sam3, then once `.venv-sam3/bin/hf auth login`
(the token goes in `$HF_HOME`, on the shared storage). Jobs run with `HF_HUB_OFFLINE=1`, so a missing weight
fails immediately instead of downloading on a compute node; set `HF_HUB_OFFLINE=0` at submission to allow it.

## 3. Where things go (`slurm/env.sh`)

| Variable | Default | Content |
|---|---|---|
| `LAPSCALE_SHARED` | `/home/shared/nearmrs/$USER` | root of everything below |
| `LAPSCALE_DATA` | `$LAPSCALE_SHARED/lapscale/data` | videos, stereo calibration files |
| `LAPSCALE_RUNS` | `.../lapscale/runs` | stage caches per video, overlays |
| `LAPSCALE_BENCH` | `.../lapscale/bench` | segbench and depthbench folders |
| `LAPSCALE_WEIGHTS` | `.../lapscale/weights` | SurgeNetXL teacher, fine-tuned checkpoints |
| caches | `.../cache/{huggingface,torch,triton,xdg,matplotlib,ultralytics,uv}` | never `$HOME`, never the node's `/tmp` |
| `TMPDIR` | `.../tmp/job_<jobid>` | private per job, deleted at the end |

Camera files are small and stay in the repo (`configs/camera/`). Every job also unsets `PYTHONPATH`, sets
`PYTHONNOUSERSITE=1` and unbuffered output, and must be submitted **from the repo root** (`logs/` must exist).

## 4. Jobs

| Script | QoS (default) | GPU | Does | Envs used |
|---|---|---|---|---|
| `verify_env.sbatch [envs]` | gpulight | 3g.20gb | torch + GPU, `lapscale check` per env, pytest (main) | each |
| `pipeline.sbatch VIDEO CAMERA DIAM` | gpu | 3g.20gb | depth -> tool -> scale (-> overlay, -> eval), one job | depth/seg env + main |
| `depthbench.sbatch BENCH` | gpuwide | 4g.20gb | stereo GT if missing -> each depth backend -> table | main, da3 |
| `segbench.sbatch BENCH CAMERA` | gpulight | 3g.20gb | each segmentation backend / SAM 3 prompt -> table | sam3, main |
| `segtrain.sbatch surgenet\|yolo ...` | gpuheavy-20 | 4g.20gb | fine-tuning | main |
| `stream.sbatch VIDEO CAMERA DIAM` | gpuwide | 4g.20gb | real-time mode on a recording: overlay + timings | main |

* At most one submitted job per QoS and two running per user: the defaults use different QoS so that a
  pipeline, a depth benchmark, a segmentation benchmark and a training can be queued together.
* Every option can be overridden at submission, e.g. MoGe-3 vitg (does not fit 20 GB):
  `sbatch --qos=gpuheavy-40 --gres=gpu:40gb:1 --export=ALL,DEPTH_CKPT=Ruicheng/moge-3-vitg slurm/pipeline.sbatch ...`
* **Resubmitting continues**: depth and tool caches skip the frames already done; `depthbench` skips the
  backends already scored (`FORCE=1` to redo).
* `stream.sbatch` needs segmentation and depth in one process, i.e. one environment: MoGe with YOLO,
  SurgeNetXL or masks. The final real-time machine is the OR workstation; on Mufasa this is for profiling.

## 5. A full session

```bash
# on the login node, repo root
bash setup_env.sh main sam3 da3                    # once (+ commit requirements/*.txt)
bash slurm/download_weights.sh                     # once
source slurm/env.sh                                # to have $LAPSCALE_DATA etc. in this shell

# 1) depth networks on the colleague's stereo videos
sbatch --export=ALL,LEFT=$LAPSCALE_DATA/stereo/VideoSX.mp4,RIGHT=$LAPSCALE_DATA/stereo/VideoDX.mp4,CALIB=$LAPSCALE_DATA/stereo/stereo_calib_par_30_refined.xml,CROP=0,960,320,1600 \
       slurm/depthbench.sbatch $LAPSCALE_BENCH/dbench_davinci

# 2) whole pipeline on the same frames, evaluated (da Vinci shaft: 8 mm)
sbatch --export=ALL,SEG=sam3,DEPTH=moge,MOGE_VERSION=v3,EVAL_BENCH=$LAPSCALE_BENCH/dbench_davinci \
       slurm/pipeline.sbatch $LAPSCALE_BENCH/dbench_davinci/images $LAPSCALE_BENCH/dbench_davinci/camera.yaml 8

# 3) segmentation benchmark (frames annotated on the PC, see 6)
sbatch --export=ALL,BACKENDS=sam3,PROMPTS="surgical instrument;laparoscopic instrument" \
       slurm/segbench.sbatch $LAPSCALE_BENCH/seg_ieo configs/camera/lap_ieo.yaml

# 4) IEO monocular videos
sbatch --export=ALL,SEG=sam3,DEPTH=moge,STRIDE=5,OVERLAY=1 \
       slurm/pipeline.sbatch $LAPSCALE_DATA/ieo/case01.mp4 configs/camera/lap_ieo.yaml 5

squeue -u $USER ; tail -f logs/lapscale_pipeline-<jobid>.out
```

Light CPU commands (`lapscale calib ...`, `segbench sample`, `segbench export-yolo`, `scale`, `viz`) can run in a
short CPU allocation instead of the login node:
`srun --partition=jobs --qos=nogpu --cpus-per-task=4 --mem=16G --time=1:00:00 .venv/bin/lapscale calib board ...`

## 6. Annotation (on your PC)

The annotation tools are GUIs and the cluster OpenCV is headless. On the PC: `uv pip install -e .`, then
`uv pip uninstall opencv-python-headless && uv pip install opencv-python`. Sample frames on Mufasa (or on the PC),
copy the benchmark folder, annotate, copy it back:

```bash
rsync -av mufasa:/home/shared/nearmrs/<user>/lapscale/bench/seg_ieo/ ./seg_ieo/      # to the PC
lapscale segbench annotate --bench seg_ieo --diameter-mm 5 --display-scale 1.5
rsync -av ./seg_ieo/ mufasa:/home/shared/nearmrs/<user>/lapscale/bench/seg_ieo/      # back
```

(`mufasa` = your ssh host alias.) Mask annotation with SAM 3 proposals needs SAM 3 on the PC; without it,
`annotate-masks` works with brush and polygons only.

## 7. Troubleshooting

| What you see | Cause and fix |
|---|---|
| `missing environment .../.venv-sam3` | build it on the login node: `bash setup_env.sh sam3` |
| `libtorch_cpu.so: failed to map segment` | torch imported on the login node: run it in a job |
| `LocalEntryNotFoundError` / offline errors | weight not in `$HF_HOME`: `bash slurm/download_weights.sh <repo>` (or submit with `HF_HUB_OFFLINE=0`) |
| SAM 3 `401` / gated | access not granted yet, or no `hf auth login` |
| MoGe-3: `Failed to find C compiler` | Triton compiles a small launcher: load a gcc module in the job (`module avail gcc`) or set `CC` |
| `ImportError: libGL.so.1` | an `opencv-python` came back: `uv pip uninstall --python .venv-X/bin/python opencv-python` and reinstall `opencv-python-headless` |
| DA3: xformers / torch resolution fails | `REPIN=1 bash setup_env.sh da3`; if still failing, pin a torch version compatible with the latest xformers wheel in `setup_env.sh` |
| `CUDA out of memory` with MoGe-3 vitg or SAM 3 | larger slice: `--qos=gpuheavy-40 --gres=gpu:40gb:1` |
| `torch.cuda.is_available()` False in a job | no `--gres`, or torch not `+cu130` (check `requirements/<env>.txt`) |
| job pending for a long time | `squeue -u $USER`: maybe another job of yours in the same QoS |
