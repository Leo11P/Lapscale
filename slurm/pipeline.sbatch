#!/bin/bash
#SBATCH --job-name=lapscale_pipeline
#SBATCH --partition=jobs
#SBATCH --qos=gpu
#SBATCH --gres=gpu:3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# Whole pipeline on one video, in ONE job (one QoS slot): depth -> tool -> scale (-> overlay, -> eval).
# Each stage uses the environment of its backend; the stages exchange files, and resume from their caches if
# the job is resubmitted after a timeout.
#
#   sbatch slurm/pipeline.sbatch VIDEO CAMERA DIAMETER_MM
#   sbatch --export=ALL,SEG=sam3,DEPTH=moge,MOGE_VERSION=v3,STRIDE=5,OVERLAY=1 slurm/pipeline.sbatch \
#          $LAPSCALE_DATA/ieo/case01.mp4 configs/camera/lap_ieo.yaml 5
#
# Variables (with defaults): SEG=sam3 (sam3 | yolo | surgenet | masks), DEPTH=moge (moge | da3 | folder),
#   MOGE_VERSION=v3, DEPTH_CKPT (checkpoint / HF id), PROMPT="surgical instrument", YOLO_WEIGHTS, SURGENET_CKPT,
#   MASKS_DIR, DEPTH_DIR, STRIDE=1 or MAX_FRAMES, MC=100, RUN_DIR (default $LAPSCALE_RUNS/<video stem>),
#   OVERLAY=1 to render the overlay video, EVAL_BENCH=<depthbench folder> to evaluate (video = <bench>/images).
# MoGe-3 vitg does not fit a 20 GB slice: sbatch --qos=gpuheavy-40 --gres=gpu:40gb:1 ...
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source slurm/env.sh
job_header

VIDEO=${1:?usage: sbatch slurm/pipeline.sbatch VIDEO CAMERA DIAMETER_MM}
CAMERA=${2:?camera YAML}
DIAMETER=${3:?shaft diameter in mm}
SEG=${SEG:-sam3}
DEPTH=${DEPTH:-moge}
MOGE_VERSION=${MOGE_VERSION:-v3}
STEM=$(basename "${VIDEO%.*}")
RUN=${RUN_DIR:-$LAPSCALE_RUNS/$STEM}
FRAMES=(--stride "${STRIDE:-1}")
[ -n "${MAX_FRAMES:-}" ] && FRAMES=(--max-frames "$MAX_FRAMES")

DEPTH_ARGS=(--depth-backend "$DEPTH")
case "$DEPTH" in
    moge)   DEPTH_TAG=moge-$MOGE_VERSION; DEPTH_ARGS+=(--moge-version "$MOGE_VERSION") ;;
    da3)    DEPTH_TAG=da3-$(basename "${DEPTH_CKPT:-DA3MONO-LARGE}") ;;
    folder) DEPTH_TAG=folder-$(basename "${DEPTH_DIR:?DEPTH_DIR}"); DEPTH_ARGS+=(--depth-dir "$DEPTH_DIR") ;;
    *) echo "unknown DEPTH=$DEPTH"; exit 1 ;;
esac
[ -n "${DEPTH_CKPT:-}" ] && DEPTH_ARGS+=(--depth-checkpoint "$DEPTH_CKPT")
DEPTH_TAG=${DEPTH_TAG_OVERRIDE:-$DEPTH_TAG}

SEG_ARGS=(--seg-backend "$SEG")
case "$SEG" in
    sam3)     TOOL_TAG=sam3; SEG_ARGS+=(--prompt "${PROMPT:-surgical instrument}") ;;
    yolo)     TOOL_TAG=yolo; SEG_ARGS+=(--yolo-weights "${YOLO_WEIGHTS:?YOLO_WEIGHTS}") ;;
    surgenet) TOOL_TAG=surgenet; SEG_ARGS+=(--surgenet-checkpoint "${SURGENET_CKPT:?SURGENET_CKPT}") ;;
    masks)    TOOL_TAG=masks; SEG_ARGS+=(--masks-dir "${MASKS_DIR:?MASKS_DIR}") ;;
    *) echo "unknown SEG=$SEG"; exit 1 ;;
esac
TOOL_TAG=${TOOL_TAG_OVERRIDE:-$TOOL_TAG}

D=$(lapscale_bin "$(env_for_depth "$DEPTH")")
T=$(lapscale_bin "$(env_for_seg "$SEG")")
C=$(lapscale_bin main)
echo "video $VIDEO -> $RUN | depth $DEPTH_TAG ($D) | tool $TOOL_TAG ($T)"

"$D" depth --video "$VIDEO" --camera "$CAMERA" --run-dir "$RUN" --tag "$DEPTH_TAG" --device cuda \
     "${DEPTH_ARGS[@]}" "${FRAMES[@]}"
"$T" tool --video "$VIDEO" --camera "$CAMERA" --run-dir "$RUN" --tag "$TOOL_TAG" --device cuda \
     --diameter-mm "$DIAMETER" --mc "${MC:-100}" "${SEG_ARGS[@]}" "${FRAMES[@]}"
"$C" scale --run-dir "$RUN" --depth-tag "$DEPTH_TAG" --tool-tag "$TOOL_TAG"
if [ -n "${OVERLAY:-}" ]; then
    "$C" viz --run-dir "$RUN" --video "$VIDEO" --camera "$CAMERA" --depth-tag "$DEPTH_TAG" --tool-tag "$TOOL_TAG"
fi
if [ -n "${EVAL_BENCH:-}" ]; then
    for V in scale_only affine; do
        "$C" eval --bench "$EVAL_BENCH" --run-dir "$RUN" --depth-tag "$DEPTH_TAG" --tool-tag "$TOOL_TAG" --variant "$V"
    done
fi
echo "end $(date) | results in $RUN"
