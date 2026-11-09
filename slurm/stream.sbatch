#!/bin/bash
#SBATCH --job-name=lapscale_stream
#SBATCH --partition=jobs
#SBATCH --qos=gpuwide
#SBATCH --gres=gpu:4g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# Real-time mode on a recorded video, for PROFILING (no display on the cluster): overlay video + per-frame
# timings. Segmentation and depth run in the same process, so they must share an environment: the main one
# (MoGe + YOLO / SurgeNet / masks). SAM 3 cannot be combined with MoGe in one environment (numpy).
#
#   sbatch --export=ALL,SEG=surgenet,SURGENET_CKPT=...,MOGE_CKPT=Ruicheng/moge-2-vits-normal \
#          slurm/stream.sbatch VIDEO CAMERA DIAMETER_MM
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source slurm/env.sh
job_header
VIDEO=${1:?usage: sbatch slurm/stream.sbatch VIDEO CAMERA DIAMETER_MM}
CAMERA=${2:?}
DIAMETER=${3:?}
SEG=${SEG:-yolo}
[ "$SEG" = sam3 ] && { echo "SAM 3 and MoGe need different environments: use SEG=yolo|surgenet|masks"; exit 1; }
case "$SEG" in
    yolo) SEG_ARGS=(--yolo-weights "${YOLO_WEIGHTS:?}") ;;
    surgenet) SEG_ARGS=(--surgenet-checkpoint "${SURGENET_CKPT:?}") ;;
    masks) SEG_ARGS=(--masks-dir "${MASKS_DIR:?}") ;;
esac
RUN=${RUN_DIR:-$LAPSCALE_RUNS/$(basename "${VIDEO%.*}")}
mkdir -p "$RUN"
"$(lapscale_bin main)" run --mode stream --video "$VIDEO" --camera "$CAMERA" --diameter-mm "$DIAMETER" \
    --run-dir "$RUN" --device cuda --seg-backend "$SEG" "${SEG_ARGS[@]}" \
    --depth-backend moge --moge-version "${MOGE_VERSION:-v2}" --depth-checkpoint "${MOGE_CKPT:-Ruicheng/moge-2-vits-normal}" \
    --out-video "$RUN/stream_overlay.mp4" ${MAX_FRAMES:+--max-frames "$MAX_FRAMES"}
echo "end $(date) | $RUN/stream_log.csv"
