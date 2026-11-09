#!/bin/bash
#SBATCH --job-name=segbench
#SBATCH --partition=jobs
#SBATCH --qos=gpulight
#SBATCH --gres=gpu:3g.20gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# Segmentation benchmark in one job: each backend (and each SAM 3 prompt) on the annotated frames -> table.
# The frames are annotated on your PC (GUI) and the benchmark folder copied to the shared storage.
#
#   sbatch slurm/segbench.sbatch $LAPSCALE_BENCH/seg_ieo configs/camera/lap_ieo.yaml
#   sbatch --export=ALL,BACKENDS="sam3 surgenet:$LAPSCALE_WEIGHTS/surgenet_ft/best.pt",PROMPTS="surgical instrument;laparoscopic grasper" \
#          slurm/segbench.sbatch BENCH CAMERA
#
# BACKENDS: sam3  yolo:<weights>  surgenet:<checkpoint>  masks:<dir>      PROMPTS: ';'-separated, one run each
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source slurm/env.sh
job_header
BENCH=${1:?usage: sbatch slurm/segbench.sbatch BENCH CAMERA}
CAMERA=${2:?camera YAML}
BACKENDS=${BACKENDS:-sam3}
PROMPTS=${PROMPTS:-surgical instrument}

run_one() {   # run_one NAME ENV args...
    local name=$1 env=$2; shift 2
    echo "=== $name ($env)"
    "$(lapscale_bin "$env")" segbench run --bench "$BENCH" --camera "$CAMERA" --name "$name" --device cuda "$@" \
        || echo "FAILED: $name (continuing)"
}
for SPEC in $BACKENDS; do
    KIND=${SPEC%%:*}
    ARG=${SPEC#*:}
    case "$KIND" in
        sam3) IFS=';' read -r -a PS <<< "$PROMPTS"
              for P in "${PS[@]}"; do run_one "sam3-${P// /_}" sam3 --seg-backend sam3 --prompt "$P"; done ;;
        yolo) run_one "yolo-$(basename "${ARG%.*}")" main --seg-backend yolo --yolo-weights "$ARG" ;;
        surgenet) run_one "surgenet-$(basename "$(dirname "$ARG")")" main --seg-backend surgenet --surgenet-checkpoint "$ARG" ;;
        masks) run_one "masks-$(basename "$ARG")" main --seg-backend masks --masks-dir "$ARG" ;;
        *) echo "unknown backend spec $SPEC"; exit 1 ;;
    esac
done
"$(lapscale_bin main)" segbench compare --bench "$BENCH"
echo "end $(date)"
