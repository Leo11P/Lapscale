#!/bin/bash
#SBATCH --job-name=segtrain
#SBATCH --partition=jobs
#SBATCH --qos=gpuheavy-20
#SBATCH --gres=gpu:4g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# Fine-tuning of the instrument segmentation (only if segbench shows zero-shot SAM 3 is not enough).
#
#   sbatch slurm/segtrain.sbatch surgenet --bench $LAPSCALE_BENCH/seg_train --out $LAPSCALE_WEIGHTS/surgenet_ft \
#          --pretrained $LAPSCALE_WEIGHTS/surgenet/SurgeNetXL_checkpoint_epoch0050_teacher.pth --epochs 30
#   sbatch slurm/segtrain.sbatch yolo --data $LAPSCALE_BENCH/yolo_ds/data.yaml --model yolo26n-seg.pt
# Everything after the first argument goes to `lapscale segtrain <kind>`.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source slurm/env.sh
job_header
KIND=${1:?usage: sbatch slurm/segtrain.sbatch surgenet|yolo [lapscale segtrain options]}
shift
# YOLO downloads its base weights (from GitHub) at the first run unless --model is a local file: if the compute
# nodes have no internet, download it on the login node and pass the path.
"$(lapscale_bin main)" segtrain "$KIND" --device cuda "$@"
echo "end $(date)"
