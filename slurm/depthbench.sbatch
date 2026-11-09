#!/bin/bash
#SBATCH --job-name=depthbench
#SBATCH --partition=jobs
#SBATCH --qos=gpuwide
#SBATCH --gres=gpu:4g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# Depth benchmark in one job: (prepare the stereo ground truth if needed) -> each backend -> comparison table.
#
#   sbatch --export=ALL,LEFT=$LAPSCALE_DATA/stereo/VideoSX.mp4,RIGHT=$LAPSCALE_DATA/stereo/VideoDX.mp4,\
#   CALIB=$LAPSCALE_DATA/stereo/stereo_calib_par_30_refined.xml,CROP=0,960,320,1600 \
#          slurm/depthbench.sbatch $LAPSCALE_BENCH/dbench_davinci
#
# BACKENDS (default below): moge:<v2|v3>[=<checkpoint>]  da3:<HF id or path>  folder:<dir>
# The prepare step runs only if <bench>/manifest.json does not exist; backends already scored are skipped
# unless FORCE=1.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source slurm/env.sh
job_header
BENCH=${1:?usage: sbatch slurm/depthbench.sbatch BENCH_DIR}
BACKENDS=${BACKENDS:-"moge:v3 moge:v2=Ruicheng/moge-2-vitl moge:v2=Ruicheng/moge-2-vits-normal da3:depth-anything/DA3MONO-LARGE da3:depth-anything/DA3METRIC-LARGE"}
C=$(lapscale_bin main)

if [ ! -f "$BENCH/manifest.json" ]; then
    "$C" depthbench prepare-stereo --left "${LEFT:?LEFT}" --right "${RIGHT:?RIGHT}" --calib "${CALIB:?CALIB}" \
        --bench "$BENCH" --n "${N_FRAMES:-50}" --offset "${OFFSET:-0}" ${CROP:+--crop "$CROP"} \
        --calib-frame "${CALIB_FRAME:-full}" --z-range-mm "${Z_RANGE_MM:-20,300}"
fi

for SPEC in $BACKENDS; do
    KIND=${SPEC%%:*}
    ARG=${SPEC#*:}
    case "$KIND" in
        moge)   VER=${ARG%%=*}; CK=""; [ "$ARG" != "$VER" ] && CK=${ARG#*=}
                NAME=moge-$VER${CK:+-$(basename "$CK")}
                ARGS=(--depth-backend moge --moge-version "$VER" ${CK:+--depth-checkpoint "$CK"}) ; ENV=main ;;
        da3)    NAME=da3-$(basename "$ARG"); ARGS=(--depth-backend da3 --depth-checkpoint "$ARG"); ENV=da3 ;;
        folder) NAME=folder-$(basename "$ARG"); ARGS=(--depth-backend folder --depth-dir "$ARG"); ENV=main ;;
        *) echo "unknown backend spec $SPEC"; exit 1 ;;
    esac
    if [ -f "$BENCH/results/$NAME/summary.json" ] && [ -z "${FORCE:-}" ]; then
        echo "=== $NAME already scored (FORCE=1 to redo)"; continue
    fi
    echo "=== $NAME ($ENV)"
    "$(lapscale_bin "$ENV")" depthbench run --bench "$BENCH" --name "$NAME" --device cuda "${ARGS[@]}" \
        || echo "FAILED: $NAME (continuing with the next backend)"
done
"$C" depthbench compare --bench "$BENCH"
echo "end $(date)"
