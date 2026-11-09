# shellcheck shell=bash
# Common environment of every lapscale job; sourced by slurm/*.sbatch after `cd "$SLURM_SUBMIT_DIR"`.
# Paths can be overridden before submitting: export LAPSCALE_SHARED=... / LAPSCALE_DATA=... etc.

REPO=$PWD
grep -q 'name = "lapscale"' pyproject.toml 2>/dev/null || { echo "submit from the lapscale repo root"; exit 1; }

SHARED=${LAPSCALE_SHARED:-/home/shared/nearmrs/$USER}          # shared group storage, not $HOME (small)
export LAPSCALE_DATA=${LAPSCALE_DATA:-$SHARED/lapscale/data}     # videos, calibration files
export LAPSCALE_RUNS=${LAPSCALE_RUNS:-$SHARED/lapscale/runs}     # stage caches, overlays
export LAPSCALE_BENCH=${LAPSCALE_BENCH:-$SHARED/lapscale/bench}  # segbench / depthbench folders
export LAPSCALE_WEIGHTS=${LAPSCALE_WEIGHTS:-$SHARED/lapscale/weights}
export LAPSCALE_SURGENET_REPO=${LAPSCALE_SURGENET_REPO:-$REPO/third_party/SurgeNet}

# every cache on the shared storage (never the node's /tmp, never $HOME)
export HF_HOME=$SHARED/cache/huggingface TORCH_HOME=$SHARED/cache/torch TRITON_CACHE_DIR=$SHARED/cache/triton \
       XDG_CACHE_HOME=$SHARED/cache/xdg MPLCONFIGDIR=$SHARED/cache/matplotlib YOLO_CONFIG_DIR=$SHARED/cache/ultralytics
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}        # weights pre-downloaded on the login node (slurm/download_weights.sh)
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
unset PYTHONPATH                                  # a PYTHONPATH from .bashrc can shadow the venv's packages
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$YOLO_CONFIG_DIR" \
         "$LAPSCALE_RUNS" "$LAPSCALE_BENCH" "$SHARED/tmp"

JOBTMP=$SHARED/tmp/job_${SLURM_JOB_ID:-$$}      # private temporary folder, removed at the end
mkdir -p "$JOBTMP"
export TMPDIR=$JOBTMP
trap 'rm -rf "$JOBTMP"' EXIT

lapscale_bin() {   # lapscale_bin main|sam3|da3 -> path of the lapscale command of that environment
    local d=$REPO/.venv
    [ "$1" = main ] || d=$REPO/.venv-$1
    [ -x "$d/bin/lapscale" ] || { echo "missing environment $d: run 'bash setup_env.sh $1' on the login node" >&2; exit 1; }
    echo "$d/bin/lapscale"
}
env_for_depth() { case "$1" in da3) echo da3 ;; *) echo main ;; esac; }   # moge, folder -> main
env_for_seg()   { case "$1" in sam3) echo sam3 ;; *) echo main ;; esac; } # yolo, surgenet, masks -> main

job_header() {
    echo "node $(hostname)  job ${SLURM_JOB_ID:-local}  start $(date)"
    if command -v nvidia-smi >/dev/null; then nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; fi
}
