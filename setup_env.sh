#!/bin/bash
# Build the uv environments of lapscale on the Mufasa LOGIN node (docs/mufasa.md).
# Never imports torch (the login node cannot); a verification job is submitted at the end.
#
#   bash setup_env.sh                  # main env only: .venv  (lapscale, torch cu130, MoGe v2/v3, YOLO, SurgeNetXL)
#   bash setup_env.sh main sam3 da3    # also .venv-sam3 (SAM 3) and .venv-da3 (Depth Anything 3)
#   REPIN=1 bash setup_env.sh sam3     # ignore requirements/sam3.txt: resolve again and rewrite the pins
#   NO_VERIFY=1 bash setup_env.sh      # do not submit the verification job
#
# One environment per conflicting stack: MoGe needs numpy>=2; SAM 3 needs numpy<2 and timm>=1.0.17; DA3 needs
# numpy<2 and xformers; SurgeNet pins timm==1.0.13. The stages exchange files, so one job can use several envs.
set -euo pipefail
cd "$(dirname "$0")"
REPO=$PWD
export PATH="$HOME/.local/bin:$PATH"
SHARED=${LAPSCALE_SHARED:-/home/shared/nearmrs/$USER}
export UV_CACHE_DIR="$SHARED/cache/uv" TMPDIR="$SHARED/tmp" UV_LINK_MODE=copy
# The login node caps virtual memory at 2 GB (ulimit -v, hard): keep uv's address space small
# (few threads, one download/install at a time), else "memory mapping the file failed (os error 12)".
export TOKIO_WORKER_THREADS=2 RAYON_NUM_THREADS=2 UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_INSTALLS=1 UV_CONCURRENT_BUILDS=1
mkdir -p "$UV_CACHE_DIR" "$TMPDIR" logs requirements third_party
PYVER=3.12                                         # SAM 3 needs >= 3.12, DA3 <= 3.13
TORCH_INDEX=https://download.pytorch.org/whl/cu130 # driver CUDA 13.0 on the nodes; torch brings its own runtime
ENVS=("$@")
[ ${#ENVS[@]} -eq 0 ] && ENVS=(main)

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

venv_dir() { if [ "$1" = main ]; then echo "$REPO/.venv"; else echo "$REPO/.venv-$1"; fi; }

install() {   # uv pip install into $V, PyPI + the PyTorch CUDA index
    uv pip install --python "$V/bin/python" --extra-index-url "$TORCH_INDEX" --index-strategy unsafe-best-match "$@"
}

headless_opencv() {   # opencv-python needs libGL (absent on the nodes) and owns the same cv2 module as headless
    uv pip uninstall --python "$V/bin/python" opencv-python opencv-contrib-python >/dev/null 2>&1 || true
    install --reinstall "opencv-python-headless>=4.8"
}

spec() {     # what each environment contains (first build, or REPIN=1)
    case "$1" in
        main) install torch torchvision "moge @ git+https://github.com/microsoft/MoGe.git" "ultralytics>=8.3" \
                      "timm==1.0.13" -e ".[dev]" ;;
        sam3) install torch torchvision "sam3 @ git+https://github.com/facebookresearch/sam3.git" \
                      "setuptools<81" einops open_clip_torch psutil scipy pycocotools matplotlib \
                      scikit-image scikit-learn pandas numba python-rapidjson -e . ;;   # undeclared by sam3; pkg_resources gone in setuptools 81
        da3)  install torch torchvision xformers \
                      "depth-anything-3 @ git+https://github.com/ByteDance-Seed/Depth-Anything-3.git" \
                      addict scikit-learn -e . ;;  # imported by DA3 but not declared
        *) echo "unknown environment '$1' (main | sam3 | da3)"; exit 1 ;;
    esac
}

for E in "${ENVS[@]}"; do
    V=$(venv_dir "$E")
    PINS="requirements/$E.txt"
    AVAIL=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
    [ "$AVAIL" -ge 10 ] || { echo "Less than 10 GB free in \$HOME ($AVAIL GB): a torch env takes ~7 GB"; exit 1; }
    echo "=== $E -> $V"
    [ -x "$V/bin/python" ] || uv venv --python "$PYVER" "$V"
    if [ -f "$PINS" ] && [ -z "${REPIN:-}" ]; then
        echo "installing the pinned versions of $PINS"
        install -r "$PINS"
        install --no-deps -e "$([ "$E" = main ] && echo ".[dev]" || echo ".")"
    else
        spec "$E"
        headless_opencv
        # pins for the next builds (commit them); lapscale itself is always installed editable
        uv pip freeze --python "$V/bin/python" | grep -v -E '^-e |^lapscale( |=|@)' > "$PINS"
        echo "wrote $PINS ($(wc -l < "$PINS") packages): commit it"
    fi
    if [ "$E" = main ] && [ ! -d third_party/SurgeNet ]; then     # authors' code, not a package
        git clone --depth 1 https://github.com/TimJaspers0801/SurgeNet third_party/SurgeNet
        echo "SurgeNet at commit $(git -C third_party/SurgeNet rev-parse --short HEAD)"
    fi
done

if [ -z "${NO_VERIFY:-}" ]; then
    J=$(sbatch --parsable slurm/verify_env.sbatch "${ENVS[@]}")
    echo "Verification job $J: tail -f logs/verify_env-$J.out"
fi
