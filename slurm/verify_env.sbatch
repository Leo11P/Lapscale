#!/bin/bash
#SBATCH --job-name=verify_env
#SBATCH --partition=jobs
#SBATCH --qos=gpulight
#SBATCH --gres=gpu:3g.20gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# Usage: sbatch slurm/verify_env.sbatch [main] [sam3] [da3]     (submitted by setup_env.sh)
# Per environment: torch + GPU, then `lapscale check` with the weights in the cache; main also runs pytest.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source slurm/env.sh
job_header
ENVS=("$@")
[ ${#ENVS[@]} -eq 0 ] && ENVS=(main)
FAIL=0
for E in "${ENVS[@]}"; do
    L=$(lapscale_bin "$E")
    PY=$(dirname "$L")/python
    echo "=== $E"
    "$PY" -c "import torch; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')" || FAIL=1
    command -v gcc >/dev/null || echo "WARNING: no gcc on this node: Triton (MoGe-3) compiles a small launcher and needs a C compiler"
    case "$E" in
        main) "$L" check --device cuda --moge ${CHECK_MOGE:-v2 v3} \
                  ${CHECK_SURGENET:+--surgenet-pretrained "$CHECK_SURGENET"} || FAIL=1
              "$PY" -m pytest -q -p no:cacheprovider tests || FAIL=1 ;;
        sam3) "$L" check --device cuda --sam3 || FAIL=1 ;;
        da3)  "$L" check --device cuda --da3 ${CHECK_DA3:-depth-anything/DA3MONO-LARGE depth-anything/DA3METRIC-LARGE} || FAIL=1 ;;
    esac
done
echo "end $(date): $([ $FAIL = 0 ] && echo ALL OK || echo SOME CHECKS FAILED, see above)"
exit $FAIL
