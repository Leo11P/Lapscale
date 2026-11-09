#!/bin/bash
# Pre-download the weights into the shared caches, on the LOGIN node (internet, no torch import).
# Jobs then run with HF_HUB_OFFLINE=1.
#   bash slurm/download_weights.sh                 # default set
#   bash slurm/download_weights.sh Ruicheng/moge-3-vitg depth-anything/DA3NESTED-GIANT-LARGE-1.1   # extra repos
# SAM 3 is gated: request access on huggingface.co/facebook/sam3, then once: .venv-sam3/bin/hf auth login
set -euo pipefail
cd "$(dirname "$0")/.."
source slurm/env.sh
export HF_HUB_OFFLINE=0
PY=.venv/bin/python
[ -x "$PY" ] || { echo "build the main environment first: bash setup_env.sh"; exit 1; }
REPOS=(Ruicheng/moge-3-vitl Ruicheng/moge-2-vitl Ruicheng/moge-2-vits-normal Ruicheng/moge-2-vitb-normal
       depth-anything/DA3MONO-LARGE depth-anything/DA3METRIC-LARGE facebook/sam3 "$@")
"$PY" - "${REPOS[@]}" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
for repo in sys.argv[1:]:
    try:
        path = snapshot_download(repo)
        print(f"OK    {repo} -> {path}")
    except Exception as e:                      # gated / missing: report and continue
        print(f"FAIL  {repo}: {type(e).__name__}: {str(e).splitlines()[0][:160]}")
PYEOF
mkdir -p "$LAPSCALE_WEIGHTS/surgenet"
SN=$LAPSCALE_WEIGHTS/surgenet/SurgeNetXL_checkpoint_epoch0050_teacher.pth
if [ ! -f "$SN" ]; then
    curl -L --fail -o "$SN.part" \
        "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/SurgeNetXL_checkpoint_epoch0050_teacher.pth?download=true"
    mv "$SN.part" "$SN"
fi
ls -lh "$SN"
echo "HF cache: $HF_HOME ($(du -sh "$HF_HOME" | cut -f1))"
