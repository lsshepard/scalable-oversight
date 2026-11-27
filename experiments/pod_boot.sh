#!/usr/bin/env bash
# Runs on a rented GPU pod at start (see experiments/runpod_driver.py): fetch this branch of the
# repository, install what the training needs, start the job server on port 8000.
# Usage: pod_boot.sh <branch>
set -u
BRANCH=${1:-claude/sweet-brown-bx2roi}
cd /workspace 2>/dev/null || cd /root
rm -rf so
git clone --depth 1 -b "$BRANCH" https://github.com/lsshepard/scalable-oversight.git so || exit 1
cd so
pip install -q transformers peft accelerate scipy safetensors scikit-learn matplotlib > /workspace/pip.log 2>&1
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
PY
exec python experiments/pod_server.py 8000
