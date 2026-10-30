#!/usr/bin/env bash
# One-time setup for the 3D helper (run from Git Bash in this folder): a Python 3.12 venv, PyTorch for this PC's
# GPU, MapAnything from Meta's GitHub, and both MapAnything checkpoints from Hugging Face (4.9 GB each).
set -euo pipefail
cd "$(dirname "$0")"

[ -d vendor/map-anything ] || git clone --depth 1 https://github.com/facebookresearch/map-anything.git vendor/map-anything
[ -d .venv ] || uv venv .venv --python 3.12
PY=".venv/Scripts/python.exe"

uv pip install --python "$PY" torch torchvision --torch-backend=auto
uv pip install --python "$PY" -e vendor/map-anything --torch-backend=auto
uv pip install --python "$PY" -r requirements.txt

"$PY" -c "import torch; print('torch', torch.__version__, 'GPU:', torch.cuda.is_available())"
"$PY" -c "
from huggingface_hub import snapshot_download
for repo in ('facebook/map-anything-apache', 'facebook/map-anything'):
    print(repo, '->', snapshot_download(repo))
"
