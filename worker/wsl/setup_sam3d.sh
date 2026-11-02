#!/usr/bin/env bash
# SAM 3D Objects in WSL on the RTX 3070, following its doc/setup.md (conda env from environments/default.yml: Python
# 3.11, CUDA 12.1, gcc 12), minus the dead pypi.ngc.nvidia.com index and nvidia-pyindex, as on Modal.
set -euo pipefail
COMMIT=f91db411c50efee93d8db7aeb323885650f6f722
ROOT=~/sam3d
mkdir -p "$ROOT" && cd "$ROOT"
echo "== $(date) micromamba"
command -v bzip2 >/dev/null || (apt-get update -qq && apt-get install -y -qq bzip2)
# system libraries some of its imports load (OpenGL and X), as in the Modal image
apt-get install -y -qq libgl1 libglib2.0-0 libegl1 libxrender1 libxi6 libxkbcommon0 libsm6 libxext6 libxfixes3 libxxf86vm1
if [ ! -x bin/micromamba ]; then
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba
fi
export MAMBA_ROOT_PREFIX="$ROOT/mamba"
echo "== $(date) clone"
if [ ! -d sam-3d-objects ]; then
  git clone -q https://github.com/facebookresearch/sam-3d-objects
fi
cd sam-3d-objects
git checkout -q "$COMMIT"
sed -i '/^nvidia-pyindex/d' requirements.txt
echo "== $(date) conda env"
if [ ! -d "$MAMBA_ROOT_PREFIX/envs/sam3d" ]; then
  "$ROOT/bin/micromamba" create -y -q -n sam3d -f environments/default.yml
fi
eval "$("$ROOT/bin/micromamba" shell hook -s bash)"
set +u  # conda's activate scripts read unset variables
micromamba activate sam3d
export CUDA_HOME="$CONDA_PREFIX" FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.6" MAX_JOBS=8 PIP_NO_CACHE_DIR=1
export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu121"
echo "== $(date) requirements"
pip install -q -e '.[dev]'
pip install -q hatchling hatch-requirements-txt editables
echo "== $(date) pytorch3d + flash-attn"
pip install -q --no-build-isolation -r requirements.p3d.txt
echo "== $(date) kaolin + gsplat"
PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html" \
  pip install -q --no-build-isolation -r requirements.inference.txt
python patching/hydra
pip install -q 'huggingface-hub[cli]<1.0' 'setuptools<81'
echo "== $(date) check"
cd notebook && python - <<'EOF'
import os
os.environ.setdefault("LIDRA_SKIP_INIT", "true")
import torch
import inference  # noqa: F401
print("imports ok; torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
EOF
echo "== $(date) done"
