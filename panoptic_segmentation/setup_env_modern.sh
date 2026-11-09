#!/usr/bin/env bash
# Create the modernized MM-OR panoptic-segmentation environment (Python 3.12, torch 2.9.1 + CUDA 12.8).
#
#   cd panoptic_segmentation
#   ./setup_env_modern.sh                 # venv in ./.venv-panoptic
#   ./setup_env_modern.sh --env /path/to/venv --locked
#
# Options:
#   --env DIR           venv location (default: .venv-panoptic next to this script)
#   --python EXE        Python 3.12 interpreter used to create the venv (default: python3.12)
#   --locked            install the exact verified package set (requirements-modern/lock.txt)
#                       instead of the top-level pins in requirements-modern/requirements.txt
#   --cuda-home DIR     CUDA toolkit used to compile (default: $CUDA_HOME, else /usr/local/cuda)
#
# Requirements: Linux x86_64, an NVIDIA driver supporting CUDA >= 12.8, Python 3.12, and a
# CUDA 12.x toolkit (nvcc, 12.8 or newer for Blackwell) with a matching host compiler:
# detectron2 and the MSDeformAttn op are compiled from source.
# Compilation is limited by MAX_JOBS (default 4) and targets TORCH_CUDA_ARCH_LIST (default: the
# GPUs visible during the build), e.g. TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0".
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ="$HERE/requirements-modern"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
DETECTRON2="git+https://github.com/facebookresearch/detectron2.git@247691a1c292654867deb35cabb27f18d75a1363"
PANOPTICAPI="git+https://github.com/cocodataset/panopticapi.git@7bb4655548f98f3fedc07bf37e9040a992b054b0"

ENV_DIR="$HERE/.venv-panoptic"; PYTHON=python3.12; LOCKED=0; CUDA_DIR="${CUDA_HOME:-/usr/local/cuda}"
while [ $# -gt 0 ]; do
    case "$1" in
        --env)       ENV_DIR="$2"; shift 2 ;;
        --python)    PYTHON="$2"; shift 2 ;;
        --locked)    LOCKED=1; shift ;;
        --cuda-home) CUDA_DIR="$2"; shift 2 ;;
        -h|--help)   sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

export CUDA_HOME="$CUDA_DIR" PATH="$CUDA_DIR/bin:$PATH" MAX_JOBS="${MAX_JOBS:-4}"
NVCC_VERSION="$("$CUDA_HOME/bin/nvcc" --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p')" || true
case "$NVCC_VERSION" in
    12.*) echo ">>> nvcc $NVCC_VERSION from $CUDA_HOME" ;;
    *) echo "ERROR: need a CUDA 12.x toolkit (found '${NVCC_VERSION:-none}' in $CUDA_HOME; pass --cuda-home)" >&2; exit 1 ;;
esac

if [ ! -x "$ENV_DIR/bin/python" ]; then
    "$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' || {
        echo "ERROR: need Python 3.12 (pass --python /path/to/python3.12)" >&2; exit 1; }
    echo ">>> creating venv $ENV_DIR"
    "$PYTHON" -m venv "$ENV_DIR"
fi
PY="$ENV_DIR/bin/python"
"$PY" -m pip install -q --upgrade pip

if [ $LOCKED -eq 1 ]; then
    echo ">>> [1/4] exact verified package set (lock.txt)"
    "$PY" -m pip install --extra-index-url "$TORCH_INDEX" -r "$REQ/lock.txt"
else
    echo ">>> [1/4] torch 2.9.1 + cu128"
    "$PY" -m pip install --index-url "$TORCH_INDEX" -r "$REQ/torch.txt"
    echo ">>> [1/4] remaining packages"
    "$PY" -m pip install -r "$REQ/requirements.txt" -c "$REQ/torch.txt"
fi

# detectron2 and the op are compiled against the installed torch, hence --no-build-isolation.
echo ">>> [2/4] detectron2 (pinned commit, compiled; MAX_JOBS=$MAX_JOBS)"
"$PY" -c 'import detectron2' 2>/dev/null || "$PY" -m pip install --no-build-isolation "$DETECTRON2"

echo ">>> [3/4] panopticapi"
"$PY" -c 'import panopticapi' 2>/dev/null || "$PY" -m pip install --no-build-isolation "$PANOPTICAPI"

echo ">>> [4/4] MSDeformAttn CUDA op (Mask2Former), built from a copy so the source tree stays clean"
if ! "$PY" -c 'import torch, MultiScaleDeformableAttention' 2>/dev/null; then
    BUILD="$(mktemp -d)"
    cp -r "$HERE/mask2former/modeling/pixel_decoder/ops/." "$BUILD/"
    (cd "$BUILD" && "$PY" -m pip install --no-build-isolation .)
    rm -rf "$BUILD"
fi

echo ">>> checking the install"
"$PY" - <<'EOF'
import torch, detectron2, panopticapi, numpy, PIL
import MultiScaleDeformableAttention  # noqa: F401
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}), detectron2 {detectron2.__version__}, "
      f"numpy {numpy.__version__}, Pillow {PIL.__version__}, MSDeformAttn built")
print(f"CUDA available: {torch.cuda.is_available()}"
      + (f", {torch.cuda.device_count()} GPU(s): {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
EOF
echo ">>> done. Activate with: source $ENV_DIR/bin/activate"
