#!/usr/bin/env bash
# Create the modernized MM-OR scene-graph environment (Python 3.12, torch 2.9.1 + CUDA 12.8).
#
#   cd scene_graph_generation
#   ./setup_env_modern.sh                 # venv in ./.venv-mmor
#   ./setup_env_modern.sh --env /path/to/venv --locked
#
# Options:
#   --env DIR           venv location (default: .venv-mmor next to this script)
#   --python EXE        Python 3.12 interpreter used to create the venv (default: python3.12)
#   --locked            install the exact verified package set (requirements-modern/lock.txt)
#                       instead of the top-level pins in requirements.txt
#   --no-flash-attn     skip flash-attn. PointTransformerV3 then falls back to a different
#                       attention grouping, which changes the numerics: published metrics were
#                       verified WITH flash-attn.
#   --no-ptv3-init      skip creating LLaVA/.../pointtransformerv3_weights.pth (needs a GPU)
#
# Requirements: Linux x86_64, an NVIDIA driver supporting CUDA >= 12.8, Python 3.12.
# No CUDA toolkit or compiler is needed: everything installs from prebuilt wheels.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ="$HERE/requirements-modern"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
FLASH_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

ENV_DIR="$HERE/.venv-mmor"; PYTHON=python3.12; LOCKED=0; FLASH=1; PTV3_INIT=1
while [ $# -gt 0 ]; do
    case "$1" in
        --env)           ENV_DIR="$2"; shift 2 ;;
        --python)        PYTHON="$2"; shift 2 ;;
        --locked)        LOCKED=1; shift ;;
        --no-flash-attn) FLASH=0; shift ;;
        --no-ptv3-init)  PTV3_INIT=0; shift ;;
        -h|--help)       sed -n '2,19p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

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
    "$PY" -m pip install -r "$HERE/requirements.txt"
fi

if [ $FLASH -eq 1 ]; then
    echo ">>> [2/4] flash-attn 2.8.3 (prebuilt release wheel, no compilation)"
    "$PY" -m pip install --no-deps "$FLASH_WHEEL"
else
    echo ">>> [2/4] skipping flash-attn"
fi

echo ">>> [3/4] LLaVA (editable, --no-deps: versions are owned by requirements.txt)"
"$PY" -m pip install --no-deps -e "$HERE/LLaVA"

PTV3_W="$HERE/LLaVA/llava/model/multimodal_projector/pointtransformerv3_weights.pth"
if [ -f "$PTV3_W" ]; then
    echo ">>> [4/4] PointTransformerV3 init weights already present"
elif [ $PTV3_INIT -eq 1 ]; then
    # Building the model asserts that this file exists, for evaluation as well as training
    # (the checkpoint's own weights replace it afterwards). It is created once, on a GPU.
    echo ">>> [4/4] creating PointTransformerV3 init weights"
    (cd "$HERE/LLaVA" && "$PY" -m llava.model.multimodal_projector.pointtransformerv3 > /dev/null)
    [ -f "$PTV3_W" ] && echo "    wrote $PTV3_W"
else
    echo ">>> [4/4] skipped. Before the first run, on a GPU machine:"
    echo "    cd LLaVA && $PY -m llava.model.multimodal_projector.pointtransformerv3"
fi

echo ">>> checking the install"
"$PY" - <<'EOF'
import torch, transformers, peft, bitsandbytes, spconv, llava
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}), transformers {transformers.__version__}, "
      f"peft {peft.__version__}, bitsandbytes {bitsandbytes.__version__}")
try:
    import flash_attn; print(f"flash-attn {flash_attn.__version__}")
except ImportError:
    print("flash-attn: not installed")
print(f"CUDA available: {torch.cuda.is_available()}"
      + (f", {torch.cuda.device_count()} GPU(s): {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
EOF
echo ">>> done. Activate with: source $ENV_DIR/bin/activate"
