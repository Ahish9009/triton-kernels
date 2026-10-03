#!/usr/bin/env bash
#
# One-shot setup for the remote GPU server.
# Creates (or updates) the `triton-kernels` conda env, installs Triton,
# verifies the install, and runs the test suite.
#
# Usage:  ./setup.sh
#
set -euo pipefail

ENV_NAME="triton-kernels"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found on PATH. Install Miniconda/Anaconda first." >&2
  exit 1
fi

# Make `conda activate` usable inside this non-interactive script.
eval "$(conda shell.bash hook)"

# Create the env from environment.yml, or update it in place if it exists.
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo ">> Updating existing env '${ENV_NAME}' from environment.yml..."
  conda env update -n "${ENV_NAME}" -f "${SCRIPT_DIR}/environment.yml" --prune
else
  echo ">> Creating env '${ENV_NAME}' from environment.yml..."
  conda env create -f "${SCRIPT_DIR}/environment.yml"
fi

conda activate "${ENV_NAME}"

# Triton is GPU/Linux-only, so it lives here rather than in environment.yml.
echo ">> Installing Triton..."
pip install --upgrade triton

echo ">> Verifying install..."
python - <<'PY'
import torch, triton
print(f"  torch   {torch.__version__}")
print(f"  triton  {triton.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("  WARNING: no CUDA device detected — kernels won't run here.")
PY

echo ">> Running tests..."
pytest -q

echo ""
echo "Setup complete. Activate the env with:  conda activate ${ENV_NAME}"
