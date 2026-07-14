#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="${ROOT_DIR}/third_party/dcnv4"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "DCNv4 must be built on the Linux training server." >&2
  exit 1
fi

if [[ -z "${CUDA_HOME:-}" ]]; then
  if [[ -x /usr/local/cuda-12.4/bin/nvcc ]]; then
    export CUDA_HOME=/usr/local/cuda-12.4
  else
    echo "Set CUDA_HOME to the CUDA Toolkit 12.4 installation directory." >&2
    exit 1
  fi
fi

if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "nvcc was not found at ${CUDA_HOME}/bin/nvcc." >&2
  exit 1
fi

NVCC_VERSION="$(${CUDA_HOME}/bin/nvcc --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')"
if [[ "${NVCC_VERSION}" != "12.4" ]]; then
  echo "Expected nvcc 12.4, found ${NVCC_VERSION:-unknown}." >&2
  exit 1
fi

python - <<'PY'
import sys
import torch

errors = []
if torch.__version__.split('+', 1)[0] != '2.5.1':
    errors.append(f"expected torch 2.5.1, found {torch.__version__}")
if torch.version.cuda != '12.4':
    errors.append(f"expected torch CUDA 12.4, found {torch.version.cuda}")
if not torch.cuda.is_available():
    errors.append("torch.cuda.is_available() is False")

if errors:
    print("DCNv4 environment check failed:", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    raise SystemExit(1)

print(f"PyTorch: {torch.__version__}")
print(f"CUDA runtime: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

python -c "import ninja" 2>/dev/null || {
  echo "ninja is missing. Install project dependencies before building DCNv4." >&2
  exit 1
}

python -m pip install --no-build-isolation -v -e "${VENDOR_DIR}"
python "${ROOT_DIR}/scripts/test_dcnv4.py" --resolution 64 --steps 1

echo "DCNv4 build and smoke test completed successfully."
