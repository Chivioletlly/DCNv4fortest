#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="${ROOT_DIR}/third_party/dcnv4"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "DCNv4 must be built on the Linux training server." >&2
  exit 1
fi

if [[ -z "${CUDA_HOME:-}" || ! -x "${CUDA_HOME:-}/bin/nvcc" ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    NVCC_PATH="$(readlink -f "$(command -v nvcc)")"
    DETECTED_CUDA_HOME="$(dirname "$(dirname "${NVCC_PATH}")")"
    if [[ -n "${CUDA_HOME:-}" ]]; then
      echo "CUDA_HOME=${CUDA_HOME} does not contain nvcc; using ${DETECTED_CUDA_HOME}." >&2
    fi
    export CUDA_HOME="${DETECTED_CUDA_HOME}"
  elif [[ -x /usr/local/cuda-12.4/bin/nvcc ]]; then
    export CUDA_HOME=/usr/local/cuda-12.4
  elif [[ -x /usr/local/cuda-12.1/bin/nvcc ]]; then
    export CUDA_HOME=/usr/local/cuda-12.1
  else
    echo "Could not find nvcc. Install CUDA Toolkit 12.4, or set CUDA_HOME to a CUDA 12.1/12.4 toolkit." >&2
    exit 1
  fi
fi

NVCC_VERSION="$(${CUDA_HOME}/bin/nvcc --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')"
if [[ "${NVCC_VERSION}" != "12.4" && "${NVCC_VERSION}" != "12.1" ]]; then
  echo "Expected nvcc 12.4 or the CUDA-minor-compatible 12.1 fallback, found ${NVCC_VERSION:-unknown}." >&2
  exit 1
fi

if [[ "${NVCC_VERSION}" == "12.1" ]]; then
  echo "Warning: building PyTorch cu124 DCNv4 with nvcc 12.1; this is an unverified CUDA minor-version fallback." >&2
  echo "The smoke test below must pass before training." >&2
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

python - <<'PY'
import importlib.util
import sys

missing = [name for name in ("setuptools",) if importlib.util.find_spec(name) is None]
if missing:
    print(
        "Missing Python build tools: " + ", ".join(missing) +
        ". Install them in the active environment before building DCNv4.",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY

for compiler in gcc g++; do
  if ! command -v "${compiler}" >/dev/null 2>&1; then
    echo "${compiler} is missing. Install a Linux C/C++ build toolchain before building DCNv4." >&2
    exit 1
  fi
done

# Build through the upstream setup.py entry point. The editable pip path starts a
# nested PEP 517 environment without torch, while the wheel path can fail metadata
# verification for this legacy extension package. Direct setup.py installation uses
# the active torch environment and does not contact a package index.
export MAX_JOBS="${MAX_JOBS:-4}"
echo "Building DCNv4 with MAX_JOBS=${MAX_JOBS}, CUDA_HOME=${CUDA_HOME}."
(
  cd "${VENDOR_DIR}"
  python setup.py build install
)

python -c "from DCNv4.modules.dcnv4 import DCNv4; print('DCNv4 import test passed.')"
python "${ROOT_DIR}/scripts/test_dcnv4.py" --resolution 64 --steps 1

echo "DCNv4 build and smoke test completed successfully."
