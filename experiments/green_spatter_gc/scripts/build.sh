#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
NVCC="${NVCC:-$CUDA_HOME/bin/nvcc}"

if [[ ! -x "$NVCC" ]]; then
  echo "nvcc not found at $NVCC" >&2
  exit 1
fi

"$NVCC" -O3 -lineinfo --ptx src/spatter_gather.cu -o spatter_gather.ptx
g++ -O3 -std=c++17 src/green_spatter_split.cpp \
  -I"$CUDA_HOME/include" -L"$CUDA_HOME/lib64" -lcuda \
  -o green_spatter_split

g++ -O3 -std=c++17 src/query_sm_count.cpp \
  -I"$CUDA_HOME/include" -L"$CUDA_HOME/lib64" -lcuda \
  -o query_sm_count

if [[ -f src/cublas_gemm_split.cpp ]]; then
  g++ -O3 -std=c++17 src/cublas_gemm_split.cpp \
    -I"$CUDA_HOME/include" -L"$CUDA_HOME/lib64" -lcuda -lcublas -lcudart \
    -o cublas_gemm_split
fi

echo "Build complete."
