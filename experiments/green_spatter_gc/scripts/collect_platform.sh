#!/usr/bin/env bash
set -euo pipefail

OUTDIR="${1:?usage: collect_platform.sh OUTDIR}"
mkdir -p "$OUTDIR"

{
  echo "hostname=$(hostname)"
  date -Is
  uname -a
  if [[ -f /etc/nv_tegra_release ]]; then
    cat /etc/nv_tegra_release
  fi
  if command -v nvcc >/dev/null 2>&1; then
    nvcc --version
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
  fi
  python3 - <<'PY' || true
try:
    import torch
    print("torch.cuda=", torch.version.cuda)
    print("torch.cuda.is_available=", torch.cuda.is_available())
except Exception as exc:
    print("torch unavailable:", exc)
PY
} > "$OUTDIR/platform.txt"

if sudo -n /usr/bin/jetson_clocks --show > "$OUTDIR/clock_state.txt" 2>/dev/null; then
  true
else
  echo "jetson_clocks --show unavailable or sudo password required" > "$OUTDIR/clock_state.txt"
fi

cat /sys/devices/gpu.0/devfreq/*/cur_freq > "$OUTDIR/gpu_cur_freq.txt" 2>/dev/null || true
cat /sys/kernel/debug/bpmp/debug/clk/emc/rate > "$OUTDIR/emc_rate.txt" 2>/dev/null || true

