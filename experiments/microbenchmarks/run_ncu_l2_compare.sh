#!/usr/bin/env bash
# Run the Orin-A L2 cache probe for device memory vs mapped zero-copy.
#
# Usage on Orin-A:
#   cd <repo>/experiments/microbenchmarks
#   /usr/local/cuda-11.4/bin/nvcc -O3 -lineinfo l2_bench_v2.cu -o l2_bench_v2
#   ./run_ncu_l2_compare.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p ../../results/orin_a/l2_cache_probe

NCU="${NCU:-/opt/nvidia/nsight-compute/2024.3.0/ncu}"
BIN="${BIN:-./l2_bench_v2}"
OUT_DIR="../../results/orin_a/l2_cache_probe"
METRICS="lts__t_sector_hit_rate.pct,lts__t_sectors.sum,lts__t_sectors_op_read.sum,dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum"

run_one () {
  local mode="$1"
  local mb="$2"
  local out="${OUT_DIR}/v2_${mode}_${mb}mb.csv"
  echo ">>> ${mode} ${mb}MB"
  sudo "$NCU" \
    --replay-mode application \
    --metrics "$METRICS" \
    --kernel-name regex:"read_kernel" \
    --launch-count 1 \
    --csv \
    "$BIN" "$mode" "$mb" > "$out" 2>&1 || true
  echo "    saved -> $out"
}

run_one device 1
run_one zerocopy 1
run_one device 64
run_one zerocopy 64

echo
echo "================================================================"
echo "  Summary parsed from CSV"
echo "================================================================"
python3 <<'PY'
import csv
import os

def parse(path):
    out = {}
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 14 and row[0].isdigit():
                out[row[12]] = (row[14], row[13])
    return out

files = [
    ("device 1MB", "../../results/orin_a/l2_cache_probe/v2_device_1mb.csv"),
    ("zerocopy 1MB", "../../results/orin_a/l2_cache_probe/v2_zerocopy_1mb.csv"),
    ("device 64MB", "../../results/orin_a/l2_cache_probe/v2_device_64mb.csv"),
    ("zerocopy 64MB", "../../results/orin_a/l2_cache_probe/v2_zerocopy_64mb.csv"),
]
keys = [
    "lts__t_sector_hit_rate.pct",
    "lts__t_sectors.sum",
    "lts__t_sectors_op_read.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "gpu__time_duration.sum",
]
short = {
    "lts__t_sector_hit_rate.pct": "L2_hit%",
    "lts__t_sectors.sum": "L2_sectors",
    "lts__t_sectors_op_read.sum": "L2_rd_sec",
    "dram__bytes_read.sum": "DRAM_rd",
    "dram__bytes_write.sum": "DRAM_wr",
    "gpu__time_duration.sum": "duration",
}

print(f"{'condition':<16} | " + " | ".join(f"{short[k]:>12}" for k in keys))
print("-" * 104)
for label, path in files:
    if not os.path.exists(path):
        print(f"{label:<16} | (no file)")
        continue
    metrics = parse(path)
    cells = []
    for key in keys:
        value, unit = metrics.get(key, ("-", "-"))
        cells.append(f"{value:>10} {unit[:2]}")
    print(f"{label:<16} | " + " | ".join(cells))
PY
