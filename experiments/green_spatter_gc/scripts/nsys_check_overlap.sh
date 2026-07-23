#!/usr/bin/env bash
# Verifies with nsys what the wall-clock numbers only imply:
#   1. whether the kernels actually overlap
#   2. how long each kernel takes, split by which buffer it read
#
# Usage:  scripts/nsys_check_overlap.sh [outdir]
#
# For single_stream_path_mix the expected answer is ZERO overlap (one stream is
# ordered), which is the point: if zero-copy still wins there, the win cannot
# come from concurrency.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="${1:-/tmp/nsys_overlap}"
mkdir -p "$OUT"

for only in 2 3; do
  nsys profile -t cuda -o "$OUT/mix_only$only" --force-overwrite true \
    ./single_stream_path_mix --mb=1 --iters=1 --chunks=5 --trials=3 \
    --pat=uniform --only=$only >/dev/null 2>&1
  nsys export --type sqlite --force-overwrite true \
    -o "$OUT/mix_only$only.sqlite" "$OUT/mix_only$only.nsys-rep" >/dev/null 2>&1
done

python3 - "$OUT" <<'PY'
import sqlite3, statistics, sys
out = sys.argv[1]
for only, label in [(2, 'case2  dev + dev'), (3, 'case3  dev + ZC ')]:
    rows = sqlite3.connect(f'{out}/mix_only{only}.sqlite').execute(
        "SELECT start, end, streamId FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"
    ).fetchall()
    overlaps = sum(1 for i in range(1, len(rows)) if rows[i][0] < rows[i-1][1])
    streams = len({r[2] for r in rows})
    # launches alternate A,B,A,B; drop the 5 warm-up pairs
    dur = [(e - s) / 1000 for s, e, _ in rows][10:]
    a, b = dur[0::2], dur[1::2]
    gaps = [(rows[i][0] - rows[i-1][1]) / 1000 for i in range(1, len(rows))]
    print(f"{label} | kernels={len(rows)}  streams={streams}  overlapping pairs={overlaps}")
    print(f"   A (device)      median {statistics.median(a):6.2f} us")
    print(f"   B (2nd buffer)  median {statistics.median(b):6.2f} us")
    print(f"   A+B pair        median {statistics.median(a)+statistics.median(b):6.2f} us")
    print(f"   gap between     median {statistics.median(gaps):6.2f} us\n")
PY
