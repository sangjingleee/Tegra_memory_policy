#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TRIALS="${TRIALS:-51}"
SIZES="${SIZES:-512 1024 2048 4096}"
HOST="$(hostname)"
OUTDIR="${OUTDIR:-$ROOT/results/$HOST}"
mkdir -p "$OUTDIR"

sudo -n /usr/bin/jetson_clocks >/dev/null 2>&1 || {
  echo "warning: sudo -n /usr/bin/jetson_clocks failed; clocks may not be fixed" >&2
}

bash "$ROOT/scripts/build.sh"
bash "$ROOT/scripts/collect_platform.sh" "$OUTDIR"

OUT="$OUTDIR/cublas_gemm.csv"
TMP="$OUT.tmp"
rm -f "$TMP"
first=1

for n in $SIZES; do
  line="$(./cublas_gemm_split --n="$n" --trials="$TRIALS")"
  if [[ "$first" -eq 1 ]]; then
    echo "$line" | head -n 1 > "$TMP"
    first=0
  fi
  echo "$line" | tail -n 1 >> "$TMP"
  echo "done GEMM n=$n"
done

mv "$TMP" "$OUT"
echo "Wrote $OUT"
