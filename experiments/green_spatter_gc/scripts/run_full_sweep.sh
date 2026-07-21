#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TRIALS="${TRIALS:-101}"
STRIDE="${STRIDE:-17}"
SIZES="${SIZES:-1 2 4 8 16 32 64}"
REUSES="${REUSES:-1 2 4 8 16 32}"
PATTERNS="${PATTERNS:-uniform strided scatter}"
TARGET_MB_PER_PATH="${TARGET_MB_PER_PATH:-512}"

HOST="$(hostname)"
OUTDIR="${OUTDIR:-$ROOT/results/$HOST}"
mkdir -p "$OUTDIR"

sudo -n /usr/bin/jetson_clocks >/dev/null 2>&1 || {
  echo "warning: sudo -n /usr/bin/jetson_clocks failed; clocks may not be fixed" >&2
}

bash "$ROOT/scripts/build.sh"
bash "$ROOT/scripts/collect_platform.sh" "$OUTDIR"

SM_COUNT="$(./query_sm_count)"
if [[ -z "$SM_COUNT" || "$SM_COUNT" -le 1 ]]; then
  echo "failed to infer SM count" >&2
  exit 2
fi
DEV_SMS=$((SM_COUNT / 2))
ZC_SMS=$((SM_COUNT - DEV_SMS))

OUT="$OUTDIR/spatter_dense_grid.csv"
TMP="$OUT.tmp"
rm -f "$TMP"
first=1
n=0
pattern_count=$(wc -w <<< "$PATTERNS")
size_count=$(wc -w <<< "$SIZES")
reuse_count=$(wc -w <<< "$REUSES")
total=$((pattern_count * size_count * reuse_count))

for pat in $PATTERNS; do
  for mb in $SIZES; do
    for reuse in $REUSES; do
      work=$((mb * reuse))
      batches=$(((TARGET_MB_PER_PATH + work / 2) / work))
      if [[ "$batches" -lt 1 ]]; then batches=1; fi
      if [[ "$batches" -gt 128 ]]; then batches=128; fi

      line="$(./green_spatter_split \
        --dev-sms="$DEV_SMS" --zc-sms="$ZC_SMS" \
        --mb="$mb" --iters="$reuse" --batches="$batches" \
        --trials="$TRIALS" --pat="$pat" --stride="$STRIDE")"

      if [[ "$first" -eq 1 ]]; then
        echo "$line" | head -n 1 > "$TMP"
        first=0
      fi
      echo "$line" | tail -n 1 >> "$TMP"
      n=$((n + 1))
      ratio="$(echo "$line" | tail -n 1 | cut -d, -f18)"
      echo "[$n/$total] host=$HOST split=${DEV_SMS}/${ZC_SMS} pat=$pat size=${mb}MB reuse=$reuse stride=$STRIDE ratio=${ratio}%"
    done
  done
done

mv "$TMP" "$OUT"
echo "Wrote $OUT"
