#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# sweep-checkpoints.sh — find the checkpoint that GENERALISES best
#
# WHY
#   Training BCER falls monotonically; real-world accuracy does not. Past some
#   point the model stops learning Kannada and starts memorising these specific
#   synthetic renders, and real-scan CER climbs while BCER keeps improving.
#   Measured 2026-08-05: the BCER-0.000 checkpoint scored 44.3% CER on real
#   scans against stock Tesseract's 19.5% — fine-tuning had made it WORSE than
#   the model it started from.
#
#   The only way to find the turning point is to package several checkpoints
#   and measure each on data the model never trained on.
#
# USAGE
#   ./scripts/sweep-checkpoints.sh                # sweep 5 checkpoints
#   ./scripts/sweep-checkpoints.sh 8              # sweep 8
#   SOURCE=classical ./scripts/sweep-checkpoints.sh
#
# Restores your original best/kan_hist.traineddata when finished.
# ═══════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)/.."

N="${1:-5}"
SOURCE="${SOURCE:-scan}"
COUNT="${COUNT:-2}"
TD=tessdata_expanded/kan.traineddata
[ -f "$TD" ] || TD=tessdata_best/kan.traineddata

# Preserve whatever is currently installed.
BACKUP=/tmp/kan_hist_backup_$$.traineddata
[ -f best/kan_hist.traineddata ] && cp best/kan_hist.traineddata "$BACKUP"
restore() {
  if [ -f "$BACKUP" ]; then
    cp "$BACKUP" best/kan_hist.traineddata
    cp "$BACKUP" tessdata_best/kan_hist.traineddata
    rm -f "$BACKUP"
    echo "  (restored your original best/kan_hist.traineddata)"
  fi
}
trap restore EXIT

# Pick N checkpoints spread across the iteration range, so the sweep covers the
# whole training trajectory rather than clustering at one end.
mapfile -t ALL < <(ls output/kan_hist_[0-9]*.checkpoint 2>/dev/null \
  | awk -F_ '{it=$NF; sub(/\.checkpoint/,"",it); print it+0, $0}' | sort -n | awk '{print $2}')
TOTAL=${#ALL[@]}
[ "$TOTAL" -gt 0 ] || { echo "✗ no checkpoints in output/"; exit 1; }

PICKS=()
if [ "$TOTAL" -le "$N" ]; then
  PICKS=("${ALL[@]}")
else
  for i in $(seq 0 $((N-1))); do
    idx=$(( i * (TOTAL-1) / (N-1) ))
    PICKS+=("${ALL[$idx]}")
  done
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Checkpoint sweep — measuring generalisation, not training error"
echo "  source: $SOURCE   samples: $COUNT   checkpoints: ${#PICKS[@]} of $TOTAL"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
printf "  %-46s %-10s %8s %8s\n" "checkpoint" "train BCER" "CER" "WER"

for cp in "${PICKS[@]}"; do
  name=$(basename "$cp")
  bcer=$(echo "$name" | awk -F_ '{print $3}')

  lstmtraining --stop_training --continue_from "$cp" \
    --traineddata "$TD" --model_output /tmp/sweep.traineddata >/dev/null 2>&1 || {
      printf "  %-46s %-10s %8s %8s\n" "$name" "$bcer" "export✗" "-"; continue; }

  cp /tmp/sweep.traineddata best/kan_hist.traineddata
  cp /tmp/sweep.traineddata tessdata_best/kan_hist.traineddata

  out=$(python3 corpus/verify-ocr.py --source "$SOURCE" --count "$COUNT" \
        --out /tmp/sweep_report.html 2>/dev/null | grep -E "^\s+kan_hist" || true)
  cer=$(echo "$out" | grep -oE "CER [0-9.]+%" | head -1 | tr -d 'CER %')
  wer=$(echo "$out" | grep -oE "WER [0-9.]+%" | head -1 | tr -d 'WER %')
  printf "  %-46s %-10s %7s%% %7s%%\n" "$name" "$bcer" "${cer:-?}" "${wer:-?}"
done

echo ""
echo "  Compare against the baseline you must beat:"
echo "     stock tessdata_best on $SOURCE — run verify-ocr and read the first row."
echo ""
echo "  If CER RISES as training BCER falls, the model is overfitting to the"
echo "  synthetic renders. Package the checkpoint at the turning point, and"
echo "  consider stopping training there in future runs."
