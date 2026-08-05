#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# run-experiment.sh — train on a small, focused slice and measure it
#
# WHY
#   The full set is 646K samples and takes many hours per attempt. When you are
#   debugging *whether the pipeline works at all*, that feedback loop is far too
#   slow — and a bad result tells you nothing about which of the dozen moving
#   parts was responsible.
#
#   This trains on ONE TITLE across a chosen set of fonts, typically ~15K
#   samples, which converges in a fraction of the time. If the model cannot
#   learn one book in three fonts, more data will not help; if it can, you have
#   a working baseline to widen from.
#
# USAGE
#   ./scripts/run-experiment.sh                                  # default: harischandrakavya, 3 revival fonts
#   ./scripts/run-experiment.sh kavirajamarga                    # another title
#   ./scripts/run-experiment.sh harischandrakavya "kan_gmp"      # single font
#   ITERATIONS=20000 ./scripts/run-experiment.sh                 # shorter/longer run
#
# Output goes to output/exp_<title>/ so it never disturbs the main run.
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$SCRIPT_DIR/.."
cd "$ROOT"

TITLE="${1:-harischandrakavya}"
FONTS="${2:-kan_gmp kan_wmp kan_kittel}"
ITERATIONS="${ITERATIONS:-15000}"

EXP_DIR="$ROOT/output/exp_${TITLE}"
EXP_LIST="$EXP_DIR/list.txt"
mkdir -p "$EXP_DIR"

hr() { printf '━%.0s' $(seq 1 60); echo; }
hr
echo "  Focused experiment"
echo "  title      : $TITLE"
echo "  fonts      : $FONTS"
echo "  iterations : $ITERATIONS"
echo "  output     : output/exp_${TITLE}/   (main run untouched)"
hr

[ -f lstmf/list.txt ] || { echo "✗ lstmf/list.txt not found — run ④ Make lstmf first."; exit 1; }

# ── Build the focused list ───────────────────────────────────────────────────
# Paths look like  lstmf/classical/<title>__<font>_<style>/pageNNNN_lineNNN.lstmf
PATTERN=""
for f in $FONTS; do
  PATTERN="${PATTERN}${PATTERN:+|}${TITLE}__${f}_"
done
grep -E "$PATTERN" lstmf/list.txt > "$EXP_LIST" || true

# Always include the character inventory: it is what teaches the individual
# glyph shapes, and at ~14K samples it would otherwise be swamped even here.
grep "/inventory/" lstmf/list.txt >> "$EXP_LIST" || true

TOTAL=$(wc -l < "$EXP_LIST" | tr -d ' ')
CLASSICAL=$(grep -cE "$PATTERN" "$EXP_LIST" || true)
INV=$(grep -c "/inventory/" "$EXP_LIST" || true)

echo ""
echo "  training set:"
printf "    classical (%s)  %6s\n" "$TITLE" "$CLASSICAL"
printf "    inventory        %6s\n" "$INV"
printf "    TOTAL            %6s\n" "$TOTAL"

if [ "$TOTAL" -lt 100 ]; then
  echo ""
  echo "✗ Only $TOTAL samples matched. Check the title name against:"
  ls classical-corpus-kannada/a5-pages/ | sed 's/^/     /'
  exit 1
fi

# ── Train ────────────────────────────────────────────────────────────────────
TESSDATA="$ROOT/tessdata_expanded"
[ -f "$TESSDATA/kan.traineddata" ] || TESSDATA="$ROOT/tessdata_best"
BASE="$ROOT/output/kan.lstm"
[ -f "$BASE" ] || { echo "✗ output/kan.lstm not found — run ① Prep base first."; exit 1; }

OLD_TD=""
if [ "$TESSDATA" != "$ROOT/tessdata_best" ]; then
  # Continuing from base weights into an expanded unicharset needs the original
  # traineddata so lstmtraining can remap the output layer.
  OLD_TD="--old_traineddata $ROOT/tessdata_best/kan.traineddata"
fi

echo ""
hr
echo "  Training — logs to logs/experiment_${TITLE}.log"
hr
lstmtraining \
  --traineddata   "$TESSDATA/kan.traineddata" \
  $OLD_TD \
  --model_output  "$EXP_DIR/exp" \
  --continue_from "$BASE" \
  --train_listfile "$EXP_LIST" \
  --max_iterations "$ITERATIONS" \
  --target_error_rate -1 \
  2>&1 | tee "logs/experiment_${TITLE}.log" | grep -E "At iteration|Error|error rate|wrote"

# ── Package and measure ──────────────────────────────────────────────────────
BEST_CP=$(ls -t "$EXP_DIR"/exp*.checkpoint 2>/dev/null | head -1 || true)
[ -n "$BEST_CP" ] || { echo "✗ no checkpoint produced"; exit 1; }

echo ""
hr
echo "  Packaging $(basename "$BEST_CP")"
hr
lstmtraining --stop_training \
  --continue_from "$BEST_CP" \
  --traineddata   "$TESSDATA/kan.traineddata" \
  --model_output  "$EXP_DIR/kan_exp.traineddata" 2>&1 | tail -3

echo ""
echo "  Model: output/exp_${TITLE}/kan_exp.traineddata"
echo ""
echo "  Measure it — the number that matters is on REAL scans:"
echo "     cp output/exp_${TITLE}/kan_exp.traineddata /tmp/ && \\"
echo "     tesseract scan-input/<a-scan>.png stdout --tessdata-dir /tmp -l kan_exp --psm 6"
echo ""
echo "  Or against the training slice, to confirm it learned anything at all:"
echo "     python3 corpus/verify-ocr.py --source classical --count 20"
hr
