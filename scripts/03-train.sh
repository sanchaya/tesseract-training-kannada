#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# 03-train.sh
#
# Fine-tunes kan.traineddata on Karnata historical fonts to
# produce kan_hist — a model for historical Kannada letterpress.
#
# What to expect:
#   Iters    0–1000:  BCER ~5–15% (adjusting to new letterforms)
#   Iters 1000–5000:  BCER drops steadily
#   Iters 5000+:      Plateau toward 1–3% range
#   Stop when BCER shows no new best for ~10,000 iterations.
#
# Checkpoints are saved every 100 iterations.
# Safe to stop with Ctrl+C — resume by re-running this script.
#
# Usage:
#   caffeinate -i ./scripts/03-train.sh > training.log 2>&1 &
#   tail -f training.log
#
# To resume from a specific checkpoint:
#   CONTINUE_FROM=output/kan_hist_2.1_50000.checkpoint ./scripts/03-train.sh
# ═══════════════════════════════════════════════════════════════
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$SCRIPT_DIR/.."
cd "$ROOT"

TESSDATA_BEST="$ROOT/tessdata_best"
OUTPUT="$ROOT/output"
LSTMF_DIR="$ROOT/lstmf"
MODEL_NAME="kan_hist"

[ -f "$LSTMF_DIR/list.txt" ] || {
    echo "ERROR: lstmf/list.txt not found. Run 02-make-lstmf.sh first."
    exit 1
}
LSTMF_COUNT=$(wc -l < "$LSTMF_DIR/list.txt" | tr -d ' ')
[ "$LSTMF_COUNT" -gt 0 ] || {
    echo "ERROR: lstmf/list.txt is empty."
    exit 1
}

MAX_ITERATIONS=100000
LEARNING_RATE=0.001

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Step 3 — LSTM Fine-tuning (kan_hist)"
echo "  Base:           output/kan.lstm (from tessdata_best)"
echo "  Training data:  $LSTMF_COUNT .lstmf files"
echo "  Iterations:     $MAX_ITERATIONS max"
echo "  Learning rate:  $LEARNING_RATE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── Checkpoint selection ───────────────────────────────────────
# Resume from the most recent checkpoint if available.
# Otherwise start from the extracted kan.lstm base.
if [ -z "${CONTINUE_FROM:-}" ]; then
    # Look for named checkpoints (e.g. kan_hist_1.23_50000.checkpoint)
    RECENT=$(ls "$OUTPUT"/${MODEL_NAME}_*.checkpoint 2>/dev/null | \
             grep -v '_checkpoint$' | sort -t_ -k3 -n | tail -1 || true)
    if [ -n "$RECENT" ]; then
        CONTINUE_FROM="$RECENT"
        echo "→ Resuming from: $(basename $CONTINUE_FROM)"
    elif [ -f "$OUTPUT/${MODEL_NAME}_checkpoint" ]; then
        CONTINUE_FROM="$OUTPUT/${MODEL_NAME}_checkpoint"
        echo "→ Resuming from rolling checkpoint"
    elif [ -f "$OUTPUT/kan.lstm" ]; then
        CONTINUE_FROM="$OUTPUT/kan.lstm"
        echo "→ Starting fresh from kan.lstm"
    else
        echo "ERROR: output/kan.lstm not found. Run 01-prep-base.sh first."
        exit 1
    fi
else
    echo "→ Using explicit checkpoint: $(basename $CONTINUE_FROM)"
fi

echo ""
echo "  Started: $(date)"
echo ""

lstmtraining \
    --continue_from   "$CONTINUE_FROM" \
    --model_output    "$OUTPUT/$MODEL_NAME" \
    --traineddata     "$TESSDATA_BEST/kan.traineddata" \
    --train_listfile  "$LSTMF_DIR/list.txt" \
    --learning_rate   "$LEARNING_RATE" \
    --max_iterations  "$MAX_ITERATIONS" \
    --target_error_rate -1

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Training done."
echo "  Finished: $(date)"
echo "  NEXT: ./scripts/04-package.sh"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
