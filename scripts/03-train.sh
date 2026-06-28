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

OUTPUT="$ROOT/output"
LSTMF_DIR="$ROOT/lstmf"
MODEL_NAME="kan_hist"

# ── Unicharset / traineddata selection ────────────────────────────────────────
#
# The expanded unicharset (tessdata_expanded/) adds ಋ ಙ ಝ ಱ to the model but
# produces a different recoder code range (116) than tessdata_best (140). Tesseract
# cannot continue from an existing fine-tuned checkpoint when the code range changes
# like this — the network output layer size would need to change.
#
# Two training modes:
#
#   TRAIN_MODE=resume  (default)
#     Continue from the existing kan_hist_checkpoint using tessdata_best (140 codes).
#     All fine-tuning progress is preserved. ಋ ಙ ಝ ಱ remain unsupported.
#
#   TRAIN_MODE=fresh
#     Start from output/kan.lstm (base weights) using tessdata_expanded (includes
#     ಋ ಙ ಝ ಱ). Fine-tuning progress in the checkpoint is NOT carried over.
#     Use this after running 00c-expand-unicharset.sh when no checkpoint exists yet
#     or when you are willing to re-train from the base model.
#
TRAIN_MODE="${TRAIN_MODE:-resume}"

TESSDATA_BEST="$ROOT/tessdata_best"

if [ "$TRAIN_MODE" = "fresh" ]; then
    if [ -f "$ROOT/tessdata_expanded/kan.traineddata" ]; then
        TESSDATA_BEST="$ROOT/tessdata_expanded"
        echo "  ✓ TRAIN_MODE=fresh — using tessdata_expanded/ (ಋ ಙ ಝ ಱ included)"
        echo "  ⚠  Will start from output/kan.lstm — checkpoint progress is NOT carried over"
    else
        echo "  ⚠  TRAIN_MODE=fresh requested but tessdata_expanded/ not found"
        echo "     Run ①+ Expand chars first, then re-run Train"
        exit 1
    fi
elif [ -f "$ROOT/tessdata_expanded/kan.traineddata" ]; then
    echo "  ℹ  tessdata_expanded/ exists but TRAIN_MODE=resume — using tessdata_best/"
    echo "     To include ಋ ಙ ಝ ಱ in a new training run, set TRAIN_MODE=fresh"
fi

[ -f "$LSTMF_DIR/list.txt" ] || {
    echo "ERROR: lstmf/list.txt not found. Run 02-make-lstmf.sh first."
    exit 1
}
LSTMF_COUNT=$(wc -l < "$LSTMF_DIR/list.txt" | tr -d ' ')
[ "$LSTMF_COUNT" -gt 0 ] || {
    echo "ERROR: lstmf/list.txt is empty."
    exit 1
}

MAX_ITERATIONS=${MAX_ITERATIONS:-500000}   # override: MAX_ITERATIONS=200000 ./scripts/03-train.sh
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
# Priority: 1) explicit CONTINUE_FROM env var
#           2) rolling checkpoint (most recent state, preserves optimizer)
#           3) highest-iteration named checkpoint
#           4) kan.lstm base model (fresh start)
if [ -z "${CONTINUE_FROM:-}" ]; then
    if [ "$TRAIN_MODE" = "fresh" ]; then
        # fresh mode: always start from base LSTM weights, not a fine-tuned checkpoint.
        # Remove any existing rolling checkpoint so lstmtraining doesn't auto-resume from it.
        if [ -f "$OUTPUT/${MODEL_NAME}_checkpoint" ]; then
            echo "  Renaming existing checkpoint → ${MODEL_NAME}_checkpoint.bak (preserving it)"
            mv "$OUTPUT/${MODEL_NAME}_checkpoint" "$OUTPUT/${MODEL_NAME}_checkpoint.bak"
        fi
        if [ -f "$OUTPUT/kan.lstm" ]; then
            CONTINUE_FROM="$OUTPUT/kan.lstm"
            echo "→ Starting from base weights (kan.lstm) with expanded unicharset"
        else
            echo "ERROR: output/kan.lstm not found. Run ① Prep base first."
            exit 1
        fi
    elif [ -f "$OUTPUT/${MODEL_NAME}_checkpoint" ]; then
        # Rolling checkpoint = the actual latest training state
        CONTINUE_FROM="$OUTPUT/${MODEL_NAME}_checkpoint"
        echo "→ Resuming from rolling checkpoint (latest state)"
    else
        # Fall back to the named checkpoint with the highest iteration count
        RECENT=$(ls "$OUTPUT"/${MODEL_NAME}_*.checkpoint 2>/dev/null | \
                 grep -v '_checkpoint$' | sort -t_ -k4 -n | tail -1 || true)
        if [ -n "$RECENT" ]; then
            CONTINUE_FROM="$RECENT"
            echo "→ Resuming from: $(basename $CONTINUE_FROM)"
        elif [ -f "$OUTPUT/kan.lstm" ]; then
            CONTINUE_FROM="$OUTPUT/kan.lstm"
            echo "→ Starting fresh from kan.lstm"
        else
            echo "ERROR: output/kan.lstm not found. Run ① Prep base first."
            exit 1
        fi
    fi
else
    echo "→ Using explicit checkpoint: $(basename $CONTINUE_FROM)"
fi

echo ""
echo "  Started: $(date)"
echo ""

# When the recoder in tessdata_expanded differs from the one in kan.lstm /
# the checkpoint, --old_traineddata tells Tesseract how to remap the old
# output-layer weights to the new code scheme. Required whenever TRAIN_MODE=fresh
# pairs kan.lstm (140-code recoder) with tessdata_expanded (116-code recoder).
OLD_TD_ARG=""
if [ "$TRAIN_MODE" = "fresh" ] && [ -f "$ROOT/tessdata_best/kan.traineddata" ]; then
    OLD_TD_ARG="--old_traineddata $ROOT/tessdata_best/kan.traineddata"
    echo "  Using --old_traineddata for recoder remapping (140 → 116 codes)"
fi

lstmtraining \
    --continue_from   "$CONTINUE_FROM" \
    --model_output    "$OUTPUT/$MODEL_NAME" \
    --traineddata     "$TESSDATA_BEST/kan.traineddata" \
    $OLD_TD_ARG \
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
