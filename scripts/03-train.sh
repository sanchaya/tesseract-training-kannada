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
#   caffeinate -i ./scripts/03-train.sh          # logs to logs/training.log
#   tail -f logs/training.log
#
# The script writes logs/training.log itself when run interactively, so the
# portal's Live Log works regardless of how training was started. Previously
# the log existed ONLY if the caller added `> logs/training.log 2>&1`, so a
# plain `./scripts/03-train.sh` produced no log and the portal showed nothing.
# Set TRAINOCR_NO_TEE=1 to disable (the portal sets it — it captures output
# itself, and teeing would double every line).
#
# To resume from a specific checkpoint:
#   CONTINUE_FROM=output/kan_hist_2.1_50000.checkpoint ./scripts/03-train.sh
# ═══════════════════════════════════════════════════════════════
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$SCRIPT_DIR/.."
cd "$ROOT"

# ── Self-logging ──────────────────────────────────────────────────────────────
# Always mirror output into logs/training.log — the file the portal's Live Log
# tails — however this script was invoked.
#
# Two cases must NOT tee, or every line would be written twice:
#   • the portal, which already points the child's stdout at that same file
#     (it sets TRAINOCR_NO_TEE=1);
#   • a caller that redirected to logs/training.log explicitly, detected with
#     `-ef` (same device + inode).
#
# A caller redirecting somewhere else — e.g. the old `> training.log` habit that
# dropped the log in the repo root — still gets logs/training.log populated, so
# the portal keeps working regardless.
TRAIN_LOG="$ROOT/logs/training.log"
mkdir -p "$ROOT/logs"
if [ -z "$TRAINOCR_NO_TEE" ] && ! [ /dev/stdout -ef "$TRAIN_LOG" ] 2>/dev/null; then
  echo "[trainocr] logging to logs/training.log"
  exec > >(tee -a "$TRAIN_LOG") 2>&1
fi

OUTPUT="$ROOT/output"
LSTMF_DIR="$ROOT/lstmf"
MODEL_NAME="kan_hist"

# ── Unicharset / traineddata selection ────────────────────────────────────────
#
# The expanded unicharset (tessdata_expanded/) adds ಋ ಙ ಝ ಱ to the model but
# uses a different recoder code range (116) than tessdata_best (140).  Tesseract
# supports switching between them via --old_traineddata, which remaps the output
# layer during the FIRST run after a switch.  After that, checkpoints are already
# 116-code and no remapping is needed.
#
# Three training modes:
#
#   TRAIN_MODE=resume  (default)
#     Continue from the existing checkpoint.  If a previous "expand" run has
#     already written output/.tessdata_mode=expanded, tessdata_expanded is used
#     automatically.  Otherwise tessdata_best is used (140 codes).
#
#   TRAIN_MODE=expand
#     Continue from the existing checkpoint AND switch to tessdata_expanded in
#     one step using --old_traineddata for output-layer remapping.  Named
#     checkpoints are preserved.  Use once after running 00c-expand-unicharset.sh
#     when a checkpoint already exists.  After this run, TRAIN_MODE=resume also
#     uses tessdata_expanded automatically.
#
#   TRAIN_MODE=fresh
#     Start from output/kan.lstm (base kan weights) using tessdata_expanded.
#     ALL existing checkpoints are backed up to output/checkpoint_backup_<date>/.
#     Use only when you want a completely clean training run.
#
TRAIN_MODE="${TRAIN_MODE:-resume}"

TESSDATA_BEST="$ROOT/tessdata_best"
TESSDATA_EXPANDED="$ROOT/tessdata_expanded"
MODE_FILE="$OUTPUT/.tessdata_mode"

# Auto-detect if a previous expand run already switched this project
if [ "$TRAIN_MODE" = "resume" ] && [ "$(cat "$MODE_FILE" 2>/dev/null)" = "expanded" ]; then
    if [ -f "$TESSDATA_EXPANDED/kan.traineddata" ]; then
        TESSDATA_BEST="$TESSDATA_EXPANDED"
        echo "  ✓ Detected previous expand run — using tessdata_expanded/ (ಋ ಙ ಝ ಱ included)"
    fi
fi

if [ "$TRAIN_MODE" = "fresh" ]; then
    if [ -f "$TESSDATA_EXPANDED/kan.traineddata" ]; then
        TESSDATA_BEST="$TESSDATA_EXPANDED"
        echo "  ✓ TRAIN_MODE=fresh — using tessdata_expanded/ (ಋ ಙ ಝ ಱ included)"
        echo "  ⚠  Will start from output/kan.lstm — ALL checkpoint progress will be backed up"
    else
        echo "  ⚠  TRAIN_MODE=fresh requested but tessdata_expanded/ not found"
        echo "     Run ①+ Expand chars first, then re-run Train"
        exit 1
    fi
elif [ "$TRAIN_MODE" = "expand" ]; then
    if [ -f "$TESSDATA_EXPANDED/kan.traineddata" ]; then
        TESSDATA_BEST="$TESSDATA_EXPANDED"
        echo "  ✓ TRAIN_MODE=expand — continuing existing checkpoint with tessdata_expanded/"
        echo "     --old_traineddata remapping will align the output layer (140 → 116 codes)"
        echo "     Existing named checkpoints are preserved"
    else
        echo "  ⚠  TRAIN_MODE=expand requested but tessdata_expanded/ not found"
        echo "     Run ①+ Expand chars first, then re-run Train"
        exit 1
    fi
elif [ "$TRAIN_MODE" = "resume" ] && [ -f "$TESSDATA_EXPANDED/kan.traineddata" ] && [ "$(cat "$MODE_FILE" 2>/dev/null)" != "expanded" ]; then
    echo "  ℹ  tessdata_expanded/ exists — to switch to expanded chars, use TRAIN_MODE=expand"
    echo "     Current run uses tessdata_best/ (ಋ ಙ ಝ ಱ not included)"
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
LEARNING_RATE=${LEARNING_RATE:-0.001}      # override: LEARNING_RATE=0.0001 ./scripts/03-train.sh

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
        # fresh mode: always start from base LSTM weights.
        # Back up ALL existing checkpoints to a timestamped directory so nothing is lost.
        BACKUP_DIR="$OUTPUT/checkpoint_backup_$(date +%Y%m%d_%H%M%S)"
        # Collect ALL checkpoint files:
        #   named:   kan_hist_<bcer>_<examples>_<iter>.checkpoint
        #   rolling: kan_hist_checkpoint   ← no .checkpoint suffix, often missed
        FOUND_CHECKPOINTS=$(ls \
            "$OUTPUT"/${MODEL_NAME}_*.checkpoint \
            "$OUTPUT"/${MODEL_NAME}_checkpoint \
            2>/dev/null | head -1 || true)
        if [ -n "$FOUND_CHECKPOINTS" ]; then
            mkdir -p "$BACKUP_DIR"
            echo "  Backing up existing checkpoints → $(basename $BACKUP_DIR)/"
            mv "$OUTPUT"/${MODEL_NAME}_*.checkpoint  "$BACKUP_DIR/" 2>/dev/null || true
            mv "$OUTPUT"/${MODEL_NAME}_checkpoint    "$BACKUP_DIR/" 2>/dev/null || true
            echo "  $(ls "$BACKUP_DIR" | wc -l | tr -d ' ') checkpoint files moved to backup"
        fi
        if [ -f "$OUTPUT/kan.lstm" ]; then
            CONTINUE_FROM="$OUTPUT/kan.lstm"
            echo "→ Starting from base weights (kan.lstm) with expanded unicharset"
        else
            echo "ERROR: output/kan.lstm not found. Run ① Prep base first."
            exit 1
        fi

    elif [ "$TRAIN_MODE" = "expand" ]; then
        # expand mode: continue from the existing checkpoint but switch to tessdata_expanded.
        # Named checkpoints are preserved — only the rolling checkpoint is used as start point.
        if [ -f "$OUTPUT/${MODEL_NAME}_checkpoint" ]; then
            CONTINUE_FROM="$OUTPUT/${MODEL_NAME}_checkpoint"
            echo "→ Continuing from rolling checkpoint with expanded unicharset (recoder remapping)"
        else
            # Fall back to named checkpoint with highest iteration
            RECENT=$(ls "$OUTPUT"/${MODEL_NAME}_*.checkpoint 2>/dev/null | \
                     grep -v '_checkpoint$' | sort -t_ -k5 -n | tail -1 || true)
            if [ -n "$RECENT" ]; then
                CONTINUE_FROM="$RECENT"
                echo "→ Continuing from: $(basename $CONTINUE_FROM) with expanded unicharset"
            elif [ -f "$OUTPUT/kan.lstm" ]; then
                CONTINUE_FROM="$OUTPUT/kan.lstm"
                echo "→ No checkpoint found — starting from kan.lstm with expanded unicharset"
            else
                echo "ERROR: No checkpoint or kan.lstm found. Run ① Prep base first."
                exit 1
            fi
        fi

    elif [ -f "$OUTPUT/${MODEL_NAME}_checkpoint" ]; then
        # resume mode: rolling checkpoint = the actual latest training state
        CONTINUE_FROM="$OUTPUT/${MODEL_NAME}_checkpoint"
        echo "→ Resuming from rolling checkpoint (latest state)"
    else
        # resume mode fallback: named checkpoint with the highest iteration count
        # Sort by field 5 (iteration) — format: kan_hist_<BCER>_<EXAMPLES>_<ITERATION>.checkpoint
        RECENT=$(ls "$OUTPUT"/${MODEL_NAME}_*.checkpoint 2>/dev/null | \
                 grep -v '_checkpoint$' | sort -t_ -k5 -n | tail -1 || true)
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

# --old_traineddata is required whenever the checkpoint was created with tessdata_best
# (140-code recoder) but training now targets tessdata_expanded (116-code recoder).
# After the FIRST such run the rolling checkpoint is already 116-code so subsequent
# resume runs do NOT need remapping.
OLD_TD_ARG=""
if [ "$TRAIN_MODE" = "fresh" ] || [ "$TRAIN_MODE" = "expand" ]; then
    if [ -f "$ROOT/tessdata_best/kan.traineddata" ]; then
        OLD_TD_ARG="--old_traineddata $ROOT/tessdata_best/kan.traineddata"
        echo "  Using --old_traineddata for recoder remapping (tessdata_best → tessdata_expanded)"
    fi
fi

# After this run the project is in "expanded" mode — write state file so future
# resume runs auto-select tessdata_expanded without needing --old_traineddata.
if [ "$TRAIN_MODE" = "expand" ] || [ "$TRAIN_MODE" = "fresh" ]; then
    mkdir -p "$OUTPUT"
    echo "expanded" > "$MODE_FILE"
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
