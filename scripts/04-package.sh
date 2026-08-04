#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# 04-package.sh
#
# Packages the best training checkpoint into kan_hist.traineddata.
#
# Checkpoint selection:
#   Picks the checkpoint with the LOWEST BCER among checkpoints
#   where iteration ≤ MAX_ITER (default 150000).
#
#   WHY THE CAP: BCER measures error on TRAINING data only.
#   A model at 0.000% BCER has memorised the training set and
#   will perform WORSE on real scans than one at 0.1% BCER.
#   The cap prevents the packaging script from always selecting
#   the most overfitted checkpoint.
#
#   Override with:
#     export BEST_CHECKPOINT=/path/to/specific.checkpoint
#     export MAX_ITER=200000        (adjust iteration cap)
#
# Output:
#   output/kan_hist.traineddata
#   best/kan_hist.traineddata      (copy for sharing / PR)
# ═══════════════════════════════════════════════════════════════
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$SCRIPT_DIR/.."
cd "$ROOT"

TESSDATA_BEST="$ROOT/tessdata_best"
TESSDATA_EXP="$ROOT/tessdata_expanded"
OUTPUT="$ROOT/output"
BEST_DIR="$ROOT/best"
MODEL_NAME="kan_hist"

# Use the expanded unicharset traineddata if it exists (training with
# expanded unicharset produces checkpoints with code range 140, not 116;
# packaging against the wrong base causes "Code range changed" fatal error).
if [ -f "$TESSDATA_EXP/kan.traineddata" ]; then
    BASE_TRAINEDDATA="$TESSDATA_EXP/kan.traineddata"
    echo "  Using tessdata_expanded/kan.traineddata as base (expanded unicharset)"
else
    BASE_TRAINEDDATA="$TESSDATA_BEST/kan.traineddata"
    echo "  Using tessdata_best/kan.traineddata as base"
fi

[ -f "$OUTPUT/${MODEL_NAME}_checkpoint" ] || {
    echo "ERROR: output/${MODEL_NAME}_checkpoint not found."
    echo "  Run 03-train.sh first."
    exit 1
}

# ── Select best checkpoint ─────────────────────────────────────
# Iteration cap: avoid selecting overfit late checkpoints.
# BCER on training data goes to 0% if you train long enough, but that
# means the model has memorised the training set — it will perform WORSE
# on real scans than a checkpoint at 0.1% BCER from an earlier iteration.
MAX_ITER=${MAX_ITER:-150000}

if [ -z "${BEST_CHECKPOINT:-}" ]; then
    # Checkpoint filename format: modelname_BCER_examples_ITERATION.checkpoint
    #   e.g. kan_hist_0.105_4947_49300.checkpoint
    #   Split by '_': $1=kan $2=hist $3=0.105 $4=4947 $5=49300.checkpoint
    #   $3 = BCER,  $NF = iteration (after stripping .checkpoint)
    #
    # Select lowest BCER among checkpoints with iteration ≤ MAX_ITER.
    # This prevents selecting the most-overfit late checkpoint.
    # NF==5 matches "…/kan_hist_BCER_examples_ITERATION.checkpoint" only,
    # excluding chartrain variants like "kan_hist_chartrain_kan_gmp_…" (NF=8).
    BEST_CHECKPOINT=$(ls "$OUTPUT"/${MODEL_NAME}_*.checkpoint 2>/dev/null | \
        grep -v '_checkpoint$' | \
        awk -F'_' -v cap="$MAX_ITER" 'NF==5 {
            iter_str = $NF; sub(/\.checkpoint.*/, "", iter_str);
            iter = iter_str + 0;
            bcer = $3 + 0;
            if (iter > 0 && iter <= cap) print $0, bcer
        }' | sort -k2 -n | head -1 | awk '{print $1}' || true)

    # If nothing found under the cap, expand to 2× (warn user)
    if [ -z "${BEST_CHECKPOINT:-}" ]; then
        echo "⚠  No checkpoint found with iteration ≤ ${MAX_ITER}."
        echo "   Expanding search to iteration ≤ $((MAX_ITER * 2))..."
        BEST_CHECKPOINT=$(ls "$OUTPUT"/${MODEL_NAME}_*.checkpoint 2>/dev/null | \
            grep -v '_checkpoint$' | \
            awk -F'_' -v cap="$((MAX_ITER * 2))" 'NF==5 {
                iter_str = $NF; sub(/\.checkpoint.*/, "", iter_str);
                iter = iter_str + 0;
                bcer = $3 + 0;
                if (iter > 0 && iter <= cap) print $0, bcer
            }' | sort -k2 -n | head -1 | awk '{print $1}' || true)
    fi
fi

[ -n "${BEST_CHECKPOINT:-}" ] || BEST_CHECKPOINT="$OUTPUT/${MODEL_NAME}_checkpoint"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Step 4 — Packaging kan_hist model"
echo "  Source: $(basename $BEST_CHECKPOINT)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Export LSTM weights from best checkpoint
lstmtraining \
    --stop_training \
    --continue_from   "$BEST_CHECKPOINT" \
    --traineddata     "$BASE_TRAINEDDATA" \
    --model_output    "$OUTPUT/${MODEL_NAME}.lstm"

# Strip macOS Tesseract 5.5+ corrupt header if present
python3 - "$OUTPUT/${MODEL_NAME}.lstm" << 'PYEOF'
import sys
path = sys.argv[1]
data = open(path, 'rb').read()
needle = b'\x00\x06\x00\x00\x00Series'
idx = data.find(needle)
if idx > 0:
    print(f"  Stripping {idx}-byte corrupt header from .lstm")
    open(path, 'wb').write(data[idx:])
else:
    print("  .lstm header clean")
PYEOF

# Bundle into traineddata (start from base to preserve unicharset/language data)
cp "$BASE_TRAINEDDATA" "$OUTPUT/${MODEL_NAME}.traineddata"
combine_tessdata \
    -o "$OUTPUT/${MODEL_NAME}.traineddata" \
    "$OUTPUT/${MODEL_NAME}.lstm"

ls -lh "$OUTPUT/${MODEL_NAME}.traineddata"

# Copy to best/
mkdir -p "$BEST_DIR"
cp "$OUTPUT/${MODEL_NAME}.traineddata" "$BEST_DIR/${MODEL_NAME}.traineddata"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Model ready:"
echo "    output/${MODEL_NAME}.traineddata"
echo "    best/${MODEL_NAME}.traineddata      ← submit this in a PR"
echo ""
echo "  To install and test:"
echo "    cp best/${MODEL_NAME}.traineddata /opt/homebrew/share/tessdata/"
echo "    tesseract your-scan.tif result -l ${MODEL_NAME}"
echo ""
echo "  NEXT: ./scripts/05-test.sh test-images/sample.tif"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
