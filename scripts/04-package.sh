#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# 04-package.sh
#
# Packages the best training checkpoint into kan_hist.traineddata.
#
# Checkpoint selection:
#   Picks the checkpoint with the LOWEST BCER (the number in the
#   filename, e.g. kan_hist_1.19_98900.checkpoint → 1.19%).
#   Override with: export BEST_CHECKPOINT=/path/to/specific.checkpoint
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
OUTPUT="$ROOT/output"
BEST_DIR="$ROOT/best"
MODEL_NAME="kan_hist"

[ -f "$OUTPUT/${MODEL_NAME}_checkpoint" ] || {
    echo "ERROR: output/${MODEL_NAME}_checkpoint not found."
    echo "  Run 03-train.sh first."
    exit 1
}

# ── Select best checkpoint ─────────────────────────────────────
if [ -z "${BEST_CHECKPOINT:-}" ]; then
    # Sort by BCER (field 2 of checkpoint name, numeric ascending)
    BEST_CHECKPOINT=$(ls "$OUTPUT"/${MODEL_NAME}_*.checkpoint 2>/dev/null | \
        grep -v '_checkpoint$' | \
        awk -F'_' '{print $0, $3+0}' | sort -k2 -n | head -1 | awk '{print $1}' || true)
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
    --traineddata     "$TESSDATA_BEST/kan.traineddata" \
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

# Bundle into traineddata (start from kan base to preserve unicharset/language data)
cp "$TESSDATA_BEST/kan.traineddata" "$OUTPUT/${MODEL_NAME}.traineddata"
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
