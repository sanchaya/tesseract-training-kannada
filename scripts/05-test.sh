#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# 05-test.sh — OCR test on a single image, comparing kan vs kan_hist.
#
# Usage:
#   ./scripts/05-test.sh <image> [ground-truth.txt]
#
# Examples:
#   ./scripts/05-test.sh test-images/sample.tif
#   ./scripts/05-test.sh test-images/book_page.png test-images/book_page.gt.txt
# ═══════════════════════════════════════════════════════════════
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$SCRIPT_DIR/.."
cd "$ROOT"

IMAGE="${1:-}"
GT="${2:-}"

[ -n "$IMAGE" ] || { echo "Usage: $0 <image> [ground-truth.txt]"; exit 1; }
[ -f "$IMAGE" ] || { echo "ERROR: image not found: $IMAGE"; exit 1; }

# Detect tessdata directory
SYSTEM_TESSDATA=""
for d in /opt/homebrew/share/tessdata /usr/share/tesseract-ocr/5/tessdata \
          /usr/share/tesseract-ocr/4.00/tessdata /usr/local/share/tessdata; do
    if [ -d "$d" ]; then SYSTEM_TESSDATA="$d"; break; fi
done

LOCAL_BEST="$ROOT/output/kan_hist.traineddata"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  OCR test: $(basename $IMAGE)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

run_ocr() {
    local label="$1"
    local tessdata="$2"
    local model="$3"
    local oem="${4:-1}"

    if [ -f "$tessdata/$model.traineddata" ] || \
       ([ "$tessdata" = "$(dirname $LOCAL_BEST)" ] && [ -f "$LOCAL_BEST" ]); then
        echo "── $label ──"
        tesseract "$IMAGE" stdout \
            --tessdata-dir "$tessdata" \
            -l "$model" --psm 6 --oem "$oem" 2>/dev/null \
            | head -20 || echo "  (no output)"
        echo ""
    else
        echo "── $label: model not found ──"
        echo ""
    fi
}

# Base Kannada model
[ -n "$SYSTEM_TESSDATA" ] && run_ocr "kan (base tessdata_best)" "$SYSTEM_TESSDATA" "kan"

# kan_hist — local build
run_ocr "kan_hist (this build)" "$(dirname $LOCAL_BEST)" "kan_hist"

# kan_hist — installed
[ -n "$SYSTEM_TESSDATA" ] && run_ocr "kan_hist (installed)" "$SYSTEM_TESSDATA" "kan_hist"

# Ground truth
if [ -n "$GT" ] && [ -f "$GT" ]; then
    echo "── Ground truth ──"
    cat "$GT"
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
