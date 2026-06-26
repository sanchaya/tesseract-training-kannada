#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# 01-prep-base.sh
#
# One-time setup for kan_hist LSTM training.
#
# What this does:
#   A. Clones all font repos listed in fonts.yml into fonts/
#   B. Downloads tessdata_best/kan.traineddata (Kannada LSTM base)
#   C. Extracts output/kan.lstm (the raw LSTM weights for fine-tuning)
#
# Why fine-tune from kan.traineddata?
#   A Kannada LSTM model already exists in tessdata_best. We fine-tune
#   it on Karnata fonts (historical letterpress revivals) so the model
#   learns the specific stroke shapes and conjunct forms of these
#   19th-century typefaces, producing kan_hist.
#
# Run once before any training. Safe to re-run (skips if already done).
# ═══════════════════════════════════════════════════════════════
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$SCRIPT_DIR/.."
cd "$ROOT"

TESSDATA_BEST="$ROOT/tessdata_best"
OUTPUT="$ROOT/output"
FONTS_DIR="$ROOT/fonts"
FONTS_YML="$ROOT/fonts.yml"

mkdir -p "$TESSDATA_BEST" "$OUTPUT" "$FONTS_DIR"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Step 1 — kan_hist training setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── A: Clone font repos ────────────────────────────────────────
echo ""
echo "→ [A] Fetching Karnata fonts from GitHub..."

python3 - "$FONTS_YML" "$FONTS_DIR" << 'PYEOF'
import sys, subprocess, yaml
from pathlib import Path

fonts_yml, fonts_dir = sys.argv[1], Path(sys.argv[2])

with open(fonts_yml) as f:
    data = yaml.safe_load(f)

for font in data['fonts']:
    fid   = font['id']
    repo  = font['repo']
    dest  = fonts_dir / fid

    print(f"  [{fid}] {repo}")
    if (dest / '.git').exists():
        print(f"    already cloned — pulling latest")
        subprocess.run(['git', '-C', str(dest), 'pull', '--ff-only'],
                       check=True, capture_output=True)
    else:
        subprocess.run(['git', 'clone', '--depth', '1', repo, str(dest)],
                       check=True)
    print(f"    → {dest}")
PYEOF

# ── B: Download kan base model ─────────────────────────────────
echo ""
echo "→ [B] Base model (kan.traineddata)..."
if [ ! -f "$TESSDATA_BEST/kan.traineddata" ]; then
    echo "  Downloading tessdata_best/kan.traineddata..."
    curl -fL --progress-bar \
        "https://github.com/tesseract-ocr/tessdata_best/raw/main/kan.traineddata" \
        -o "$TESSDATA_BEST/kan.traineddata"
    echo "  Downloaded: $(du -h $TESSDATA_BEST/kan.traineddata | cut -f1)"
else
    echo "  Already present: tessdata_best/kan.traineddata"
fi

# ── C: Extract LSTM weights ────────────────────────────────────
echo ""
echo "→ [C] Extracting kan.lstm..."
combine_tessdata -e "$TESSDATA_BEST/kan.traineddata" "$OUTPUT/kan.lstm" 2>/dev/null
ls -lh "$OUTPUT/kan.lstm"
echo "  (Fine-tuning will start from these weights)"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Setup complete."
echo ""
echo "  Base model:      tessdata_best/kan.traineddata"
echo "  LSTM weights:    output/kan.lstm"
echo ""
echo "  NEXT STEPS:"
echo "    python3 corpus/clean-corpus.py    (clean your raw Kannada text)"
echo "    python3 corpus/render-corpus.py   (render training images)"
echo "    ./scripts/02-make-lstmf.sh        (create .lstmf files)"
echo "    caffeinate -i ./scripts/03-train.sh > training.log 2>&1 &"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
