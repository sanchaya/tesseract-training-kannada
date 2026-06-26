#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# 02-make-lstmf.sh
#
# Converts PNG+gt.txt pairs (from render-corpus.py) into
# .lstmf training files that lstmtraining consumes.
#
# Scans two directories:
#   rendered/      — synthetic images from render-corpus.py
#   scan-input/    — real scan images you supply (PNG + .gt.txt)
#
# Output:
#   lstmf/rendered/   — lstmf from synthetic images
#   lstmf/scan/       — lstmf from real scan images
#   lstmf/list.txt    — combined training file list
#
# Prerequisites:
#   Run 01-prep-base.sh first (needs tessdata_best/kan.traineddata)
# ═══════════════════════════════════════════════════════════════
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$SCRIPT_DIR/.."
cd "$ROOT"

TESSDATA_BEST="$ROOT/tessdata_best"
LSTMF_DIR="$ROOT/lstmf"
RENDERED_DIR="$ROOT/rendered"
SCAN_DIR="$ROOT/scan-input"

[ -f "$TESSDATA_BEST/kan.traineddata" ] || {
    echo "ERROR: tessdata_best/kan.traineddata not found."
    echo "  Run ./scripts/01-prep-base.sh first."
    exit 1
}

mkdir -p "$LSTMF_DIR/rendered" "$LSTMF_DIR/scan"
> "$LSTMF_DIR/list.txt"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Step 2 — PNG+GT → lstmf"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

process_dir() {
    local src_dir="$1"
    local out_dir="$2"
    local label="$3"

    if [ ! -d "$src_dir" ]; then
        echo "  $label: directory not found, skipping ($src_dir)"
        return
    fi

    local count
    count=$(find "$src_dir" -maxdepth 1 \( -name "*.png" -o -name "*.tif" \) \
            2>/dev/null | wc -l | tr -d ' ')
    if [ "$count" -eq 0 ]; then
        echo "  $label: no images found in $src_dir"
        return
    fi

    echo ""
    echo "→ Processing $label ($count images)..."

    python3 - "$src_dir" "$out_dir" "$LSTMF_DIR/list.txt" "$label" \
             "$TESSDATA_BEST" << 'PYEOF'
import sys, subprocess, shutil
from pathlib import Path

src_dir, out_dir, list_txt, label, tdata = \
    sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]

def make_lstmf(img_path, out_dir, list_file, tessdata):
    img_path = Path(img_path)
    gt_path  = img_path.with_suffix('.gt.txt')
    if not img_path.exists() or not gt_path.exists():
        return False

    out_dir  = Path(out_dir)
    stem     = img_path.stem
    dst_img  = out_dir / (stem + img_path.suffix)
    lstmf    = out_dir / (stem + '.lstmf')

    shutil.copy2(img_path, dst_img)

    # Try PSM 7 (single text line) first, fall back to PSM 6
    for psm in ("7", "6"):
        result = subprocess.run(
            ["tesseract", str(dst_img), str(out_dir / stem),
             "--tessdata-dir", tessdata,
             "--dpi", "150", "--psm", psm,
             "-l", "kan", "lstm.train"],
            capture_output=True, text=True
        )
        if lstmf.exists():
            with open(list_file, 'a', encoding='utf-8') as lf:
                lf.write(str(lstmf.resolve()) + '\n')
            return True

    dst_img.unlink(missing_ok=True)
    return False

imgs = sorted(
    p for p in Path(src_dir).iterdir()
    if p.suffix.lower() in ('.png', '.tif', '.tiff', '.jpg')
)
ok = fail = skip = 0
for img in imgs:
    if not img.with_suffix('.gt.txt').exists():
        skip += 1
        continue
    if make_lstmf(img, out_dir, list_txt, tessdata):
        ok += 1
    else:
        fail += 1

print(f"  {label}: {ok} OK  {fail} failed  {skip} skipped (no gt.txt)")
PYEOF
}

process_dir "$RENDERED_DIR"  "$LSTMF_DIR/rendered"  "Synthetic rendered"
process_dir "$SCAN_DIR"      "$LSTMF_DIR/scan"      "Real scan images"

TOTAL=$(wc -l < "$LSTMF_DIR/list.txt" | tr -d ' ')

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Done. lstmf/list.txt contains $TOTAL files."
echo ""
echo "  NEXT: caffeinate -i ./scripts/03-train.sh > training.log 2>&1 &"
echo "        tail -f training.log"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
