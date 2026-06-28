#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# 00c-expand-unicharset.sh
#
# Adds missing Kannada characters (ಋ ಙ ಝ ಱ) to kan.traineddata's
# unicharset, producing tessdata_expanded/kan.traineddata.
#
# WHY: The tessdata_best/kan.traineddata unicharset does not include
# these 4 characters. lstmtraining skips (or errors on) any training
# line that contains them. After running this script, retraining from
# your existing checkpoint will auto-expand the output layer to cover
# the new characters.
#
# What it does:
#   1. Downloads kan/ langdata from GitHub (first run only; cached)
#      https://github.com/tesseract-ocr/langdata_lstm/tree/main/kan
#   2. Uses kan/kan.unicharset as the authoritative base (not the
#      extracted lstm-unicharset) to stay aligned with upstream
#   3. Merges the 4 missing characters into kan.unicharset
#   4. Copies the expanded unicharset as Kannada/Kannada.unicharset
#      for the combine_lang_model script_dir
#   5. Runs combine_lang_model → tessdata_expanded/kan.traineddata
#
# After this script:
#   • Re-run ④ Make lstmf (filter for these chars is auto-removed)
#   • Re-run ⑤ Train (auto-uses tessdata_expanded/kan.traineddata)
#
# Usage:
#   ./scripts/00c-expand-unicharset.sh           # normal (cached downloads)
#   ./scripts/00c-expand-unicharset.sh --force   # re-download all, re-run combine_lang_model
# ═══════════════════════════════════════════════════════════════
set -e

FORCE=0
for arg in "$@"; do
    [ "$arg" = "--force" ] && FORCE=1
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$SCRIPT_DIR/.."
cd "$ROOT"

TESSDATA_BEST="$ROOT/tessdata_best"
WORK_DIR="$ROOT/tmp/unicharset_work"
LANGDATA_DIR="$ROOT/tmp/langdata_lstm"
OUTPUT_TESSDATA="$ROOT/tessdata_expanded"

MISSING_CHARS="ಋ ಙ ಝ ಱ"
BASE_URL="https://raw.githubusercontent.com/tesseract-ocr/langdata_lstm/main/kan"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Expand Kannada Unicharset"
echo "  Adding: $MISSING_CHARS"
echo "  Base:   tessdata_best/kan.traineddata lstm-unicharset (binary)"
echo "  Output: tessdata_expanded/kan.traineddata"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── Prerequisites ──────────────────────────────────────────────
for tool in combine_lang_model combine_tessdata merge_unicharsets unicharset_extractor; do
    command -v "$tool" >/dev/null 2>&1 || {
        echo "ERROR: $tool not found. Install Tesseract training tools."
        echo "  macOS: brew install tesseract"
        exit 1
    }
done

mkdir -p "$WORK_DIR" "$LANGDATA_DIR/kan" "$LANGDATA_DIR/Kannada" "$LANGDATA_DIR/Latin" "$OUTPUT_TESSDATA"

# ── Step 1: Download kan/ langdata from GitHub ────────────────
if [ "$FORCE" = "1" ]; then
    echo "→ --force: clearing cached downloads and output..."
    rm -rf "$LANGDATA_DIR/kan" "$LANGDATA_DIR/radical-stroke.txt" \
           "$LANGDATA_DIR/Kannada.unicharset" "$LANGDATA_DIR/Kannada/Kannada.unicharset" \
           "$LANGDATA_DIR/Latin.unicharset"   "$LANGDATA_DIR/Latin/Latin.unicharset" \
           "$WORK_DIR" "$OUTPUT_TESSDATA/kan.traineddata"
    mkdir -p "$WORK_DIR" "$LANGDATA_DIR/kan" "$LANGDATA_DIR/Kannada" "$LANGDATA_DIR/Latin" "$OUTPUT_TESSDATA"
fi

if [ ! -f "$LANGDATA_DIR/kan/kan.unicharset" ]; then
    echo "→ Step 1: Downloading kan/ langdata from GitHub..."
    echo "  Source: https://github.com/tesseract-ocr/langdata_lstm/tree/main/kan"
    echo ""
    for f in kan.unicharset kan.config kan.wordlist kan.puncs kan.numbers kan.unicharambigs; do
        if curl -fsSL "$BASE_URL/$f" -o "$LANGDATA_DIR/kan/$f" 2>/dev/null; then
            echo "  ✓ $f"
        else
            echo "  ⚠  $f not available (optional)"
            rm -f "$LANGDATA_DIR/kan/$f"
        fi
    done
else
    echo "→ Step 1: langdata already cached (tmp/langdata_lstm/kan/)"
fi

# ── radical-stroke.txt — combine_lang_model needs this unconditionally ──
# combine_lang_model tries to parse this CJK radical table regardless of language.
# An empty file causes "Failed to read data from" — must download the real file.
if [ ! -s "$LANGDATA_DIR/radical-stroke.txt" ]; then
    echo "→ Downloading radical-stroke.txt (required by combine_lang_model)..."
    if curl -fsSL "https://raw.githubusercontent.com/tesseract-ocr/langdata_lstm/main/radical-stroke.txt" \
            -o "$LANGDATA_DIR/radical-stroke.txt" 2>/dev/null; then
        echo "  ✓ radical-stroke.txt"
    else
        echo "ERROR: Could not download radical-stroke.txt from GitHub."
        echo "  Manually place it at: tmp/langdata_lstm/radical-stroke.txt"
        echo "  Source: https://raw.githubusercontent.com/tesseract-ocr/langdata_lstm/main/radical-stroke.txt"
        exit 1
    fi
fi

# ── Script unicharsets — combine_lang_model loads these for all scripts used ──
# The Kannada unicharset includes Latin-script chars (digits, punctuation), so
# combine_lang_model also needs Latin.unicharset. Download once; both locations
# are symlinked so either lookup path (root or subdirectory) resolves correctly.
LATIN_URL="https://raw.githubusercontent.com/tesseract-ocr/langdata_lstm/main/Latin/Latin.unicharset"
if [ ! -s "$LANGDATA_DIR/Latin/Latin.unicharset" ]; then
    echo "→ Downloading Latin.unicharset (required by combine_lang_model)..."
    if curl -fsSL "$LATIN_URL" -o "$LANGDATA_DIR/Latin/Latin.unicharset" 2>/dev/null; then
        ln -sf "Latin/Latin.unicharset" "$LANGDATA_DIR/Latin.unicharset"
        echo "  ✓ Latin.unicharset"
    else
        echo "  ⚠  Latin.unicharset not available — combine_lang_model will warn but continue"
    fi
fi

[ -f "$LANGDATA_DIR/kan/kan.unicharset" ] || {
    echo ""
    echo "ERROR: Could not download kan/kan.unicharset from GitHub."
    echo "  Check your internet connection, or manually place the file at:"
    echo "  tmp/langdata_lstm/kan/kan.unicharset"
    echo "  Source: https://github.com/tesseract-ocr/langdata_lstm/tree/main/kan"
    exit 1
}

# ── Step 2: Extract lstm-unicharset from tessdata_best/kan.traineddata ──────
# IMPORTANT: the merge base must be the lstm-unicharset embedded in the BINARY
# tessdata_best/kan.traineddata, NOT the upstream text kan.unicharset.
# The binary lstm-unicharset (140 entries) is what existing checkpoints were
# trained with. tessdata_expanded must be a SUPERSET of that set, otherwise
# lstmtraining --continue_from fails with "Code range changed".
echo ""
echo "→ Step 2: Extracting lstm-unicharset from tessdata_best/kan.traineddata..."
combine_tessdata -u "$TESSDATA_BEST/kan.traineddata" "$WORK_DIR/kan_base" 2>/dev/null || true
# combine_tessdata -u writes kan_base.lstm-unicharset
BASE_UNICHARSET="$WORK_DIR/kan_base.lstm-unicharset"
[ -f "$BASE_UNICHARSET" ] || {
    echo "ERROR: Could not extract lstm-unicharset from tessdata_best/kan.traineddata"
    exit 1
}
BASE_COUNT=$(head -1 "$BASE_UNICHARSET")
echo "  Extracted: $BASE_COUNT entries (binary lstm-unicharset)"

# Check which of the 4 chars are missing from the BINARY unicharset
python3 -c "
data = open('$BASE_UNICHARSET').read()
missing = []
for ch in 'ಋಙಝಱ':
    if ch not in data:
        missing.append(f'{ch} (U+{ord(ch):04X})')
if missing:
    print('  Missing from lstm-unicharset: ' + ', '.join(missing))
else:
    print('  All 4 characters already in lstm-unicharset.')
    exit(1)
" || {
    echo "  Nothing to add — tessdata_expanded not needed."
    echo "  tessdata_best/kan.traineddata already covers all 4 chars."
    exit 0
}

# ── Step 3: Create corpus with the missing characters ─────────
echo ""
echo "→ Step 3: Creating corpus with missing characters..."
cat > "$WORK_DIR/new_chars.txt" << 'EOF'
ಋ ಙ ಝ ಱ
ಋಷಿ ಋಣ ಋತು ಋಗ್ವೇದ ಋಜು ಋಕ್ಷ
ಪಂಚಾಂಗ ಸಂಗ ಅಂಗ ಮಂಗ ರಂಗ ಲಿಂಗ
ಝರ ಝಲ ಝಳ ಝಂಕ ಝಗ ಝಲಕ
ಗಾಱ ಅಱ ಕಱ ತಱ ಮಱ
EOF

# ── Step 4: Extract unicharset from corpus ─────────────────────
echo "→ Step 4: Extracting unicharset from new characters..."
unicharset_extractor \
    --output_unicharset "$WORK_DIR/new_chars.unicharset" \
    "$WORK_DIR/new_chars.txt"

# ── Step 5: Merge lstm-unicharset + new chars ──────────────────
echo "→ Step 5: Merging into lstm-unicharset..."
merge_unicharsets \
    "$BASE_UNICHARSET" \
    "$WORK_DIR/new_chars.unicharset" \
    "$WORK_DIR/kan_expanded.unicharset"

NEW_COUNT=$(head -1 "$WORK_DIR/kan_expanded.unicharset")
echo "  Unicharset: $BASE_COUNT → $NEW_COUNT entries"

# Verify
python3 -c "
data = open('$WORK_DIR/kan_expanded.unicharset').read()
all_ok = True
for ch in 'ಋಙಝಱ':
    status = '✓' if ch in data else '✗ STILL MISSING'
    print(f'  {ch} (U+{ord(ch):04X}): {status}')
    if ch not in data:
        all_ok = False
if not all_ok:
    exit(1)
"

# ── Step 6a: Write result back to kan/kan.unicharset ─────────────
# kan/kan.unicharset is the single source of truth in our langdata dir.
# After expansion it holds the merged (lstm-binary + 4 new chars) set.
# Symlink at both locations for combine_lang_model compatibility.
cp "$WORK_DIR/kan_expanded.unicharset" "$LANGDATA_DIR/kan/kan.unicharset"
ln -sf "../kan/kan.unicharset" "$LANGDATA_DIR/Kannada/Kannada.unicharset"
ln -sf "kan/kan.unicharset" "$LANGDATA_DIR/Kannada.unicharset"
echo "  ✓ kan/kan.unicharset updated (Kannada.unicharset + Kannada/Kannada.unicharset → symlinks)"

# ── Step 6: Rebuild traineddata with expanded unicharset ───────
if [ -f "$OUTPUT_TESSDATA/kan.traineddata" ] && [ "$FORCE" != "1" ]; then
    echo ""
    echo "→ Step 6: tessdata_expanded/kan.traineddata already exists — skipping."
    echo "  (Run with --force to rebuild)"
    SIZE=$(du -h "$OUTPUT_TESSDATA/kan.traineddata" | cut -f1)
    echo "  Existing: tessdata_expanded/kan.traineddata ($SIZE)"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  ✓ Already done — unicharset expansion is complete."
    echo "  Next: re-run ④ Make lstmf  →  ⑤ Train"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    exit 0
fi

echo ""
echo "→ Step 6: Running combine_lang_model..."

EXTRA_FLAGS=""
# --words and --puncs must be used together; combine_lang_model errors if words
# is provided without a non-empty puncs list. Create a minimal fallback if needed.
if [ -s "$LANGDATA_DIR/kan/kan.wordlist" ]; then
    if [ ! -s "$LANGDATA_DIR/kan/kan.puncs" ]; then
        echo "  ⚠  kan.puncs missing or empty — creating minimal fallback"
        printf '.\n,\n!\n?\n;\n:\n"\n'"'"'\n(\n)\n-\n—\n…\n' \
            > "$LANGDATA_DIR/kan/kan.puncs"
    fi
    EXTRA_FLAGS="$EXTRA_FLAGS --words $LANGDATA_DIR/kan/kan.wordlist"
    EXTRA_FLAGS="$EXTRA_FLAGS --puncs $LANGDATA_DIR/kan/kan.puncs"
fi
[ -s "$LANGDATA_DIR/kan/kan.numbers" ] && EXTRA_FLAGS="$EXTRA_FLAGS --numbers $LANGDATA_DIR/kan/kan.numbers"
# Note: --unicharambigs is not supported by all combine_lang_model versions — omitted

combine_lang_model \
    --input_unicharset "$WORK_DIR/kan_expanded.unicharset" \
    --script_dir "$LANGDATA_DIR" \
    --lang kan \
    --output_dir "$OUTPUT_TESSDATA/" \
    $EXTRA_FLAGS

echo ""

# combine_lang_model writes to <output_dir>/<lang>/<lang>.traineddata
# Move it up one level so 03-train.sh finds it at tessdata_expanded/kan.traineddata
NESTED="$OUTPUT_TESSDATA/kan/kan.traineddata"
if [ -f "$NESTED" ]; then
    mv "$NESTED" "$OUTPUT_TESSDATA/kan.traineddata"
    rmdir "$OUTPUT_TESSDATA/kan" 2>/dev/null || true
fi

if [ -f "$OUTPUT_TESSDATA/kan.traineddata" ]; then
    SIZE=$(du -h "$OUTPUT_TESSDATA/kan.traineddata" | cut -f1)
    NEW_COUNT=$(head -1 "$WORK_DIR/kan_expanded.unicharset")
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  ✓ Done!"
    echo ""
    echo "  Created: tessdata_expanded/kan.traineddata ($SIZE)"
    echo "  Unicharset: $BASE_COUNT (upstream) → $NEW_COUNT entries"
    echo "  Added: ಋ (U+0C8B)  ಙ (U+0C99)  ಝ (U+0C9D)  ಱ (U+0CB1)"
    echo ""
    echo "  Next steps:"
    echo "  1. Re-run ④ Make lstmf  — filter is now lifted for these chars"
    echo "  2. Re-run ⑤ Train       — will use tessdata_expanded/ automatically"
    echo "     (first ~5000 iters may show higher BCER as new nodes initialise)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
else
    echo "ERROR: tessdata_expanded/kan.traineddata was not created."
    echo "  combine_lang_model may have written to a different path."
    echo "  Check for: tessdata_expanded/kan/kan.traineddata"
    ls -R "$OUTPUT_TESSDATA/" 2>/dev/null || true
    exit 1
fi
