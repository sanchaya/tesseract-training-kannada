#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# 00c-expand-unicharset.sh
#
# Adds missing Kannada characters to kan.traineddata's unicharset,
# producing tessdata_expanded/kan.traineddata.
#
# Characters added (absent from tessdata_best/kan.traineddata):
#   ಋ (U+0C8B)  KANNADA LETTER VOCALIC R
#   ಙ (U+0C99)  KANNADA LETTER NGA
#   ಝ (U+0C9D)  KANNADA LETTER JHA
#   ಱ (U+0CB1)  KANNADA LETTER RRA
#   ೃ (U+0CC3)  KANNADA VOWEL SIGN VOCALIC R  ← very common in Sanskrit-
#               origin classical Kannada: ನೃಪ ಮೃ ಕೃ ತೃ ಕೃಷ್ಣ ಮೃತ್ಯು…
#   ಞ (U+0C9E)  KANNADA LETTER NYA  ← present in the base unicharset only
#               INSIDE cluster units such as ಜ್ಞ; standalone ಞ cannot be
#               encoded, so any line containing it is skipped (177 corpus lines)
#   ೞ (U+0CDE)  KANNADA LETTER FA (archaic ḷa) ← appears in 9,610 classical
#               ground-truth files — the single largest source of encoding
#               failures in the classical corpus
#
# WHY: The tessdata_best unicharset omits these characters.
# lstmtraining skips any training line that contains them, producing
# "Encoding of string failed" / "Compute CTC targets failed" errors.
# After running this script, retraining from the existing checkpoint
# auto-expands the output layer to cover the new characters.
#
# What it does:
#   1. Downloads kan/ langdata from GitHub (first run only; cached)
#      https://github.com/tesseract-ocr/langdata_lstm/tree/main/kan
#   2. Extracts the binary lstm-unicharset from tessdata_best so the
#      expanded set is a proper superset (required by lstmtraining)
#   3. Merges the missing characters into kan.unicharset
#   4. Copies the expanded unicharset as Kannada/Kannada.unicharset
#   5. Runs combine_lang_model → tessdata_expanded/kan.traineddata
#
# After this script:
#   • Re-run ④ Make lstmf  (character filters auto-lifted)
#   • Re-run ⑤ Train       (auto-uses tessdata_expanded/kan.traineddata)
#
# If you previously ran this script (before ೃ was added), rebuild with:
#   ./scripts/00c-expand-unicharset.sh --force
# Then: rm -rf lstmf/classical/ && re-run ④ Make lstmf → ⑤ Train
#
# Usage:
#   ./scripts/00c-expand-unicharset.sh           # normal (cached downloads)
#   ./scripts/00c-expand-unicharset.sh --force   # re-download all, rebuild
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

MISSING_CHARS="ಋ ಙ ಝ ಱ ೃ ಞ ೞ"
BASE_URL="https://raw.githubusercontent.com/tesseract-ocr/langdata_lstm/main/kan"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Expand Kannada Unicharset"
echo "  Adding: $MISSING_CHARS"
echo "  Base:   tessdata_best/kan.traineddata (binary lstm-unicharset)"
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
# combine_lang_model also needs Latin.unicharset.  combine_lang_model looks for:
#   $LANGDATA_DIR/Latin/Latin.unicharset   (subdirectory form, preferred)
#   $LANGDATA_DIR/Latin.unicharset         (root symlink, fallback)
# We try four routes in order of reliability:
#   1. GitHub langdata_lstm raw file (direct, ~20 KB)
#   2. Extract from system Latin.traineddata (if installed)
#   3. Extract from system eng.traineddata  (always present after brew install tesseract)
#   4. Download script/Latin.traineddata from tessdata_best and extract (reliable)
_install_latin_unicharset() {
    local dest_file="$LANGDATA_DIR/Latin/Latin.unicharset"
    local dest_link="$LANGDATA_DIR/Latin.unicharset"
    local tmp_td="$WORK_DIR/_lextract"
    local GOT=0

    # 1. GitHub langdata_lstm
    if curl -fsSL \
            "https://raw.githubusercontent.com/tesseract-ocr/langdata_lstm/main/Latin/Latin.unicharset" \
            -o "$dest_file" 2>/dev/null \
       && [ -s "$dest_file" ]; then
        echo "  ✓ Latin.unicharset (GitHub langdata_lstm)"
        GOT=1
    fi

    # 2. System Latin.traineddata
    if [ "$GOT" = "0" ]; then
        for _td in /opt/homebrew/share/tessdata /usr/local/share/tessdata /usr/share/tessdata \
                   /opt/homebrew/share/tessdata/script /usr/local/share/tessdata/script; do
            if [ -f "$_td/Latin.traineddata" ]; then
                combine_tessdata -u "$_td/Latin.traineddata" "$tmp_td" 2>/dev/null || true
                if [ -s "${tmp_td}.lstm-unicharset" ]; then
                    cp "${tmp_td}.lstm-unicharset" "$dest_file"
                    rm -f "${tmp_td}".*
                    echo "  ✓ Latin.unicharset (extracted from $_td/Latin.traineddata)"
                    GOT=1; break
                fi
                rm -f "${tmp_td}".*
            fi
        done
    fi

    # 3. System eng.traineddata (eng is always installed by brew install tesseract)
    if [ "$GOT" = "0" ]; then
        # Find tessdata dir via TESSDATA_PREFIX or common Homebrew/system paths
        _tdata_dirs="${TESSDATA_PREFIX:-}"
        for _td in /opt/homebrew/share/tessdata /usr/local/share/tessdata /usr/share/tessdata; do
            _tdata_dirs="$_tdata_dirs $_td"
        done
        for _td in $_tdata_dirs; do
            [ -f "$_td/eng.traineddata" ] || continue
            combine_tessdata -u "$_td/eng.traineddata" "$tmp_td" 2>/dev/null || true
            if [ -s "${tmp_td}.lstm-unicharset" ]; then
                cp "${tmp_td}.lstm-unicharset" "$dest_file"
                rm -f "${tmp_td}".*
                echo "  ✓ Latin.unicharset (derived from $_td/eng.traineddata)"
                GOT=1; break
            fi
            rm -f "${tmp_td}".*
        done
    fi

    # 4. Download script/Latin.traineddata from tessdata_best, then extract
    if [ "$GOT" = "0" ]; then
        echo "  → GitHub routes failed; downloading script/Latin.traineddata (~22 MB)..."
        local _lat_td="$WORK_DIR/Latin.traineddata"
        if curl -fL --progress-bar \
                "https://github.com/tesseract-ocr/tessdata_best/raw/main/script/Latin.traineddata" \
                -o "$_lat_td" 2>/dev/null \
           && [ -s "$_lat_td" ]; then
            combine_tessdata -u "$_lat_td" "$tmp_td" 2>/dev/null || true
            if [ -s "${tmp_td}.lstm-unicharset" ]; then
                cp "${tmp_td}.lstm-unicharset" "$dest_file"
                rm -f "${tmp_td}".* "$_lat_td"
                echo "  ✓ Latin.unicharset (from tessdata_best script/Latin.traineddata)"
                GOT=1
            fi
            rm -f "${tmp_td}".* "$_lat_td"
        fi
    fi

    if [ "$GOT" = "1" ]; then
        # Create/refresh the root-level symlink (relative so it survives mv)
        ln -sf "Latin/Latin.unicharset" "$dest_link" 2>/dev/null || \
            cp "$dest_file" "$dest_link"
        return 0
    else
        echo "  ✗ Latin.unicharset not found — combine_lang_model WILL fail."
        echo "    Fix: brew install tesseract  (installs eng.traineddata)"
        echo "    Or:  curl -fsSL https://raw.githubusercontent.com/tesseract-ocr/langdata_lstm/main/Latin/Latin.unicharset \\"
        echo "              -o tmp/langdata_lstm/Latin/Latin.unicharset"
        return 1
    fi
}

if [ ! -s "$LANGDATA_DIR/Latin/Latin.unicharset" ]; then
    echo "→ Fetching Latin.unicharset (required by combine_lang_model)..."
    _install_latin_unicharset || exit 1
fi

[ -f "$LANGDATA_DIR/kan/kan.unicharset" ] || {
    echo ""
    echo "ERROR: Could not download kan/kan.unicharset from GitHub."
    echo "  Check your internet connection, or manually place the file at:"
    echo "  tmp/langdata_lstm/kan/kan.unicharset"
    echo "  Source: https://github.com/tesseract-ocr/langdata_lstm/tree/main/kan"
    exit 1
}

# ── Save a pristine copy of the langdata unicharset ───────────────────────────
# Step 6a overwrites kan/kan.unicharset with our merged output. The pristine
# download is used in the two-pass merge (Step 5) so standard chars like ಙ
# that are in the upstream langdata but not in tessdata_best are never lost.
# Refreshed on --force; otherwise cached across runs.
if [ "$FORCE" = "1" ] || [ ! -f "$WORK_DIR/kan_langdata_pristine.unicharset" ]; then
    cp "$LANGDATA_DIR/kan/kan.unicharset" "$WORK_DIR/kan_langdata_pristine.unicharset"
fi

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

# Check which chars are missing from the BINARY unicharset
python3 -c "
data = open('$BASE_UNICHARSET').read()
missing = []
for ch in 'ಋಙಝಱೃ':
    if ch not in data:
        missing.append(f'{ch} (U+{ord(ch):04X})')
if missing:
    print('  Missing from lstm-unicharset: ' + ', '.join(missing))
else:
    print('  All characters already in lstm-unicharset — nothing to add.')
    exit(1)
" || {
    echo "  Nothing to add — tessdata_expanded not needed."
    echo "  tessdata_best/kan.traineddata already covers all required chars."
    exit 0
}

# ── Step 3: Create corpus with the missing characters ─────────
echo ""
echo "→ Step 3: Creating corpus with missing characters..."
cat > "$WORK_DIR/new_chars.txt" << 'EOF'
ಋ ಙ ಝ ಱ ಞ ೞ
ಋಷಿ ಋಣ ಋತು ಋಗ್ವೇದ ಋಜು ಋಕ್ಷ
ಝರ ಝಲ ಝಳ ಝಂಕ ಝಗ ಝಲಕ
ಗಾಱ ಅಱ ಕಱ ತಱ ಮಱ
ಮಙ ಲಙ ಅಙ ಪಙ ಮಙ್ಕ ಪಙ್ಕ
ನೃಪ ನೃಪತಿ ಮೃತ್ಯು ಕೃತ್ಯ ಕೃಷ್ಣ ತೃಪ್ತಿ ಮೃದು ವೃತ್ತಿ
ಞಾನ ಅಞ ಮಞ ಪಞ ವಿಜ್ಞಾನ ಸಂಜ್ಞೆ ಆಜ್ಞೆ ಯಜ್ಞ
ೞ ಪೞ ಕೞ ಮೞ ಎೞ ತೞ ಪೞೆಯ ಕೞಿ ಮೞೆ ಎೞ್ತು
EOF
# IMPORTANT: ೃ (U+0CC3, KANNADA VOWEL SIGN VOCALIC R) must NOT appear
# standalone here — unicharset_extractor rejects bare combining chars:
#   "Invalid start of grapheme sequence:M=0xcc3"
#   "Normalization failed for string '...ೃ'"
# This causes the ENTIRE LINE to be skipped, losing all other chars on it
# (including ಙ).  ೃ is correctly extracted from word context: ನೃ, ಮೃ, ಕೃ…

# ── Step 4: Extract unicharset from corpus ─────────────────────
echo "→ Step 4: Extracting unicharset from new characters..."
unicharset_extractor \
    --output_unicharset "$WORK_DIR/new_chars.unicharset" \
    "$WORK_DIR/new_chars.txt"

# ── Step 5: Two-pass merge ─────────────────────────────────────────────────────
# Pass A: BASE (tessdata_best binary) ∪ pristine langdata kan.unicharset
#   The downloaded langdata unicharset has ALL standard Kannada chars including
#   ಙ (U+0C99) — which IS in the upstream file but was missing from tessdata_best.
#   Using the pristine copy saved earlier (not kan/kan.unicharset which Step 6a
#   will overwrite) ensures chars are never lost across consecutive script runs.
# Pass B: intermediate ∪ new_chars
#   Adds chars not in either above (e.g. ೃ U+0CC3 if absent from langdata).
echo "→ Step 5: Merging into expanded unicharset (two-pass)..."

LANGDATA_UC="$WORK_DIR/kan_langdata_pristine.unicharset"
[ -f "$LANGDATA_UC" ] || LANGDATA_UC="$LANGDATA_DIR/kan/kan.unicharset"

merge_unicharsets \
    "$BASE_UNICHARSET" \
    "$LANGDATA_UC" \
    "$WORK_DIR/kan_intermediate.unicharset"

merge_unicharsets \
    "$WORK_DIR/kan_intermediate.unicharset" \
    "$WORK_DIR/new_chars.unicharset" \
    "$WORK_DIR/kan_expanded.unicharset"

NEW_COUNT=$(head -1 "$WORK_DIR/kan_expanded.unicharset")
echo "  Unicharset: $BASE_COUNT (base) → $NEW_COUNT entries"

# Verify
python3 -c "
data = open('$WORK_DIR/kan_expanded.unicharset').read()
all_ok = True
for ch in 'ಋಙಝಱೃ':
    status = '✓' if ch in data else '✗ STILL MISSING'
    print(f'  {ch} (U+{ord(ch):04X}): {status}')
    if ch not in data:
        all_ok = False
if not all_ok:
    exit(1)
"

# ── Step 6a: Write expanded unicharset back to kan/kan.unicharset ─────────────
# This overwrites the langdata download — but a pristine copy was saved in
# tmp/unicharset_work/kan_langdata_pristine.unicharset for future runs.
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
    echo "  Unicharset: $BASE_COUNT (base) → $NEW_COUNT entries"
    echo "  Added: ಋ (U+0C8B)  ಙ (U+0C99)  ಝ (U+0C9D)  ಱ (U+0CB1)  ೃ (U+0CC3)"
    echo ""
    echo "  Next steps:"
    echo "  1. Regenerate classical lstmf files (old ones used limited unicharset):"
    echo "       rm -rf lstmf/classical/"
    echo "       CLASSICAL_A5_DIR=<path>/a5-pages ./scripts/02-make-lstmf.sh"
    echo "  2. Re-run ⑤ Train — will use tessdata_expanded/ + full classical data"
    echo "     (first ~5000 iters may show higher BCER as new output nodes warm up)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
else
    echo "ERROR: tessdata_expanded/kan.traineddata was not created."
    echo "  combine_lang_model may have written to a different path."
    echo "  Check for: tessdata_expanded/kan/kan.traineddata"
    ls -R "$OUTPUT_TESSDATA/" 2>/dev/null || true
    exit 1
fi
