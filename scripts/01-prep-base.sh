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

SKIP_DIRS = {'webfonts', 'Source', 'source', '.git', 'Tests'}

def has_font_files(d: Path) -> bool:
    """True when the directory already holds usable font files at any depth."""
    if not d.is_dir():
        return False
    for p in list(d.rglob('*.ttf')) + list(d.rglob('*.otf')):
        if not any(part in SKIP_DIRS for part in p.parts):
            return True
    return False

def is_git_remote(url: str) -> bool:
    """
    Cheap check for a clonable remote.

    Not every font is distributed as a git repo — Google Fonts families are
    downloaded from a specimen page. Attempting `git clone` on such a URL fails
    with exit 128 and, under `set -e`, used to abort the whole prep step before
    the base model was ever downloaded.
    """
    u = (url or '').lower()
    if u.endswith('.git') or u.startswith('git@'):
        return True
    return any(h in u for h in ('github.com', 'gitlab.com', 'bitbucket.org', 'codeberg.org'))

failed, manual = [], []

for font in data['fonts']:
    fid   = font['id']
    repo  = font.get('repo', '')
    dest  = fonts_dir / fid

    print(f"  [{fid}] {repo or '(no repo)'}")

    # Presence is judged by actual font files, not by a .git dir — repos may be
    # cloned then stripped of .git, and downloaded families never have one.
    if has_font_files(dest):
        print(f"    ✓ font files present — skipping")
        continue

    if font.get('clone') is False or not is_git_remote(repo):
        print(f"    ⚠ not a git remote — download manually into fonts/{fid}/")
        print(f"      then re-run this step. Source: {repo}")
        manual.append(fid)
        continue

    # One unreachable repo must not abort prep for every other font.
    r = subprocess.run(['git', 'clone', '--depth', '1', repo, str(dest)])
    if r.returncode == 0:
        print(f"    → cloned to {dest}")
    else:
        print(f"    ✗ clone failed (exit {r.returncode})")
        failed.append(fid)

if manual:
    print(f"\n  ⚠ {len(manual)} font(s) need manual download: {', '.join(manual)}")
if failed:
    print(f"\n  ✗ {len(failed)} font(s) failed to clone: {', '.join(failed)}")
    sys.exit(1)
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
