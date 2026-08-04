#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# check-line-crops.sh — prove the line-cropping fix before committing
#                       hours to a full classical re-render
#
# WHY
#   measureLinesInPage() used to group characters by matching rect tops. On a
#   Kannada line the glyph tops vary a lot — a bare consonant, one with an
#   ascender matra, and a below-base ottu all start at different heights — so a
#   single visual line was split into several groups, each holding part of the
#   text but only a sliver of the height. That produced crops like 386x10 with a
#   full 16-character transcription and 0% ink: the glyphs were outside the box.
#
#   Tesseract reported those as "Failed to read boxes" and "truncated file",
#   which reads like disk corruption and sends the investigation the wrong way.
#
# WHAT THIS DOES
#   Renders a few pages to a temp directory with --lines and checks every crop:
#     • height >= 16px          (a sliver cannot contain a text line)
#     • ink >= 0.1%             (a blank crop cannot match its transcription)
#     • CTC feasible            (timesteps at 48px height >= label count)
#     • ground truth present and non-empty
#
#   Nothing under classical-corpus-kannada/ is touched.
#
# USAGE
#   ./scripts/check-line-crops.sh                 # 3 pages, default font
#   ./scripts/check-line-crops.sh 6               # 6 pages
#   ./scripts/check-line-crops.sh 3 kan_wmp       # specific font id
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$SCRIPT_DIR/.."
cd "$ROOT"

PAGES="${1:-3}"
FONT_ID="${2:-}"

# Validate rather than trust: a pasted trailing comment ("./check-line-crops.sh
# 6   # more thorough") arrives as a literal '#' argument in some shells, which
# then blew up deep inside the checker with an unrelated int() error.
case "$PAGES" in
  ''|*[!0-9]*) echo "✗ page count must be a number, got: '$PAGES'"
               echo "  usage: $0 [pages] [font_id]"; exit 1 ;;
esac
case "$FONT_ID" in
  *[!A-Za-z0-9_.-]*) echo "✗ font id looks wrong: '$FONT_ID' — ignoring it"; FONT_ID="" ;;
esac
OUT="$(mktemp -d /tmp/linecheck.XXXXXX)"
CORPUS="${CLASSICAL_CORPUS_DIR:-$ROOT/classical-corpus-kannada}"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Line-crop verification — $PAGES page(s)"
echo "  Output (temporary): $OUT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

[ -d "$CORPUS" ] || { echo "✗ classical corpus not found: $CORPUS"; exit 1; }

# Pick the first title with a .txt source
TITLE_TXT="$(find "$CORPUS" -maxdepth 2 -name '*.txt' | head -1)"
[ -n "$TITLE_TXT" ] || { echo "✗ no source .txt found under $CORPUS"; exit 1; }
TITLE="$(basename "$(dirname "$TITLE_TXT")")"
echo "  Title: $TITLE"

ARGS=(--corpus-dir "$CORPUS" --output-dir "$OUT" --lines --title "$TITLE" --workers 1)
[ -n "$FONT_ID" ] && echo "  Font filter: $FONT_ID"

echo ""
echo "→ Rendering (this touches nothing in the corpus)…"
python3 -u corpus/render-a5-pages.py "${ARGS[@]}" 2>&1 | tail -6 || {
  echo "✗ render failed — fix that before the full run"; exit 1; }

echo ""
echo "→ Checking every crop produced…"
python3 - "$OUT" "$PAGES" <<'PY'
import sys
from pathlib import Path
try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("  ✗ Pillow/numpy required for the check"); sys.exit(1)

out = Path(sys.argv[1])
crops = sorted(out.rglob('*_line*.png'))
if not crops:
    print("  ✗ no line crops produced — the render did not run in line mode")
    sys.exit(1)

limit = int(sys.argv[2]) * 60          # cap the check; pages have ~15-40 lines
crops = crops[:limit]

short = blank = noctc = nogt = ok = 0
examples = []
for f in crops:
    gt = Path(str(f)[:-4] + '.gt.txt')
    txt = gt.read_text(encoding='utf-8').strip() if gt.exists() else ''
    with Image.open(f) as im:
        w, h = im.size
        ink = float((np.asarray(im.convert('L')) < 128).mean())
    if not txt:
        nogt += 1;  examples.append(('no ground truth', f.name, f'{w}x{h}')); continue
    if h < 16:
        short += 1; examples.append(('sliver crop', f.name, f'{w}x{h}')); continue
    if ink < 0.001:
        blank += 1; examples.append(('blank', f.name, f'{w}x{h} {ink*100:.2f}% ink')); continue
    if int(w * (48.0 / h)) < len(txt):
        noctc += 1; examples.append(('CTC infeasible', f.name, f'{len(txt)} labels')); continue
    ok += 1

tot = len(crops)
print(f"  checked        : {tot} crops")
print(f"  usable         : {ok}  ({100*ok/tot:.1f}%)")
print(f"  sliver (<16px) : {short}")
print(f"  blank          : {blank}")
print(f"  CTC infeasible : {noctc}")
print(f"  missing gt     : {nogt}")

if examples:
    print("\n  first failures:")
    for kind, name, detail in examples[:6]:
        print(f"    {kind:16s} {name:28s} {detail}")

bad = short + blank + noctc + nogt
print()
if bad == 0:
    print("  ✓ PASS — every crop is usable. Safe to run the full re-render:")
    print("      ./scripts/run-pipeline.sh --with-classical --from classical")
    sys.exit(0)
else:
    pct = 100 * bad / tot
    print(f"  ✗ FAIL — {bad} unusable crop(s) ({pct:.1f}%). Do NOT start the full")
    print("    re-render; the cropping still needs work.")
    sys.exit(1)
PY
rc=$?

echo ""
echo "  Temp output left at: $OUT"
echo "  Remove with: rm -rf $OUT"
exit $rc
