#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# run-pipeline.sh — ordered, gated, measurable end-to-end rebuild
#
# WHY THIS EXISTS
#   The steps have a strict dependency order, and getting it wrong fails
#   silently rather than loudly:
#
#     • the unicharset defines the recoder, so changing it invalidates every
#       checkpoint AND every .lstmf built against the old one
#     • .lstmf files cache the ground truth, so a corpus fix has no effect
#       until they are rebuilt
#     • 02-make-lstmf.sh resumes from existing .lstmf files, so a stale build
#       silently survives unless cleared
#
#   Running the steps by hand in the wrong order produced a training run that
#   spent 23% of every batch on samples it could never learn from.
#
# USAGE
#   ./scripts/run-pipeline.sh                 # full rebuild, then stop before training
#   ./scripts/run-pipeline.sh --train         # ... and start training at the end
#   ./scripts/run-pipeline.sh --from lstmf    # resume partway
#   ./scripts/run-pipeline.sh --dry-run       # show the plan and current state only
#   ./scripts/run-pipeline.sh --with-classical  # include the ~hours-long A5 line render
#
# Every stage prints a metric line so successive runs can be compared.
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$SCRIPT_DIR/.."
cd "$ROOT"

mkdir -p logs
PIPELINE_LOG="$ROOT/logs/pipeline.log"

# Root of the classical corpus — the folder holding per-title .txt sources and
# the generated a5-pages/. Override for a corpus kept outside the repo:
#   CLASSICAL_CORPUS_DIR=/path/to/classical-corpus-kannada ./scripts/run-pipeline.sh
CLASSICAL_DIR="${CLASSICAL_CORPUS_DIR:-$ROOT/classical-corpus-kannada}"
CLASSICAL_A5="$CLASSICAL_DIR/a5-pages"

DRY_RUN=0; DO_TRAIN=0; WITH_CLASSICAL=0; FROM="unicharset"
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run)        DRY_RUN=1 ;;
    --train)          DO_TRAIN=1 ;;
    --with-classical) WITH_CLASSICAL=1 ;;
    --from)           FROM="${2:-}"; shift ;;
    -h|--help)        sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
  shift
done

STAGES=(unicharset corpus render inventory classical lstmf validate train)
started=0
should_run() {
  [ "$1" = "$FROM" ] && started=1
  [ $started -eq 1 ]
}

hr()  { printf '━%.0s' $(seq 1 60); echo; }
say() { echo "$*" | tee -a "$PIPELINE_LOG"; }
stage() { hr; say "▶ $1"; hr; }

metric() { say "   📊 $1"; }

say ""
say "═══ pipeline run $(date '+%Y-%m-%d %H:%M:%S') ═══"

# ── Baseline snapshot ────────────────────────────────────────────────────────
snapshot() {
  local inv rend cls lst
  inv=$(find inventory -name '*.png' 2>/dev/null | wc -l | tr -d ' ')
  rend=$(ls rendered/*.png 2>/dev/null | wc -l | tr -d ' ')
  cls=$(find "$CLASSICAL_A5" -name '*_line*.png' 2>/dev/null | wc -l | tr -d ' ')
  lst=$(wc -l < lstmf/list.txt 2>/dev/null | tr -d ' ' || echo 0)
  say "   inventory=$inv  rendered=$rend  classical-lines=$cls  lstmf-list=$lst"
}

# ── Preflight: fail fast on a toolchain that cannot produce valid output ─────
# Checked up front rather than at the lstmf stage, so a wrong Tesseract is
# caught before spending minutes on rendering.
TESS_VER="$(tesseract --version 2>&1 | head -1 || echo 'not found')"
case "$TESS_VER" in
  "tesseract 5."*) say "Tesseract   : $TESS_VER  ✓" ;;
  "tesseract 4."*)
      say ""
      say "✗ $TESS_VER — Tesseract 4 cannot encode the Kannada virama (್)."
      say "  It writes .lstmf files that only fail later, during training."
      say "  Install Tesseract 5:  brew install tesseract"
      exit 1 ;;
  *)
      say ""
      say "✗ Tesseract not found. Install version 5:  brew install tesseract"
      exit 1 ;;
esac

say ""
say "Current state:"; snapshot

if [ $DRY_RUN -eq 1 ]; then
  say ""
  say "Plan (from '$FROM'):"
  started=0
  for s in "${STAGES[@]}"; do
    [ "$s" = "train" ] && [ $DO_TRAIN -eq 0 ] && continue
    [ "$s" = "classical" ] && [ $WITH_CLASSICAL -eq 0 ] && continue
    should_run "$s" && say "   • $s"
  done
  say ""
  say "(dry run — nothing executed)"
  exit 0
fi

if pgrep -x lstmtraining >/dev/null 2>&1; then
  say ""
  say "⚠  lstmtraining is running. Rebuilding data underneath it will not affect"
  say "   the live job (it has already read its list) and the new data will only"
  say "   apply on restart. Stop it first:  pkill lstmtraining"
  say ""
  read -r -p "   Continue anyway? [y/N] " reply
  [ "$reply" = "y" ] || exit 1
fi

started=0

# ── 1. Unicharset ────────────────────────────────────────────────────────────
# MUST be first: it defines the recoder. Changing it later invalidates
# everything built before it.
if should_run unicharset; then
  stage "1/7  Expand unicharset  (adds ಋ ಙ ಝ ಱ ೃ ಞ ೞ)"
  ./scripts/00c-expand-unicharset.sh --force 2>&1 | tee -a "$PIPELINE_LOG" | tail -5
  metric "unicharset units: $(
    combine_tessdata -u tessdata_expanded/kan.traineddata /tmp/_pl. >/dev/null 2>&1 &&
    head -1 /tmp/_pl.lstm-unicharset 2>/dev/null || echo '?')"
fi

# ── 2. Corpus ────────────────────────────────────────────────────────────────
if should_run corpus; then
  stage "2/7  Clean corpus  (strip unassigned codepoints)"
  python3 corpus/clean-corpus.py 2>&1 | tee -a "$PIPELINE_LOG" | tail -6
  metric "corpus lines: $(wc -l < corpus/kan_corpus.txt | tr -d ' ')"
fi

# ── 3. Render ────────────────────────────────────────────────────────────────
# --force because a corpus or shaping change must overwrite existing images;
# without it render-corpus.py skips every file that already exists.
if should_run render; then
  stage "3/7  Render synthetic lines  (--force)"
  python3 corpus/render-corpus.py --force 2>&1 | tee -a "$PIPELINE_LOG" | tail -3
  metric "rendered images: $(ls rendered/*.png 2>/dev/null | wc -l | tr -d ' ')"
fi

# ── 4. Inventory ─────────────────────────────────────────────────────────────
if should_run inventory; then
  stage "4/7  Character inventory"
  python3 corpus/generate-inventory.py 2>&1 | tee -a "$PIPELINE_LOG" | tail -5
  metric "inventory images: $(find inventory -name '*.png' | wc -l | tr -d ' ')"
fi

# ── 5. Classical (optional, slow) ────────────────────────────────────────────
if [ $WITH_CLASSICAL -eq 1 ] && should_run classical; then
  stage "5/7  Classical A5 → LINE images  (hours; ~430K files)"

  if [ ! -d "$CLASSICAL_DIR" ]; then
    say "   ✗ classical corpus not found at: $CLASSICAL_DIR"
    say "     Set CLASSICAL_CORPUS_DIR=/path/to/classical-corpus-kannada"
    exit 1
  fi
  titles=$(find "$CLASSICAL_DIR" -maxdepth 2 -name '*.txt' 2>/dev/null | wc -l | tr -d ' ')
  if [ "$titles" -eq 0 ]; then
    say "   ✗ no .txt source texts under $CLASSICAL_DIR — nothing to render"
    exit 1
  fi
  say "   corpus: $CLASSICAL_DIR  ($titles source texts)"

  python3 corpus/render-a5-pages.py \
    --corpus-dir "$CLASSICAL_DIR" \
    --lines --workers 4 2>&1 | tee -a "$PIPELINE_LOG" | tail -5
  metric "classical line images: $(find "$CLASSICAL_A5" -name '*_line*.png' 2>/dev/null | wc -l | tr -d ' ')"
fi

# ── 6. lstmf ─────────────────────────────────────────────────────────────────
# Clear first: 02-make-lstmf.sh resumes from existing .lstmf files. After a
# unicharset or corpus change the cached ones are stale and would be re-admitted.
if should_run lstmf; then
  stage "6/7  Build lstmf  (clearing stale cache first)"
  rm -rf lstmf/rendered lstmf/inventory lstmf/classical lstmf/font-test lstmf/list.txt 2>/dev/null || true
  INVENTORY_DIR="$ROOT/inventory" \
  CLASSICAL_A5_DIR="$CLASSICAL_A5" \
    ./scripts/02-make-lstmf.sh 2>&1 | tee -a "$PIPELINE_LOG" | tail -8
  metric "lstmf entries: $(wc -l < lstmf/list.txt | tr -d ' ')"
fi

# ── 7. Validate before committing hours of GPU time ──────────────────────────
if should_run validate; then
  stage "7/7  Validate training set"
  python3 - <<'PY' 2>&1 | tee -a "$PIPELINE_LOG"
import subprocess, tempfile, os, collections
from pathlib import Path
from PIL import Image

td = 'tessdata_expanded/kan.traineddata'
if not os.path.exists(td):
    td = 'tessdata_best/kan.traineddata'
with tempfile.TemporaryDirectory() as t:
    p = os.path.join(t, 'kan.')
    subprocess.run(['combine_tessdata', '-u', td, p], capture_output=True, check=True)
    U = {l.split(' ')[0] for l in
         open(p + 'lstm-unicharset', encoding='utf-8', errors='replace').read().split('\n')[1:]
         if l.strip()}
M = max(len(u) for u in U); EX = set(' \t\n್')

def enc(s):
    i, n = 0, len(s)
    while i < n:
        if s[i] in EX: i += 1; continue
        for k in range(min(M, n - i), 0, -1):
            if s[i:i+k] in U: i += k; break
        else: return False
    return True

srcs = {}
for root in ('rendered', 'inventory', 'test-images', 'classical-corpus-kannada'):
    rp = Path(root)
    if rp.exists():
        for g in rp.rglob('*.gt.txt'):
            srcs[Path(str(g)[:-7]).name] = g

by_kind = collections.Counter(); bad_enc = []; bad_ctc = []; total = 0
for ln in open('lstmf/list.txt', encoding='utf-8'):
    ln = ln.strip()
    if not ln: continue
    total += 1
    parts = Path(ln).parts
    by_kind[parts[parts.index('lstmf') + 1] if 'lstmf' in parts else '?'] += 1
    stem = Path(ln).stem
    g = srcs.get(stem)
    if not g: continue
    txt = g.read_text(encoding='utf-8', errors='ignore').strip()
    if txt and not enc(txt):
        bad_enc.append(stem); continue
    img = None
    for ext in ('.png', '.tif'):
        c = Path(str(g)[:-7] + ext)
        if c.exists(): img = c; break
    if img:
        w, h = Image.open(img).size
        if txt and int(w * (48.0 / h)) < len(txt):
            bad_ctc.append(stem)

print(f"   total entries : {total}")
for k, v in by_kind.most_common():
    print(f"     {k:12s} {v}")
print(f"   unencodable   : {len(bad_enc)}")
print(f"   CTC infeasible: {len(bad_ctc)}")
problems = len(bad_enc) + len(bad_ctc)
pct = 100 * problems / max(1, total)
print(f"   → expected skip ratio ≈ {pct:.1f}%")
if pct > 5:
    print(f"   ⚠  ABOVE 5% — investigate before training. Samples: "
          f"{(bad_enc + bad_ctc)[:3]}")
else:
    print("   ✓ training set is clean")
PY
fi

# ── 8. Train ─────────────────────────────────────────────────────────────────
if [ $DO_TRAIN -eq 1 ] && should_run train; then
  stage "Training  (TRAIN_MODE=fresh — required after a unicharset change)"
  say "   logs → logs/training.log"
  TRAIN_MODE=fresh ./scripts/03-train.sh
else
  hr
  say "Pipeline complete. Review the validation numbers above, then start training:"
  say "   TRAIN_MODE=fresh caffeinate -i ./scripts/03-train.sh"
  hr
fi

say ""
say "Final state:"; snapshot
say "Full log: logs/pipeline.log"
