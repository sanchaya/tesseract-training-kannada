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

# The portal's Live Log tails logs/training.log and only that file. Writing
# pipeline output solely to pipeline.log made the portal look frozen for the
# hours a rebuild takes — nothing was broken, it was just watching a different
# file. Mirror into both: pipeline.log is this script's own record, training.log
# is the portal's stream.
PORTAL_LOG="$ROOT/logs/training.log"
LOG_TARGETS=("$PIPELINE_LOG" "$PORTAL_LOG")

# Root of the classical corpus — the folder holding per-title .txt sources and
# the generated a5-pages/. Override for a corpus kept outside the repo:
#   CLASSICAL_CORPUS_DIR=/path/to/classical-corpus-kannada ./scripts/run-pipeline.sh
CLASSICAL_DIR="${CLASSICAL_CORPUS_DIR:-$ROOT/classical-corpus-kannada}"
CLASSICAL_A5="$CLASSICAL_DIR/a5-pages"

DRY_RUN=0; DO_TRAIN=0; WITH_CLASSICAL=0; FROM="coverage"
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

STAGES=(coverage unicharset corpus render inventory classical scans lstmf validate train)
started=0
should_run() {
  [ "$1" = "$FROM" ] && started=1
  [ $started -eq 1 ]
}

hr()  { printf '━%.0s' $(seq 1 60); echo; }
say() { echo "$*" | tee -a "${LOG_TARGETS[@]}"; }
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

# ── 0. Coverage analysis ─────────────────────────────────────────────────────
# MUST precede the unicharset stage: it mines the corpus for grapheme clusters
# that cannot currently be encoded and appends real example words to
# 00c-expand-unicharset.sh, which the next stage turns into unicharset units.
# Without this, gaps are found one at a time by training failures.
if should_run coverage; then
  stage "1/9  Coverage analysis  (find clusters the unicharset cannot encode)"
  python3 -u scripts/find-missing-clusters.py --kannada-only --update-00c 2>&1 \
    | tee -a "${LOG_TARGETS[@]}" | tail -14
fi

# ── 1. Unicharset ────────────────────────────────────────────────────────────
# MUST be first: it defines the recoder. Changing it later invalidates
# everything built before it.
if should_run unicharset; then
  stage "2/9  Expand unicharset  (adds ಋ ಙ ಝ ಱ ೃ ಞ ೞ)"
  ./scripts/00c-expand-unicharset.sh --force 2>&1 | tee -a "${LOG_TARGETS[@]}" | tail -5
  metric "unicharset units: $(
    combine_tessdata -u tessdata_expanded/kan.traineddata /tmp/_pl. >/dev/null 2>&1 &&
    head -1 /tmp/_pl.lstm-unicharset 2>/dev/null || echo '?')"
fi

# ── 2. Corpus ────────────────────────────────────────────────────────────────
if should_run corpus; then
  stage "3/9  Clean corpus  (strip unassigned codepoints)"
  python3 -u corpus/clean-corpus.py 2>&1 | tee -a "${LOG_TARGETS[@]}" | tail -6
  metric "corpus lines: $(wc -l < corpus/kan_corpus.txt | tr -d ' ')"
fi

# ── 3. Render ────────────────────────────────────────────────────────────────
# --force because a corpus or shaping change must overwrite existing images;
# without it render-corpus.py skips every file that already exists.
if should_run render; then
  stage "4/9  Render synthetic lines  (--force)"
  python3 -u corpus/render-corpus.py --force 2>&1 \
    | tee -a "${LOG_TARGETS[@]}" \
    | awk 'NR%10==0 || /Total|ERROR|fail/'
  metric "rendered images: $(ls rendered/*.png 2>/dev/null | wc -l | tr -d ' ')"
fi

# ── 4. Inventory ─────────────────────────────────────────────────────────────
if should_run inventory; then
  stage "5/9  Character inventory  (complete coverage set)"
  mkdir -p corpus/coverage
  python3 -u corpus/generate-inventory.py \
    --complete --attested-only \
    --emit-wordlist corpus/coverage/kannada-units.txt 2>&1 \
    | tee -a "${LOG_TARGETS[@]}" | tail -8
  metric "inventory images: $(find inventory -name '*.png' | wc -l | tr -d ' ')"
fi

# ── 4b. Portal gallery ───────────────────────────────────────────────────────
# test-images/ drives the portal's Images page and the 1:1 OCR test. It is not
# training data, but it is what you LOOK at to judge whether shaping is correct —
# so a stale gallery hides the very defect a rebuild was meant to fix.
if should_run inventory; then
  stage "5b/9 Portal gallery  (test-images/)"
  python3 -u scripts/gen-char-images.py 2>&1 | tee -a "${LOG_TARGETS[@]}" | tail -3
  metric "gallery images: $(find test-images -name '*.png' 2>/dev/null | wc -l | tr -d ' ')"
fi

# ── 5. Classical (optional, slow) ────────────────────────────────────────────
if [ $WITH_CLASSICAL -eq 1 ] && should_run classical; then
  stage "6/9  Classical A5 → LINE images  (hours; ~430K files)"

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

  # --force: the renderer skips existing output, so without it a rebuild after a
  # shaping or font-feature change silently preserves every old image. This is
  # exactly how aalt-rendered GTN/WMP pages survived earlier rebuilds.
  # NOT `| tail -N`: tail buffers until EOF, so a multi-hour stage shows nothing
  # until it finishes and then prints five lines. Pass progress through instead —
  # awk keeps only the periodic percentage lines so the terminal stays readable
  # while the full output still lands in both logs.
  python3 -u corpus/render-a5-pages.py \
    --corpus-dir "$CLASSICAL_DIR" \
    --lines --force --workers 4 2>&1 \
    | tee -a "${LOG_TARGETS[@]}" \
    | awk 'NR%20==0 || /Done|ERROR|Total|fail=[1-9]/'
  metric "classical line images: $(find "$CLASSICAL_A5" -name '*_line*.png' 2>/dev/null | wc -l | tr -d ' ')"
fi

# ── 6. lstmf ─────────────────────────────────────────────────────────────────
# Clear first: 02-make-lstmf.sh resumes from existing .lstmf files. After a
# unicharset or corpus change the cached ones are stale and would be re-admitted.
# ── 7. Real scans → line images ─────────────────────────────────────────────
# MUST run before lstmf. 02-make-lstmf.sh reads scan-lines/, not scan-input/,
# because a whole scanned page is CTC-infeasible: the LSTM normalises input to
# 48px tall, so an A5 page collapses to ~30 timesteps and cannot emit 500+
# characters. Both real scans were silently rejected on exactly that guard,
# which is why the model had never trained on a single real scan while scoring
# 44-53% CER on them against stock Tesseract's 19.5%.
if should_run scans; then
  stage "7/9  Segment real scans  (whole pages → line images)"
  if ls scan-input/*.png scan-input/*.jpg scan-input/*.tif >/dev/null 2>&1; then
    python3 -u corpus/segment-scans.py 2>&1 | tee -a "${LOG_TARGETS[@]}" | tail -20
    metric "scan line samples: $(find scan-lines -name 'line*.png' 2>/dev/null | wc -l | tr -d ' ')"
  else
    say "  no pages in scan-input/ — skipping."
    say "  Real scans are the highest-value training data you can add."
  fi
fi

if should_run lstmf; then
  stage "8/9  Build lstmf  (clearing stale cache first)"
  rm -rf lstmf/rendered lstmf/inventory lstmf/classical lstmf/font-test lstmf/list.txt 2>/dev/null || true
  INVENTORY_DIR="$ROOT/inventory" \
  CLASSICAL_A5_DIR="$CLASSICAL_A5" \
    ./scripts/02-make-lstmf.sh 2>&1 | tee -a "${LOG_TARGETS[@]}" | tail -8
  metric "lstmf entries: $(wc -l < lstmf/list.txt | tr -d ' ')"
fi

# ── 7. Validate before committing hours of GPU time ──────────────────────────
if should_run validate; then
  stage "9/9  Validate training set"
  python3 -u - <<'PY' 2>&1 | tee -a "${LOG_TARGETS[@]}"
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
M = max(len(u) for u in U)
# Whitespace only. The virama used to be exempt here, which made this check MORE
# permissive than Tesseract's: a bare virama has no unit of its own (the
# unicharset stores virama+consonant as ONE unit), so exempting it let words
# like ರ್ಘ pass validation and then fail in training with "Can't encode
# transcription" — hiding exactly the gaps this stage exists to find.
EX = set(' \t\n')

def enc(s):
    i, n = 0, len(s)
    while i < n:
        if s[i] in EX: i += 1; continue
        for k in range(min(M, n - i), 0, -1):
            if s[i:i+k] in U: i += k; break
        else: return False
    return True

srcs = {}
# scan-lines/ included: hand-transcribed scan text is the likeliest place for an
# unencodable cluster, and it was the one source this check never looked at.
for root in ('rendered', 'inventory', 'test-images', 'classical-corpus-kannada',
             'scan-lines'):
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

  # Report the held-out split at the gate, so nobody commits hours to training
  # on a set whose only error signal would come from its own training data.
  python3 -u scripts/make-eval-split.py --report 2>&1 \
    | tee -a "${LOG_TARGETS[@]}" | sed -n '2,20p'
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
