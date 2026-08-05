#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# diagnose-model.sh — resolve "BCER says 0.4%, inference says 60%"
#
# Those two numbers cannot both describe the same model. Exactly one of these
# is true, and this script tells you which:
#
#   A. The CHECKPOINT is good and PACKAGING breaks it.
#      lstmeval on the checkpoint ≈ BCER, but the packaged .traineddata scores
#      far worse. Fix 04-package.sh; the training is fine.
#
#   B. The CHECKPOINT is bad and BCER is not measuring what we assume.
#      lstmeval ≈ the inference number. Training itself is the problem — no
#      amount of re-rendering or re-cleaning data will help.
#
# lstmeval runs the same evaluation path as training, reading .lstmf directly,
# so it isolates the model from Tesseract's inference pipeline (binarisation,
# layout analysis, line finding) entirely.
# ═══════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)/.."

TD=tessdata_expanded/kan.traineddata
[ -f "$TD" ] || TD=tessdata_best/kan.traineddata
CP=$(ls -t output/kan_hist_[0-9]*.checkpoint 2>/dev/null | head -1)
[ -n "$CP" ] || { echo "✗ no checkpoint in output/"; exit 1; }

# The held-out list built by make-eval-split.py. This used to be
# `tail -200 lstmf/list.txt` — the TRAINING list. Evaluating a model on its own
# training data reports memorisation, and acting on that number is how we
# concluded the iteration cap should be raised when it was already correct.
mkdir -p /tmp/diag
if [ -s lstmf/list.eval.txt ]; then
  head -300 lstmf/list.eval.txt > /tmp/diag/eval.txt
else
  echo "  ✗ lstmf/list.eval.txt not found."
  echo "    Run:  python3 scripts/make-eval-split.py"
  echo "    Refusing to fall back to the training list — the resulting number"
  echo "    would look excellent and mean nothing."
  exit 1
fi
echo "  checkpoint : $(basename "$CP")"
echo "  traineddata: $TD"
echo "  eval set   : $(wc -l < /tmp/diag/eval.txt) lstmf entries"
echo ""

echo "── 1. lstmeval on the CHECKPOINT ───────────────────────────"
lstmeval --model "$CP" --traineddata "$TD" --eval_listfile /tmp/diag/eval.txt 2>&1 | tail -6

echo ""
echo "── 2. lstmeval on the PACKAGED model ───────────────────────"
if [ -f best/kan_hist.traineddata ]; then
  lstmeval --model best/kan_hist.traineddata --eval_listfile /tmp/diag/eval.txt 2>&1 | tail -6
else
  echo "  best/kan_hist.traineddata not found — run ./scripts/04-package.sh"
fi

echo ""
echo "── How to read this ────────────────────────────────────────"
echo "  checkpoint ≈ 0.4%  and packaged ≫ that  → PACKAGING is broken (case A)"
echo "  both ≈ 60%                              → the MODEL is bad   (case B)"
echo "  checkpoint ≈ 0.4%  and packaged ≈ 0.4%  → model+packaging fine;"
echo "                                             the gap is Tesseract's"
echo "                                             inference pipeline, not training"
