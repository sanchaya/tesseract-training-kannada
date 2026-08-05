#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════
# make-eval-split.py — split lstmf/list.txt into a train list and an eval list
#
# WHY THIS EXISTS
#   tesstrain's Makefile splits the sample list (RATIO_TRAIN = 0.90) and passes
#   the remainder to lstmtraining as --eval_listfile. We never did. 03-train.sh
#   passed only --train_listfile, so every number lstmtraining printed was the
#   error on data it was actively training on.
#
#   That single omission explains the central mystery of this project: BCER fell
#   to 0.003% while real-scan CER sat at 44–53%, and the checkpoint sweep found
#   no overfitting knee. There was no knee to find, because there was never a
#   held-out curve to bend. We were watching the model memorise and calling it
#   learning.
#
# WHY THE SPLIT IS BY GROUP, NOT BY LINE
#   A uniform random 10% would be almost as useless as no split at all. 96.5% of
#   our samples are synthetic renders of four fonts over a handful of titles, so
#   a randomly held-out line is a near-duplicate of lines still in training —
#   same font, same page, same rendering pipeline, often the same sentence
#   fragment. The model can memorise its way to a good score on that.
#
#   So we hold out whole GROUPS. A group is one page of one title in one font
#   (classical), or one font's inventory sheet. Held-out pages were never seen in
#   any form, which is the only way the eval number means anything.
#
#   Real scans are handled separately and never enter the eval list here: they
#   are too scarce to spend on it, and scan-holdout/ already reserves whole pages
#   for end-to-end measurement via verify-ocr.py.
#
# USAGE
#   python3 scripts/make-eval-split.py                 # 90/10 group-aware
#   python3 scripts/make-eval-split.py --ratio 0.95
#   python3 scripts/make-eval-split.py --report        # show the split, write nothing
#
# OUTPUT
#   lstmf/list.train.txt   → --train_listfile
#   lstmf/list.eval.txt    → --eval_listfile
# ═══════════════════════════════════════════════════════════════════════════
import argparse
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LSTMF = ROOT / 'lstmf'
FULL = LSTMF / 'list.txt'
TRAIN = LSTMF / 'list.train.txt'
EVAL = LSTMF / 'list.eval.txt'

SEED = 20260805      # fixed: the split must be identical across reruns, or
                     # "training" and "held out" swap between runs and the eval
                     # number silently becomes a training number again.


def group_of(path):
    """
    The unit we hold out as a whole.

    classical/<title>__<font>_<style>/pageNNNN_lineNNN.lstmf
        → ('classical', '<title>__<font>_<style>/pageNNNN')
          One physical page. Lines from the same page share layout, ink
          weight and hyphenation, so they must not straddle the split.

    inventory/<font>/char_XXXX.lstmf
        → ('inventory', '<font>/char_XXXX')
          Per-character, NOT per-font. Holding out a whole font's inventory
          would leave that font's glyph shapes untaught for the sake of ~1.6K
          eval lines. Individual inventory entries are genuinely distinct
          labels — deliberately enumerated combinations, not near-duplicate
          lines off the same page — so a per-character split is an honest test
          without starving any font.

    rendered/<...>/<file>.lstmf        → ('rendered', '<dir>')
    scan/<page>/lineNNNN.lstmf         → ('scan', '<page>')
    """
    p = str(path)
    i = p.rfind('/lstmf/')
    rel = p[i + 7:] if i >= 0 else p
    parts = rel.split('/')
    kind = parts[0] if parts else 'other'

    if kind == 'classical' and len(parts) >= 3:
        m = re.match(r'(page\d+)', parts[2])
        page = m.group(1) if m else parts[2]
        return kind, f'{parts[1]}/{page}'
    if kind == 'inventory' and len(parts) >= 3:
        return kind, f'{parts[1]}/{Path(parts[2]).stem}'
    if len(parts) >= 2:
        return kind, parts[1]
    return kind, rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ratio', type=float, default=0.90,
                    help='fraction of groups used for training (default 0.90)')
    ap.add_argument('--report', action='store_true',
                    help='print the split and exit without writing')
    args = ap.parse_args()

    if not FULL.exists():
        print(f'✗ {FULL} not found — run ./scripts/02-make-lstmf.sh first.')
        return 1

    lines = [l.strip() for l in FULL.read_text(encoding='utf-8').splitlines() if l.strip()]
    if not lines:
        print('✗ lstmf/list.txt is empty.')
        return 1

    groups = defaultdict(list)
    for l in lines:
        groups[group_of(l)].append(l)

    # Split within each kind, so the eval set keeps the same composition as the
    # training set. Holding out a random 10% of ALL groups could, by chance,
    # take every inventory font and no classical pages.
    by_kind = defaultdict(list)
    for key in groups:
        by_kind[key[0]].append(key)

    rng = random.Random(SEED)
    train_keys, eval_keys = [], []
    for kind, keys in by_kind.items():
        keys = sorted(keys)
        rng.shuffle(keys)
        if kind == 'scan':
            # Real scans are too scarce to spend on the eval list. scan-holdout/
            # already reserves whole pages for honest end-to-end measurement.
            train_keys += keys
            continue
        n_eval = max(1, round(len(keys) * (1 - args.ratio))) if len(keys) > 1 else 0
        eval_keys += keys[:n_eval]
        train_keys += keys[n_eval:]

    train = [l for k in train_keys for l in groups[k]]
    ev = [l for k in eval_keys for l in groups[k]]
    rng.shuffle(train)
    rng.shuffle(ev)

    print('━' * 72)
    print('  Train / eval split — held out by GROUP, not by line')
    print(f'  seed {SEED}   train ratio {args.ratio}')
    print('━' * 72)
    print(f'  {"kind":14} {"groups":>8} {"→ train":>10} {"→ eval":>9} '
          f'{"train ln":>10} {"eval ln":>9}')
    print('  ' + '─' * 68)
    for kind in sorted(by_kind):
        gk = by_kind[kind]
        te = [k for k in gk if k in set(eval_keys)]
        tt = [k for k in gk if k not in set(eval_keys)]
        print(f'  {kind:14} {len(gk):>8} {len(tt):>10} {len(te):>9} '
              f'{sum(len(groups[k]) for k in tt):>10} '
              f'{sum(len(groups[k]) for k in te):>9}')
    print('  ' + '─' * 68)
    print(f'  {"TOTAL":14} {len(groups):>8} {len(train_keys):>10} {len(eval_keys):>9} '
          f'{len(train):>10} {len(ev):>9}')

    if not ev:
        print('\n  ✗ Eval set is empty — too few groups to split.')
        print('    Add more source material, or lower --ratio.')
        return 1

    if args.report:
        print('\n  (--report: nothing written)')
        return 0

    TRAIN.write_text('\n'.join(train) + '\n', encoding='utf-8')
    EVAL.write_text('\n'.join(ev) + '\n', encoding='utf-8')

    print('')
    print('━' * 72)
    print(f'  lstmf/list.train.txt  {len(train):>8} samples')
    print(f'  lstmf/list.eval.txt   {len(ev):>8} samples  (never trained on)')
    print('')
    print('  03-train.sh now passes both to lstmtraining. Watch for the line:')
    print('     "At iteration N ... BCER train=X%  ... eval=Y%"')
    print('  When eval stops falling while train keeps falling, that is the knee.')
    print('  Package the checkpoint at the knee, not the last one.')
    print('━' * 72)
    return 0


if __name__ == '__main__':
    sys.exit(main())
