#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════
# segment-scans.py — turn whole-page scans into line images for training
#
# WHY THIS EXISTS
#   02-make-lstmf.sh reads scan-input/ looking for PNG + .gt.txt pairs, and it
#   found both of ours. It then rejected both, and the rejection was correct:
#
#     scan_2026-06-28T12-43-21.png   1252x1778   gt=574 chars   timesteps=33
#     Screenshot 2026-06-27 ...png   1048x1650   gt=1322 chars  timesteps=30
#
#   The LSTM normalises input to 36px tall and then Mp3,3 divides the width by
#   three, so a full A5 page collapses to a few dozen horizontal timesteps. CTC
#   needs at least one timestep per output label, and the recoder expands each
#   unicharset unit into several labels. A page can never align, so the guard
#   threw both away.
#
#   Net effect: the model has been trained on ZERO real scans. Every measurement
#   on 2026-08-05 showed 44–53% CER on real scans against stock Tesseract's
#   19.5% — fine-tuning made the model WORSE than the model it started from, and
#   the checkpoint sweep found no overfitting knee. That is not a checkpoint
#   selection problem. It is the training set never having seen a real scan.
#
#   Tesseract trains on LINES, not pages. This script produces them.
#
# WHAT IT DOES
#   1. Runs Tesseract page segmentation on the scan to get line bounding boxes.
#   2. Sorts them into reading order and crops each with padding.
#   3. Pairs crop N with line N of the .gt.txt.
#   4. REFUSES to write anything if the counts disagree.
#
#   Step 4 is the whole point of the design. Misaligned ground truth is worse
#   than no ground truth: the model trains confidently on wrong labels and you
#   cannot tell from the loss curve. So a mismatch is a hard stop with a report,
#   not a best-effort guess.
#
# USAGE
#   python3 corpus/segment-scans.py                      # segment everything
#   python3 corpus/segment-scans.py --holdout 2          # reserve N pages for eval
#   python3 corpus/segment-scans.py --page myscan.png    # one page
#   python3 corpus/segment-scans.py --dry-run            # report only, write nothing
#
# OUTPUT
#   scan-lines/<page-stem>/line0001.png + line0001.gt.txt
#   scan-holdout/  — pages deliberately withheld, for honest evaluation
# ═══════════════════════════════════════════════════════════════════════════
import argparse
import csv
import json
import re
import shutil
import os
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCAN_DIR = ROOT / 'scan-input'
OUT_DIR = ROOT / 'scan-lines'
HOLDOUT_DIR = ROOT / 'scan-holdout'
# Two directories, two jobs — and preferring the wrong one here would have been
# silent and total.
#
#   LAYOUT_TESSDATA  drives `tesseract ... tsv`, which needs REAL WEIGHTS.
#                    tessdata_expanded has none — 00c builds it with
#                    combine_lang_model, producing a starter traineddata with
#                    unicharset, recoder and dawgs but no `lstm`. Pointed at it,
#                    layout analysis returns no line boxes, every page fails the
#                    box-count check, and not one scan is ever segmented. Since
#                    real scans are the whole point of this script, that would
#                    have looked like "the segmenter doesn't work".
#
#   UNITS_TESSDATA   supplies the unicharset for counting labels, where the
#                    expanded 273-unit set IS the right one.
LAYOUT_TESSDATA = ROOT / 'tessdata_best'
UNITS_TESSDATA = ROOT / 'tessdata_expanded' \
    if (ROOT / 'tessdata_expanded' / 'kan.traineddata').exists() else ROOT / 'tessdata_best'
TESSDATA = LAYOUT_TESSDATA          # kept: referenced by detect_lines()

# Padding around each detected line box. Tesseract's boxes hug the ink, which
# clips the tops of ೀ/ೈ and the bottoms of ottu conjuncts — exactly the marks
# that distinguish one Kannada grapheme from another. Generous vertical padding
# is cheap; a clipped vowel sign is a permanently mislabelled sample.
PAD_X = 8
PAD_Y = 10

MIN_H = 16          # below this the LSTM has no vertical resolution to work with
MIN_W = 40
# Derived from the network spec, not guessed:
#   [1,36,0,1 Ct3,3,16 Mp3,3 Lfys64 Lfx96 Lrx96 Lfx384 O1c1]
# Input is normalised to 36px tall, then Mp3,3 divides width by 3, so the
# timestep budget is w*(36/h)/3 = w*12/h. And a character is not a label — the
# recoder expands each unicharset unit into several codes and CTC needs blanks
# between repeats, measured at ~2.5 codes per character on this corpus.
CTC_STEPS_PER_PX = 12.0
CTC_STEPS_PER_UNIT = 2.0    # measured: 0.03% good rejected, 97% of failures caught


def log(msg=''):
    print(msg, flush=True)


def clean_gt(raw):
    """Normalise a ground-truth line the same way 02-make-lstmf.sh does."""
    t = unicodedata.normalize('NFC', raw)
    t = t.replace('​', '').replace('‌', '').replace('‍', '')
    t = re.sub(r'[ \t]+', ' ', t)
    return t.strip()


def gt_lines(gt_path):
    """Non-empty ground-truth lines, in order."""
    raw = gt_path.read_text(encoding='utf-8')
    return [c for c in (clean_gt(l) for l in raw.splitlines()) if c]


def detect_lines(img_path, psm=3):
    """
    Ask Tesseract for line bounding boxes via TSV.

    We use the stock model purely as a page-layout engine here — its Kannada
    recognition accuracy is irrelevant, we throw the text away and keep only
    the geometry. Layout analysis is script-agnostic connected-component work.
    """
    cmd = ['tesseract', str(img_path), 'stdout',
           '--tessdata-dir', str(TESSDATA), '-l', 'kan',
           '--psm', str(psm), 'tsv']
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except FileNotFoundError:
        log('  ✗ tesseract not found on PATH')
        return None
    except subprocess.TimeoutExpired:
        log('  ✗ tesseract timed out')
        return None
    if res.returncode != 0:
        log(f'  ✗ tesseract failed: {res.stderr.strip().splitlines()[:1]}')
        return None

    boxes = []
    reader = csv.DictReader(res.stdout.splitlines(), delimiter='\t',
                            quoting=csv.QUOTE_NONE)
    for row in reader:
        try:
            if int(row['level']) != 4:      # level 4 == text line
                continue
            boxes.append({
                'x': int(row['left']), 'y': int(row['top']),
                'w': int(row['width']), 'h': int(row['height']),
                'block': int(row['block_num']), 'par': int(row['par_num']),
                'line': int(row['line_num']),
            })
        except (KeyError, ValueError, TypeError):
            continue

    # Reading order: Tesseract's own block/paragraph/line numbering already
    # encodes it, and it handles multi-column layouts that a naive top-to-bottom
    # y-sort would interleave into nonsense.
    boxes.sort(key=lambda b: (b['block'], b['par'], b['line']))
    return boxes


_UNITS, _MAXUNIT = None, 1


def _load_units():
    """Unicharset units, so we count LABELS rather than characters."""
    global _UNITS, _MAXUNIT
    if _UNITS is not None:
        return
    _UNITS = set()
    with tempfile.TemporaryDirectory() as t:
        pre = os.path.join(t, 'x.')
        r = subprocess.run(['combine_tessdata', '-u',
                            str(UNITS_TESSDATA / 'kan.traineddata'), pre],
                           capture_output=True)
        f = Path(pre + 'lstm-unicharset')
        if r.returncode == 0 and f.exists():
            _UNITS = {l.split(' ')[0] for l in
                      f.read_text(encoding='utf-8', errors='replace').split('\n')[1:]
                      if l.strip()}
    _MAXUNIT = max((len(u) for u in _UNITS), default=1)


def nunits(text):
    _load_units()
    if not _UNITS:
        return len(text)
    i, n, c = 0, len(text), 0
    while i < n:
        for k in range(min(_MAXUNIT, n - i), 0, -1):
            if text[i:i + k] in _UNITS:
                i += k
                c += 1
                break
        else:
            i += 1
            c += 1
    return c


def ctc_ok(w, h, text):
    """Will this crop survive the CTC feasibility guard in 02-make-lstmf.sh?

    Counts unicharset UNITS, not characters. A unit is often several characters
    (ರ್ಘ, ತ್ತುಂ), so a character count overstates the labels CTC must emit and
    the two thresholds cannot be made to agree.
    """
    if h <= 0:
        return False, 0
    steps = int(w * CTC_STEPS_PER_PX / h)
    return steps >= nunits(text) * CTC_STEPS_PER_UNIT, steps


def segment_page(img_path, out_root, dry_run=False, psm=3):
    from PIL import Image

    gt_path = img_path.with_suffix('.gt.txt')
    if not gt_path.exists():
        return {'page': img_path.name, 'status': 'no-gt', 'lines': 0}

    truth = gt_lines(gt_path)
    boxes = detect_lines(img_path, psm=psm)
    if boxes is None:
        return {'page': img_path.name, 'status': 'tesseract-failed', 'lines': 0}

    result = {
        'page': img_path.name,
        'gt_lines': len(truth),
        'detected': len(boxes),
        'status': 'ok',
        'lines': 0,
        'rejected': [],
    }

    if len(boxes) != len(truth):
        # Hard stop. See the module docstring: a silent off-by-one here would
        # mislabel every subsequent line on the page, and nothing downstream
        # would ever tell you.
        result['status'] = 'count-mismatch'
        return result

    img = Image.open(img_path).convert('L')
    W, H = img.size
    page_dir = out_root / img_path.stem.replace(' ', '_')

    crops = []
    for i, (b, text) in enumerate(zip(boxes, truth), start=1):
        x0 = max(0, b['x'] - PAD_X)
        y0 = max(0, b['y'] - PAD_Y)
        x1 = min(W, b['x'] + b['w'] + PAD_X)
        y1 = min(H, b['y'] + b['h'] + PAD_Y)
        w, h = x1 - x0, y1 - y0

        if h < MIN_H or w < MIN_W:
            result['rejected'].append((i, f'too small {w}x{h}', text[:24]))
            continue
        ok, steps = ctc_ok(w, h, text)
        if not ok:
            result['rejected'].append(
                (i, f'CTC {steps} steps, needs ~{int(nunits(text)*CTC_STEPS_PER_UNIT)}',
                 text[:24]))
            continue
        crops.append((i, (x0, y0, x1, y1), text))

    result['lines'] = len(crops)
    if dry_run or not crops:
        return result

    if page_dir.exists():
        shutil.rmtree(page_dir)
    page_dir.mkdir(parents=True)

    for i, box, text in crops:
        stem = page_dir / f'line{i:04d}'
        img.crop(box).save(stem.with_suffix('.png'))
        # Trailing newline: tesstrain's convention, and Tesseract's gt reader
        # is happier with it.
        stem.with_suffix('.gt.txt').write_text(text + '\n', encoding='utf-8')

    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--holdout', type=int, default=0,
                    help='reserve N pages (unsegmented) for honest evaluation')
    ap.add_argument('--page', help='segment a single page by filename')
    ap.add_argument('--psm', type=int, default=3,
                    help='page segmentation mode for layout analysis (default 3)')
    ap.add_argument('--dry-run', action='store_true',
                    help='report what would happen, write nothing')
    args = ap.parse_args()

    try:
        import PIL  # noqa: F401
    except ImportError:
        log('✗ Pillow required:  pip3 install --break-system-packages Pillow')
        return 1

    if not LAYOUT_TESSDATA.joinpath('kan.traineddata').exists():
        log(f'✗ no kan.traineddata in {LAYOUT_TESSDATA.name}/ — run ① Prep base first.')
        return 1

    pages = sorted(p for p in SCAN_DIR.glob('*')
                   if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
                   and p.with_suffix('.gt.txt').exists())
    if args.page:
        pages = [p for p in pages if p.name == args.page]
        if not pages:
            log(f'✗ {args.page} not found in scan-input/ (or has no .gt.txt)')
            return 1

    if not pages:
        log('✗ No image + .gt.txt pairs in scan-input/.')
        log('  Add scanned pages there, each with a matching .gt.txt whose')
        log('  line count equals the number of text lines on the page.')
        return 1

    log('━' * 72)
    log('  Scan segmentation — whole pages → line images')
    log(f'  source   : scan-input/  ({len(pages)} page{"s" if len(pages) != 1 else ""})')
    log(f'  layout   : {LAYOUT_TESSDATA.name}/  (needs real weights)')
    log(f'  units    : {UNITS_TESSDATA.name}/  (unicharset only)')
    if args.holdout:
        log(f'  holdout  : {args.holdout} page(s) reserved for evaluation')
    if args.dry_run:
        log('  MODE     : dry run — nothing will be written')
    log('━' * 72)

    held = []
    if args.holdout > 0 and not args.dry_run:
        # Take the holdout from the END of the sorted list so it is stable as
        # pages are added, and MOVE it out of scan-input/ so no later pipeline
        # run can quietly pull it back into training.
        held = pages[-args.holdout:]
        pages = pages[:-args.holdout]
        HOLDOUT_DIR.mkdir(exist_ok=True)
        for p in held:
            for f in (p, p.with_suffix('.gt.txt')):
                if f.exists():
                    shutil.move(str(f), str(HOLDOUT_DIR / f.name))
        log(f'\n  Moved to scan-holdout/ (never trained on):')
        for p in held:
            log(f'    {p.name}')

    if not args.dry_run:
        OUT_DIR.mkdir(exist_ok=True)

    results = [segment_page(p, OUT_DIR, args.dry_run, args.psm) for p in pages]

    log('')
    log(f'  {"page":44} {"gt":>5} {"found":>6} {"kept":>6}  status')
    log('  ' + '─' * 70)
    total = 0
    problems = []
    for r in results:
        total += r['lines']
        mark = '✓' if r['status'] == 'ok' and r['lines'] else '✗'
        log(f'  {r["page"][:44]:44} {r.get("gt_lines","-"):>5} '
            f'{r.get("detected","-"):>6} {r["lines"]:>6}  {mark} {r["status"]}')
        if r['status'] != 'ok' or r.get('rejected'):
            problems.append(r)

    log('')
    log(f'  Total line samples: {total}')

    for r in problems:
        if r['status'] == 'count-mismatch':
            log('')
            log(f'  ✗ {r["page"]}')
            log(f'      ground truth has {r["gt_lines"]} lines, '
                f'layout analysis found {r["detected"]}')
            log('      Nothing was written — pairing them would mislabel the page.')
            log('      Fix by either:')
            log('        • editing the .gt.txt so its line count matches the page')
            log('        • re-running with --psm 4 (single column) or --psm 6 (uniform block)')
            log('        • splitting a two-page spread into two images')
        elif r['status'] == 'no-gt':
            log(f'\n  ✗ {r["page"]} — no .gt.txt alongside it')
        elif r['status'] == 'tesseract-failed':
            log(f'\n  ✗ {r["page"]} — tesseract could not process it')
        for i, why, snippet in r.get('rejected', [])[:6]:
            log(f'      line {i:>3} dropped: {why}   “{snippet}…”')

    if total and not args.dry_run:
        (OUT_DIR / 'manifest.json').write_text(
            json.dumps({'pages': results, 'total_lines': total}, ensure_ascii=False,
                       indent=2), encoding='utf-8')
        log('')
        log('━' * 72)
        log(f'  Wrote {total} line samples to scan-lines/')
        log('')
        log('  Next:  ./scripts/02-make-lstmf.sh    (picks up scan-lines/ automatically)')
        log('━' * 72)

    return 0 if total or args.dry_run else 1


if __name__ == '__main__':
    sys.exit(main())
