#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════
# remediate-coverage.py — turn the coverage report into targeted training data
#
# WHY THE OBVIOUS VERSION DOESN'T WORK
#   The tempting loop is "find failing graphemes → render more images of them →
#   train more". It is worth being precise about why that is wrong, because it
#   fails quietly and looks like progress.
#
#   Everything the coverage report measures comes from inventory/, and inventory
#   images are ALREADY in the training set. A grapheme failing there is one the
#   model has seen thousands of times and still gets wrong. Rendering more of
#   the same isolated glyph is not new information — it is the same information
#   at higher volume, and the usual result is that common characters get worse
#   while the target does not improve.
#
#   So this script diagnoses before it generates, and for two of the three
#   diagnoses it generates nothing at all.
#
# THE TRIAGE
#   1. STRUCTURAL — the grapheme is not in the unicharset.
#      No amount of training can produce it; the output layer has no label for
#      it. All 52 English letters are in this class today: the kan unicharset
#      has ASCII digits and punctuation but not one A-Z or a-z. Fix belongs in
#      00c-expand-unicharset.sh, not here.
#
#   2. FONT-SPECIFIC — reads correctly in some fonts, fails in others.
#      Usually the image is wrong, not the model. This is exactly the signature
#      the forced-Indic-features bug produced: conjuncts rendered in reversed
#      order for three fonts while Kittel was fine. Training harder on a
#      mis-rendered image teaches the model the wrong shape and is worse than
#      doing nothing, so these are reported for inspection and skipped.
#
#   3. UNIVERSAL — encodable, rendered correctly, fails in every font.
#      This is the only class more data can fix, and it gets WORD CONTEXT rather
#      than more isolated glyphs. Tesseract reads lines: a glyph is recognised
#      partly by what sits either side of it, and an isolated-glyph diet teaches
#      a context-free shape that does not survive running text.
#
#   Confusions are mined too. If ಠ is consistently read as ಥ, the model needs
#   the two in contrast, not more ಠ alone — so both are pulled into the same
#   lines wherever the corpus provides them.
#
# USAGE
#   python3 corpus/remediate-coverage.py                 # triage + report only
#   python3 corpus/remediate-coverage.py --generate      # also write GT lines
#   python3 corpus/remediate-coverage.py --render        # ... and render images
#   python3 corpus/remediate-coverage.py --threshold 0.5 # what counts as failing
#
# OUTPUT
#   corpus/remediation.txt                    word-context lines
#   rendered/<tag>_remed####.png + .gt.txt    picked up by 02-make-lstmf.sh
#   output/reports/remediation.json
# ═══════════════════════════════════════════════════════════════════════════
import argparse
import collections
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))

REPORT = ROOT / 'output' / 'reports' / 'unit-coverage.json'
OUT_GT = ROOT / 'corpus' / 'remediation.txt'
OUT_JSON = ROOT / 'output' / 'reports' / 'remediation.json'
RENDERED = ROOT / 'rendered'
CORPORA = [ROOT / 'corpus' / 'kan_corpus.txt']
CLASSICAL = ROOT / 'classical-corpus-kannada'

SEED = 20260805
WORDS_PER_LINE = 6

# How many lines one grapheme may contribute.
#
# Uncapped oversampling is its own failure mode: push a rare conjunct to
# thousands of lines and the model starts over-predicting it everywhere, so
# targeted accuracy rises while overall CER gets worse. The cap keeps
# remediation a correction to the distribution rather than a replacement for it.
MAX_LINES_PER_GRAPHEME = 40
MIN_WORDS_NEEDED = 3

# A grapheme is "font-specific" only if a real MAJORITY of fonts read it.
# Anything less is a global failure that happens to have got lucky somewhere.
FONT_SPECIFIC_MIN_OK = 0.5


def log(m=''):
    print(m, flush=True)


def load_report():
    if not REPORT.exists():
        log('✗ output/reports/unit-coverage.json not found.')
        log('  Run the coverage report first — it is what this script triages:')
        log('     python3 corpus/unit-coverage-report.py')
        return None
    d = json.loads(REPORT.read_text(encoding='utf-8'))
    if not d.get('meta', {}).get('measured'):
        log('✗ The coverage report has no measurements (it was run with --static).')
        log('  Triage needs per-font results. Re-run:')
        log('     python3 corpus/unit-coverage-report.py')
        return None
    return d


_UNITS, _MAXU = None, 1


def _load_units():
    """Unicharset units, for rejecting words training would discard anyway."""
    global _UNITS, _MAXU
    if _UNITS is not None:
        return
    import subprocess, tempfile, os
    td = ROOT / 'tessdata_expanded' / 'kan.traineddata'
    if not td.exists():
        td = ROOT / 'tessdata_best' / 'kan.traineddata'
    _UNITS = set()
    with tempfile.TemporaryDirectory() as t:
        pre = os.path.join(t, 'x.')
        r = subprocess.run(['combine_tessdata', '-u', str(td), pre], capture_output=True)
        f = Path(pre + 'lstm-unicharset')
        if r.returncode == 0 and f.exists():
            _UNITS = {l.split(' ')[0] for l in
                      f.read_text(encoding='utf-8', errors='replace').split('\n')[1:]
                      if l.strip()}
    _MAXU = max((len(u) for u in _UNITS), default=1)


def encodable(text):
    _load_units()
    if not _UNITS:
        return True
    i, n = 0, len(text)
    while i < n:
        if text[i] in ' \t\n':
            i += 1
            continue
        for k in range(min(_MAXU, n - i), 0, -1):
            if text[i:i + k] in _UNITS:
                i += k
                break
        else:
            return False
    return True


def build_word_index(graphemes):
    """Map grapheme → corpus words containing it. One pass over the corpus.

    Words the unicharset cannot encode are rejected here rather than downstream.
    They come mostly from classical-corpus-kannada/, which is raw transcription
    and never went through clean-corpus.py.

    Skipping them matters more than it looks. Lines are six words joined, and
    02-make-lstmf.sh discards a line if ANY word in it fails to encode — so one
    bad word takes five good ones with it. Measured on the first run: 38% of
    remediation images were built and then thrown away, and the loss fell
    unevenly across exactly the graphemes the remediation existed to fix.
    """
    targets = sorted(graphemes, key=len, reverse=True)
    index = collections.defaultdict(list)
    seen = collections.defaultdict(set)
    rejected = [0]
    files = [p for p in CORPORA if p.exists()]
    if CLASSICAL.exists():
        files += sorted(CLASSICAL.glob('*/*.txt'))

    for f in files:
        for line in f.read_text(encoding='utf-8', errors='ignore').splitlines():
            for w in line.split():
                w = w.strip('।॥|.,;:!?()[]"\'')
                if not (2 <= len(w) <= 24):
                    continue
                if not encodable(w):
                    rejected[0] += 1
                    continue
                for g in targets:
                    if g in w and w not in seen[g]:
                        seen[g].add(w)
                        index[g].append(w)
    if rejected[0]:
        log(f'  ({rejected[0]:,} corpus word(s) skipped — not encodable in the '
            f'unicharset, so any line containing one would be discarded)')
    return index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threshold', type=float, default=0.5,
                    help='accuracy at or below which a grapheme counts as failing')
    ap.add_argument('--generate', action='store_true', help='write remediation GT lines')
    ap.add_argument('--render', action='store_true', help='also render images')
    ap.add_argument('--max-lines', type=int, default=MAX_LINES_PER_GRAPHEME)
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--force', action='store_true',
                    help='re-render images that already exist')
    args = ap.parse_args()

    data = load_report()
    if not data:
        return 1
    rng = random.Random(SEED)

    # unit-coverage-report.py already filters combinations to the standard mark
    # set (nukta and the length marks are dropped). Nothing to re-filter here —
    # but assert it, because triage silently generating training data for ಕ಼
    # would be worse than not running at all.
    STRAY = set('಼ೕೖೄ')
    stray = [g['g'] for g in data['graphemes'] if STRAY & set(g['g'])]
    if stray:
        log(f'  ⚠  {len(stray)} grapheme(s) in the report use non-standard marks')
        log(f'     ({" ".join(stray[:12])}). Re-run the coverage report — it should')
        log('     have excluded these. Skipping them here.')
        data['graphemes'] = [g for g in data['graphemes'] if not (STRAY & set(g['g']))]

    structural, font_specific, universal, healthy = [], [], [], []
    for g in data['graphemes']:
        if g['category'] in ('special',) or not g.get('n'):
            continue
        if not g['encodable']:
            structural.append(g)
            continue
        if g['acc'] > args.threshold:
            healthy.append(g)
            continue
        # Bucket by the PROPORTION of fonts that fail, not by "did any font
        # succeed". With 9-16 fonts per grapheme, a single lucky hit was enough
        # to file a grapheme as font-specific — which is how 257 graphemes ended
        # up in that bucket while every font was failing 200+ of them. That is
        # not a font problem being described, it is a global one wearing the
        # wrong label, and the label decides whether we generate data or go
        # looking at images.
        oks = sum(1 for f in g['fonts'] if f['ok'])
        total = len(g['fonts']) or 1
        (font_specific if oks / total >= FONT_SPECIFIC_MIN_OK else universal).append(g)

    # Structural failures with no sample at all (the Latin block) still matter.
    for g in data['graphemes']:
        if not g['encodable'] and not g.get('n'):
            structural.append(g)

    log('━' * 72)
    log('  Coverage remediation — triage before generation')
    log(f'  report   : {data["meta"]["when"]}   model {data["meta"]["model"]}')
    log(f'  failing  : accuracy ≤ {args.threshold:.0%}')
    log('━' * 72)
    log('')
    log(f'  {len(healthy):>5}  reading correctly')
    log(f'  {len(structural):>5}  STRUCTURAL     not in the unicharset — training cannot fix')
    log(f'  {len(font_specific):>5}  FONT-SPECIFIC  works in some fonts — suspect the image, not the model')
    log(f'  {len(universal):>5}  UNIVERSAL      fails everywhere — more context data may help')
    log('')
    log('  Read these against how they were measured: one grapheme per image,')
    log('  in isolation, at PSM 13. A model trained on running text is being')
    log('  asked to read a single floating glyph with no neighbours, which is')
    log('  harder than its real job. Treat the failures as a ranked list of')
    log('  weak spots, not as an accuracy figure for the model.')

    # ── 1. Structural ───────────────────────────────────────────────────────
    if structural:
        by_cat = collections.Counter(g['category'] for g in structural)
        log('')
        log('  ── STRUCTURAL ─────────────────────────────────────────────────')
        log('  The output layer has no label for these. Training is not the fix.')
        for c, n in by_cat.most_common():
            ex = ' '.join(g['g'] for g in structural if g['category'] == c)[:56]
            log(f'    {c:18} {n:>4}   {ex}')
        if any(g['category'].startswith('latin') for g in structural):
            log('')
            log('    English letters are missing from the kan unicharset entirely.')
            log('    Every font here has full A-Z/a-z glyphs, so this is fixable —')
            log('    but it changes the unicharset, which invalidates existing')
            log('    checkpoints and forces a fresh run. Deliberate decision:')
            log('       ./scripts/00c-expand-unicharset.sh --with-latin')

    # ── 2. Font-specific ────────────────────────────────────────────────────
    if font_specific:
        log('')
        log('  ── FONT-SPECIFIC ──────────────────────────────────────────────')
        log('  Reads fine in some fonts, not others. That pattern usually means the')
        log('  IMAGE is wrong — it is the signature the forced-Indic-features bug')
        log('  produced. Look at these before training on them; training on a')
        log('  mis-rendered glyph teaches the wrong shape.')
        badfont = collections.Counter()
        for g in font_specific:
            for f in g['fonts']:
                if not f['ok']:
                    badfont[f['font']] += 1
        for f, n in badfont.most_common(10):
            log(f'    {f:38} {n:>5} failing graphemes')
        log('')
        log('    Sample — check these images by eye:')
        for g in font_specific[:6]:
            bad = [f["font"].replace("karnata", "") for f in g['fonts'] if not f['ok']]
            got = next((f['got'] for f in g['fonts'] if not f['ok']), '')
            log(f'      {g["g"]:6} read as {got or "nothing":8} in {", ".join(bad[:3])}')

    # ── 3. Universal → the only trainable class ─────────────────────────────
    confusion = collections.Counter()
    for g in universal:
        for f in g['fonts']:
            if not f['ok'] and f['got']:
                confusion[(g['g'], f['got'])] += 1

    log('')
    log('  ── UNIVERSAL ──────────────────────────────────────────────────')
    if not universal:
        log('  None. Nothing here that more data would fix.')
    else:
        log(f'  {len(universal)} grapheme(s) encodable but never read correctly.')
        if confusion:
            log('')
            log('  Most consistent confusions (target → what the model reads):')
            for (a, b), n in confusion.most_common(10):
                log(f'    {a:8} → {b:8}  ({n} font(s))')
            log('')
            log('  Confused pairs need CONTRAST, not volume: lines containing both,')
            log('  so the model learns the boundary rather than seeing more of one.')

    plan = {'structural': [g['g'] for g in structural],
            'font_specific': [g['g'] for g in font_specific],
            'universal': [g['g'] for g in universal],
            'confusions': [{'target': a, 'read_as': b, 'fonts': n}
                           for (a, b), n in confusion.most_common(40)]}

    if not args.generate:
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(plan, ensure_ascii=False, indent=1),
                            encoding='utf-8')
        log('')
        log('━' * 72)
        log('  Triage only. To generate word-context training lines for the')
        log('  UNIVERSAL set:   python3 corpus/remediate-coverage.py --generate')
        log('━' * 72)
        return 0

    if not universal:
        log('\n  Nothing to generate.')
        return 0

    # ── Generate word-context lines ─────────────────────────────────────────
    log('')
    log('  Mining the corpus for words containing each target…')
    wanted = {g['g'] for g in universal} | {b for (_, b) in confusion}
    index = build_word_index(wanted)

    lines, stats, starved = [], {}, []
    for g in universal:
        key = g['g']
        words = list(index.get(key, []))
        # Pull in whatever it gets confused WITH, so the pair appears together.
        for (a, b), _ in confusion.items():
            if a == key:
                words += index.get(b, [])[:len(words) or 20]
        if len(words) < MIN_WORDS_NEEDED:
            starved.append(key)
            continue
        rng.shuffle(words)
        n = min(args.max_lines, max(1, len(words) // WORDS_PER_LINE))
        made = 0
        for i in range(n):
            chunk = words[i * WORDS_PER_LINE:(i + 1) * WORDS_PER_LINE]
            if len(chunk) < 2:
                break
            lines.append(' '.join(chunk))
            made += 1
        stats[key] = made

    rng.shuffle(lines)
    OUT_GT.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    log('')
    log(f'  {len(lines)} line(s) covering {len(stats)} grapheme(s) → '
        f'{OUT_GT.relative_to(ROOT)}')
    if starved:
        log('')
        log(f'  {len(starved)} grapheme(s) have almost no corpus words:')
        log(f'    {" ".join(starved[:20])}')
        log('    Real words are the only honest source here. Inventing them would')
        log('    teach letter shapes in sequences the language never produces, so')
        log('    these are left alone — add source texts that contain them.')

    plan['generated_lines'] = len(lines)
    plan['per_grapheme'] = stats
    plan['starved'] = starved
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding='utf-8')

    if not args.render:
        log('')
        log('━' * 72)
        log('  Next:  python3 corpus/remediate-coverage.py --generate --render')
        log('  then:  ./scripts/02-make-lstmf.sh   (picks up rendered/)')
        log('━' * 72)
        return 0

    # ── Render ──────────────────────────────────────────────────────────────
    import os
    import multiprocessing
    import yaml
    from shaping_render import SHAPING_AVAILABLE, check_and_warn
    check_and_warn()
    if not SHAPING_AVAILABLE:
        log('✗ HarfBuzz shaping unavailable — refusing to render.')
        log('  Pillow cannot form conjuncts, and mis-shaped conjuncts are exactly')
        log('  what this script exists to avoid creating.')
        return 1

    cfg = yaml.safe_load(open(ROOT / 'fonts.yml', encoding='utf-8'))
    tasks = []
    for font in cfg['fonts']:
        fid = font['id']
        fdir = font.get('font_dir', 'fonts')
        for ff in font['font_files']:
            fp = ROOT / 'fonts' / fid / fdir / ff
            if not fp.exists():
                continue
            style = Path(ff).stem.lower().replace('-', '_').split('_')[-1]
            tag = f'{fid}_{style}'
            aalt = 'aalt' in (font.get('font_features') or '')
            degrade = font.get('degrade', False)
            for i, text in enumerate(lines):
                tasks.append((str(fp), tag, i, text, degrade, aalt,
                              hash((tag, i)) & 0xFFFFFFFF, args.force))

    log('')
    log(f'  Rendering {len(tasks)} images across '
        f'{len({t[1] for t in tasks})} font styles…')

    workers = args.workers or max(1, (os.cpu_count() or 4) - 1)
    with multiprocessing.Pool(workers) as pool:
        res = pool.map(_render, tasks, chunksize=16)
    ok = sum(1 for r in res if r == 'ok')
    skipped = sum(1 for r in res if r == 'skip')
    failed = len(res) - ok - skipped
    # These were one number before, and it read as total failure: a rerun that
    # correctly skipped 36,873 already-rendered images reported
    # "0 rendered, 36873 skipped/failed" — indistinguishable from nothing working.
    log(f'  rendered {ok}, already present {skipped}, failed {failed} → rendered/')
    if failed:
        log(f'  ⚠  {failed} render(s) failed. Re-run with --force to retry them.')
    if skipped and not ok:
        log('  Everything was already rendered — the images are on disk and ready.')
    log('')
    log('━' * 72)
    log('  Next:  ./scripts/02-make-lstmf.sh   then retrain')
    log('  Re-run the coverage report afterwards to see whether it moved —')
    log('  remediation that does not move the number is remediation that failed.')
    log('━' * 72)
    return 0


def _render(t):
    """One image. Mirrors render-corpus.py's _render_task exactly — same
    signature order, same degradation, same seeding — so remediation images are
    indistinguishable from ordinary corpus images to everything downstream."""
    import random as _rng
    from PIL import ImageFilter as _IF
    from shaping_render import render_text

    fp, tag, idx, text, degrade, aalt, seed, force = t
    stem = RENDERED / f'{tag}_remed{idx:04d}'
    png = Path(str(stem) + '.png')
    gt = Path(str(stem) + '.gt.txt')
    if not force and png.exists() and gt.exists():
        return 'skip'
    try:
        # NOTE the argument order: render_text(font_path, text, ...). Reversing
        # these two silently renders the font path as if it were the text.
        img = render_text(fp, text, font_size=36, padding_x=20, padding_y=12,
                          min_height=60, bg_color=255, ink_color=0, aalt=aalt)
    except Exception:                                    # noqa: BLE001
        return 'fail'
    if img is None:
        return 'fail'
    if degrade:
        r = _rng.Random(seed)
        img = img.filter(_IF.GaussianBlur(radius=0.6))
        w, h = img.size
        px = img.load()
        for _ in range(int(w * h * 0.003)):
            px[r.randint(0, w - 1), r.randint(0, h - 1)] = r.choice([0, 255])
        img = img.rotate(r.uniform(-0.8, 0.8), expand=False, fillcolor=255)
    img.save(str(png), dpi=(150, 150))
    gt.write_text(text, encoding='utf-8')
    return 'ok'


if __name__ == '__main__':
    sys.exit(main())
