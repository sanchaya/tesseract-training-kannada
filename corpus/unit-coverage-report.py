#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════
# unit-coverage-report.py — what the model can emit vs what it actually reads
#
# WHY
#   "BCER 0.003%" says nothing about which characters work. It is one number
#   averaged over hundreds of thousands of synthetic lines, dominated by the
#   common ones, and it hid the fact that basic vowels were not being recognised
#   at all. What you need instead is a per-grapheme picture: this conjunct is
#   read correctly in six fonts and fails in three, that vowel sign is never
#   read at all.
#
#   Two different questions get conflated here, so the report answers both:
#
#     CAN it emit the unit?   A unicharset question. If ್ಘ is not a unit, no
#                             amount of training will ever produce it. Static,
#                             needs no OCR.
#
#     DOES it read it?        A model question, and only measurable by running
#                             the model on images of that grapheme and checking
#                             the output.
#
#   A grapheme can be encodable and still unread; that gap is the interesting
#   part, and it is where training effort actually pays.
#
# HOW IT MEASURES
#   inventory/<font>/char_*.png is a systematic enumeration of Kannada
#   graphemes, one per image, each with its own .gt.txt. That is a ready-made
#   labelled test set covering the whole script, so the report runs the model
#   over it and scores exact matches.
#
#   Inventory images ARE in the training set, so these numbers are optimistic —
#   they measure "did the model learn this shape", not "will it generalise".
#   A grapheme failing here has definitively not been learned.
#
# USAGE
#   python3 corpus/unit-coverage-report.py                    # all fonts, kan_hist vs stock
#   python3 corpus/unit-coverage-report.py --fonts karnatafkittel
#   python3 corpus/unit-coverage-report.py --limit 200        # quick sample
#   python3 corpus/unit-coverage-report.py --no-baseline      # skip stock, ~2x faster
#   python3 corpus/unit-coverage-report.py --static           # unicharset only, no OCR
#
# OUTPUT
#   output/reports/unit-coverage.html   (self-contained)
#   output/reports/unit-coverage.json
# ═══════════════════════════════════════════════════════════════════════════
import argparse
import collections
import html
import json
import multiprocessing
import os
import subprocess
import sys
import tempfile
import string
import unicodedata
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INVENTORY = ROOT / 'inventory'
REPORTS = ROOT / 'output' / 'reports'
VIRAMA = '್'
ZWNJ = '‌'

# Where the packaged model actually lands. 04-package.sh writes to best/, and
# reading from tessdata_best/ instead is how an entire day of measurements ended
# up describing a model from a month earlier.
MODEL_DIRS = [ROOT / 'best', ROOT / 'tessdata_best']

# Two different needs, and conflating them reported stock Tesseract at 0.0%.
#
#   UNITS_DIRS     where the UNICHARSET comes from. tessdata_expanded is right:
#                  it carries the 273-unit set training actually used.
#
#   BASELINE_DIRS  where the BASELINE RECOGNISER comes from. tessdata_expanded
#                  is wrong, because it has no `lstm` component at all —
#                  00c-expand-unicharset.sh builds it with combine_lang_model,
#                  which produces a *starter* traineddata: unicharset, recoder
#                  and dawgs, no weights. It is 2.6MB against stock's 9.8MB and
#                  recognises nothing. Pointed at it, every baseline comparison
#                  returned an empty string and scored zero — which reads as
#                  "stock Tesseract cannot read Kannada at all" rather than
#                  "this file was never a model".
UNITS_DIRS = [ROOT / 'tessdata_expanded', ROOT / 'tessdata_best']
BASELINE_DIRS = [ROOT / 'tessdata_best']


def has_lstm(traineddata):
    """Does this traineddata contain recogniser weights?

    A starter traineddata is a trap: same name, same extension, loads without
    complaint, and silently recognises nothing.
    """
    try:
        out = subprocess.run(['combine_tessdata', '-d', str(traineddata)],
                             capture_output=True, text=True, timeout=30)
        return any(l.split(':')[1:2] == ['lstm']
                   for l in (out.stdout + out.stderr).splitlines() if ':' in l)
    except Exception:                                    # noqa: BLE001
        return True                                      # don't block on a probe failure

VOWEL_NAMES = {'A', 'AA', 'I', 'II', 'U', 'UU', 'E', 'EE', 'AI', 'O', 'OO', 'AU',
               'VOCALIC R', 'VOCALIC RR', 'VOCALIC L', 'VOCALIC LL'}

# The combining marks that belong in a Kannada conjunct/syllable coverage test.
#
#   ಾ ಿ ೀ ು ೂ ೃ ೆ ೇ ೈ ೊ ೋ ೌ   the twelve productive vowel signs
#   ಂ ಃ                          anusvara, visarga
#   ್                            virama (the ottu former)
#
# Deliberately EXCLUDED, though Unicode defines them for Kannada:
#   ೄ  U+0CC4  VOWEL SIGN VOCALIC RR   — vanishingly rare, effectively unused
#   ೕ  U+0CD5  LENGTH MARK             — a composition artefact, not a mark a
#   ೖ  U+0CD6  AI LENGTH MARK            reader ever sees on its own
#   ಼  U+0CBC  NUKTA                   — for foreign sounds; absent from
#                                         historical Kannada printing entirely
#
# The nukta is the one that actually mattered: 26 combinations (ಕ಼ ಖ಼ ಗ಼ …) came
# in from 00c-expand-unicharset.sh's word list and were being scored as OCR
# failures. They are not failures worth chasing — no page in this corpus
# contains them.
COVERAGE_MARKS = set('ಾಿೀುೂೃೆೇೈೊೋೌಂಃ್')

# Every combining mark Unicode assigns to Kannada, so exclusions can be
# reported rather than silently applied.
ALL_KANNADA_MARKS = set('ಾಿೀುೂೃೄೆೇೈೊೋೌ್ಂಃ಼ೕೖ')


def marks_in(g):
    return {c for c in g if c in ALL_KANNADA_MARKS}


def wanted_grapheme(g, all_marks=False):
    """Should this grapheme be part of the coverage test?

    Keeps single characters (the base alphabet) always. For combinations, every
    combining mark must be one we actually test — otherwise a grapheme no reader
    will ever meet is scored alongside ones that matter, and a red cell for ಕ಼
    reads exactly like a red cell for ಕ್ಕ.
    """
    if all_marks or len(g) == 1:
        return True
    ms = marks_in(g)
    return not ms or ms <= COVERAGE_MARKS


def find_model(dirs, lang):
    for d in dirs:
        if (d / f'{lang}.traineddata').exists():
            return d
    return None


def load_units(traineddata):
    with tempfile.TemporaryDirectory() as t:
        pre = os.path.join(t, 'x.')
        r = subprocess.run(['combine_tessdata', '-u', str(traineddata), pre],
                           capture_output=True)
        if r.returncode != 0:
            return []
        f = Path(pre + 'lstm-unicharset')
        if not f.exists():
            return []
        raw = f.read_text(encoding='utf-8', errors='replace').split('\n')[1:]
        return [l.split(' ')[0] for l in raw if l.strip()]


def classify(g):
    """Bucket a grapheme for display. Order matters: most specific first."""
    if g in ('NULL', 'Joined', '|Broken|0|1'):
        return 'special'
    if VIRAMA in g:
        # Two viramas = a stacked/triple conjunct, which Tesseract's recoder
        # has no single unit for. Worth separating: these are expected to fail.
        return 'conjunct-stacked' if g.count(VIRAMA) > 1 else 'conjunct'
    if len(g) == 1:
        n = unicodedata.name(g, '')
        if 'KANNADA LETTER' in n:
            return 'vowel' if n.replace('KANNADA LETTER ', '') in VOWEL_NAMES else 'consonant'
        if 'KANNADA VOWEL SIGN' in n:
            return 'vowel-sign'
        if 'KANNADA DIGIT' in n:
            return 'digit'
        if 'KANNADA' in n:
            return 'sign'
        if g in string.ascii_uppercase:
            return 'latin-upper'
        if g in string.ascii_lowercase:
            return 'latin-lower'
        if g in string.digits:
            return 'ascii-digit'
        if g.isascii():
            return 'punctuation'
        return 'other'
    if any('KANNADA VOWEL SIGN' in unicodedata.name(c, '') for c in g[1:]):
        return 'cv-syllable'
    if any(unicodedata.name(c, '').endswith('ANUSVARA') or
           unicodedata.name(c, '').endswith('VISARGA') for c in g):
        return 'anusvara-visarga'
    return 'cluster'


CATEGORY_ORDER = ['vowel', 'consonant', 'vowel-sign', 'cv-syllable',
                  'anusvara-visarga', 'conjunct', 'conjunct-stacked', 'cluster',
                  'sign', 'digit', 'latin-upper', 'latin-lower', 'ascii-digit',
                  'punctuation', 'other', 'special']

# Always shown, even with no unicharset unit and no sample.
#
# Latin coverage has to be visible rather than merely absent. The kan unicharset
# contains ASCII digits and punctuation but NOT ONE letter — no A-Z, no a-z — so
# kan_hist physically cannot emit an English character no matter how it is
# trained. Any line with an English word in it is unencodable and gets dropped.
# Without these reference rows that hole is invisible: a character that is in no
# unicharset and in no inventory simply never appears in the report.
REFERENCE = set(string.ascii_letters + string.digits + '.,;:!?()[]-\'"/&%')

CATEGORY_LABEL = {
    'vowel': 'Independent vowels  ಅ ಆ ಇ',
    'consonant': 'Consonants  ಕ ಖ ಗ',
    'vowel-sign': 'Vowel signs (matra)  ಾ ಿ ೀ',
    'cv-syllable': 'Consonant + vowel sign  ಕಾ ಕಿ',
    'anusvara-visarga': 'Anusvara / visarga  ಂ ಃ',
    'conjunct': 'Conjuncts (ottu)  ಕ್ಕ ತ್ತ',
    'conjunct-stacked': 'Stacked conjuncts  ರ್ಘ್ಯ',
    'cluster': 'Other clusters',
    'sign': 'Signs',
    'digit': 'Kannada digits  ೦ ೧',
    'latin-upper': 'English capitals  A B C',
    'latin-lower': 'English lowercase  a b c',
    'ascii-digit': 'Western digits  0 1 2',
    'punctuation': 'Punctuation',
    'other': 'Other',
    'special': 'Tesseract internal',
}


def encodable(text, units, maxu):
    """Greedy longest-match. Verified against full backtracking search on
    162,572 corpus words with zero disagreements."""
    i, n = 0, len(text)
    while i < n:
        if text[i] in ' \t\n':
            i += 1
            continue
        for k in range(min(maxu, n - i), 0, -1):
            if text[i:i + k] in units:
                i += k
                break
        else:
            return False
    return True


# ── OCR worker ──────────────────────────────────────────────────────────────
_CFG = {}


def _init(cfg):
    _CFG.update(cfg)


def _ocr(png, tessdir, lang, psm):
    r = subprocess.run(
        ['tesseract', png, 'stdout', '--tessdata-dir', tessdir, '-l', lang,
         '--psm', str(psm)],
        capture_output=True, text=True, timeout=60,
        env={**os.environ, 'OMP_THREAD_LIMIT': '1'})
    return r.stdout.strip() if r.returncode == 0 else ''


def _norm(s):
    return unicodedata.normalize('NFC', s.replace(ZWNJ, '').strip())


def _job(item):
    font, png, truth = item
    out = {'font': font, 'gt': truth}
    try:
        got = _ocr(png, _CFG['model_dir'], _CFG['model'], _CFG['psm'])
        out['hist'] = _norm(got)
        out['hist_ok'] = _norm(got) == _norm(truth)
        if _CFG.get('base_dir'):
            b = _ocr(png, _CFG['base_dir'], _CFG['base'], _CFG['psm'])
            out['base'] = _norm(b)
            out['base_ok'] = _norm(b) == _norm(truth)
    except Exception as e:                      # noqa: BLE001
        out['error'] = str(e)[:80]
        out['hist_ok'] = False
    return out


def collect_samples(fonts, limit):
    samples = []
    for d in sorted(INVENTORY.iterdir()):
        if not d.is_dir():
            continue
        if fonts and d.name not in fonts:
            continue
        pairs = sorted(d.glob('*.png'))
        if limit:
            pairs = pairs[:limit]
        for p in pairs:
            g = p.with_suffix('.gt.txt')
            if not g.exists():
                continue
            t = g.read_text(encoding='utf-8').strip()
            if t:
                samples.append((d.name, str(p), t))
    return samples


# ── HTML ────────────────────────────────────────────────────────────────────
def colour(acc, n):
    if n == 0:
        return '#e5e7eb', '#6b7280'
    if acc >= 0.95:
        return '#065f46', '#ffffff'
    if acc >= 0.75:
        return '#10b981', '#062e21'
    if acc >= 0.5:
        return '#fbbf24', '#432c00'
    if acc > 0:
        return '#f97316', '#ffffff'
    return '#dc2626', '#ffffff'


def render_html(data, path):
    graphemes = data['graphemes']
    meta = data['meta']
    by_cat = collections.defaultdict(list)
    for g in graphemes:
        by_cat[g['category']].append(g)

    parts = ["""<!doctype html><meta charset="utf-8">
<title>Kannada coverage — what the model reads</title>
<style>
 :root{--bg:#fff;--fg:#111827;--mute:#6b7280;--line:#e5e7eb;--brand:#0f766e}
 *{box-sizing:border-box}
 body{margin:0;padding:28px;font:14px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
      color:var(--fg);background:var(--bg)}
 h1{font-size:1.4rem;margin:0 0 4px} h2{font-size:1rem;margin:26px 0 10px}
 .sub{color:var(--mute);font-size:.83rem;margin-bottom:18px}
 .bar{display:flex;gap:16px;flex-wrap:wrap;margin:16px 0 22px}
 .stat{border:1px solid var(--line);border-radius:8px;padding:10px 16px;min-width:130px}
 .stat b{display:block;font-size:1.5rem;line-height:1.2}
 .stat span{font-size:.72rem;color:var(--mute);text-transform:uppercase;letter-spacing:.4px}
 .legend{display:flex;gap:10px;flex-wrap:wrap;font-size:.74rem;align-items:center;margin-bottom:20px}
 .sw{display:inline-block;width:13px;height:13px;border-radius:3px;vertical-align:-2px;margin-right:4px}
 .grid{display:flex;flex-wrap:wrap;gap:5px}
 .cell{min-width:46px;padding:7px 6px 5px;border-radius:6px;text-align:center;cursor:pointer;
       border:1px solid rgba(0,0,0,.08)}
 .cell .g{font-size:1.15rem;line-height:1.25}
 .cell .p{font-size:.62rem;opacity:.85}
 .cell.unenc{outline:2px dashed #dc2626;outline-offset:1px}
 details{border:1px solid var(--line);border-radius:8px;padding:10px 14px;margin-bottom:8px}
 summary{cursor:pointer;font-weight:600}
 table{border-collapse:collapse;font-size:.78rem;margin-top:8px;width:100%}
 th,td{border-bottom:1px solid var(--line);padding:4px 8px;text-align:left}
 th{color:var(--mute);font-weight:600}
 .miss{color:#b91c1c}
 #panel{position:fixed;right:20px;bottom:20px;width:330px;max-height:62vh;overflow:auto;
        background:#fff;border:1px solid var(--line);border-radius:10px;padding:14px;
        box-shadow:0 8px 30px rgba(0,0,0,.14);display:none;z-index:9}
 #panel h3{margin:0 0 6px;font-size:1.6rem}
 .x{float:right;cursor:pointer;color:var(--mute)}
 .note{background:#fffbeb;border:1px solid #fcd34d;border-radius:8px;padding:10px 14px;
       font-size:.8rem;margin-bottom:18px}
</style>"""]

    p = parts.append
    p(f"<h1>Kannada coverage — what the model reads</h1>")
    p(f"<div class='sub'>{html.escape(meta['when'])} &nbsp;·&nbsp; "
      f"model <b>{html.escape(meta['model'])}</b> from {html.escape(meta['model_dir'])}/"
      + (f" &nbsp;·&nbsp; baseline <b>{html.escape(meta.get('base',''))}</b>"
         if meta.get('base') else '') + "</div>")

    if meta['measured']:
        p("<div class='note'><b>These numbers are optimistic.</b> Inventory images are part "
          "of the training set, so this measures “did the model learn this shape”, not "
          "“will it generalise to a real scan”. A grapheme failing <i>here</i> has "
          "definitively not been learned.</div>")
    else:
        p("<div class='note'>Static mode: unicharset coverage only. Re-run without "
          "<code>--static</code> to measure what the model actually reads.</div>")

    s = meta['summary']
    p("<div class='bar'>")
    p(f"<div class='stat'><b>{s['units']}</b><span>unicharset units</span></div>")
    p(f"<div class='stat'><b>{s['graphemes']}</b><span>graphemes tested</span></div>")
    if meta['measured']:
        p(f"<div class='stat'><b>{s['acc']:.1f}%</b><span>read correctly</span></div>")
        if s.get('base_acc') is not None:
            p(f"<div class='stat'><b>{s['base_acc']:.1f}%</b><span>stock kan</span></div>")
        p(f"<div class='stat'><b>{s['never']}</b><span>never read, any font</span></div>")
    p(f"<div class='stat'><b>{s['unencodable']}</b><span>not encodable</span></div>")
    p("</div>")

    p("<div class='legend'>")
    for lab, col in [('≥95%', '#065f46'), ('75–95%', '#10b981'), ('50–75%', '#fbbf24'),
                     ('&lt;50%', '#f97316'), ('never', '#dc2626'), ('untested', '#e5e7eb')]:
        p(f"<span><i class='sw' style='background:{col}'></i>{lab}</span>")
    p("<span style='margin-left:8px'><i class='sw' style='outline:2px dashed #dc2626;"
      "background:#fff'></i>not in unicharset — cannot ever be emitted</span></div>")

    mc = meta.get('markCoverage') or {}
    if mc:
        thin = [m for m, v in mc.items() if v['combos'] < 20]
        p("<h2>Mark coverage &nbsp;<span style='color:var(--mute);font-weight:400'>"
          "(how many consonant combinations exist per mark)</span></h2>")
        p("<table style='max-width:640px'><tr><th>mark</th><th>combinations</th>"
          "<th>tested</th><th>accuracy</th></tr>")
        for m, v in mc.items():
            acc = f"{v['acc']:.0f}%" if v.get('acc') is not None else '—'
            warn = " style='color:#b45309'" if v['combos'] < 20 else ""
            p(f"<tr{warn}><td style='font-size:1.15rem'>{html.escape(m)} "
              f"<span style='color:var(--mute);font-size:.7rem'>U+{ord(m):04X}</span></td>"
              f"<td>{v['combos']}</td><td>{v['tested']}</td><td>{acc}</td></tr>")
        p("</table>")
        if thin:
            p(f"<p class='note' style='margin-top:10px'><b>{len(thin)} mark(s) have almost "
              f"nothing to test: {' '.join(html.escape(m) for m in thin)}.</b> "
              "That is a gap in the <i>inventory</i>, not a model failure — the generator "
              "ran with <code>--attested-only</code>, so combinations absent from the "
              "corpus were never rendered, and nothing here can report on them either "
              "way. Regenerate the inventory without that restriction to close it.</p>")

    if meta.get('excluded'):
        ex = meta['excluded']
        p(f"<h2>Excluded from the test &nbsp;<span style='color:var(--mute);font-weight:400'>"
          f"({len(ex)})</span></h2>")
        p("<p style='font-size:.8rem;color:var(--mute);margin:0 0 8px'>Combinations using "
          "marks outside the standard set — nukta, length marks, vocalic RR. No page in "
          "this corpus contains them, so scoring them as failures is noise. "
          "<code>--all-marks</code> puts them back.</p>")
        p("<div class='grid'>" + ''.join(
            f"<div class='cell' style='background:#f1f5f9;color:#64748b'>"
            f"<div class='g'>{html.escape(g)}</div><div class='p'>skip</div></div>"
            for g in ex[:60]) + "</div>")

    for cat in CATEGORY_ORDER:
        items = by_cat.get(cat)
        if not items:
            continue
        items.sort(key=lambda g: (g.get('acc', -1), g['g']))
        n_bad = sum(1 for g in items if meta['measured'] and g.get('n', 0) and g['acc'] == 0)
        head = f"{CATEGORY_LABEL.get(cat, cat)} &nbsp;<span style='color:var(--mute);font-weight:400'>({len(items)}"
        head += f", {n_bad} never read" if n_bad else ""
        head += ")</span>"
        p(f"<h2>{head}</h2><div class='grid'>")
        for g in items:
            bg, fg = colour(g.get('acc', 0), g.get('n', 0))
            cls = 'cell unenc' if not g['encodable'] else 'cell'
            pct = f"{g['acc']*100:.0f}%" if g.get('n') else '—'
            p(f"<div class='{cls}' style='background:{bg};color:{fg}' "
              f"data-k='{html.escape(g['g'])}'>"
              f"<div class='g'>{html.escape(g['g'])}</div><div class='p'>{pct}</div></div>")
        p("</div>")

    p("<div id='panel'><span class='x' onclick=\"document.getElementById('panel').style.display='none'\">✕</span>"
      "<div id='pbody'></div></div>")
    p("<script>const D=" + json.dumps({g['g']: g for g in graphemes}, ensure_ascii=False) + ";")
    p("""
document.querySelectorAll('.cell').forEach(c=>c.onclick=()=>{
  const g=D[c.dataset.k]; if(!g) return;
  let h='<h3>'+g.g+'</h3>';
  h+='<div style="color:#6b7280;font-size:.75rem;margin-bottom:8px">'+g.codepoints+'<br>'+g.category+'</div>';
  if(!g.encodable) h+='<div style="color:#b91c1c;font-size:.78rem;margin-bottom:8px">'
     +'Not representable in the unicharset — the model can never output this, '
     +'regardless of training.</div>';
  if(g.fonts && g.fonts.length){
    h+='<table><tr><th>font</th><th>read as</th><th></th></tr>';
    g.fonts.forEach(f=>{h+='<tr><td>'+f.font.replace(/^karnata/,'')+'</td><td>'
      +(f.got||'<i style="color:#9ca3af">nothing</i>')+'</td><td>'
      +(f.ok?'✓':'<span class="miss">✗</span>')+'</td></tr>';});
    h+='</table>';
  } else h+='<div style="color:#6b7280;font-size:.8rem">No inventory sample for this unit.</div>';
  document.getElementById('pbody').innerHTML=h;
  document.getElementById('panel').style.display='block';
});""")
    p("</script>")
    path.write_text('\n'.join(parts), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='kan_hist')
    ap.add_argument('--baseline', default='kan')
    ap.add_argument('--no-baseline', action='store_true')
    ap.add_argument('--fonts', help='comma-separated font dir names')
    ap.add_argument('--limit', type=int, help='max samples per font (quick run)')
    ap.add_argument('--psm', default='13', help='default 13 = raw line, as trained')
    ap.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument('--static', action='store_true', help='unicharset only, no OCR')
    ap.add_argument('--all-marks', action='store_true',
                    help='include combinations using marks outside the standard '
                         'set (nukta, length marks, vocalic RR) — off by default')
    args = ap.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)
    fonts = set(args.fonts.split(',')) if args.fonts else None

    units_dir = find_model(UNITS_DIRS, args.baseline)
    if not units_dir:
        print(f'✗ {args.baseline}.traineddata not found. Run ① Prep base first.')
        return 1
    units = set(load_units(units_dir / f'{args.baseline}.traineddata'))

    base_dir = find_model(BASELINE_DIRS, args.baseline)
    if base_dir and not has_lstm(base_dir / f'{args.baseline}.traineddata'):
        print(f'  ⚠  {base_dir.name}/{args.baseline}.traineddata has no lstm layer —')
        print('     it cannot recognise anything. Skipping the baseline rather than')
        print('     reporting it as 0%.')
        base_dir = None
    if not units:
        print('✗ could not read the unicharset (is combine_tessdata on PATH?)')
        return 1
    maxu = max(len(u) for u in units)

    model_dir = find_model(MODEL_DIRS, args.model)
    measured = bool(model_dir) and not args.static
    if not model_dir and not args.static:
        print(f'✗ {args.model}.traineddata not found in best/ or tessdata_best/.')
        print('  Run ③ Package, or pass --static for unicharset coverage only.')
        return 1

    print('━' * 72)
    print('  Kannada coverage report')
    print(f'  unicharset : {len(units)} units from {units_dir.name}/')
    if base_dir:
        print(f'  baseline   : {args.baseline} from {base_dir.name}/')
    if measured:
        print(f'  model      : {args.model} from {model_dir.name}/  (psm {args.psm})')
    print('━' * 72)

    # Collect inventory labels even in --static mode. Without them the grid shows
    # only the 273 unicharset units, which is the alphabet Tesseract can emit but
    # not the script you actually care about reading — ಕಾ, ಕಿ, ಕ್ಕ and friends are
    # sequences of units, not units, and they are what a reader sees on the page.
    samples = collect_samples(fonts, args.limit)
    results = []
    if not measured:
        samples = []
        seen = set()
        for f, png, t in collect_samples(fonts, args.limit):
            seen.add(t)
        _static_labels = seen
    else:
        _static_labels = set()
    if samples:
        print(f'  measuring {len(samples)} inventory samples on {args.jobs} workers…')
        cfg = {'model_dir': str(model_dir), 'model': args.model, 'psm': args.psm}
        if not args.no_baseline and base_dir:
            cfg['base_dir'] = str(base_dir)
            cfg['base'] = args.baseline
        with multiprocessing.Pool(args.jobs, _init, (cfg,)) as pool:
            for i, r in enumerate(pool.imap_unordered(_job, samples, chunksize=16), 1):
                results.append(r)
                if i % 250 == 0:
                    ok = sum(1 for x in results if x.get('hist_ok'))
                    print(f'    {i}/{len(samples)}  ({100*ok/len(results):.1f}% correct)',
                          flush=True)

    # ── Aggregate per grapheme ──────────────────────────────────────────────
    per = collections.defaultdict(list)
    for r in results:
        per[r['gt']].append(r)

    keys = set(per) | set(units) | _static_labels | REFERENCE
    excluded = sorted(k for k in keys if not wanted_grapheme(k, args.all_marks))
    keys = {k for k in keys if wanted_grapheme(k, args.all_marks)}
    graphemes = []
    for g in sorted(keys):
        rs = per.get(g, [])
        n = len(rs)
        ok = sum(1 for r in rs if r.get('hist_ok'))
        graphemes.append({
            'g': g,
            'category': classify(g),
            'encodable': encodable(g, units, maxu),
            'codepoints': ' '.join(f'U+{ord(c):04X}' for c in g),
            'n': n,
            'acc': (ok / n) if n else 0.0,
            'fonts': [{'font': r['font'], 'got': r.get('hist', ''),
                       'ok': bool(r.get('hist_ok'))} for r in
                      sorted(rs, key=lambda r: r['font'])],
        })

    tested = [g for g in graphemes if g['n']]
    total_ok = sum(g['acc'] * g['n'] for g in tested)
    total_n = sum(g['n'] for g in tested)
    base_ok = sum(1 for r in results if r.get('base_ok'))
    has_base = any('base_ok' in r for r in results)

    summary = {
        'units': len(units),
        'graphemes': len(tested) if tested else len(units),
        'acc': (100 * total_ok / total_n) if total_n else 0,
        'base_acc': (100 * base_ok / len(results)) if has_base and results else None,
        'never': sum(1 for g in tested if g['acc'] == 0),
        'unencodable': sum(1 for g in graphemes if not g['encodable']),
    }
    # How well is each tested mark actually represented? A mark with one sample
    # is not being tested, it is being sampled — and a green cell there means
    # much less than a green cell for ್ with a thousand.
    mark_cov = {}
    for m in sorted(COVERAGE_MARKS):
        n = sum(1 for g in graphemes if m in g['g'] and len(g['g']) > 1)
        n_tested = sum(1 for g in graphemes if m in g['g'] and len(g['g']) > 1 and g['n'])
        ok = sum(g['acc'] * g['n'] for g in graphemes
                 if m in g['g'] and len(g['g']) > 1 and g['n'])
        tot = sum(g['n'] for g in graphemes if m in g['g'] and len(g['g']) > 1 and g['n'])
        mark_cov[m] = {'combos': n, 'tested': n_tested,
                       'acc': (100 * ok / tot) if tot else None,
                       'name': unicodedata.name(m, '')}

    data = {
        'meta': {
            'when': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'model': args.model,
            'model_dir': model_dir.name if model_dir else '-',
            'base': args.baseline if has_base else None,
            'measured': measured and bool(tested),
            'summary': summary,
            'excluded': excluded,
            'markCoverage': mark_cov,
            'allMarks': args.all_marks,
        },
        'graphemes': graphemes,
    }

    (REPORTS / 'unit-coverage.json').write_text(
        json.dumps(data, ensure_ascii=False, indent=1), encoding='utf-8')
    render_html(data, REPORTS / 'unit-coverage.html')

    print('')
    print(f'  unicharset units : {summary["units"]}')
    print(f'  graphemes tested : {len(tested)}')
    if tested:
        print(f'  read correctly   : {summary["acc"]:.1f}%'
              + (f'   (stock kan: {summary["base_acc"]:.1f}%)'
                 if summary['base_acc'] is not None else ''))
        print(f'  never read, any font : {summary["never"]}')
        worst = collections.Counter()
        for g in tested:
            if g['acc'] == 0:
                worst[g['category']] += 1
        for c, k in worst.most_common(6):
            print(f'     {c:20} {k}')
    print(f'  not encodable    : {summary["unencodable"]}')

    if excluded:
        print('')
        print(f'  excluded {len(excluded)} combination(s) using marks outside the')
        print('  standard set (nukta, length marks, vocalic RR) — no page in this')
        print('  corpus contains them, so scoring them as failures is noise:')
        print(f'    {" ".join(excluded[:24])}')
        print('    (--all-marks to include them)')

    print('')
    print('  Mark coverage — how many consonant combinations exist per mark:')
    print(f'    {"mark":6} {"combos":>7} {"tested":>7} {"acc":>7}')
    thin = []
    for m, v in mark_cov.items():
        acc = f'{v["acc"]:.0f}%' if v['acc'] is not None else '—'
        flag = ''
        if v['combos'] < 20:
            flag = '  ← thin'
            thin.append(m)
        print(f'    {m:6} {v["combos"]:>7} {v["tested"]:>7} {acc:>7}{flag}')
    if thin:
        print('')
        print(f'  {len(thin)} mark(s) have almost no combinations to test: {" ".join(thin)}')
        print('  That is a gap in the INVENTORY, not a model failure — the generator')
        print('  ran with --attested-only, so combinations absent from the corpus were')
        print('  never rendered. Nothing here can report on them either way. To close')
        print('  it, regenerate the inventory without that restriction.')
    print('')
    print('━' * 72)
    print(f'  {REPORTS.relative_to(ROOT)}/unit-coverage.html')
    print('━' * 72)
    return 0


if __name__ == '__main__':
    sys.exit(main())
