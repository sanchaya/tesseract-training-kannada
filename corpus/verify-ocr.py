#!/usr/bin/env python3
"""
verify-ocr.py  —  OCR Verification & Visual Diff Tool

Samples training images, runs them through Tesseract (tessdata_best and
kan_hist if available), computes character error rate, and generates a
self-contained HTML report with visual character-level diffs.

Usage:
    python3 corpus/verify-ocr.py                     # 20 rendered + 10 classical
    python3 corpus/verify-ocr.py --count 40          # more samples
    python3 corpus/verify-ocr.py --source rendered   # only line images
    python3 corpus/verify-ocr.py --source classical  # only A5 pages
    python3 corpus/verify-ocr.py --out my-report.html

Output: ocr-verification-report.html (self-contained, no external deps)
"""

import argparse
import base64
import io
import random
import re
import subprocess
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent.parent
TESSDATA     = ROOT / 'tessdata_best'
TESS_EXPANDED = ROOT / 'tessdata_expanded'
RENDERED_DIR = ROOT / 'rendered'
CLASSICAL_DIR = ROOT / 'classical-corpus-kannada' / 'a5-pages'

# ── OCR ────────────────────────────────────────────────────────────────────────
def run_ocr(img_path: Path, tessdata_dir: Path, lang: str, psm: int) -> str:
    result = subprocess.run(
        ['tesseract', str(img_path), 'stdout',
         '--tessdata-dir', str(tessdata_dir),
         '-l', lang, '--psm', str(psm),
         '-c', 'preserve_interword_spaces=1'],
        capture_output=True, text=True, errors='replace'
    )
    return result.stdout.strip()

# ── Metrics ────────────────────────────────────────────────────────────────────
def levenshtein(a: str, b: str) -> int:
    """Fast Levenshtein distance."""
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j] + (ca != cb), prev[j+1] + 1, curr[j] + 1))
        prev = curr
    return prev[-1]

def cer(gt: str, ocr: str) -> float:
    gt_c  = gt.replace(' ', '')
    ocr_c = ocr.replace(' ', '')
    if not gt_c: return 0.0
    return min(levenshtein(gt_c, ocr_c) / len(gt_c), 1.0)

def wer(gt: str, ocr: str) -> float:
    gt_w  = gt.split()
    ocr_w = ocr.split()
    if not gt_w: return 0.0
    return min(levenshtein(gt_w, ocr_w) / len(gt_w), 1.0)

# ── Character confusion ────────────────────────────────────────────────────────
def extract_confusions(gt: str, ocr: str, confusion: dict):
    """Collect character-level substitutions from alignment."""
    sm = SequenceMatcher(None, gt, ocr, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == 'replace':
            for gc, oc in zip(gt[i1:i2], ocr[j1:j2]):
                if gc != oc:
                    confusion[(gc, oc)] += 1

# ── Diff HTML ──────────────────────────────────────────────────────────────────
def diff_html(gt: str, ocr: str) -> tuple[str, str]:
    """
    Returns (gt_html, ocr_html) with inline colour spans.
    GT:  deletions in red,    substitutions underlined red
    OCR: insertions in orange, substitutions underlined orange
    """
    sm = SequenceMatcher(None, gt, ocr, autojunk=False)
    gt_parts, ocr_parts = [], []

    def esc(s):
        return s.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')

    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == 'equal':
            t = esc(gt[i1:i2])
            gt_parts.append(t)
            ocr_parts.append(t)
        elif op == 'replace':
            gt_parts.append(f'<span class="sub-del">{esc(gt[i1:i2])}</span>')
            ocr_parts.append(f'<span class="sub-ins">{esc(ocr[j1:j2])}</span>')
        elif op == 'delete':
            gt_parts.append(f'<span class="del">{esc(gt[i1:i2])}</span>')
        elif op == 'insert':
            ocr_parts.append(f'<span class="ins">{esc(ocr[j1:j2])}</span>')

    return ''.join(gt_parts), ''.join(ocr_parts)

# ── Image thumbnail → base64 ───────────────────────────────────────────────────
def img_to_b64(path: Path, max_w: int = 800, max_h: int = 200) -> str:
    if HAS_PIL:
        img = PILImage.open(path).convert('RGB')
        img.thumbnail((max_w, max_h), PILImage.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, 'JPEG', quality=80)
        return base64.b64encode(buf.getvalue()).decode()
    else:
        return base64.b64encode(path.read_bytes()).decode()

def img_tag(path: Path, max_w: int = 800, max_h: int = 200) -> str:
    ext = 'jpeg' if HAS_PIL else path.suffix.lstrip('.').lower()
    return f'<img src="data:image/{ext};base64,{img_to_b64(path, max_w, max_h)}" style="max-width:100%;max-height:{max_h}px;">'

# ── Sample collection ──────────────────────────────────────────────────────────
def collect_rendered(n: int) -> list[tuple[Path, str, int]]:
    """Returns list of (img_path, gt_text, psm)."""
    pairs = [
        (p, p.with_suffix('.gt.txt'))
        for p in RENDERED_DIR.glob('*.png')
        if p.with_suffix('.gt.txt').exists()
    ]
    random.shuffle(pairs)
    out = []
    for img, gt_f in pairs[:n]:
        out.append((img, gt_f.read_text(encoding='utf-8').strip(), 7))
    return out

def collect_classical(n: int) -> list[tuple[Path, str, int]]:
    """Returns list of (img_path, gt_text, psm) from a5-pages."""
    pairs = []
    if CLASSICAL_DIR.exists():
        for gt_f in CLASSICAL_DIR.rglob('*.gt.txt'):
            # Replace .gt.txt suffix with .png (page0001.gt.txt → page0001.png)
            img = gt_f.with_name(gt_f.name.removesuffix('.gt.txt') + '.png')
            if img.exists():
                pairs.append((img, gt_f))
    random.shuffle(pairs)
    out = []
    for img, gt_f in pairs[:n]:
        gt = gt_f.read_text(encoding='utf-8').strip()
        # Join multi-line gt.txt into single string for display
        gt = re.sub(r'\s+', ' ', gt).strip()
        out.append((img, gt, 6))
    return out

# ── HTML template ──────────────────────────────────────────────────────────────
HTML_HEAD = """<!DOCTYPE html>
<html lang="kn">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OCR Verification Report</title>
<style>
  :root {
    --bg: #0f1117; --card: #1a1d27; --border: #2d3045;
    --text: #e2e8f0; --muted: #8892a4; --accent: #6c8ebf;
    --green: #22c55e; --red: #ef4444; --orange: #f97316;
    --yellow: #eab308;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: var(--bg);
         color: var(--text); padding: 24px; }
  h1  { font-size: 1.5rem; margin-bottom: 4px; }
  h2  { font-size: 1.1rem; color: var(--muted); margin: 24px 0 12px; }
  .subtitle { color: var(--muted); font-size: .85rem; margin-bottom: 24px; }

  /* Summary bar */
  .summary-grid { display: grid; grid-template-columns: repeat(auto-fit,minmax(160px,1fr));
                  gap: 12px; margin-bottom: 28px; }
  .stat-card { background: var(--card); border: 1px solid var(--border);
               border-radius: 8px; padding: 16px; text-align: center; }
  .stat-card .label { font-size: .75rem; color: var(--muted); text-transform: uppercase;
                      letter-spacing: .05em; margin-bottom: 6px; }
  .stat-card .value { font-size: 1.8rem; font-weight: 700; }
  .good  { color: var(--green); }
  .ok    { color: var(--yellow); }
  .bad   { color: var(--red); }

  /* Model comparison header */
  .model-row { display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; }
  .model-badge { background: var(--card); border: 1px solid var(--border);
                 border-radius: 20px; padding: 6px 14px; font-size: .8rem; }
  .model-badge span { font-weight: 700; color: var(--accent); }

  /* Sample cards */
  .card { background: var(--card); border: 1px solid var(--border);
          border-radius: 8px; margin-bottom: 16px; overflow: hidden; }
  .card-header { display: flex; align-items: center; gap: 12px;
                 padding: 10px 14px; border-bottom: 1px solid var(--border);
                 font-size: .8rem; color: var(--muted); background: rgba(0,0,0,.2); }
  .card-header .filename { flex: 1; font-family: monospace; font-size: .75rem; }
  .cer-badge { border-radius: 4px; padding: 2px 8px; font-weight: 700;
               font-size: .8rem; }
  .card-body { padding: 14px; }
  .img-wrap { margin-bottom: 12px; border-radius: 4px; overflow: hidden;
              border: 1px solid var(--border); background: #fff; }
  .rows { display: flex; flex-direction: column; gap: 8px; }
  .row  { display: grid; grid-template-columns: 80px 1fr;
          gap: 10px; align-items: baseline; }
  .row-label { font-size: .7rem; font-weight: 700; text-transform: uppercase;
               letter-spacing: .05em; color: var(--muted); text-align: right;
               padding-top: 2px; }
  .row-text { font-family: "Noto Sans Kannada", "Noto Serif Kannada",
              serif; font-size: 1rem; line-height: 1.6;
              word-break: break-word; }
  .gt-text { color: #a8d8a8; }

  /* Diff colours */
  .del     { background: rgba(239,68,68,.3);  color: #fca5a5;
             text-decoration: line-through; border-radius: 2px; }
  .ins     { background: rgba(249,115,22,.3); color: #fdba74;
             border-radius: 2px; }
  .sub-del { background: rgba(239,68,68,.25); color: #fca5a5;
             text-decoration: underline; border-radius: 2px; }
  .sub-ins { background: rgba(249,115,22,.25); color: #fdba74;
             text-decoration: underline; border-radius: 2px; }

  /* Confusion table */
  .confusion-wrap { overflow-x: auto; }
  table { border-collapse: collapse; width: 100%; font-size: .85rem; }
  th { background: rgba(255,255,255,.05); padding: 8px 12px;
       text-align: left; font-size: .75rem; text-transform: uppercase;
       letter-spacing: .05em; color: var(--muted); border-bottom: 1px solid var(--border); }
  td { padding: 7px 12px; border-bottom: 1px solid rgba(255,255,255,.05); }
  tr:hover td { background: rgba(255,255,255,.03); }
  .char-cell { font-family: "Noto Sans Kannada", serif; font-size: 1.1rem; }
  .count-bar { display: inline-block; height: 8px; background: var(--accent);
               border-radius: 4px; margin-left: 8px; vertical-align: middle;
               opacity: .6; }

  /* Legend */
  .legend { display: flex; gap: 16px; margin-bottom: 16px;
            font-size: .75rem; flex-wrap: wrap; }
  .legend-item { display: flex; align-items: center; gap: 6px; }
  .legend-dot { width: 12px; height: 12px; border-radius: 2px; }

  /* Tabs */
  .tabs { display: flex; gap: 0; border-bottom: 1px solid var(--border);
          margin-bottom: 20px; }
  .tab { padding: 10px 20px; cursor: pointer; font-size: .85rem;
         border-bottom: 2px solid transparent; color: var(--muted);
         transition: all .15s; }
  .tab.active { color: var(--accent); border-bottom-color: var(--accent); }
  .tab-pane { display: none; }
  .tab-pane.active { display: block; }

  /* Filter bar */
  .filter-bar { display: flex; gap: 10px; margin-bottom: 16px; align-items: center;
                flex-wrap: wrap; }
  .filter-bar label { font-size: .8rem; color: var(--muted); }
  .filter-bar select, .filter-bar input {
    background: var(--card); border: 1px solid var(--border); color: var(--text);
    padding: 5px 10px; border-radius: 6px; font-size: .8rem; }
  .count-label { margin-left: auto; font-size: .8rem; color: var(--muted); }
</style>
</head>
<body>
"""

HTML_FOOT = """
<script>
// Tab switching
document.querySelectorAll('.tab').forEach(t => {
  t.addEventListener('click', () => {
    const panel = t.dataset.tab;
    document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(x => x.classList.remove('active'));
    t.classList.add('active');
    document.getElementById(panel).classList.add('active');
  });
});

// Filter samples
function applyFilter() {
  const src   = document.getElementById('filter-src').value;
  const model = document.getElementById('filter-model').value;
  const sort  = document.getElementById('filter-sort').value;
  const cards = Array.from(document.querySelectorAll('.card[data-src]'));

  // Filter
  let visible = cards.filter(c => {
    if (src   !== 'all' && c.dataset.src   !== src)   return false;
    return true;
  });

  // Sort
  visible.sort((a, b) => {
    const ka = parseFloat(a.dataset[model] || 0);
    const kb = parseFloat(b.dataset[model] || 0);
    return sort === 'worst' ? kb - ka : ka - kb;
  });

  // Hide all, then show in order
  cards.forEach(c => { c.style.display = 'none'; c.parentNode.appendChild(c); });
  visible.forEach((c, i) => {
    c.style.display = '';
    c.parentNode.insertBefore(c, c.parentNode.querySelector('.card[data-src]') || null);
  });

  // Reorder visible ones
  const container = document.getElementById('samples-container');
  visible.forEach(c => container.appendChild(c));

  document.getElementById('visible-count').textContent =
    `Showing ${visible.length} of ${cards.length} samples`;
}
document.getElementById('filter-src')  ?.addEventListener('change', applyFilter);
document.getElementById('filter-model')?.addEventListener('change', applyFilter);
document.getElementById('filter-sort') ?.addEventListener('change', applyFilter);
</script>
</body></html>
"""

# ── CER colour ─────────────────────────────────────────────────────────────────
def cer_class(v: float) -> str:
    return 'good' if v < 0.1 else ('ok' if v < 0.35 else 'bad')

def cer_badge_style(v: float) -> str:
    if v < 0.1:   return 'background:rgba(34,197,94,.2);color:#4ade80;'
    if v < 0.35:  return 'background:rgba(234,179,8,.2);color:#fde047;'
    return 'background:rgba(239,68,68,.2);color:#f87171;'

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--count', type=int, default=30,
                    help='Total number of samples (default 30)')
    ap.add_argument('--source', choices=['rendered','classical','both'], default='both',
                    help='Image source (default: both)')
    ap.add_argument('--seed', type=int, default=42, help='Random seed')
    ap.add_argument('--out', type=str, default='ocr-verification-report.html')
    args = ap.parse_args()

    random.seed(args.seed)
    out_path = ROOT / args.out

    # Determine available models
    models = []
    if (TESSDATA / 'kan.traineddata').exists():
        models.append(('tessdata_best', TESSDATA, 'kan', '#6c8ebf'))
    if (TESSDATA / 'kan_hist.traineddata').exists():
        models.append(('kan_hist', TESSDATA, 'kan_hist', '#a78bfa'))
    if (TESS_EXPANDED / 'kan.traineddata').exists():
        models.append(('tessdata_expanded', TESS_EXPANDED, 'kan', '#34d399'))
    if not models:
        print("ERROR: No tessdata found.")
        sys.exit(1)

    print(f"Models: {[m[0] for m in models]}")

    # Collect samples
    n = args.count
    samples = []
    if args.source in ('rendered', 'both'):
        n_r = n * 2 // 3 if args.source == 'both' else n
        s = collect_rendered(n_r)
        for img, gt, psm in s:
            samples.append({'img': img, 'gt': gt, 'psm': psm, 'src': 'rendered',
                            'label': img.stem})
        print(f"Collected {len(s)} rendered samples")
    if args.source in ('classical', 'both'):
        n_c = n // 3 if args.source == 'both' else n
        s = collect_classical(n_c)
        for img, gt, psm in s:
            rel = img.relative_to(CLASSICAL_DIR)
            samples.append({'img': img, 'gt': gt, 'psm': psm, 'src': 'classical',
                            'label': str(rel)})
        print(f"Collected {len(s)} classical samples")

    if not samples:
        print("ERROR: No samples found.")
        sys.exit(1)

    # Run OCR on all samples
    print(f"Running OCR on {len(samples)} images across {len(models)} models...")
    confusion = {m[0]: defaultdict(int) for m in models}

    for i, s in enumerate(samples):
        if i % 10 == 0:
            print(f"  {i}/{len(samples)}...", flush=True)
        s['ocr'] = {}
        for mname, tdata, lang, _ in models:
            try:
                ocr_text = run_ocr(s['img'], tdata, lang, s['psm'])
            except Exception as e:
                ocr_text = f'[ERROR: {e}]'
            s['ocr'][mname] = ocr_text
            extract_confusions(s['gt'].replace(' ',''),
                               ocr_text.replace(' ',''),
                               confusion[mname])

    print("  Done.")

    # Compute CER/WER per sample
    for s in samples:
        s['cer'] = {}
        s['wer'] = {}
        for mname, _, _, _ in models:
            s['cer'][mname] = cer(s['gt'], s['ocr'][mname])
            s['wer'][mname] = wer(s['gt'], s['ocr'][mname])

    # Overall stats per model
    overall = {}
    for mname, _, _, _ in models:
        cers = [s['cer'][mname] for s in samples]
        wers = [s['wer'][mname] for s in samples]
        overall[mname] = {
            'cer': sum(cers) / len(cers) if cers else 0,
            'wer': sum(wers) / len(wers) if wers else 0,
        }

    # ── Build HTML ─────────────────────────────────────────────────────────────
    html = [HTML_HEAD]

    html.append('<h1>🔍 OCR Verification Report</h1>')
    html.append(f'<p class="subtitle">'
                f'{len(samples)} samples &nbsp;·&nbsp; '
                f'{", ".join(m[0] for m in models)}'
                f'</p>')

    # Summary stats
    html.append('<div class="summary-grid">')
    for mname, _, _, col in models:
        v = overall[mname]['cer']
        cls = cer_class(v)
        html.append(f'''<div class="stat-card">
  <div class="label">{mname} CER</div>
  <div class="value {cls}">{v:.1%}</div>
</div>''')
    for mname, _, _, col in models:
        v = overall[mname]['wer']
        cls = cer_class(v)
        html.append(f'''<div class="stat-card">
  <div class="label">{mname} WER</div>
  <div class="value {cls}">{v:.1%}</div>
</div>''')
    html.append(f'''<div class="stat-card">
  <div class="label">Samples</div>
  <div class="value" style="color:#94a3b8">{len(samples)}</div>
</div>''')
    html.append('</div>')  # summary-grid

    # Tabs
    html.append('''<div class="tabs">
  <div class="tab active" data-tab="tab-samples">Sample Pages</div>
  <div class="tab" data-tab="tab-confusion">Character Confusion</div>
</div>''')

    # ── Tab: Samples ───────────────────────────────────────────────────────────
    html.append('<div class="tab-pane active" id="tab-samples">')

    # Legend + filter bar
    html.append('''<div class="legend">
  <div class="legend-item"><div class="legend-dot" style="background:rgba(34,197,94,.5)"></div> Correct</div>
  <div class="legend-item"><div class="legend-dot del"></div> Deleted (in GT, missed by OCR)</div>
  <div class="legend-item"><div class="legend-dot ins"></div> Inserted (OCR hallucination)</div>
  <div class="legend-item"><div class="legend-dot sub-del"></div> Substituted</div>
</div>''')

    src_opts = '<option value="all">All sources</option>'
    if any(s['src'] == 'rendered'   for s in samples): src_opts += '<option value="rendered">Rendered lines</option>'
    if any(s['src'] == 'classical'  for s in samples): src_opts += '<option value="classical">Classical A5 pages</option>'

    model_opts = ''.join(f'<option value="{m[0]}">{m[0]}</option>' for m in models)

    html.append(f'''<div class="filter-bar">
  <label>Source: <select id="filter-src">{src_opts}</select></label>
  <label>Sort by: <select id="filter-model">{model_opts}</select>
    <select id="filter-sort">
      <option value="worst">Worst first</option>
      <option value="best">Best first</option>
    </select>
  </label>
  <span class="count-label" id="visible-count">Showing {len(samples)} samples</span>
</div>''')

    html.append('<div id="samples-container">')

    # Sort worst-first by first model
    first_model = models[0][0]
    samples_sorted = sorted(samples, key=lambda s: s['cer'].get(first_model, 0), reverse=True)

    for s in samples_sorted:
        # data-* for JS sorting
        data_attrs = ' '.join(f'data-{m[0].replace("-","_")}="{s["cer"][m[0]]:.4f}"'
                              for m, *_ in [(m,) for m in models])
        html.append(f'<div class="card" data-src="{s["src"]}" {data_attrs}>')

        # Card header
        cer_v = s['cer'][first_model]
        html.append(f'''<div class="card-header">
  <span class="filename">{s["label"]}</span>
  <span style="color:var(--muted)">{s["src"]}</span>
  {''.join(f'<span class="cer-badge" style="{cer_badge_style(s["cer"][m[0]])}">CER {s["cer"][m[0]]:.1%}</span>' for m in models)}
</div>''')

        html.append('<div class="card-body">')

        # Image
        try:
            html.append(f'<div class="img-wrap">{img_tag(s["img"], 900, 160)}</div>')
        except Exception:
            html.append(f'<div class="img-wrap" style="padding:8px;color:#666">image unavailable</div>')

        html.append('<div class="rows">')

        # GT row
        html.append(f'''<div class="row">
  <div class="row-label">GT</div>
  <div class="row-text gt-text">{s["gt"][:500]}</div>
</div>''')

        # OCR rows for each model
        for mname, _, _, col in models:
            ocr_t = s['ocr'][mname]
            gt_h, ocr_h = diff_html(s['gt'], ocr_t)
            html.append(f'''<div class="row">
  <div class="row-label" style="color:{col}">{mname}</div>
  <div class="row-text">{ocr_h[:1000]}</div>
</div>''')

        html.append('</div>')  # rows
        html.append('</div>')  # card-body
        html.append('</div>')  # card

    html.append('</div>')  # samples-container
    html.append('</div>')  # tab-pane samples

    # ── Tab: Confusion ─────────────────────────────────────────────────────────
    html.append('<div class="tab-pane" id="tab-confusion">')
    html.append('<h2>Character Confusion Matrix — top 40 errors per model</h2>')

    for mname, _, _, col in models:
        conf = confusion[mname]
        top = sorted(conf.items(), key=lambda x: x[1], reverse=True)[:40]
        if not top:
            continue
        max_cnt = top[0][1] if top else 1

        html.append(f'<h2 style="color:{col}">⊕ {mname}</h2>')
        html.append('<div class="confusion-wrap"><table>')
        html.append('''<tr>
  <th>#</th><th>GT char</th><th>OCR char</th>
  <th>Unicode (GT)</th><th>Unicode (OCR)</th><th>Count</th>
</tr>''')
        for rank, ((gc, oc), cnt) in enumerate(top, 1):
            bar_w = int(120 * cnt / max_cnt)
            gu = ' '.join(f'U+{ord(c):04X}' for c in gc)
            ou = ' '.join(f'U+{ord(c):04X}' for c in oc)
            html.append(f'''<tr>
  <td style="color:var(--muted)">{rank}</td>
  <td class="char-cell" style="background:rgba(34,197,94,.1)">{gc}</td>
  <td class="char-cell" style="background:rgba(239,68,68,.1)">{oc}</td>
  <td style="font-family:monospace;font-size:.75rem;color:var(--muted)">{gu}</td>
  <td style="font-family:monospace;font-size:.75rem;color:var(--muted)">{ou}</td>
  <td>{cnt}<span class="count-bar" style="width:{bar_w}px"></span></td>
</tr>''')
        html.append('</table></div>')

    html.append('</div>')  # tab-pane confusion
    html.append(HTML_FOOT)

    out_path.write_text(''.join(html), encoding='utf-8')
    print(f"\n✓ Report saved: {out_path}")
    print(f"\n  Summary:")
    for mname, _, _, _ in models:
        print(f"    {mname:30s}  CER {overall[mname]['cer']:.1%}   WER {overall[mname]['wer']:.1%}")

if __name__ == '__main__':
    main()
