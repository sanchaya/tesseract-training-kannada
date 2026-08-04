#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# 02-make-lstmf.sh
#
# Converts PNG+gt.txt pairs (from render-corpus.py) into
# .lstmf training files that lstmtraining consumes.
#
# Scans two directories:
#   rendered/      — synthetic images from render-corpus.py
#   scan-input/    — real scan images you supply (PNG + .gt.txt)
#
# Output:
#   lstmf/rendered/   — lstmf from synthetic images
#   lstmf/scan/       — lstmf from real scan images
#   lstmf/list.txt    — combined training file list
#
# Prerequisites:
#   Run 01-prep-base.sh first (needs tessdata_best/kan.traineddata)
# ═══════════════════════════════════════════════════════════════
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$SCRIPT_DIR/.."
cd "$ROOT"

TESSDATA_BEST="$ROOT/tessdata_best"
LSTMF_DIR="$ROOT/lstmf"
RENDERED_DIR="$ROOT/rendered"
SCAN_DIR="$ROOT/scan-input"
INVENTORY_DIR="$ROOT/inventory"

# ── Classical corpus A5 pages (optional) ──────────────────────────
# Set CLASSICAL_A5_DIR to the a5-pages/ output of render-a5-pages.py.
# Each leaf directory (a5-pages/<title>/<font_tag>/) is a flat folder
# of PNG+gt.txt pairs that process_dir can consume directly.
#
# Override via environment:
#   CLASSICAL_A5_DIR=/path/to/classical-corpus-kannada/a5-pages \
#       ./scripts/02-make-lstmf.sh
CLASSICAL_A5_DIR="${CLASSICAL_A5_DIR:-}"

# ── Character inventory (optional) ───────────────────────────────────
# Set INVENTORY_DIR to the output of generate-inventory.py.
# Contains systematic Kannada character combinations for training baseline.
INVENTORY_DIR="${INVENTORY_DIR:-$INVENTORY_DIR}"

# ── Tesseract 5 check ─────────────────────────────────────────────
TESS_VER=$(tesseract --version 2>&1 | head -1)
if echo "$TESS_VER" | grep -qE "^tesseract [45]\."; then
    echo "  Tesseract: $TESS_VER"
    if echo "$TESS_VER" | grep -q "^tesseract 4\."; then
        echo ""
        echo "  ✗ ERROR: Tesseract 4 detected — refusing to build lstmf."
        echo ""
        echo "     Tesseract 4 cannot encode the Kannada virama (U+0CCD / ್),"
        echo "     which appears in virtually every conjunct in this corpus."
        echo "     It does not fail loudly: it writes .lstmf files that look"
        echo "     fine and then fail during training with"
        echo "     'Encoding of string failed', wasting the whole run."
        echo ""
        echo "     This used to be an interactive 'continue anyway?' prompt. It"
        echo "     is now a hard failure — answering yes could only ever produce"
        echo "     a corrupt training set, and the prompt also hung unattended"
        echo "     runs (portal, pipeline script, CI) waiting on stdin."
        echo ""
        echo "     Install Tesseract 5:"
        echo "       macOS:  brew install tesseract"
        echo "       Ubuntu: sudo add-apt-repository ppa:alex-p/tesseract-ocr5"
        echo "               sudo apt update && sudo apt install tesseract-ocr"
        echo ""
        echo "     Then confirm:  tesseract --version   # must report 5.x"
        exit 1
    fi
else
    echo "ERROR: Tesseract not found. Install Tesseract 5 first."
    echo "  macOS:  brew install tesseract"
    exit 1
fi

# ── Ensure tessdata configs dir exists (needed for lstm.train) ────
if [ ! -d "$TESSDATA_BEST/configs" ]; then
    echo "  Setting up tessdata configs..."
    SYS_CONFIGS=$(find /usr/local/share /opt/homebrew/share /usr/share \
        -name "configs" -path "*/tessdata/*" 2>/dev/null | head -1)
    if [ -n "$SYS_CONFIGS" ]; then
        ln -sf "$SYS_CONFIGS" "$TESSDATA_BEST/configs"
        echo "  ✓ Linked configs from $SYS_CONFIGS"
    else
        echo "ERROR: Cannot find tessdata configs dir."
        echo "  Locate it with: find / -name 'lstm.train' 2>/dev/null | head -5"
        exit 1
    fi
fi

[ -f "$TESSDATA_BEST/kan.traineddata" ] || {
    echo "ERROR: tessdata_best/kan.traineddata not found."
    echo "  Run ./scripts/01-prep-base.sh first."
    exit 1
}

mkdir -p "$LSTMF_DIR/rendered" "$LSTMF_DIR/scan"
> "$LSTMF_DIR/list.txt"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Step 2 — PNG+GT → lstmf"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

process_dir() {
    local src_dir="$1"
    local out_dir="$2"
    local label="$3"

    if [ ! -d "$src_dir" ]; then
        echo "  $label: directory not found, skipping ($src_dir)"
        return
    fi

    local count
    count=$(find "$src_dir" -maxdepth 1 \( -name "*.png" -o -name "*.tif" \) \
            2>/dev/null | wc -l | tr -d ' ')
    if [ "$count" -eq 0 ]; then
        echo "  $label: no images found in $src_dir"
        return
    fi

    mkdir -p "$out_dir"
    echo ""
    echo "→ Processing $label ($count images)..."

    python3 - "$src_dir" "$out_dir" "$LSTMF_DIR/list.txt" "$label" \
             "$TESSDATA_BEST" "${WORKERS:-0}" << 'PYEOF'
import sys, subprocess, shutil, os, re as _re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

src_dir, out_dir, list_txt, label, tdata, workers_arg = \
    sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]

# Workers: env WORKERS > CLI arg > cpu_count (capped at 8 to avoid Tesseract overload)
_ncpu = os.cpu_count() or 4
WORKERS = int(os.environ.get('WORKERS') or (workers_arg if workers_arg != '0' else 0) or min(_ncpu, 8))

# ── Classical corpus text cleaning ───────────────────────────────────────────
def _clean_gt(raw: str) -> str:
    """
    Pre-process classical Kannada ground-truth text before lstmf creation.

    Known issues across digitised texts in this corpus:
      1. ೦ (U+0CE6) used as anusvara ಂ (U+0C82) — common OCR/input error.
         e.g. 'ಬೇ೦ಟಿ' → 'ಬೇಂಟಿ'
      2. XML/HTML chapter tags: <ch>ಐದನೆಯ ಸಂಧಿ</ch>, <p>…</p>, etc. —
         editorial markup in nalacharitre and similar digitised texts.
      3. Devanagari dandas ।/॥ (U+0964/U+0965) used as verse separators.
      4. Devanagari digits ०-९ (U+0966-U+096F) used as verse numbers.
      5. Editorial single-char annotations: (ಕ), (ಚ), (ರ) — alternate readings.
      6. ASCII markup: + verse dividers, stray quotes.
      7. ASCII pipe || used as speaker/section markers in drama texts (Yakshagana).
         e.g. '|| ಸಿದ್ಧಯ್ಯ ||' — | (0x7C) is not in the Kannada unicharset.
      8. Stray Latin letters embedded in Kannada words — digitization artifacts.
         e.g. 'ಕಂದಾs' → 'ಕಂದಾ'  (ASCII 's', 0x73, not in unicharset).
    """
    t = raw
    t = t.replace('೦', 'ಂ')              # ೦ → ಂ  digit-zero → anusvara
    t = _re.sub(r'<[^>]+>', '', t)        # strip XML/HTML tags (<ch>, <p>, etc.)
    t = _re.sub(r'[।॥]', '', t)          # strip Devanagari dandas U+0964/U+0965
    t = _re.sub(r'[०-९]', '', t)         # strip Devanagari digits ०-९
    t = _re.sub(r'\([ಀ-೿]\)', '', t)     # strip editorial (X) annotations
    t = t.replace('+', '')                # strip verse-divider +
    t = _re.sub(r'[\'"ʼ]', '', t)        # strip stray quote chars
    t = t.replace('|', '')               # strip ASCII pipes (drama text || markers)
    t = _re.sub(r'[a-zA-Z]', '', t)      # strip stray Latin letters (digitization artifacts)
    # Word-final virama (U+0CCD) not followed by a Kannada consonant.
    # The unicharset has ್‌ (virama+ZWNJ) as an explicit half-form entry,
    # but has NO entry for standalone ್.  Words like ರಾವ್, ಕನ್, ಸ್ end in
    # consonant + ್ with nothing following — causing ~87% skip ratio.
    # Fix: append ZWNJ so it maps to the ್‌ unicharset slot.
    t = _re.sub(r'್(?![ಕ-ಹೞೠೡ‌])',
                r'್‌', t)
    t = _re.sub(r'\s+', ' ', t).strip()
    return t

# ── Unicharset — determine which characters to filter ────────────────────────
# tessdata_best/kan.traineddata is missing: ಋ ಙ ಝ ಱ ಩ ಴ ೃ
# tessdata_expanded adds them all via 00c-expand-unicharset.sh.
#
# Three tiers:
#   tessdata_best only          → filter all 7 known-missing chars
#   tessdata_expanded (old)     → filter ೃ only  (re-run 00c --force to fix)
#   tessdata_expanded (updated) → no filtering (all chars present)
#
# IMPORTANT: _has_vocalic_r must check tessdata_expanded *exists* — not just the
# unicharset cache file.  After 00c Step 5 the cache has ೃ, but if combine_lang_model
# failed (e.g. missing Latin.unicharset) tessdata_expanded is NOT built yet and
# tessdata_best still can't encode ೃ → "Failed to read boxes" errors.
_exp_tdata_dir = os.path.join(tdata, '..', 'tessdata_expanded')
_exp_tdata     = os.path.join(_exp_tdata_dir, 'kan.traineddata')
_uc_file       = os.path.join(tdata, '..', 'tmp', 'unicharset_work', 'kan_expanded.unicharset')
_has_expanded  = os.path.exists(_exp_tdata)
_has_vocalic_r = (
    _has_expanded                               # tessdata_expanded must be built
    and os.path.exists(_uc_file)
    and 'ೃ' in open(_uc_file).read()
)
_UNSUPPORTED   = (set()          if _has_vocalic_r  else
                  {'ೃ'}          if _has_expanded   else
                  set('ಋಙಝಱ಩಴ೃ'))
# Use tessdata_expanded when available so rare chars can be encoded correctly.
_run_tdata = _exp_tdata_dir if _has_expanded else tdata
# Ensure configs symlink exists in run tessdata dir (needed for lstm.train config)
_run_configs  = os.path.join(_run_tdata, 'configs')
_best_configs = os.path.join(tdata, 'configs')
if not os.path.exists(_run_configs) and os.path.exists(_best_configs):
    try:
        os.symlink(os.path.realpath(_best_configs), _run_configs)
    except OSError:
        pass
if _UNSUPPORTED:
    print(f"  Note: filtering chars absent from unicharset: "
          f"{' '.join(sorted(_UNSUPPORTED))} "
          f"(run 00c-expand-unicharset.sh {'--force ' if _has_expanded else ''}to add them)",
          flush=True)

# ── Authoritative encodability check ─────────────────────────────────────────
# The hardcoded _UNSUPPORTED set above is a guess maintained by hand. This reads
# the ACTUAL unicharset out of the traineddata being used, so nothing unencodable
# can reach list.txt regardless of what is stale on disk.
#
# Three classes of input fail encoding and used to spam every training epoch with
# "Can't encode transcription / Encoding of string failed":
#   • reserved codepoints that aren't characters (U+0C8D, U+0C91, U+0CA9)
#   • real characters absent from the unicharset (ಌ, ೄ, ೞ)
#   • characters valid only INSIDE a cluster — ಞ encodes within ಜ್ಞ but not alone
def _load_units():
    import subprocess, tempfile
    td = _exp_tdata if _has_expanded else os.path.join(tdata, 'kan.traineddata')
    if not os.path.exists(td):
        return None
    try:
        with tempfile.TemporaryDirectory() as tmp:
            prefix = os.path.join(tmp, 'kan.')
            subprocess.run(['combine_tessdata', '-u', td, prefix],
                           capture_output=True, check=True)
            uc = prefix + 'lstm-unicharset'
            if not os.path.exists(uc):
                return None
            with open(uc, encoding='utf-8', errors='replace') as fh:
                lines = fh.read().split('\n')
            return {l.split(' ')[0] for l in lines[1:] if l.strip()} or None
    except Exception:
        return None

_UNITS   = _load_units()
_MAXUNIT = max((len(u) for u in _UNITS), default=1) if _UNITS else 1

# Whitespace and the virama are never standalone unicharset units, yet both are
# perfectly encodable: Tesseract treats space as a word separator outside the
# unicharset, and the virama (್ U+0CCD) only ever appears fused into cluster
# units such as ್ನ. Treating them as failures would reject 96% of valid corpus
# lines — every multi-word sentence — so they are exempt from the check.
_ENCODE_EXEMPT = set(' \t\n್')

def _encodable(text):
    """Greedy longest-match segmentation into unicharset units (Tesseract's encoder)."""
    if not _UNITS:
        return True
    i, n = 0, len(text)
    while i < n:
        if text[i] in _ENCODE_EXEMPT:
            i += 1
            continue
        for size in range(min(_MAXUNIT, n - i), 0, -1):
            if text[i:i + size] in _UNITS:
                i += size
                break
        else:
            return False
    return True

print(f"  Unicharset: {len(_UNITS) if _UNITS else 'UNAVAILABLE — encodability check disabled'}"
      f"{' units' if _UNITS else ''}", flush=True)

def _make_lstmf_impl(img_path_str):
    """
    Convert one PNG+gt.txt pair to an lstmf file.
    Returns absolute path to the lstmf, or None on failure.
    Thread-safe: each call writes to a unique stem, no shared file handles.
    """
    img_path = Path(img_path_str)
    gt_path  = img_path.with_suffix('.gt.txt')

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    stem    = img_path.stem
    dst_img = out_path / (stem + img_path.suffix)
    dst_box = out_path / (stem + '.box')
    lstmf   = out_path / (stem + '.lstmf')

    # ── Read + clean the ground truth FIRST ─────────────────────────────────
    # Validation must happen BEFORE the resume check. Previously the function
    # returned an existing .lstmf immediately, so every file built by an earlier
    # run was re-admitted to list.txt without ever being validated — the guards
    # below logged zero rejections while training still failed on 1,480 lines
    # per 3,000 log lines. Resume must not mean "trust whatever is on disk".
    from PIL import Image as _PIL
    with _PIL.open(img_path) as im:
        w, h = im.size

    # _clean_gt() fixes corpus OCR errors (digit-zero→anusvara, editorial marks, +).
    # _UNSUPPORTED filters whole tokens whose chars aren't in the tessdata unicharset.
    # Both are computed once at module level above.
    _raw    = gt_path.read_text(encoding='utf-8', errors='ignore').strip()
    _raw    = _clean_gt(_raw)                        # Fix OCR / editorial issues
    _tokens = [t for t in _raw.split(' ') if not any(c in _UNSUPPORTED for c in t)]
    gt_text = _re.sub(r'\s+', ' ', ' '.join(_tokens)).strip()

    def _reject(reason):
        """Drop this pair and remove any stale artefacts a previous run left."""
        print(f"  ⊘ {stem}: {reason}", flush=True)
        for _f in (dst_img, dst_box, lstmf):
            try: _f.unlink(missing_ok=True)
            except OSError: pass
        return None

    _txt = gt_text.strip()

    # Guard 1 — encodable in the unicharset actually being used
    if _txt and not _encodable(_txt):
        _bad = ''.join(sorted({c for c in _txt if not _encodable(c)}))
        return _reject(f"not encodable in unicharset"
                       f"{f' (offending: {_bad})' if _bad else ''}")

    # Guard 2 — CTC feasibility. The LSTM scales input to 48px height, so the
    # timestep budget is roughly the width at that scale; CTC needs at least one
    # timestep per label. A full PAGE image paired with the whole page's text
    # (875x1241 → ~33 timesteps, ~700 labels) can never align, and lstmtraining
    # reports "Compute CTC targets failed". Line images pass comfortably.
    _timesteps = int(w * (48.0 / h)) if h else 0
    if _txt and _timesteps < len(_txt):
        return _reject(f"CTC infeasible — {len(_txt)} labels need > {_timesteps} "
                       f"timesteps ({w}x{h}). Page-level image? Needs line segmentation.")

    # Resume: already built AND validated above
    if lstmf.exists():
        return str(lstmf.resolve())

    shutil.copy2(img_path, dst_img)

    if not gt_text:
        # Every token was filtered (e.g. entire page is ೃ-dense Sanskrit verse).
        # An empty box file causes "Failed to read boxes from X.png" in Tesseract.
        # Skip cleanly; this page will be included once 00c-expand-unicharset.sh
        # --force is re-run to add the missing chars to tessdata_expanded.
        dst_img.unlink(missing_ok=True)
        dst_box.unlink(missing_ok=True)   # remove any stale box from a prior run
        return None

    # Split long GT into multiple WordStr box lines. Tesseract reads box text
    # into a fixed kBoxReadBufSize (1024) buffer — a single line longer than
    # ~1000 UTF-8 bytes gets truncated mid-character ("Bad UTF-8 str … at col
    # 1022" → "Failed to read boxes from X.png"). Chunks are concatenated back
    # into one full line by TrainFromBoxes, so word spacing must be preserved:
    # trailing spaces are stripped by chomp_string, so append a trailing space
    # to every chunk except the last.
    _MAX_BOX_BYTES = 1000
    _chunks = []
    _cur = ''
    for _word in gt_text.split(' '):
        _cand = _word if not _cur else _cur + ' ' + _word
        if len(_cand.encode('utf-8')) <= _MAX_BOX_BYTES:
            _cur = _cand
        else:
            if _cur:
                _chunks.append(_cur)
            _cur = _word
    if _cur:
        _chunks.append(_cur)

    with open(dst_box, 'w', encoding='utf-8') as bf:
        for _i, _chunk in enumerate(_chunks):
            _line = _chunk + (' ' if _i < len(_chunks) - 1 else '')
            bf.write(f"WordStr 0 0 {w} {h} 0 #{_line}\n")

    # (Encodability and CTC-feasibility guards ran before the resume check above,
    #  so anything reaching this point is known-valid.)

    result = subprocess.run(
        ["tesseract", str(dst_img), str(out_path / stem),
         "--tessdata-dir", _run_tdata,
         "--dpi", "150", "--psm", "6",
         "-l", "kan", "lstm.train"],
        capture_output=True, encoding='utf-8', errors='replace'
    )

    if lstmf.exists():
        return str(lstmf.resolve())

    # Surface the actual Tesseract error so we can diagnose failures.
    err_lines = (result.stderr or '').strip().splitlines()
    # Find the most useful line (skip generic "Tesseract Open Source..." header)
    reason = next((l for l in err_lines if any(k in l for k in
        ('Error', 'error', 'Failed', 'failed', 'Encoding', 'encoding',
         'Warning', 'FATAL', 'assert', 'Could not'))), None)
    if not reason and err_lines:
        reason = err_lines[-1]
    if reason:
        print(f"  ✗ {stem}: {reason.strip()}", flush=True)

    # Clean up on failure
    dst_img.unlink(missing_ok=True)
    dst_box.unlink(missing_ok=True)
    return None

def make_lstmf(img_path_str):
    """
    Wrapper around _make_lstmf_impl.

    A corrupt or transiently-unreadable image must not crash the whole run.
    Catch any exception (e.g. PIL UnidentifiedImageError on a truncated PNG,
    or OSError mid-read) and treat it as a per-image failure instead.
    """
    try:
        return _make_lstmf_impl(img_path_str)
    except Exception as _e:
        stem = Path(img_path_str).stem
        suffix = Path(img_path_str).suffix
        print(f"  ✗ {stem}: {_e}", flush=True)
        for _f in (Path(out_dir) / (stem + suffix),
                   Path(out_dir) / (stem + '.box'),
                   Path(out_dir) / (stem + '.lstmf')):
            try:
                _f.unlink(missing_ok=True)
            except OSError:
                pass
        return None

# ── Registry filter ──────────────────────────────────────────────────────────
# fonts.yml is the single source of truth for what trains. Images belonging to a
# font that is no longer registered are ignored here, whatever is left on disk.
#
# Deleting the files is not enough on its own: a purge can be declined, can fail
# partway, or can miss output written after it ran. Filtering at the point of use
# means an unregistered font can never re-enter training by accident.
#
# Naming conventions this has to match:
#   rendered/          <id>_<style>_lineNNNN.png
#   inventory/         <font-file-stem>/char_*.png        (keyed by FILE stem)
#   classical a5/      <title>/<id>_<style>/pageNNNN*.png
#   test-images/       <id>/<variant>/*.png
def _registered():
    """(font ids, font-file stems) currently declared in fonts.yml."""
    try:
        import yaml as _y
        doc = _y.safe_load(open(os.path.join(tdata, '..', 'fonts.yml'), encoding='utf-8'))
        entries = (doc or {}).get('fonts', []) or []
    except Exception:
        return None, None                      # unreadable — do not filter
    ids   = {e['id'] for e in entries if e.get('id')}
    stems = {os.path.splitext(f)[0].lower()
             for e in entries for f in (e.get('font_files') or [])}
    return ids, stems

_REG_IDS, _REG_STEMS = _registered()

def _is_registered(p):
    """False when this path belongs to a font absent from fonts.yml."""
    if not _REG_IDS:
        return True
    p = Path(p)
    parent = p.parent.name.lower()
    # inventory/<font-file-stem>/…
    if parent in _REG_STEMS:
        return True
    # <title>/<id>_<style>/…  or  <id>/<variant>/…
    for part in (parent, p.parent.parent.name.lower() if p.parent.parent else ''):
        for fid in _REG_IDS:
            if part == fid or part.startswith(fid + '_'):
                return True
    # rendered/<id>_<style>_lineNNNN.png  (flat)
    name = p.name.lower()
    if any(name.startswith(fid + '_') for fid in _REG_IDS):
        return True
    # A directory that matches no known font at all — only filter when the path
    # actually looks font-scoped, so unrelated sources (scan/) are not dropped.
    looks_font_scoped = (
        'rendered' in p.parts or 'inventory' in p.parts
        or 'a5-pages' in p.parts or 'test-images' in p.parts
    )
    return not looks_font_scoped

# Collect images that have a matching gt.txt
_all = [p for p in Path(src_dir).iterdir()
        if p.suffix.lower() in ('.png', '.tif', '.tiff', '.jpg')
           and p.with_suffix('.gt.txt').exists()]
_unreg = [p for p in _all if not _is_registered(p)]
if _unreg:
    print(f"  ⊘ {len(_unreg)} image(s) skipped — font not in fonts.yml "
          f"(e.g. {_unreg[0].name})", flush=True)
imgs = sorted(str(p) for p in _all if _is_registered(p))
skipped = sum(
    1 for p in Path(src_dir).iterdir()
    if p.suffix.lower() in ('.png', '.tif', '.tiff', '.jpg')
       and not p.with_suffix('.gt.txt').exists()
)

ok = fail = 0
lstmf_paths = []
total = len(imgs)

print(f"  Workers: {WORKERS}", flush=True)

# ThreadPoolExecutor: tesseract is an external subprocess, so threads are
# truly parallel despite the GIL. Collect paths — write list.txt at the end
# to avoid concurrent file-append races.
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    futures = {ex.submit(make_lstmf, img): img for img in imgs}
    done = 0
    for future in as_completed(futures):
        done += 1
        path = future.result()
        if path:
            lstmf_paths.append(path)
            ok += 1
        else:
            fail += 1
        if done % 200 == 0 or done == total:
            pct = done * 100 // total if total else 0
            print(f"  {done}/{total} ({pct}%)  ok={ok}  fail={fail}", flush=True)

# Write results to list.txt (single write, no race)
with open(list_txt, 'a', encoding='utf-8') as lf:
    for p in sorted(lstmf_paths):
        lf.write(p + '\n')

print(f"  {label}: {ok} OK  {fail} failed  {skipped} skipped (no gt.txt)")
PYEOF
}

# ── Process inventory first (character building blocks) ─────────────────────────
# Walk inventory/<font_tag>/ subdirs (flat dirs with char_*.png + char_*.gt.txt)
if [ -d "$INVENTORY_DIR" ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Character inventory: $INVENTORY_DIR"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    mkdir -p "$LSTMF_DIR/inventory"
    for font_dir in "$INVENTORY_DIR"/*; do
        [ -d "$font_dir" ] || continue
        font_tag=$(basename "$font_dir")
        lstmf_out="$LSTMF_DIR/inventory/$font_tag"
        process_dir "$font_dir" "$lstmf_out" "Inventory/$font_tag"
    done
fi

process_dir "$RENDERED_DIR"           "$LSTMF_DIR/rendered"   "Synthetic rendered"
process_dir "$SCAN_DIR"               "$LSTMF_DIR/scan"       "Real scan images"
process_dir "$RENDERED_DIR/font-test" "$LSTMF_DIR/font-test"  "Per-font test images"

# ── Classical corpus A5 pages ─────────────────────────────────────
if [ -n "$CLASSICAL_A5_DIR" ] && [ -d "$CLASSICAL_A5_DIR" ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Classical corpus A5 pages: $CLASSICAL_A5_DIR"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    mkdir -p "$LSTMF_DIR/classical"

    # Walk title/<font_tag>/ leaf dirs (depth 2 under CLASSICAL_A5_DIR)
    while IFS= read -r leaf_dir; do
        [ -d "$leaf_dir" ] || continue
        # Build a safe label from the relative path (slashes → __)
        rel="${leaf_dir#$CLASSICAL_A5_DIR/}"
        label="${rel//\//__}"
        lstmf_out="$LSTMF_DIR/classical/$label"
        process_dir "$leaf_dir" "$lstmf_out" "Classical/$rel"
    done < <(find "$CLASSICAL_A5_DIR" -mindepth 2 -maxdepth 2 -type d | sort)
else
    if [ -n "$CLASSICAL_A5_DIR" ]; then
        echo ""
        echo "  NOTE: CLASSICAL_A5_DIR set but not found: $CLASSICAL_A5_DIR"
        echo "        Run render-a5-pages.py first to generate A5 pages."
    fi
fi

TOTAL=$(wc -l < "$LSTMF_DIR/list.txt" | tr -d ' ')

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Done. lstmf/list.txt contains $TOTAL files."
echo ""
echo "  NEXT: caffeinate -i ./scripts/03-train.sh"
echo "        tail -f logs/training.log"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
