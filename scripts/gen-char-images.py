#!/usr/bin/env python3
"""
gen-char-images.py
──────────────────
Renders Kannada Unicode characters to PNG test images using each registered
font from fonts.yml.  Output is grouped by font+variant:

    test-images/
      <font_id>/
        <variant_name>/
          char_A.png        — single character, white background
          char_A.gt.txt     — ground-truth text (the Unicode character)
          line_vowels.png   — full vowel row as one line image
          line_vowels.gt.txt
          …
          conjunct_ksha.png
          sentence_01.png
          manifest.json

Rendering engine:
  Uses uharfbuzz + freetype-py for full OpenType shaping so that Kannada
  conjunct characters (ಕ್ಷ, ಜ್ಞ, ಶ್ರ …) render correctly.  Without shaping,
  Pillow renders each Unicode codepoint as a separate isolated glyph — the
  visual result is broken/split conjuncts.

  If uharfbuzz / freetype-py are not installed the script falls back to
  Pillow with a warning.  Fix with:
      pip install uharfbuzz freetype-py numpy --break-system-packages

Usage:
    python3 scripts/gen-char-images.py                   # all fonts
    python3 scripts/gen-char-images.py --font-id kan_gmp # one font
    python3 scripts/gen-char-images.py --dpi 150 --size 48

Requires: Pillow, pyyaml
    pip install Pillow pyyaml --break-system-packages
"""

import sys, os, json, argparse, re
from pathlib import Path

ROOT = Path(__file__).parent.parent

# ── Shaping renderer ───────────────────────────────────────────────────────
sys.path.insert(0, str(ROOT / 'corpus'))
try:
    from shaping_render import (
        SHAPING_AVAILABLE, render_text as _render_text,
        render_char_with_label as _render_char_label,
        check_and_warn as _check_and_warn,
    )
except ImportError:
    SHAPING_AVAILABLE = False
    def _check_and_warn():
        print("  ⚠  shaping_render.py not found — using Pillow (broken conjuncts)")

# ── Kannada character sets ─────────────────────────────────────────────────

VOWELS = [
    ('ಅ', '0C85', 'A'),   ('ಆ', '0C86', 'AA'),  ('ಇ', '0C87', 'I'),
    ('ಈ', '0C88', 'II'),  ('ಉ', '0C89', 'U'),   ('ಊ', '0C8A', 'UU'),
    ('ಋ', '0C8B', 'RR'),  ('ಎ', '0C8E', 'E'),   ('ಏ', '0C8F', 'EE'),
    ('ಐ', '0C90', 'AI'),  ('ಒ', '0C92', 'O'),   ('ಓ', '0C93', 'OO'),
    ('ಔ', '0C94', 'AU'),  ('ಂ', '0C82', 'Anusvara'), ('ಃ', '0C83', 'Visarga'),
]

CONSONANTS = [
    ('ಕ', '0C95', 'KA'),  ('ಖ', '0C96', 'KHA'), ('ಗ', '0C97', 'GA'),
    ('ಘ', '0C98', 'GHA'), ('ಙ', '0C99', 'NGA'), ('ಚ', '0C9A', 'CA'),
    ('ಛ', '0C9B', 'CHA'), ('ಜ', '0C9C', 'JA'),  ('ಝ', '0C9D', 'JHA'),
    ('ಞ', '0C9E', 'NYA'), ('ಟ', '0C9F', 'TTA'), ('ಠ', '0CA0', 'TTHA'),
    ('ಡ', '0CA1', 'DDA'), ('ಢ', '0CA2', 'DDHA'),('ಣ', '0CA3', 'NNA'),
    ('ತ', '0CA4', 'TA'),  ('ಥ', '0CA5', 'THA'), ('ದ', '0CA6', 'DA'),
    ('ಧ', '0CA7', 'DHA'), ('ನ', '0CA8', 'NA'),  ('ಪ', '0CAA', 'PA'),
    ('ಫ', '0CAB', 'PHA'), ('ಬ', '0CAC', 'BA'),  ('ಭ', '0CAD', 'BHA'),
    ('ಮ', '0CAE', 'MA'),  ('ಯ', '0CAF', 'YA'),  ('ರ', '0CB0', 'RA'),
    ('ಱ', '0CB1', 'RRA'), ('ಲ', '0CB2', 'LA'),  ('ಳ', '0CB3', 'LLA'),
    ('ವ', '0CB5', 'VA'),  ('ಶ', '0CB6', 'SHA'), ('ಷ', '0CB7', 'SSA'),
    ('ಸ', '0CB8', 'SA'),  ('ಹ', '0CB9', 'HA'),
]

# Conjuncts: (rendered cluster, safe filename stem)
CONJUNCTS = [
    ('ಕ್ಷ',  'ksha'),  ('ಜ್ಞ',  'jnya'),  ('ತ್ತ',  'tt'),
    ('ದ್ದ',  'dd'),   ('ನ್ನ',  'nn'),   ('ಮ್ಮ',  'mm'),
    ('ಲ್ಲ',  'll'),   ('ಸ್ತ',  'st'),   ('ಪ್ರ',  'pr'),
    ('ಗ್ರ',  'gr'),   ('ತ್ರ',  'tr'),   ('ಶ್ರ',  'shr'),
    ('ಸ್ವ',  'sv'),   ('ನ್ತ',  'nt'),   ('ರ್ಕ',  'rk'),
    ('ಕ್ಕ',  'kk'),   ('ಬ್ಬ',  'bb'),   ('ಷ್ಟ',  'sht'),
    ('ಕ್ತ',  'kt'),   ('ನ್ಮ',  'nm'),   ('ಧ್ವ',  'dhv'),
    ('ಸ್ಪ',  'sp'),   ('ನ್ದ',  'nd'),   ('ರ್ಥ',  'rth'),
]

DIGITS = [
    ('೦', '0CE6', '0'), ('೧', '0CE7', '1'), ('೨', '0CE8', '2'),
    ('೩', '0CE9', '3'), ('೪', '0CEA', '4'), ('೫', '0CEB', '5'),
    ('೬', '0CEC', '6'), ('೭', '0CED', '7'), ('೮', '0CEE', '8'),
    ('೯', '0CEF', '9'), ('।',  '0964', 'danda'), ('॥', '0965', 'ddanda'),
]

SAMPLE_SENTENCES = [
    "ಕರ್ನಾಟಕ ರಾಜ್ಯದ ಪ್ರಾಚೀನ ಶಾಸನಗಳು",
    "ಶ್ರೀ ಕೃಷ್ಣನ ಭಕ್ತಿಯಿಂದ ಮೋಕ್ಷ ಸಿದ್ಧಿಸುವುದು",
    "ವಿದ್ಯಾರ್ಥಿಗಳಿಗೆ ಜ್ಞಾನದ ಮಹತ್ತ್ವ ತಿಳಿಸಬೇಕು",
    "ಕನ್ನಡ ಸಾಹಿತ್ಯದ ಪರಂಪರೆ ಅತ್ಯಂತ ಶ್ರೀಮಂತ",
    "ಪ್ರಕೃತಿಯ ಸೌಂದರ್ಯವನ್ನು ಕಾಪಾಡಬೇಕು",
    "ರಾಷ್ಟ್ರ ಸೇವೆ ದೈವ ಸೇವೆ ಎಂದು ತಿಳಿಯಬೇಕು",
    "ಅಷ್ಟಾದಶ ಪುರಾಣಗಳ ಸಂಕ್ಷಿಪ್ತ ವಿವರ",
    "ಶ್ರದ್ಧೆ ಭಕ್ತಿ ಜ್ಞಾನ ವೈರಾಗ್ಯ ಮೋಕ್ಷ",
]

# ── Font discovery ─────────────────────────────────────────────────────────

def load_fonts_yml():
    """Load fonts.yml and return list of font dicts."""
    import yaml
    fpath = ROOT / 'fonts.yml'
    if not fpath.exists():
        return []
    with open(fpath, encoding='utf-8') as f:
        doc = yaml.safe_load(f)
    return doc.get('fonts', [])


def scan_roots(font_id, font_dir=None):
    """
    Directories to scan for a font, in priority order.

    When fonts.yml declares `font_dir`, the scan is SCOPED to that directory
    (plus its ttf/ and otf/ siblings) rather than walking the whole repo tree.
    This matters for families that ship many width variants on disk — e.g.
    Anek Kannada has 5 width families × 8 weights = 40 files, but fonts.yml
    declares only static/AnekKannada, so only those 8 weights are used.

    Falls back to a full recursive walk when no font_dir is declared.
    """
    base = ROOT / 'fonts' / font_id
    if not base.exists():
        return []

    if not font_dir:
        return [base]

    d       = base / font_dir
    parent  = d.parent
    roots   = [d, d / 'ttf', d / 'otf', parent / 'ttf', parent / 'otf', base]
    seen, out = set(), []
    for r in roots:
        if r.is_dir() and r not in seen:
            seen.add(r)
            out.append(r)
    return out or [base]


def scan_font_dir(font_id, font_dir=None):
    """
    Scan a font's directories for ALL TTF and OTF files.
    - If a stem has BOTH TTF and OTF: include both as '<Stem>-ttf' and '<Stem>-otf'
      (they may rasterise slightly differently → more training diversity)
    - If a stem has only one format: use '<Stem>' (no suffix)
    - Skips: variable fonts ([wght] / VariableFont), webfonts/, Source/ dirs, duplicates
    - Scope is limited by `font_dir` when declared in fonts.yml (see scan_roots)
    Returns dict: variant_name → Path
    """
    roots = scan_roots(font_id, font_dir)
    if not roots:
        return {}

    SKIP_DIRS = {'webfonts', 'Source', 'source'}

    # Group by stem → {ttf: Path, otf: Path}
    by_stem = {}

    # Scoped roots are scanned shallowly; an unscoped base is walked recursively.
    if font_dir:
        candidates = [p for r in roots for p in sorted(r.glob('*'))]
    else:
        candidates = sorted(roots[0].rglob('*'))

    for p in candidates:
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if '[' in p.name or ']' in p.name or 'VariableFont' in p.name:
            continue

        suffix = p.suffix.lower()
        if suffix not in ('.ttf', '.otf'):
            continue

        stem = p.stem
        fmt  = suffix[1:]   # 'ttf' or 'otf'

        if stem not in by_stem:
            by_stem[stem] = {}
        # Don't overwrite with a duplicate in a different subdir
        if fmt not in by_stem[stem]:
            by_stem[stem][fmt] = p

    # Build variant name → path mapping
    result = {}
    for stem, fmts in sorted(by_stem.items()):
        if 'ttf' in fmts and 'otf' in fmts:
            result[f'{stem}-ttf'] = fmts['ttf']
            result[f'{stem}-otf'] = fmts['otf']
        elif 'ttf' in fmts:
            result[stem] = fmts['ttf']
        else:
            result[stem] = fmts['otf']

    return result


def resolve_font_path(font_entry, filename):
    """
    Resolve actual path for a specific filename listed in fonts.yml.
    Tries TTF subdirs first, falls back to OTF counterpart.
    """
    fid     = font_entry['id']
    fontdir = font_entry.get('font_dir', 'fonts')
    base    = ROOT / 'fonts' / fid

    stem    = re.sub(r'\.[^.]+$', '', filename)
    suffix  = Path(filename).suffix.lower()

    # Priority order: exact path, ttf/ subdir, otf/ subdir, root
    search_roots = [
        base / fontdir,
        base / fontdir / 'ttf',
        base / fontdir / 'otf',
        base,
    ]
    for root in search_roots:
        p = root / filename
        if p.exists():
            return p

    # Try OTF counterpart if TTF requested but missing
    if suffix == '.ttf':
        otf = filename[:-4] + '.otf'
        for root in search_roots:
            p = root / otf
            if p.exists():
                return p

    # Fall back to directory scan
    scanned = scan_font_dir(fid, fontdir)
    return scanned.get(stem)


def get_font_variants(font_entry):
    """
    Return list of (variant_name, path) for a font entry, covering ALL TTF+OTF.

    Strategy:
      • For each stem in font_files:
          – if both TTF and OTF exist → add '<Stem>-ttf' and '<Stem>-otf'
          – if only one format exists → add '<Stem>' (no suffix)
      • Then add any extra stems found on disk that aren't already covered.
    """
    fid        = font_entry['id']
    # Scope the scan to the declared font_dir so extra width families on disk
    # (e.g. Anek's Condensed/Expanded sets) don't explode the variant count.
    scanned    = scan_font_dir(fid, font_entry.get('font_dir'))
    seen_names = set()
    variants   = []

    # 1. Walk font_files in declared order to set the primary ordering
    for filename in font_entry.get('font_files', []):
        stem = re.sub(r'\.[^.]+$', '', filename)
        added = False
        for suffix in (f'{stem}-ttf', f'{stem}-otf', stem):
            if suffix in scanned and suffix not in seen_names:
                variants.append((suffix, scanned[suffix]))
                seen_names.add(suffix)
                added = True
        if not added:
            # Last resort: resolve directly
            p = resolve_font_path(font_entry, filename)
            if p and stem not in seen_names:
                variants.append((stem, p))
                seen_names.add(stem)

    # 2. Append any remaining scanned variants (extra weights, OTF-only, etc.)
    for name in sorted(scanned):
        if name not in seen_names:
            variants.append((name, scanned[name]))
            seen_names.add(name)

    return variants


# ── Image rendering ────────────────────────────────────────────────────────

BG    = (255, 255, 255)
INK   = (20,  20,  60)
MUTED = (120, 130, 150)


def render_char_image(ch, label, font_path_str, size, dpi, out_path, write_gt=True, aalt=False,
                      show_label=False):
    """
    Render one character to PNG.

    `show_label` burns the codepoint (e.g. "U+0C95") into the image below the
    glyph. It defaults to FALSE, and that matters: these images are fed to
    Tesseract by the 1:1 OCR test, whose ground truth is the character alone.
    With a label present, Tesseract reads both the glyph AND the label, so the
    comparison fails for every character no matter how good the model is —
    a model at 0.003% BCER still scored zero here.

    The label is also drawn in the Kannada font, where Latin glyphs are often
    missing, so "U+0C95" rendered as garbage — adding noise on top of the
    mismatch.

    The codepoint is still recorded per character in manifest.json, which is
    where diagnostic metadata belongs.
    """
    if SHAPING_AVAILABLE:
        img = _render_char_label(
            char=ch,
            label=(label if show_label else ""),
            font_path=font_path_str,
            font_size=size,
            label_size=max(size // 4, 9),
            padding=max(size // 3, 16),
            dpi=dpi,
            bg=BG,
            ink=INK,
            label_ink=MUTED,
            aalt=aalt,
        )
    else:
        # Pillow fallback (broken conjuncts)
        from PIL import Image as _PI, ImageDraw, ImageFont
        try:
            font = ImageFont.truetype(font_path_str, size=size)
        except Exception:
            font = ImageFont.load_default()
        lbl_size = max(size // 4, 9)
        try:
            lbl_font = ImageFont.truetype(font_path_str, size=lbl_size)
        except Exception:
            lbl_font = ImageFont.load_default()

        pad  = max(size // 3, 16)
        tmp  = _PI.new('RGB', (1, 1))
        dtmp = ImageDraw.Draw(tmp)
        bb   = dtmp.textbbox((0, 0), ch, font=font)
        _lbl = label if show_label else ""
        lbb  = dtmp.textbbox((0, 0), _lbl, font=lbl_font)
        cw, ch_h = bb[2]-bb[0], bb[3]-bb[1]
        lw = lbb[2]-lbb[0]

        W = max(cw + pad*2, lw + pad*2, size + pad)
        H = ch_h + pad*3 + (lbb[3]-lbb[1]) + 4
        img = _PI.new('RGB', (W, H), BG)
        d   = ImageDraw.Draw(img)
        d.text(((W-cw)//2 - bb[0], pad - bb[1]), ch, font=font, fill=INK)
        if show_label:
            d.text(((W-lw)//2, pad + ch_h + pad//2 - lbb[1]), _lbl, font=lbl_font, fill=MUTED)

    img.save(str(out_path), dpi=(dpi, dpi))
    if write_gt:
        out_path.with_suffix('.gt.txt').write_text(ch, encoding='utf-8')
    return out_path


def render_line_image(items, font_path_str, dpi, out_path, text_override=None, size=48, aalt=False):
    """
    Render a row of characters as a single shaped line image.

    Uses shaping_render for correct Kannada conjunct formation.
    """
    text = text_override or '  '.join(x[0] for x in items)
    pad  = 20

    if SHAPING_AVAILABLE:
        grey = _render_text(
            font_path_str, text,
            font_size=size,
            padding_x=pad * 2,
            padding_y=pad,
            min_height=size + pad * 2,
            bg_color=255,
            ink_color=0,
            aalt=aalt,
        )
        # Convert greyscale to RGB
        from PIL import Image as _PI
        img = _PI.merge('RGB', [grey.convert('L')] * 3)
        # Re-tint to INK colour
        import numpy as np
        arr = np.array(grey, dtype=np.float32) / 255.0
        r = (arr*BG[0] + (1-arr)*INK[0]).astype(np.uint8)
        g = (arr*BG[1] + (1-arr)*INK[1]).astype(np.uint8)
        b = (arr*BG[2] + (1-arr)*INK[2]).astype(np.uint8)
        img = _PI.fromarray(np.stack([r,g,b], axis=2), mode='RGB')
    else:
        from PIL import Image as _PI, ImageDraw, ImageFont
        try:
            font = ImageFont.truetype(font_path_str, size=size)
        except Exception:
            font = ImageFont.load_default()
        tmp  = _PI.new('RGB', (1, 1))
        dtmp = ImageDraw.Draw(tmp)
        bb   = dtmp.textbbox((0, 0), text, font=font)
        lw, lh = bb[2]-bb[0], bb[3]-bb[1]
        W = lw + pad * 4
        H = lh + pad * 2
        img = _PI.new('RGB', (W, H), BG)
        ImageDraw.Draw(img).text((pad*2 - bb[0], pad - bb[1]), text, font=font, fill=INK)

    img.save(str(out_path), dpi=(dpi, dpi))
    out_path.with_suffix('.gt.txt').write_text(text, encoding='utf-8')
    return out_path


# ── Per-font generation ────────────────────────────────────────────────────

def generate_for_variant(font_id, variant_name, font_path, out_base, size, dpi, aalt=False,
                         show_labels=False):
    """Generate all character images for one font variant."""
    from PIL import ImageFont

    out_dir = out_base / font_id / variant_name
    out_dir.mkdir(parents=True, exist_ok=True)

    font_path_str = str(font_path)

    # Verify the font is loadable before generating anything
    try:
        ImageFont.truetype(font_path_str, size=size)
    except Exception as e:
        print(f"    ✗ Cannot load font {font_path}: {e}")
        return 0

    generated = []
    errors    = []

    # ── Individual characters ──────────────────────────────────
    # render_char_image now takes (ch, label, font_path_str, size, dpi, out_path)
    for ch, cp, name in VOWELS:
        fname = out_dir / f"vowel_{name.lower()}.png"
        try:
            render_char_image(ch, f"U+{cp}", font_path_str, size, dpi, fname, aalt=aalt, show_label=show_labels)
            generated.append({'type':'vowel','ch':ch,'cp':cp,'name':name,'file':fname.name})
        except Exception as e:
            errors.append(f"vowel {name}: {e}")

    for ch, cp, name in CONSONANTS:
        fname = out_dir / f"consonant_{name.lower()}.png"
        try:
            render_char_image(ch, f"U+{cp}", font_path_str, size, dpi, fname, aalt=aalt, show_label=show_labels)
            generated.append({'type':'consonant','ch':ch,'cp':cp,'name':name,'file':fname.name})
        except Exception as e:
            errors.append(f"consonant {name}: {e}")

    for ch, stem in CONJUNCTS:
        fname = out_dir / f"conjunct_{stem}.png"
        try:
            render_char_image(ch, stem, font_path_str, size, dpi, fname, aalt=aalt, show_label=show_labels)
            generated.append({'type':'conjunct','ch':ch,'name':stem,'file':fname.name})
        except Exception as e:
            errors.append(f"conjunct {stem}: {e}")

    for ch, cp, name in DIGITS:
        fname = out_dir / f"digit_{name}.png"
        try:
            render_char_image(ch, f"U+{cp}", font_path_str, size, dpi, fname, aalt=aalt, show_label=show_labels)
            generated.append({'type':'digit','ch':ch,'cp':cp,'name':name,'file':fname.name})
        except Exception as e:
            errors.append(f"digit {name}: {e}")

    # ── Line images ────────────────────────────────────────────
    # render_line_image now takes (items, font_path_str, dpi, out_path, text_override, size)
    for group_name, items in [
        ('vowels',     [(ch,cp,name) for ch,cp,name in VOWELS]),
        ('consonants', [(ch,cp,name) for ch,cp,name in CONSONANTS]),
        ('digits',     [(ch,cp,name) for ch,cp,name in DIGITS]),
    ]:
        fname = out_dir / f"line_{group_name}.png"
        try:
            render_line_image([(ch,) for ch,*_ in items], font_path_str, dpi, fname, size=size, aalt=aalt)
            generated.append({'type':'line','name':group_name,'file':fname.name})
        except Exception as e:
            errors.append(f"line {group_name}: {e}")

    # Conjuncts line
    fname = out_dir / 'line_conjuncts.png'
    try:
        render_line_image(CONJUNCTS, font_path_str, dpi, fname, size=size, aalt=aalt)
        generated.append({'type':'line','name':'conjuncts','file':fname.name})
    except Exception as e:
        errors.append(f"line conjuncts: {e}")

    # ── Sample sentences ───────────────────────────────────────
    for i, sent in enumerate(SAMPLE_SENTENCES, 1):
        fname = out_dir / f"sentence_{i:02d}.png"
        try:
            render_line_image([], font_path_str, dpi, fname, text_override=sent, size=size, aalt=aalt)
            generated.append({'type':'sentence','name':f'sentence_{i:02d}','file':fname.name})
        except Exception as e:
            errors.append(f"sentence {i}: {e}")

    if errors:
        print(f"    ⚠ {len(errors)} errors:")
        for err in errors[:5]:
            print(f"      {err}")

    # ── Manifest ───────────────────────────────────────────────
    manifest = {
        'font_id':      font_id,
        'variant':      variant_name,
        'font_path':    str(font_path),
        'size':         size,
        'dpi':          dpi,
        'count':        len(generated),
        'errors':       len(errors),
        'characters':   generated,
    }
    (out_dir / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

    return len(generated)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir',   default=str(ROOT / 'test-images'))
    parser.add_argument('--dpi',      type=int, default=150)
    parser.add_argument('--size',     type=int, default=48)
    parser.add_argument('--show-labels', action='store_true',
                        help='Burn the codepoint label (U+0C95) into each character image. '
                             'OFF by default: the 1:1 OCR test compares against the character '
                             'alone, so a label in the pixels makes every comparison fail no '
                             'matter how good the model is. The codepoint is in manifest.json.')
    parser.add_argument('--font-id',  default=None,
                        help='Generate for one font only (e.g. kan_gmp)')
    args = parser.parse_args()

    try:
        from PIL import Image
    except ImportError:
        print("ERROR: Pillow not installed.")
        print("  pip install Pillow --break-system-packages")
        sys.exit(1)

    try:
        import yaml
    except ImportError:
        print("ERROR: PyYAML not installed.")
        print("  pip install pyyaml --break-system-packages")
        sys.exit(1)

    # Warn if HarfBuzz shaping libraries are not installed
    _check_and_warn()
    print(f"  Renderer: {'HarfBuzz + FreeType (shaped)' if SHAPING_AVAILABLE else 'Pillow (BROKEN conjuncts)'}")

    out_base = Path(args.outdir)
    fonts    = load_fonts_yml()

    if not fonts:
        print("ERROR: No fonts found in fonts.yml")
        sys.exit(1)

    if args.font_id:
        fonts = [f for f in fonts if f['id'] == args.font_id]
        if not fonts:
            print(f"ERROR: font-id '{args.font_id}' not found in fonts.yml")
            sys.exit(1)

    total_images = 0
    summary = []

    print(f"\n{'━'*60}")
    print(f"  Generating Kannada Unicode test images")
    print(f"  Size {args.size}pt @ {args.dpi}dpi  →  {out_base}/")
    print(f"{'━'*60}")

    for font_entry in fonts:
        fid      = font_entry['id']
        fname    = font_entry['name']
        aalt     = 'aalt' in font_entry.get('font_features', '')
        variants = get_font_variants(font_entry)

        if not variants:
            print(f"\n  [{fid}] {fname}: ⚠ no font files found, skipping")
            continue

        print(f"\n  [{fid}] {fname}")
        for variant_name, font_path in variants:
            print(f"    {variant_name}  ({font_path.name})")
            n = generate_for_variant(fid, variant_name, font_path,
                                     out_base, args.size, args.dpi, aalt,
                                     show_labels=args.show_labels)
            print(f"    → {n} images")
            total_images += n
            summary.append({'font_id': fid, 'font_name': fname,
                             'variant': variant_name, 'count': n,
                             'dir': str(out_base / fid / variant_name)})

    # Top-level manifest
    top_manifest = {
        'total':   total_images,
        'dpi':     args.dpi,
        'size':    args.size,
        'fonts':   summary,
    }
    (out_base / 'manifest.json').write_text(
        json.dumps(top_manifest, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f"\n{'━'*60}")
    print(f"  ✓ {total_images} images across {len(summary)} variants")
    print(f"  Manifest: {out_base}/manifest.json")
    print(f"{'━'*60}\n")

    # Output JSON for server to parse
    print(json.dumps({'ok': True, 'total': total_images, 'fonts': summary}))
    return total_images


if __name__ == '__main__':
    main()
