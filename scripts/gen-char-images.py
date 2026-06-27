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


def scan_font_dir(font_id):
    """
    Scan fonts/<id>/ for ALL TTF and OTF files.
    - If a stem has BOTH TTF and OTF: include both as '<Stem>-ttf' and '<Stem>-otf'
      (they may rasterise slightly differently → more training diversity)
    - If a stem has only one format: use '<Stem>' (no suffix)
    - Skips: variable fonts ([wght]), webfonts/, Source/ dirs, duplicate files
    Returns dict: variant_name → Path
    """
    font_root = ROOT / 'fonts' / font_id
    if not font_root.exists():
        return {}

    SKIP_DIRS = {'webfonts', 'Source', 'source'}

    # Group by stem → {ttf: Path, otf: Path}
    by_stem = {}

    for p in sorted(font_root.rglob('*')):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if '[' in p.name or ']' in p.name:
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
    scanned = scan_font_dir(fid)
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
    scanned    = scan_font_dir(fid)   # variant_name → path (already suffixed when both exist)
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


def render_char_image(ch, label, font, size, dpi, out_path, write_gt=True):
    """Render a single character (+ small label) to a PNG."""
    from PIL import Image, ImageDraw, ImageFont

    pad = max(size // 3, 16)
    tmp = Image.new('RGB', (1, 1))
    dtmp = ImageDraw.Draw(tmp)

    bb = dtmp.textbbox((0, 0), ch, font=font)
    cw, ch_h = bb[2] - bb[0], bb[3] - bb[1]

    # Label font
    lbl_size = max(size // 4, 9)
    try:
        lbl_font = ImageFont.truetype(str(font.path), size=lbl_size)
    except Exception:
        lbl_font = ImageFont.load_default()
    lbl_bb = dtmp.textbbox((0, 0), label, font=lbl_font)
    lw = lbl_bb[2] - lbl_bb[0]

    W = max(cw + pad * 2, lw + pad * 2, size + pad)
    H = ch_h + pad * 3 + (lbl_bb[3] - lbl_bb[1]) + 4

    img = Image.new('RGB', (W, H), BG)
    d   = ImageDraw.Draw(img)

    cx = (W - cw) // 2 - bb[0]
    cy = pad - bb[1]
    d.text((cx, cy), ch, font=font, fill=INK)

    lx = (W - lw) // 2
    ly = cy + ch_h + pad // 2
    d.text((lx, ly), label, font=lbl_font, fill=MUTED)

    img.save(str(out_path), dpi=(dpi, dpi))
    if write_gt:
        out_path.with_suffix('.gt.txt').write_text(ch, encoding='utf-8')
    return out_path


def render_line_image(items, font, dpi, out_path, text_override=None):
    """Render a row of characters as a single line image."""
    from PIL import Image, ImageDraw

    text = text_override or '  '.join(x[0] for x in items)
    pad  = 20
    tmp  = Image.new('RGB', (1, 1))
    dtmp = ImageDraw.Draw(tmp)
    bb   = dtmp.textbbox((0, 0), text, font=font)
    lw, lh = bb[2] - bb[0], bb[3] - bb[1]

    W = lw + pad * 4
    H = lh + pad * 2
    img = Image.new('RGB', (W, H), BG)
    ImageDraw.Draw(img).text((pad * 2 - bb[0], pad - bb[1]), text, font=font, fill=INK)
    img.save(str(out_path), dpi=(dpi, dpi))
    out_path.with_suffix('.gt.txt').write_text(text, encoding='utf-8')
    return out_path


# ── Per-font generation ────────────────────────────────────────────────────

def generate_for_variant(font_id, variant_name, font_path, out_base, size, dpi):
    """Generate all character images for one font variant."""
    from PIL import ImageFont

    out_dir = out_base / font_id / variant_name
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        font = ImageFont.truetype(str(font_path), size=size)
    except Exception as e:
        print(f"    ✗ Cannot load font {font_path}: {e}")
        return 0

    generated = []
    errors    = []

    # ── Individual characters ──────────────────────────────────
    for ch, cp, name in VOWELS:
        fname = out_dir / f"vowel_{name.lower()}.png"
        try:
            render_char_image(ch, f"U+{cp}", font, size, dpi, fname)
            generated.append({'type':'vowel','ch':ch,'cp':cp,'name':name,'file':fname.name})
        except Exception as e:
            errors.append(f"vowel {name}: {e}")

    for ch, cp, name in CONSONANTS:
        fname = out_dir / f"consonant_{name.lower()}.png"
        try:
            render_char_image(ch, f"U+{cp}", font, size, dpi, fname)
            generated.append({'type':'consonant','ch':ch,'cp':cp,'name':name,'file':fname.name})
        except Exception as e:
            errors.append(f"consonant {name}: {e}")

    for ch, stem in CONJUNCTS:
        fname = out_dir / f"conjunct_{stem}.png"
        try:
            render_char_image(ch, stem, font, size, dpi, fname)
            generated.append({'type':'conjunct','ch':ch,'name':stem,'file':fname.name})
        except Exception as e:
            errors.append(f"conjunct {stem}: {e}")

    for ch, cp, name in DIGITS:
        fname = out_dir / f"digit_{name}.png"
        try:
            render_char_image(ch, f"U+{cp}", font, size, dpi, fname)
            generated.append({'type':'digit','ch':ch,'cp':cp,'name':name,'file':fname.name})
        except Exception as e:
            errors.append(f"digit {name}: {e}")

    # ── Line images ────────────────────────────────────────────
    for group_name, items in [
        ('vowels',     [(ch,cp,name) for ch,cp,name in VOWELS]),
        ('consonants', [(ch,cp,name) for ch,cp,name in CONSONANTS]),
        ('digits',     [(ch,cp,name) for ch,cp,name in DIGITS]),
    ]:
        fname = out_dir / f"line_{group_name}.png"
        try:
            render_line_image([(ch,) for ch,*_ in items], font, dpi, fname)
            generated.append({'type':'line','name':group_name,'file':fname.name})
        except Exception as e:
            errors.append(f"line {group_name}: {e}")

    # Conjuncts line
    fname = out_dir / 'line_conjuncts.png'
    try:
        render_line_image(CONJUNCTS, font, dpi, fname)
        generated.append({'type':'line','name':'conjuncts','file':fname.name})
    except Exception as e:
        errors.append(f"line conjuncts: {e}")

    # ── Sample sentences ───────────────────────────────────────
    for i, sent in enumerate(SAMPLE_SENTENCES, 1):
        fname = out_dir / f"sentence_{i:02d}.png"
        try:
            render_line_image([], font, dpi, fname, text_override=sent)
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
        variants = get_font_variants(font_entry)

        if not variants:
            print(f"\n  [{fid}] {fname}: ⚠ no font files found, skipping")
            continue

        print(f"\n  [{fid}] {fname}")
        for variant_name, font_path in variants:
            print(f"    {variant_name}  ({font_path.name})")
            n = generate_for_variant(fid, variant_name, font_path,
                                     out_base, args.size, args.dpi)
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
