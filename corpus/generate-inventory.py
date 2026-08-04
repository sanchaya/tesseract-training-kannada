#!/usr/bin/env python3
"""
generate-inventory.py — Systematic Kannada Character Inventory Generator

Creates synthetic training images for ALL Kannada character combinations:
  - Single characters (consonants, vowels, numerals)
  - Consonant + vowel sign (ಕ, ಕಾ, ಕಿ, ಕೀ, ಕು, ಕೂ, ಕೃ, ಕೄ, ಕೆ, ಕೇ, ಕೈ, ಕೊ, ಕೋ, ಕೌ)
  - Conjuncts (consonant + virama + consonant: ಕ್ಕ, ಕ್ಷ, ಗ್ಧ, etc.)
  - Anusvara/visarga combinations (ಕಂ, ಕಃ, ಕಃ, etc.)

Output: inventory/<font_tag>/char_XXXX.png + char_XXXX.gt.txt

Usage:
    python3 corpus/generate-inventory.py
    # or with options:
    python3 corpus/generate-inventory.py --fonts "kan_gtn_medium,kan_gtn_bold" --dpi 300
"""

import argparse
import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import itertools

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
CORPUS_DIR = ROOT / 'corpus'
FONTS_DIR = ROOT / 'fonts'
OUTPUT_DIR = ROOT / 'inventory'

# ── Shaping renderer ───────────────────────────────────────────────────────────
# This generator MUST shape its text. The inventory is the character-baseline
# training set — it is almost entirely conjuncts, which is exactly what bare
# Pillow renders wrong (each codepoint as an isolated glyph). Rendering ಕ್ಷ as
# ಕ + ್ + ಷ here teaches the model the wrong glyph shapes.
# See corpus/shaping_render.py for the HarfBuzz font-unit conversion.
sys.path.insert(0, str(CORPUS_DIR))
try:
    from shaping_render import (
        SHAPING_AVAILABLE, render_text as _render_text, check_and_warn as _check_and_warn,
    )
except ImportError:
    SHAPING_AVAILABLE = False
    def _check_and_warn():
        print("  ⚠  shaping_render.py not found — using Pillow (BROKEN conjuncts)")

# ── Kannada Unicode ranges ─────────────────────────────────────────────────────
KANNADA_VOWELS = [chr(i) for i in range(0x0C85, 0x0C95)]  # ಅ-ಔ
KANNADA_CONSONANTS = [chr(i) for i in range(0x0C95, 0x0CB0)]  # ಕ-ಹ (excluding ೞ)
KANNADA_VOWEL_SIGNS = [chr(i) for i in range(0x0CBE, 0x0CC5)]  # ಾ-ೄ
KANNADA_VIRAMA = chr(0x0CCD)  # ್
KANNADA_ANUSVARA = chr(0x0C82)  # ಂ
KANNADA_VISARGA = chr(0x0C83)  # ಃ
KANNADA_NUKTA = chr(0x0CBC)  # ಼
KANNADA_NUMERALS = [chr(i) for i in range(0x0CE6, 0x0CF0)]  # ೦-೯

def _aalt_map():
    """
    font-file stem (lowercased) → whether the 'aalt' GSUB feature is needed.

    GTN and WMP keep their correct conjunct (ottu) forms in the aalt feature,
    which shapers do not enable by default; GMP's correct forms live in the base
    features and aalt BREAKS them. Sourced from fonts.yml so the two stay in
    sync. See docs/CONJUNCT_RENDERING.md.
    """
    try:
        import yaml
    except ImportError:
        return {}
    fy = ROOT / 'fonts.yml'
    if not fy.exists():
        return {}
    try:
        entries = (yaml.safe_load(fy.read_text(encoding='utf-8')) or {}).get('fonts', [])
    except Exception:
        return {}

    out = {}
    for e in entries:
        needs = 'aalt' in (e.get('font_features') or '')
        for fn in e.get('font_files', []):
            out[Path(fn).stem.lower()] = needs
    return out


def _declared_stems():
    """Lowercased stems of every font file declared in fonts.yml."""
    return set(_aalt_map().keys())


# ── Unicharset validation ──────────────────────────────────────────────────────

_UNITS_CACHE = None

def _unicharset_units():
    """
    The set of unichars the training model can actually encode.

    Returns None when the unicharset cannot be read, in which case validation is
    skipped (with a warning) rather than silently dropping everything.
    """
    global _UNITS_CACHE
    if _UNITS_CACHE is not None:
        return _UNITS_CACHE or None

    mode_file = ROOT / 'output' / '.tessdata_mode'
    expanded  = mode_file.exists() and 'expanded' in mode_file.read_text(errors='ignore')
    td = ROOT / ('tessdata_expanded' if expanded else 'tessdata_best') / 'kan.traineddata'
    if not td.exists():
        _UNITS_CACHE = set()
        return None

    import subprocess, tempfile
    try:
        with tempfile.TemporaryDirectory() as tmp:
            prefix = str(Path(tmp) / 'kan.')
            subprocess.run(['combine_tessdata', '-u', str(td), prefix],
                           capture_output=True, check=True)
            uc = Path(prefix + 'lstm-unicharset')
            if not uc.exists():
                _UNITS_CACHE = set()
                return None
            lines = uc.read_text(encoding='utf-8', errors='replace').split('\n')
            # Format: first line is the count, then "<unichar> <props...>" per line.
            _UNITS_CACHE = {l.split(' ')[0] for l in lines[1:] if l.strip()}
    except Exception:
        _UNITS_CACHE = set()
        return None
    return _UNITS_CACHE or None


def _encodable(text, units):
    """
    True when *text* can be segmented entirely into unicharset units.

    Mirrors Tesseract's greedy longest-match encoder. Catches three classes of
    unusable input that this generator used to emit blindly by iterating raw
    Unicode ranges:

      • reserved codepoints that are not characters at all
        (U+0C8D, U+0C91, U+0CA9 fall inside the vowel/consonant ranges)
      • real characters absent from the unicharset (ಌ U+0C8C, ೄ U+0CC4, ೞ U+0CDE)
      • characters that exist only INSIDE clusters and cannot stand alone —
        ಞ (U+0C9E) is encodable within ಜ್ಞ but not on its own

    Each of these produced an image whose ground truth lstmtraining then rejected
    with "Can't encode transcription / Encoding of string failed".
    """
    if not units:
        return True          # validation unavailable — don't drop anything
    # Only whitespace is exempt — the virama always belongs to a cluster unit.
    # Unit matching runs BEFORE the exemption so ್‌ (virama + ZWNJ) is tried as a
    # unit rather than skipped.
    exempt = set(' \t\n')
    longest = max((len(u) for u in units), default=1)
    i, n = 0, len(text)
    while i < n:
        matched = False
        for size in range(min(longest, n - i), 0, -1):
            if text[i:i + size] in units:
                i += size
                matched = True
                break
        if matched:
            continue
        if text[i] in exempt:            # fallback only when no unit matched
            i += 1
            continue
        return False
    return True


def find_fonts(pattern=None, all_fonts=False):
    """
    Locate Kannada fonts under fonts/.

    Searches recursively so fonts installed as family folders (fonts/<id>/…)
    are found, not just loose files at the top level — the previous top-level
    glob only ever saw 2 stray files. Skips webfont/source dirs and variable
    fonts, mirroring scripts/gen-char-images.py.

    By default the result is restricted to the weights declared in fonts.yml,
    so the inventory trains on the same set as the rest of the pipeline.
    Pass all_fonts=True (CLI: --all-fonts) to use every .ttf found on disk.
    """
    if not FONTS_DIR.exists():
        print(f"ERROR: fonts/ directory not found at {FONTS_DIR}")
        sys.exit(1)

    SKIP_PARTS = {'webfonts', 'Source', 'source', '.git', 'Tests'}
    declared   = set() if all_fonts else _declared_stems()

    # .otf as well as .ttf — kan_kittel ships OTF only, and a ttf-only glob
    # silently left that family out of the inventory entirely.
    candidates = sorted(list(FONTS_DIR.rglob('*.ttf')) + list(FONTS_DIR.rglob('*.otf')))

    fonts = {}
    for fnt in candidates:
        if any(part in SKIP_PARTS for part in fnt.parts):
            continue
        if '[' in fnt.name or 'VariableFont' in fnt.name:
            continue
        tag = fnt.stem.lower()
        if declared and tag not in declared:
            continue
        if pattern and pattern not in tag:
            continue
        fonts.setdefault(tag, str(fnt))   # first match wins (ttf/ before dupes)

    if not fonts:
        print(f"ERROR: No .ttf/.otf fonts found in {FONTS_DIR}")
        if pattern:
            print(f"  (filtered by pattern: {pattern})")
        sys.exit(1)

    return fonts

def render_char(text, font_path, dpi=150, height=80, aalt=False):
    """
    Render a single character/cluster to PNG with full OpenType shaping.

    Uses shaping_render (HarfBuzz + FreeType) so conjuncts form correctly.
    Falls back to Pillow only when the shaping libraries are unavailable —
    in that case conjuncts WILL be broken and the output is unfit for training.
    """
    # DPI to pixels: assume 72 DPI as baseline
    font_size = int((height / 72) * dpi)

    if SHAPING_AVAILABLE:
        try:
            grey = _render_text(
                font_path, text,
                font_size=font_size,
                padding_x=10, padding_y=10,
                min_height=font_size + 20,
                bg_color=255, ink_color=0,
                aalt=aalt,
            )
            return grey.convert('RGB')
        except Exception as e:
            print(f"  Shaped render failed for {font_path}: {e}")
            return None

    # ── Pillow fallback (no shaping — conjuncts render broken) ──
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        print(f"  Error loading font {font_path}: {e}")
        return None

    img_tmp = Image.new('RGB', (1000, 100), color=(255, 255, 255))
    draw = ImageDraw.Draw(img_tmp)
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0] + 20
    height_actual = bbox[3] - bbox[1] + 20

    img = Image.new('RGB', (width, height_actual), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), text, font=font, fill=(0, 0, 0))

    return img

def generate_inventory(fonts, dpi=150, limit=None):
    """Generate all character combinations."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    combinations = []

    # Single vowels
    combinations.extend([(v, f'vowel_{ord(v):04x}') for v in KANNADA_VOWELS])

    # Single consonants
    combinations.extend([(c, f'consonant_{ord(c):04x}') for c in KANNADA_CONSONANTS])

    # Consonant + vowel sign
    for c in KANNADA_CONSONANTS[:5]:  # First 5 consonants to keep size manageable
        for vs in KANNADA_VOWEL_SIGNS:
            text = c + vs
            cid = f'c{ord(c):02x}_vs{ord(vs):02x}'
            combinations.append((text, cid))

    # Consonant + virama + consonant (select conjuncts)
    # Only use single consonant characters
    conjunct_pairs = [
        ('ಕ', 'ಕ'),  # ಕ್ಕ
        ('ಕ', 'ಷ'),  # ಕ್ಷ
        ('ಜ', 'ಞ'),  # ಜ್ಞ
        ('ತ', 'ರ'),  # ತ್ರ
        ('ಶ', 'ರ'),  # ಶ್ರ
        ('ಪ', 'ರ'),  # ಪ್ರ
        ('ನ', 'ದ'),  # ನ್ದ
        ('ಮ', 'ಮ'),  # ಮ್ಮ
    ]
    for c1, c2 in conjunct_pairs:
        if len(c1) == 1 and len(c2) == 1:  # Ensure single chars
            text = c1 + KANNADA_VIRAMA + c2
            cid = f'conj_{ord(c1):02x}_{ord(c2):02x}'
            combinations.append((text, cid))

    # Anusvara/visarga
    combinations.append((KANNADA_ANUSVARA, 'anusvara'))
    combinations.append((KANNADA_VISARGA, 'visarga'))

    # Numerals
    combinations.extend([(n, f'numeral_{ord(n):04x}') for n in KANNADA_NUMERALS])

    # ── Drop combinations the model cannot encode ────────────────────────────
    # Without this the generator renders images whose ground truth lstmtraining
    # rejects at every epoch ("Can't encode transcription: 'ಖೄ'"), inflating the
    # skip ratio and wasting the image entirely.
    units = _unicharset_units()
    if units is None:
        print("  ⚠  unicharset unavailable — skipping encodability validation")
    else:
        before = len(combinations)
        rejected = [(t, cid) for t, cid in combinations if not _encodable(t, units)]
        combinations = [(t, cid) for t, cid in combinations if _encodable(t, units)]
        if rejected:
            print(f"  ⊘ {before - len(combinations)} of {before} combinations are not "
                  f"encodable in the unicharset — skipping")
            shown = ', '.join(repr(t) for t, _ in rejected[:8])
            print(f"    e.g. {shown}{' …' if len(rejected) > 8 else ''}")

    if limit:
        combinations = combinations[:limit]

    print(f"Generating {len(combinations)} character combinations across {len(fonts)} fonts...")
    print(f"Renderer: {'HarfBuzz + FreeType (shaped)' if SHAPING_AVAILABLE else 'Pillow (BROKEN conjuncts)'}")
    _check_and_warn()

    aalt_by_stem = _aalt_map()

    total = 0
    for font_tag, font_path in fonts.items():
        font_dir = OUTPUT_DIR / font_tag
        font_dir.mkdir(parents=True, exist_ok=True)
        aalt = aalt_by_stem.get(Path(font_path).stem.lower(), False)

        for i, (text, cid) in enumerate(combinations):
            if i % max(1, len(combinations) // 10) == 0:
                print(f"  {font_tag}: {i}/{len(combinations)}...")

            img = render_char(text, font_path, dpi=dpi, aalt=aalt)
            if img is None:
                continue

            # Save PNG
            png_path = font_dir / f'char_{cid}.png'
            img.save(str(png_path), 'PNG')

            # Save GT
            gt_path = font_dir / f'char_{cid}.gt.txt'
            gt_path.write_text(text, encoding='utf-8')

            total += 1

    print(f"\n✓ Generated {total} inventory images in {OUTPUT_DIR}/")
    return OUTPUT_DIR

def main():
    parser = argparse.ArgumentParser(description='Generate Kannada character inventory')
    parser.add_argument('--fonts', help='Filter fonts by name (comma-separated patterns)')
    parser.add_argument('--dpi', type=int, default=150, help='Render DPI (default: 150)')
    parser.add_argument('--limit', type=int, help='Limit combinations (for testing)')
    parser.add_argument('--all-fonts', action='store_true',
                        help='Use every .ttf on disk, not just the weights declared in fonts.yml')
    args = parser.parse_args()

    font_patterns = args.fonts.split(',') if args.fonts else [None]
    all_fonts = {}
    for pattern in font_patterns:
        all_fonts.update(find_fonts(pattern.strip() if pattern else None,
                                    all_fonts=args.all_fonts))

    print(f"Found {len(all_fonts)} fonts: {', '.join(sorted(all_fonts.keys())[:5])}...")
    inventory_dir = generate_inventory(all_fonts, dpi=args.dpi, limit=args.limit)

    print(f"\nNext steps:")
    print(f"  1. Convert inventory to lstmf:")
    print(f"     INVENTORY_DIR={inventory_dir} ./scripts/02-make-lstmf.sh")
    print(f"  2. Train on inventory data:")
    print(f"     TRAIN_MODE=fresh caffeinate -i ./scripts/03-train.sh")

if __name__ == '__main__':
    main()
