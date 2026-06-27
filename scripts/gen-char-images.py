#!/usr/bin/env python3
"""
gen-char-images.py
──────────────────
Generates PNG test images of Kannada characters and conjuncts.
These images are used for OCR testing in the TrainOCR portal.

Output:  test-images/char_<name>.png   — single characters
         test-images/line_<group>.png  — full group lines
         test-images/conjunct_<name>.png — complex conjuncts

Usage:
    python3 scripts/gen-char-images.py
    python3 scripts/gen-char-images.py --outdir test-images --dpi 150

Requires: Pillow  (pip install Pillow --break-system-packages)
"""

import sys
import os
import argparse
import json
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--outdir', default='test-images')
    p.add_argument('--dpi',    type=int, default=150)
    p.add_argument('--font',   default=None,
                   help='Path to .ttf/.otf Kannada font. Auto-detected if not given.')
    p.add_argument('--size',   type=int, default=48, help='Font size in points')
    return p.parse_args()


# ── Kannada character sets ────────────────────────────────────────────────────

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

CONJUNCTS = [
    ('ಕ್ಷ',  'ksha'),  ('ಜ್ಞ',  'jnya'),  ('ತ್ತ',  'tt'),
    ('ದ್ದ',  'dd'),   ('ನ್ನ',  'nn'),   ('ಮ್ಮ',  'mm'),
    ('ಲ್ಲ',  'll'),   ('ಸ್ತ',  'st'),   ('ಪ್ರ',  'pr'),
    ('ಗ್ರ',  'gr'),   ('ತ್ರ',  'tr'),   ('ಶ್ರ',  'shr'),
    ('ಸ್ವ',  'sv'),   ('ನ್ತ',  'nt'),   ('ರ್ಕ',  'rk'),
    ('ಕ್ಕ',  'kk'),   ('ಬ್ಬ',  'bb'),   ('ಷ್ಟ',  'sht'),
    ('ಕ್ತ',  'kt'),   ('ನ್ಮ',  'nm'),
]

DIGITS = [
    ('೦', '0CE6', '0'), ('೧', '0CE7', '1'), ('೨', '0CE8', '2'),
    ('೩', '0CE9', '3'), ('೪', '0CEA', '4'), ('೫', '0CEB', '5'),
    ('೬', '0CEC', '6'), ('೭', '0CED', '7'), ('೮', '0CEE', '8'),
    ('೯', '0CEF', '9'), ('।', '0964', 'danda'), ('॥', '0965', 'ddanda'),
]

# Sample sentences using historical Kannada words (virama-heavy)
SAMPLE_SENTENCES = [
    "ಕರ್ನಾಟಕ ರಾಜ್ಯದ ಪ್ರಾಚೀನ ಶಾಸನಗಳು",
    "ಶ್ರೀ ಕೃಷ್ಣನ ಭಕ್ತಿಯಿಂದ ಮೋಕ್ಷ ಸಿದ್ಧಿಸುವುದು",
    "ವಿದ್ಯಾರ್ಥಿಗಳಿಗೆ ಜ್ಞಾನದ ಮಹತ್ತ್ವ ತಿಳಿಸಬೇಕು",
    "ಕನ್ನಡ ಸಾಹಿತ್ಯದ ಪರಂಪರೆ ಅತ್ಯಂತ ಶ್ರೀಮಂತ",
    "ಪ್ರಕೃತಿಯ ಸೌಂದರ್ಯವನ್ನು ಕಾಪಾಡಬೇಕು",
    "ರಾಷ್ಟ್ರ ಸೇವೆ ದೈವ ಸೇವೆ ಎಂದು ತಿಳಿಯಬೇಕು",
    "ಶ್ರದ್ಧೆ ಭಕ್ತಿ ಜ್ಞಾನ ವೈರಾಗ್ಯ",
    "ಅಷ್ಟಾದಶ ಪುರಾಣಗಳ ಸಂಕ್ಷಿಪ್ತ ವಿವರ",
]


def find_kannada_font():
    """Find a usable Kannada font on the system."""
    candidates = [
        # Homebrew
        '/opt/homebrew/share/fonts/NotoSansKannada-Regular.ttf',
        '/usr/local/share/fonts/NotoSansKannada-Regular.ttf',
        # macOS system
        '/System/Library/Fonts/Supplemental/Kohinoor Kannada.ttc',
        '/Library/Fonts/NotoSansKannada-Regular.ttf',
        # Project fonts dir (populated by 01-prep-base.sh)
        'fonts/kan_gmp/KarnatakaText-Regular.ttf',
        'fonts/kan_gmp/Kedage-n.ttf',
        'fonts/kan_gmp/Mallige.ttf',
        'fonts/Tunga Regular.ttf',
    ]
    # Also search project fonts/
    for p in Path('fonts').rglob('*.ttf') if Path('fonts').exists() else []:
        candidates.append(str(p))

    for c in candidates:
        if Path(c).exists():
            return c

    # Try fc-list
    import subprocess
    try:
        out = subprocess.check_output(
            ['fc-list', ':lang=kn', '--format=%{file}\n'], text=True
        ).strip().split('\n')
        for f in out:
            if f and Path(f).exists():
                return f
    except Exception:
        pass
    return None


def render_text(draw, font, text, x, y, fill=(30, 27, 75)):
    """Draw text with a label."""
    draw.text((x, y), text, font=font, fill=fill)


def make_char_image(char, name, font, out_dir, size, dpi):
    """Render a single character + its label to PNG."""
    from PIL import Image, ImageDraw, ImageFont
    pad = size // 2
    # Measure the character
    img_tmp = Image.new('RGB', (1, 1))
    d_tmp   = ImageDraw.Draw(img_tmp)
    bb = d_tmp.textbbox((0, 0), char, font=font)
    cw, ch = bb[2] - bb[0] + 1, bb[3] - bb[1] + 1

    # Label font (small, system)
    try:
        lbl_font = ImageFont.truetype(font.path, size=size // 3)
    except Exception:
        lbl_font = ImageFont.load_default()

    lbl = name
    lbl_bb = d_tmp.textbbox((0, 0), lbl, font=lbl_font)
    lbl_w = lbl_bb[2] - lbl_bb[0]

    W = max(cw + pad * 2, lbl_w + pad * 2, size * 2)
    H = ch + pad * 3 + (lbl_bb[3] - lbl_bb[1]) + 4

    img = Image.new('RGB', (W, H), (255, 255, 255))
    d   = ImageDraw.Draw(img)

    cx = (W - cw) // 2 - bb[0]
    cy = pad - bb[1]
    d.text((cx, cy), char, font=font, fill=(30, 27, 75))

    lx = (W - lbl_w) // 2
    ly = cy + ch + pad // 2
    d.text((lx, ly), lbl, font=lbl_font, fill=(107, 114, 128))

    fname = out_dir / f"char_{name.lower().replace(' ','_')}.png"
    img.save(str(fname), dpi=(dpi, dpi))
    return fname


def make_line_image(items, group_name, font, out_dir, size, dpi):
    """Render a row of characters as a single PNG (good for OCR line testing)."""
    from PIL import Image, ImageDraw, ImageFont
    pad = size // 3
    chars = [x[0] for x in items]
    line  = '  '.join(chars)

    img_tmp = Image.new('RGB', (1, 1))
    d_tmp   = ImageDraw.Draw(img_tmp)
    bb = d_tmp.textbbox((0, 0), line, font=font)
    lw = bb[2] - bb[0]
    lh = bb[3] - bb[1]

    W = lw + pad * 4
    H = lh + pad * 2
    img = Image.new('RGB', (W, H), (255, 255, 255))
    d   = ImageDraw.Draw(img)
    d.text((pad * 2 - bb[0], pad - bb[1]), line, font=font, fill=(30, 27, 75))

    fname = out_dir / f"line_{group_name}.png"
    img.save(str(fname), dpi=(dpi, dpi))
    # Also write .gt.txt for lstmf generation
    (out_dir / f"line_{group_name}.gt.txt").write_text(line, encoding='utf-8')
    return fname


def make_sentence_image(text, idx, font, out_dir, size, dpi):
    """Render a full sample sentence."""
    from PIL import Image, ImageDraw
    pad = size // 2

    img_tmp = Image.new('RGB', (1, 1))
    d_tmp   = ImageDraw.Draw(img_tmp)
    bb = d_tmp.textbbox((0, 0), text, font=font)
    lw, lh = bb[2] - bb[0], bb[3] - bb[1]

    W = lw + pad * 4
    H = lh + pad * 2
    img = Image.new('RGB', (W, H), (255, 255, 255))
    d   = ImageDraw.Draw(img)
    d.text((pad * 2 - bb[0], pad - bb[1]), text, font=font, fill=(30, 27, 75))

    fname = out_dir / f"sentence_{idx:02d}.png"
    img.save(str(fname), dpi=(dpi, dpi))
    (out_dir / f"sentence_{idx:02d}.gt.txt").write_text(text, encoding='utf-8')
    return fname


def main():
    args = parse_args()

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("ERROR: Pillow not installed.")
        print("  pip install Pillow --break-system-packages")
        sys.exit(1)

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    font_path = args.font or find_kannada_font()
    if not font_path:
        print("ERROR: No Kannada font found.")
        print("  Run  ./scripts/01-prep-base.sh  first to clone fonts,")
        print("  or specify --font /path/to/KannadaFont.ttf")
        sys.exit(1)

    print(f"  Font:   {font_path}")
    print(f"  Size:   {args.size}pt @ {args.dpi}dpi")
    print(f"  Output: {out_dir}/")

    font = ImageFont.truetype(font_path, size=args.size)

    generated = []

    # Individual characters
    print("\n→ Vowels…")
    for ch, cp, name in VOWELS:
        f = make_char_image(ch, name, font, out_dir, args.size, args.dpi)
        generated.append(str(f))

    print("→ Consonants…")
    for ch, cp, name in CONSONANTS:
        f = make_char_image(ch, name, font, out_dir, args.size, args.dpi)
        generated.append(str(f))

    print("→ Conjuncts…")
    for ch, name in CONJUNCTS:
        f = make_char_image(ch, name, font, out_dir, args.size, args.dpi)
        generated.append(str(f))

    print("→ Digits…")
    for ch, cp, name in DIGITS:
        f = make_char_image(ch, f"digit_{name}", font, out_dir, args.size, args.dpi)
        generated.append(str(f))

    # Line images (better for OCR)
    print("→ Line images…")
    f = make_line_image(VOWELS,      'vowels',      font, out_dir, args.size, args.dpi)
    generated.append(str(f))
    f = make_line_image(CONSONANTS,  'consonants',  font, out_dir, args.size, args.dpi)
    generated.append(str(f))
    f = make_line_image(CONJUNCTS,   'conjuncts',   font, out_dir, args.size, args.dpi)
    generated.append(str(f))
    f = make_line_image(DIGITS,      'digits',      font, out_dir, args.size, args.dpi)
    generated.append(str(f))

    # Sample sentences
    print("→ Sample sentences…")
    for i, sent in enumerate(SAMPLE_SENTENCES):
        f = make_sentence_image(sent, i + 1, font, out_dir, args.size, args.dpi)
        generated.append(str(f))

    print(f"\n✓ {len(generated)} images written to {out_dir}/")

    # Write manifest
    manifest = out_dir / 'manifest.json'
    manifest.write_text(json.dumps({
        'count': len(generated),
        'font':  font_path,
        'files': generated,
    }, ensure_ascii=False, indent=2))

    print(f"  Manifest: {manifest}")
    return len(generated)


if __name__ == '__main__':
    main()
