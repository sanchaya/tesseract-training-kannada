#!/usr/bin/env python3
"""
render-corpus.py

Renders corpus/kan_corpus.txt with all fonts listed in fonts.yml
into rendered/ as PNG + gt.txt pairs.

Each corpus line × each font style = one PNG training image.

Historical/letterpress fonts (degrade: true in fonts.yml) get:
  - Slight Gaussian blur (simulates ink spread)
  - Mild noise (simulates paper texture)
  - Random ±1° rotation (simulates print misalignment)

Modern fonts (degrade: false) get clean rendering.

Output:
    rendered/<font_id>_<style>_line<N>.png
    rendered/<font_id>_<style>_line<N>.gt.txt

Run from the repo root:
    python3 corpus/render-corpus.py
"""

import random
import sys
import yaml
from pathlib import Path

try:
    from PIL import Image, ImageFont, ImageDraw, ImageFilter
except ImportError:
    print("ERROR: Pillow not installed. Run: pip install Pillow")
    sys.exit(1)

ROOT       = Path(__file__).parent.parent
FONTS_YML  = ROOT / "fonts.yml"
CORPUS     = ROOT / "corpus" / "kan_corpus.txt"
FONTS_DIR  = ROOT / "fonts"
OUTDIR     = ROOT / "rendered"
OUTDIR.mkdir(exist_ok=True)

# Rendering parameters
FONT_SIZE = 36        # px at 150 DPI effective
PAD_X     = 20
PAD_Y     = 12
MIN_H     = 60        # Tesseract minimum image height
DPI       = 150       # metadata DPI saved in PNG

random.seed(42)


def add_degradation(img: Image.Image) -> Image.Image:
    """Simulate letterpress ink spread: blur + noise + tiny rotation."""
    # Gaussian blur (ink spread)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.6))

    # Salt-and-pepper noise
    import struct
    pixels = img.load()
    w, h = img.size
    n_noise = int(w * h * 0.003)   # 0.3% of pixels
    for _ in range(n_noise):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        pixels[x, y] = random.choice([0, 255])

    # Tiny rotation (±0.8°)
    angle = random.uniform(-0.8, 0.8)
    img = img.rotate(angle, expand=False, fillcolor=255)

    return img


def render_font(font_id: str, font_file: str, font_path: Path,
                lines: list[str], degrade: bool, max_pages: int) -> int:
    if not font_path.exists():
        print(f"  WARNING: font not found: {font_path}")
        print(f"    Run ./scripts/01-prep-base.sh first.")
        return 0

    try:
        font = ImageFont.truetype(str(font_path), FONT_SIZE)
    except Exception as e:
        print(f"  WARNING: cannot load {font_path}: {e}")
        return 0

    # Style tag from filename: KarnataGTN-Bold.ttf → bold
    style = Path(font_file).stem.lower().replace('-', '_')
    # Strip font family prefix if present (e.g. karnatagtn_bold → bold)
    # Keep the last underscore-separated segment that isn't a number
    parts = style.split('_')
    style_tag = parts[-1] if len(parts) > 1 else style

    tag    = f"{font_id}_{style_tag}"
    subset = lines[:max_pages]
    count  = 0

    for idx, text in enumerate(subset):
        # Measure
        dummy = Image.new("L", (1, 1))
        bbox  = ImageDraw.Draw(dummy).textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0] + PAD_X * 2
        h = max(bbox[3] - bbox[1] + PAD_Y * 2, MIN_H)

        img  = Image.new("L", (w, h), 255)
        draw = ImageDraw.Draw(img)
        draw.text((PAD_X, PAD_Y), text, font=font, fill=0)

        if degrade:
            img = add_degradation(img)

        stem = OUTDIR / f"{tag}_line{idx:04d}"
        img.save(str(stem) + ".png", dpi=(DPI, DPI))
        (stem.parent / (stem.name + ".gt.txt")).write_text(
            text, encoding='utf-8')
        count += 1

    return count


def main():
    if not CORPUS.exists():
        print(f"ERROR: Corpus not found: {CORPUS}")
        print("Run: python3 corpus/download-wiki.py")
        print("Then: python3 corpus/clean-corpus.py")
        sys.exit(1)

    lines = [ln.rstrip('\n') for ln in CORPUS.read_text(encoding='utf-8').splitlines()
             if ln.strip()]
    print(f"Corpus: {len(lines)} lines")

    with open(FONTS_YML) as f:
        config = yaml.safe_load(f)

    total = 0
    for font in config['fonts']:
        fid      = font['id']
        degrade  = font.get('degrade', False)
        max_pgs  = font.get('max_pages', 600)
        font_dir = font.get('font_dir', 'fonts')
        flag     = "degraded" if degrade else "clean"

        print(f"\n[{fid}]  ({flag} rendering, max {max_pgs} pages/style)")

        for font_file in font['font_files']:
            path  = FONTS_DIR / fid / font_dir / font_file
            count = render_font(fid, font_file, path, lines, degrade, max_pgs)
            print(f"  {font_file:<50}  {count} images")
            total += count

    print(f"\nTotal rendered: {total} PNG+gt.txt pairs → {OUTDIR}")
    print("Run: ./scripts/02-make-lstmf.sh")


if __name__ == '__main__':
    main()
