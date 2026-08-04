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

Rendering engine:
  Uses uharfbuzz + freetype-py for full OpenType shaping so that Kannada
  conjunct characters (e.g. ಕ್ಷ, ಜ್ಞ, ಶ್ರ) render as single fused glyphs
  rather than broken sequences of isolated codepoints.  Pillow's
  ImageDraw.text() does NOT apply GSUB rules and will always produce broken
  conjuncts for Kannada — the shaped renderer fixes this.

  If uharfbuzz / freetype-py are not installed the script warns and falls
  back to Pillow (conjuncts will be visually broken).  Fix with:
      pip install uharfbuzz freetype-py numpy --break-system-packages

Parallelism:
  Uses multiprocessing.Pool (fork) to render all images in parallel.
  Defaults to os.cpu_count() workers; override with --workers N.
  Already-rendered files are skipped automatically (safe to resume).

Output:
    rendered/<font_id>_<style>_line<N>.png
    rendered/<font_id>_<style>_line<N>.gt.txt

Run from the repo root:
    python3 corpus/render-corpus.py
    python3 corpus/render-corpus.py --workers 4
"""

import argparse
import multiprocessing
import os
import sys
import yaml
from pathlib import Path

try:
    from PIL import Image, ImageFont, ImageDraw, ImageFilter
except ImportError:
    print("ERROR: Pillow not installed. Run: pip install Pillow")
    sys.exit(1)

# ── Shaping renderer (HarfBuzz + FreeType) ────────────────────────────────
# Ensure corpus/ is on the path so the import works from any cwd.
sys.path.insert(0, str(Path(__file__).parent))
from shaping_render import SHAPING_AVAILABLE, render_text, check_and_warn

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


# ── Parallel worker ────────────────────────────────────────────────────────
# Must be a module-level function for multiprocessing (fork) to pickle it.
# With fork, all module globals (SHAPING_AVAILABLE, render_text, etc.) are
# inherited by child processes — no need to re-import inside the worker.

def _render_task(args):
    """
    Render one (font_path, text, idx) pair and write PNG + gt.txt.
    Returns 'ok', 'skip' (already exists), or 'fail:<reason>'.
    """
    font_path_str, text, idx, tag, degrade, out_dir_str, seed, aalt, force = args
    import random as _rng_mod
    from PIL import ImageFilter as _IF

    out_dir  = Path(out_dir_str)
    stem     = out_dir / f"{tag}_line{idx:04d}"
    png_path = Path(str(stem) + ".png")
    gt_path  = Path(str(stem) + ".gt.txt")

    # Resume support: skip already-rendered images.
    # --force overwrites in place, which is how you re-render after a shaping
    # change without needing to delete the output directory first.
    if not force and png_path.exists() and gt_path.exists():
        return 'skip'

    try:
        if SHAPING_AVAILABLE:
            img = render_text(
                font_path_str, text,
                font_size=FONT_SIZE, padding_x=PAD_X, padding_y=PAD_Y,
                min_height=MIN_H, bg_color=255, ink_color=0,
                aalt=aalt,
            )
        else:
            from PIL import ImageDraw
            pil_font = ImageFont.truetype(font_path_str, FONT_SIZE)
            dummy = Image.new("L", (1, 1))
            bbox  = ImageDraw.Draw(dummy).textbbox((0, 0), text, font=pil_font)
            w = bbox[2] - bbox[0] + PAD_X * 2
            h = max(bbox[3] - bbox[1] + PAD_Y * 2, MIN_H)
            img = Image.new("L", (w, h), 255)
            ImageDraw.Draw(img).text((PAD_X, PAD_Y), text, font=pil_font, fill=0)
    except Exception as e:
        return f'fail:{e}'

    if degrade:
        # Deterministic degradation: seed is hash(tag, idx) so reruns match
        rng = _rng_mod.Random(seed)
        img = img.filter(_IF.GaussianBlur(radius=0.6))
        w, h = img.size
        pixels = img.load()
        for _ in range(int(w * h * 0.003)):
            pixels[rng.randint(0, w - 1), rng.randint(0, h - 1)] = rng.choice([0, 255])
        img = img.rotate(rng.uniform(-0.8, 0.8), expand=False, fillcolor=255)

    img.save(str(png_path), dpi=(DPI, DPI))
    gt_path.write_text(text, encoding='utf-8')
    return 'ok'


def main():
    parser = argparse.ArgumentParser(
        description='Render Kannada corpus to PNG training images (parallel)')
    parser.add_argument(
        '--workers', type=int,
        default=os.cpu_count() or 4,
        help=f'Parallel worker processes (default: {os.cpu_count() or 4} = cpu count)')
    parser.add_argument(
        '--force', action='store_true',
        help='Re-render images that already exist (use after a shaping/font change)')
    cli = parser.parse_args()
    workers = max(1, cli.workers)

    check_and_warn()

    if not CORPUS.exists():
        print(f"ERROR: Corpus not found: {CORPUS}")
        print("Run: python3 corpus/download-wiki.py")
        print("Then: python3 corpus/clean-corpus.py")
        sys.exit(1)

    lines = [ln.rstrip('\n') for ln in CORPUS.read_text(encoding='utf-8').splitlines()
             if ln.strip()]
    print(f"Corpus: {len(lines)} lines")
    print(f"Renderer: {'HarfBuzz + FreeType (shaped)' if SHAPING_AVAILABLE else 'Pillow (BROKEN conjuncts)'}")
    print(f"Workers:  {workers}")

    with open(FONTS_YML) as f:
        config = yaml.safe_load(f)

    # ── Build flat task list (all font × line combinations) ───────────────
    tasks = []
    style_info = []   # (tag, font_file, n_lines) — for the summary header

    for font in config['fonts']:
        fid      = font['id']
        degrade  = font.get('degrade', False)
        aalt     = 'aalt' in font.get('font_features', '')
        max_pgs  = font.get('max_pages', 600)
        font_dir = font.get('font_dir', 'fonts')
        flag     = "degraded" if degrade else "clean"

        print(f"\n[{fid}]  ({flag} rendering, max {max_pgs} pages/style)")

        for font_file in font['font_files']:
            font_path = FONTS_DIR / fid / font_dir / font_file

            if not font_path.exists():
                print(f"  WARNING: font not found: {font_path}")
                print(f"    Run ./scripts/01-prep-base.sh first.")
                continue
            try:
                ImageFont.truetype(str(font_path), FONT_SIZE)
            except Exception as e:
                print(f"  WARNING: cannot load {font_path}: {e}")
                continue

            # Style tag: KarnataGTN-Bold.ttf → bold
            style     = Path(font_file).stem.lower().replace('-', '_')
            parts     = style.split('_')
            style_tag = parts[-1] if len(parts) > 1 else style
            tag       = f"{fid}_{style_tag}"

            subset = lines[:max_pgs]
            for idx, text in enumerate(subset):
                seed = hash((tag, idx)) & 0xFFFFFFFF   # deterministic per image
                tasks.append((str(font_path), text, idx, tag, degrade, str(OUTDIR), seed, aalt, cli.force))

            print(f"  {font_file:<50}  queued {len(subset)} images")
            style_info.append((tag, len(subset)))

    if not tasks:
        print("\nNo tasks to render.")
        return

    total = len(tasks)
    print(f"\nRendering {total} images with {workers} workers…\n")

    ok = fail = skip = 0

    with multiprocessing.Pool(processes=workers) as pool:
        for result in pool.imap_unordered(_render_task, tasks, chunksize=16):
            if result == 'ok':
                ok += 1
            elif result == 'skip':
                skip += 1
            else:
                fail += 1

            done = ok + skip + fail
            if done % 200 == 0 or done == total:
                pct = done * 100 // total
                print(f"  {done}/{total} ({pct}%)  new={ok}  skipped={skip}  failed={fail}",
                      flush=True)

    print(f"\nTotal rendered: {ok} new  {skip} skipped  {fail} failed → {OUTDIR}")
    print("Run: ./scripts/02-make-lstmf.sh")


if __name__ == '__main__':
    # Use fork start method: safe for CPU-bound PIL work on macOS/Linux.
    # (spawn would require this script to be importable as a module,
    #  but the filename contains a hyphen which Python can't import.)
    try:
        multiprocessing.set_start_method('fork')
    except RuntimeError:
        pass  # already set (e.g. called twice in same process)
    main()
