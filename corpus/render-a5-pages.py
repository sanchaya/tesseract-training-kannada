#!/usr/bin/env python3
"""
render-a5-pages.py
──────────────────
Renders classical Kannada corpus TXT files as A5-sized page images
with matching .gt.txt ground truth files for Tesseract OCR training.

Uses headless Chrome (via corpus/browser_render.js) for rendering — the
same text pipeline as fonts.sanchaya.net.  This correctly shapes Kannada
conjuncts for all fonts including historical fonts (GMP, WMP, GTN TTFs)
whose OpenType GSUB tables are incomplete.

All fonts declared in fonts.yml are used — 9 styles across 4 typefaces:

    kan_gtn  × 6 weights  (clean)
    kan_gmp  × 1          (degraded – letterpress simulation)
    kan_wmp  × 1          (degraded – letterpress simulation)
    kan_kittel × 1        (degraded – letterpress simulation)

A5 at 150 DPI ≈ 875 × 1241 px.

Output layout
─────────────
    <corpus-dir>/a5-pages/
        <title>/
            <font_tag>/
                page0001.png
                page0001.gt.txt
                page0002.png
                page0002.gt.txt
                …

The .gt.txt for each page contains the source text chunk rendered on that
page.  The browser handles actual line-wrapping; gt.txt preserves the
original paragraph structure — both are valid for Tesseract page-level GT.

Usage
─────
    # From the tesseract-training-kannada project root:
    python3 corpus/render-a5-pages.py \\
        --corpus-dir /path/to/classical-corpus-kannada

    # Override font size or page dimensions:
    python3 corpus/render-a5-pages.py \\
        --corpus-dir /path/to/classical-corpus-kannada \\
        --font-size 28 --page-w 875 --page-h 1241

    # Limit parallelism:
    python3 corpus/render-a5-pages.py \\
        --corpus-dir /path/to/classical-corpus-kannada \\
        --concurrency 2

Resume support
──────────────
    Already-rendered PNG+gt.txt pairs are skipped automatically.
    Safe to Ctrl-C and re-run.

Dependencies
────────────
    Node.js + puppeteer (npm install in project root)
    Chrome / Chromium (or: npx puppeteer browsers install chrome)
    Python: pyyaml   (pip install pyyaml --break-system-packages)
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

# ── Project paths ──────────────────────────────────────────────────────────────
_CORPUS_DIR  = Path(__file__).parent
_ROOT        = _CORPUS_DIR.parent
_FONTS_YML   = _ROOT / "fonts.yml"
_FONTS_DIR   = _ROOT / "fonts"
_BROWSER_JS  = _CORPUS_DIR / "browser_render.js"

# A5 at 150 DPI
_DEFAULT_PAGE_W = 875
_DEFAULT_PAGE_H = 1241
_DEFAULT_MARGIN_X = 50
_DEFAULT_MARGIN_Y = 60
_DEFAULT_FONT_SIZE = 32

# Characters per page chunk (rough — browser handles actual wrapping)
# At 32px ~35 chars/line × 18 lines ≈ 630.  Use 900 to allow shorter lines.
_CHARS_PER_PAGE = 900


# ══════════════════════════════════════════════════════════════════════════════
# Text chunking (paragraph-aware)
# ══════════════════════════════════════════════════════════════════════════════

def _chunk_to_pages(text: str, chars_per_page: int = _CHARS_PER_PAGE) -> list[str]:
    """
    Split text into page-sized chunks at paragraph boundaries.

    Paragraphs are separated by blank lines in the source.  We accumulate
    paragraphs until we hit chars_per_page, then start a new chunk.
    This keeps related stanzas/verses together and avoids mid-stanza breaks.

    Returns a list of page text strings (each is the gt.txt for one page).
    """
    # Collect paragraphs
    paragraphs: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            current.append(stripped)
        else:
            if current:
                paragraphs.append("\n".join(current))
                current = []
    if current:
        paragraphs.append("\n".join(current))

    if not paragraphs:
        return []

    # Group paragraphs into pages
    pages: list[str] = []
    current_page: list[str] = []
    current_chars = 0

    for para in paragraphs:
        para_len = len(para)
        # Start a new page if adding this paragraph would overflow
        # (but always include at least one paragraph per page)
        if current_chars > 0 and current_chars + para_len > chars_per_page:
            pages.append("\n\n".join(current_page))
            current_page = []
            current_chars = 0
        current_page.append(para)
        current_chars += para_len

    if current_page:
        pages.append("\n\n".join(current_page))

    return pages


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render classical Kannada corpus as A5 page images for OCR training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--corpus-dir", required=True,
        help="Path to the classical-corpus-kannada folder",
    )
    parser.add_argument(
        "--output-dir",
        help="Where to write a5-pages/ (default: <corpus-dir>/a5-pages)",
    )
    parser.add_argument(
        "--font-size", type=int, default=_DEFAULT_FONT_SIZE,
        help=f"Font size in pixels (default: {_DEFAULT_FONT_SIZE})",
    )
    parser.add_argument(
        "--page-w", type=int, default=_DEFAULT_PAGE_W,
        help=f"Page width in px (default: {_DEFAULT_PAGE_W})",
    )
    parser.add_argument(
        "--page-h", type=int, default=_DEFAULT_PAGE_H,
        help=f"Page height in px (default: {_DEFAULT_PAGE_H})",
    )
    parser.add_argument(
        "--margin-x", type=int, default=_DEFAULT_MARGIN_X,
        help=f"Left/right margin in px (default: {_DEFAULT_MARGIN_X})",
    )
    parser.add_argument(
        "--margin-y", type=int, default=_DEFAULT_MARGIN_Y,
        help=f"Top/bottom margin in px (default: {_DEFAULT_MARGIN_Y})",
    )
    parser.add_argument(
        "--chars-per-page", type=int, default=_CHARS_PER_PAGE,
        help=f"Approximate chars per page chunk (default: {_CHARS_PER_PAGE})",
    )
    parser.add_argument(
        "--lines", action="store_true",
        help="Emit one cropped image per TEXT LINE (pageNNNN_lineNNN.png) instead "
             "of one image per page. Required for LSTM training — a full-page "
             "image paired with the whole page's text cannot be aligned by CTC "
             "and lstmtraining reports 'Compute CTC targets failed'.",
    )
    parser.add_argument(
        "--line-pad", type=int, default=6,
        help="Padding in px around each cropped line (default: 6)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel Chrome processes (default: 1; max useful = number of fonts = 9)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=2,
        help="Parallel browser pages per Node process (default: 2; raise to 4 if pages render fast)",
    )
    parser.add_argument(
        "--fonts-yml",
        help="Path to fonts.yml (default: auto-detected from project root)",
    )
    parser.add_argument(
        "--title", action="append", metavar="TITLE",
        help="Process only this title (repeatable; default: all)",
    )
    args = parser.parse_args()

    corpus_dir = Path(args.corpus_dir).resolve()
    out_root   = Path(args.output_dir).resolve() if args.output_dir \
                 else corpus_dir / "a5-pages"
    fonts_yml  = Path(args.fonts_yml).resolve() if args.fonts_yml else _FONTS_YML

    # ── Pre-flight checks ─────────────────────────────────────────────────
    if not _BROWSER_JS.exists():
        print(f"ERROR: browser_render.js not found: {_BROWSER_JS}")
        sys.exit(1)

    result = subprocess.run(["node", "--version"], capture_output=True, text=True)
    if result.returncode != 0:
        print("ERROR: Node.js not found.  Install from https://nodejs.org")
        sys.exit(1)

    if not fonts_yml.exists():
        print(f"ERROR: fonts.yml not found: {fonts_yml}")
        sys.exit(1)

    # ── Startup banner ────────────────────────────────────────────────────
    print("=" * 60)
    print("render-a5-pages.py  (browser rendering)")
    print("=" * 60)
    print(f"Corpus dir  : {corpus_dir}")
    print(f"Output dir  : {out_root}")
    print(f"Fonts yml   : {fonts_yml}")
    print(f"Page size   : {args.page_w} × {args.page_h} px")
    print(f"Font size   : {args.font_size} px")
    print(f"Margins     : {args.margin_x} px L/R,  {args.margin_y} px T/B")
    print(f"Chars/page  : ~{args.chars_per_page}")
    print(f"Workers     : {args.workers} Chrome process(es)")
    print(f"Concurrency : {args.concurrency} pages per process")
    print(f"Renderer    : headless Chrome via browser_render.js")
    print(f"Output mode : {'LINE images (LSTM-ready)' if args.lines else 'PAGE images (not usable for LSTM training)'}")
    print()

    # ── Load fonts.yml ────────────────────────────────────────────────────
    with open(fonts_yml) as f:
        config = yaml.safe_load(f)

    # ── Collect title TXT files ───────────────────────────────────────────
    filter_titles = set(args.title) if args.title else None
    titles: dict[str, Path] = {}

    # Also look for txt files directly in corpus_dir itself (single-folder case)
    direct_txts = sorted(corpus_dir.glob("*.txt"))
    if direct_txts:
        name = corpus_dir.name
        if not filter_titles or name in filter_titles:
            titles[name] = direct_txts[0]
    else:
        # Traverse one level of subdirectories; within each, search recursively
        # for the first .txt file so nested layouts (e.g. title/src/text.txt) work.
        for item in sorted(corpus_dir.iterdir()):
            if not item.is_dir() or item.name.startswith("."):
                continue
            if filter_titles and item.name not in filter_titles:
                continue
            # rglob finds txt files at any depth inside this subdirectory
            txts = sorted(item.rglob("*.txt"))
            if txts:
                titles[item.name] = txts[0]

    if not titles:
        print(f"ERROR: no title subfolders with .txt files found in {corpus_dir}")
        sys.exit(1)

    print(f"Titles: {len(titles)}")
    for name, path in sorted(titles.items()):
        size_kb = path.stat().st_size // 1024
        print(f"  {name:<50}  {size_kb:>6} KB")
    print()

    # ── Build jobs list ───────────────────────────────────────────────────
    # Each job: one A5 page for one title × font combination.
    jobs: list[dict] = []
    skipped_fonts = 0

    for font in config["fonts"]:
        fid      = font["id"]
        degrade  = font.get("degrade", False)
        features = font.get("font_features", "")
        font_dir = font.get("font_dir", "fonts")
        flag     = "degraded" if degrade else "clean"

        print(f"[{fid}]  ({flag})")

        for font_file in font["font_files"]:
            font_path = _FONTS_DIR / fid / font_dir / font_file

            if not font_path.exists():
                print(f"  SKIP  {font_path.name} — not found (run 01-prep-base.sh?)")
                skipped_fonts += 1
                continue

            # Derive style tag: KarnataGTN-Bold.ttf → kan_gtn_bold
            stem_lower = Path(font_file).stem.lower().replace("-", "_")
            parts      = stem_lower.split("_")
            style_tag  = parts[-1] if len(parts) > 1 else stem_lower
            tag        = f"{fid}_{style_tag}"

            pages_total = 0
            for title_name, txt_path in titles.items():
                text = txt_path.read_text(encoding="utf-8")
                page_chunks = _chunk_to_pages(text, args.chars_per_page)
                out_dir = out_root / title_name / tag

                for page_idx, chunk in enumerate(page_chunks):
                    stem     = f"page{page_idx + 1:04d}"
                    png_path = out_dir / f"{stem}.png"
                    gt_path  = out_dir / f"{stem}.gt.txt"
                    # browser_render.js will skip if both exist (resume)
                    jobs.append({
                        "font":     str(font_path),
                        "text":     chunk,
                        "out":      str(png_path),
                        "size":     args.font_size,
                        "degrade":  degrade,
                        "features": features,
                        "seed":     hash((tag, title_name, page_idx)) & 0xFFFF_FFFF,
                        "page_w":   args.page_w,
                        "page_h":   args.page_h,
                        "margin_x": args.margin_x,
                        "margin_y": args.margin_y,
                        "lines":    args.lines,
                        "line_pad": args.line_pad,
                    })
                    pages_total += 1

            print(f"  queued {pages_total} pages across {len(titles)} titles  →  {tag}")

    print()

    if not jobs:
        print("No jobs to run — check that font files are present.")
        return

    print(f"Total page jobs : {len(jobs)}")

    # ── Split jobs across workers (grouped by font for cache efficiency) ──
    # Group by font path so each worker only pre-caches its own fonts.
    num_workers = min(args.workers, len(jobs))
    font_to_jobs: dict[str, list] = {}
    for job in jobs:
        font_to_jobs.setdefault(job["font"], []).append(job)

    font_keys = list(font_to_jobs.keys())
    chunks: list[list] = [[] for _ in range(num_workers)]
    for i, fk in enumerate(font_keys):
        chunks[i % num_workers].extend(font_to_jobs[fk])

    print(f"Workers         : {num_workers}")
    for i, chunk in enumerate(chunks):
        fonts_in_chunk = sorted({Path(j["font"]).name for j in chunk})
        print(f"  W{i}: {len(chunk):>5} jobs  ({', '.join(fonts_in_chunk)})")
    print()

    # Create output directories ahead of time
    dirs_needed = {Path(j["out"]).parent for j in jobs}
    for d in dirs_needed:
        d.mkdir(parents=True, exist_ok=True)

    # Write per-worker jobs files
    jobs_files: list[Path] = []
    for i, chunk in enumerate(chunks):
        jf = _ROOT / f"a5_render_jobs_w{i}.json"
        jf.write_text(json.dumps(chunk, ensure_ascii=False), encoding="utf-8")
        jobs_files.append(jf)

    # ── Launch all workers ────────────────────────────────────────────────
    print("Launching browser renderer(s)…")
    print("Headless Chrome has no visible window — that is normal.")
    print()
    print("To monitor progress in another terminal:")
    print(f"  watch -n5 'find \"{out_root}\" -name \"*.png\" | wc -l'")
    print()
    if num_workers == 1:
        print("Startup sequence (expect ~30 s before first pages appear):")
        print("  Finding Chrome → Pre-caching fonts → Warm-up → Batch")
    else:
        print(f"Starting {num_workers} Chrome processes in parallel.")
        print("Each will print startup messages prefixed [W0], [W1], …")
    print()

    env = {**os.environ, "BROWSER_CONCURRENCY": str(args.concurrency)}
    procs: list[subprocess.Popen] = []
    try:
        for i, (chunk, jf) in enumerate(zip(chunks, jobs_files)):
            cmd = [
                "node", str(_BROWSER_JS),
                "--batch", str(jf),
                "--label", f"W{i}",
            ]
            p = subprocess.Popen(
                cmd, env=env, cwd=str(_ROOT),
                # Each worker streams directly to the terminal
                stdout=subprocess.DEVNULL,  # summary JSON — not needed here
                stderr=None,                # inherit → streams to terminal
            )
            procs.append(p)

        # Wait for all workers to finish
        exit_codes = [p.wait() for p in procs]

    except KeyboardInterrupt:
        print("\nInterrupted — terminating workers…")
        for p in procs:
            try: p.terminate()
            except Exception: pass
        print("Partially rendered pages are safe. Re-run to resume.")
        sys.exit(0)
    finally:
        for jf in jobs_files:
            jf.unlink(missing_ok=True)

    failed_workers = [i for i, rc in enumerate(exit_codes) if rc != 0]
    if failed_workers:
        print(f"\nWARNING: workers {failed_workers} exited with non-zero codes.")

    # ── Final tally ───────────────────────────────────────────────────────
    total_png = sum(1 for j in jobs if Path(j["out"]).exists())
    print()
    print(f"Done.  {total_png}/{len(jobs)} pages written to {out_root}")
    print()
    print("Next: feed into training with scripts/02-make-lstmf.sh:")
    print(f"  CLASSICAL_A5_DIR={out_root} ./scripts/02-make-lstmf.sh")


if __name__ == "__main__":
    main()
