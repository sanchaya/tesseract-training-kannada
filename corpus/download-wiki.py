#!/usr/bin/env python3
"""
download-wiki.py

Downloads the Kannada Wikipedia text dump and extracts clean Kannada prose
into corpus/raw_kannada.txt, ready for clean-corpus.py.

Also prepends character-coverage lines so every Kannada Unicode codepoint
appears in training at least once.

Usage:
    python3 corpus/download-wiki.py [--lines N]

    --lines N   Number of Wikipedia prose lines to extract (default: 5000)

Requirements: requests (pip install requests)
Disk: ~150 MB for the bz2 dump (cached in corpus/cache/)
"""

import argparse
import bz2
import re
import sys
import urllib.request
from pathlib import Path

CORPUS_DIR  = Path(__file__).parent
CACHE_DIR   = CORPUS_DIR / "cache"
OUTPUT_FILE = CORPUS_DIR / "raw_kannada.txt"
WIKI_URL    = ("https://dumps.wikimedia.org/knwiki/latest/"
               "knwiki-latest-pages-articles.xml.bz2")

MIN_KAN_RATIO = 0.75
MAX_LINE_LEN  = 80


def is_kannada(c: str) -> bool:
    return 0x0C80 <= ord(c) <= 0x0CFF


def kan_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if is_kannada(c)) / len(text)


# ── Character coverage lines ───────────────────────────────────────────────────

def build_coverage_lines() -> list[str]:
    """Return lines that exercise every Kannada Unicode codepoint."""
    lines = []

    vowels      = [chr(c) for c in range(0x0C85, 0x0C95)]    # ಅ–ಔ
    consonants  = [chr(c) for c in range(0x0C95, 0x0CB9 + 1)] # ಕ–ಹ
    vowel_signs = [chr(c) for c in range(0x0CBE, 0x0CCC + 1)] # ಾ–ೌ
    virama      = chr(0x0CCD)
    digits_kan  = [chr(c) for c in range(0x0CE6, 0x0CEF + 1)] # ೦–೯

    # All vowels on one line
    lines.append("  ".join(vowels))

    # Consonants in chunks
    for i in range(0, len(consonants), 12):
        lines.append("  ".join(consonants[i:i + 12]))

    # Each consonant × all vowel signs (CV combinations)
    for cons in consonants:
        lines.append("  ".join(cons + vs for vs in vowel_signs))

    # Consonant clusters (ka + virama + each consonant)
    for c1 in consonants:
        row = "  ".join(c1 + virama + c2 for c2 in consonants)
        # Split into ≤ MAX_LINE_LEN chunks
        while len(row) > MAX_LINE_LEN:
            split = row[:MAX_LINE_LEN].rfind("  ")
            if split < 0:
                split = MAX_LINE_LEN
            lines.append(row[:split].strip())
            row = row[split:].strip()
        if row:
            lines.append(row)

    # Kannada digits
    lines.append("  ".join(digits_kan))

    # Punctuation in context
    sample = "ಕನ್ನಡ"
    for punct in [".", ",", "?", "!", ":", "।", "॥", "-", "(", ")"]:
        lines.append(sample + punct + "  " + punct + sample)

    return [ln for ln in lines if ln.strip()]


# ── Wikipedia extraction ───────────────────────────────────────────────────────

def download_dump(dest: Path) -> Path:
    if dest.exists():
        print(f"  Wiki dump cached: {dest}", file=sys.stderr)
        return dest
    print(f"  Downloading {WIKI_URL}", file=sys.stderr)
    print("  (~150 MB bz2 — may take a few minutes)", file=sys.stderr)
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(WIKI_URL, dest,
        reporthook=lambda n, bs, ts: print(
            f"\r  {n*bs/1e6:.1f}/{ts/1e6:.1f} MB", end='', file=sys.stderr))
    print(file=sys.stderr)
    return dest


def extract_wiki_lines(dump_path: Path, max_lines: int) -> list[str]:
    lines = []
    in_text = False
    print(f"  Extracting up to {max_lines} lines…", file=sys.stderr)

    open_fn = bz2.open if str(dump_path).endswith('.bz2') else open
    with open_fn(dump_path, 'rt', encoding='utf-8', errors='ignore') as fh:
        for raw in fh:
            raw = raw.rstrip('\n')
            if raw.strip().startswith('<') and raw.strip().endswith('>'):
                in_text = '<text' in raw
                continue
            if not in_text:
                continue
            for sentence in re.split(r'[।॥\n]', raw):
                # Strip markup
                sentence = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', sentence)
                sentence = re.sub(r'\{\{[^}]*\}\}', '', sentence)
                sentence = re.sub(r'<[^>]+>', '', sentence)
                sentence = re.sub(r'\[\d+\]', '', sentence)
                sentence = re.sub(r'\s+', ' ', sentence).strip()
                if not sentence or len(sentence) < 10:
                    continue
                if len(sentence) > MAX_LINE_LEN:
                    sentence = sentence[:MAX_LINE_LEN]
                if kan_ratio(sentence) < MIN_KAN_RATIO:
                    continue
                lines.append(sentence)
                if len(lines) >= max_lines:
                    return lines
    return lines


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lines', type=int, default=5000,
                        help='Wikipedia lines to extract (default 5000)')
    args = parser.parse_args()

    all_lines: list[str] = []

    # Coverage lines first — guarantee every glyph appears in training
    print("Building character coverage lines…", file=sys.stderr)
    coverage = build_coverage_lines()
    print(f"  {len(coverage)} coverage lines", file=sys.stderr)
    all_lines.extend(coverage)

    # Wikipedia prose
    dump = CACHE_DIR / "knwiki-latest.xml.bz2"
    try:
        download_dump(dump)
        wiki = extract_wiki_lines(dump, args.lines)
        print(f"  {len(wiki)} Wikipedia lines", file=sys.stderr)
        all_lines.extend(wiki)
    except Exception as exc:
        print(f"  WARNING: Wikipedia extraction failed: {exc}", file=sys.stderr)
        print("  Continuing with coverage lines only.", file=sys.stderr)

    # Deduplicate
    seen = set()
    deduped = []
    for ln in all_lines:
        if ln not in seen:
            seen.add(ln)
            deduped.append(ln)

    OUTPUT_FILE.write_text('\n'.join(deduped) + '\n', encoding='utf-8')
    print(f"\nWrote {len(deduped)} lines → {OUTPUT_FILE}", file=sys.stderr)
    print("Run: python3 corpus/clean-corpus.py", file=sys.stderr)


if __name__ == '__main__':
    main()
