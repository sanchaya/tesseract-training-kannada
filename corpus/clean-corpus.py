#!/usr/bin/env python3
"""
clean-corpus.py

Reads a raw Kannada text file and produces a clean corpus suitable
for rendering and Tesseract training.

Usage:
    python3 corpus/clean-corpus.py [input_file]

    input_file defaults to corpus/raw_kannada.txt
    Output is written to corpus/kan_corpus.txt

Input sources (supply one or more, concatenate into raw_kannada.txt):
  - Kannada Wikipedia dump (knwiki-latest-pages-articles.xml.bz2)
    → use: python3 corpus/download-wiki.py  (downloads & extracts)
  - Kannada Wikisource text (historical books — ideal for kan_hist)
  - Any plain-text Kannada document

Keeps:
  - Kannada Unicode block U+0C80–U+0CFF
  - Common punctuation used in Kannada: । ॥ . , : ? ! - ( ) " '
  - ASCII digits 0–9 (page/section numbers)
  - Kannada digits ೦–೯ (U+0CE6–U+0CEF)
  - ASCII space

Removes:
  - Latin letters, Devanagari, other scripts
  - URLs, HTML/XML tags, reference markers

Lines with fewer than 8 Kannada characters are dropped.
Lines longer than MAX_CHARS are split at word/space boundaries.
"""

import re
import sys
import unicodedata
from pathlib import Path

CORPUS_DIR = Path(__file__).parent
INPUT  = Path(sys.argv[1]) if len(sys.argv) > 1 else CORPUS_DIR / "raw_kannada.txt"
OUTPUT = CORPUS_DIR / "kan_corpus.txt"

MAX_CHARS = 80
MIN_KAN   = 8   # minimum Kannada chars per line

KEEP_ASCII = set(' .,;:?!-/()"\'।॥%0123456789')
# Note: * intentionally excluded — markdown bullets must not appear in GT text

def is_kannada(c: str) -> bool:
    """
    True for ASSIGNED characters in the Kannada block.

    The block 0x0C80–0x0CFF contains ~20 reserved codepoints that Unicode has
    never assigned (U+0C8D, U+0C91, U+0CA9, U+0CB4, U+0CC5, U+0CC9, …). They
    appear in scraped text as mojibake or encoding damage. A bare range check
    let them through, they were rendered into training images, and lstmtraining
    then rejected every line containing one:

        Encoding of string failed! Failure bytes: e0 b2 a9
        Can't encode transcription: '಩' in language ''

    unicodedata.name() raises for unassigned codepoints, which is the
    authoritative test and stays correct as Unicode adds characters.
    """
    if not (0x0C80 <= ord(c) <= 0x0CFF):
        return False
    try:
        unicodedata.name(c)
        return True
    except ValueError:
        return False

def clean_line(line: str) -> str:
    # Strip HTML/XML tags
    line = re.sub(r'<[^>]+>', ' ', line)
    # Strip URLs
    line = re.sub(r'https?://\S+', '', line)
    # Strip Wikipedia reference markers like [1], [2]
    line = re.sub(r'\[\d+\]', '', line)
    # Strip markdown/wiki heading markers (==, ===, #, ##, *)
    line = re.sub(r'^[=*#\s]+', '', line)
    line = re.sub(r'[=*#]+$', '', line)
    # Keep only Kannada chars + allowed ASCII
    chars = [c for c in line if is_kannada(c) or c in KEEP_ASCII]
    # Collapse whitespace
    return re.sub(r' +', ' ', ''.join(chars)).strip()

def kan_count(s: str) -> int:
    return sum(1 for c in s if is_kannada(c))

def split_at_boundary(line: str, max_len: int) -> list[str]:
    if len(line) <= max_len:
        return [line]
    parts, current, cur_len = [], [], 0
    for word in line.split(' '):
        if cur_len + len(word) + 1 > max_len and current:
            parts.append(' '.join(current))
            current, cur_len = [word], len(word)
        else:
            current.append(word)
            cur_len += len(word) + 1
    if current:
        parts.append(' '.join(current))
    return parts

if not INPUT.exists():
    print(f"ERROR: Input file not found: {INPUT}")
    print()
    print("Supply a Kannada text file as argument, or create corpus/raw_kannada.txt")
    print("See corpus/download-wiki.py to download Kannada Wikipedia text.")
    sys.exit(1)

raw = INPUT.read_text(encoding='utf-8').splitlines()
out_lines, dropped = [], 0

for line in raw:
    cleaned = clean_line(line)
    if kan_count(cleaned) < MIN_KAN:
        if cleaned.strip():
            dropped += 1
        continue
    for part in split_at_boundary(cleaned, MAX_CHARS):
        if kan_count(part) >= MIN_KAN:
            out_lines.append(part)

# Deduplicate while preserving order
seen = set()
deduped = []
for ln in out_lines:
    if ln not in seen:
        seen.add(ln)
        deduped.append(ln)

OUTPUT.write_text('\n'.join(deduped) + '\n', encoding='utf-8')
print(f"Input lines:   {len(raw)}")
print(f"Output lines:  {len(deduped)}")
print(f"Dropped:       {dropped}  (< {MIN_KAN} Kannada chars)")
print(f"Written to:    {OUTPUT}")
