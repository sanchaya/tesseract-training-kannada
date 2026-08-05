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

# ── Unicharset encodability ──────────────────────────────────────────────────
# Stripping unassigned codepoints is not enough. A line like
#     ಹ್ಕ ಹ್ಖ ಹ್ಗ … ಹ್಩ … ಹ್಴
# (a systematic conjunct grid left in raw_kannada.txt by an early specimen
# generator) still fails to encode after the reserved characters are removed,
# because several of the remaining clusters have no unicharset unit either.
# lstmtraining rejects the whole line — these produced a 60% skip ratio.
#
# So: after cleaning, verify the line is actually encodable and drop it if not.
# Uses the same greedy longest-match segmentation Tesseract's encoder uses.
def _load_units():
    """Unicharset units from the traineddata that will be used for training."""
    root = CORPUS_DIR.parent
    mode = root / 'output' / '.tessdata_mode'
    expanded = mode.exists() and 'expanded' in mode.read_text(errors='ignore')
    td = root / ('tessdata_expanded' if expanded else 'tessdata_best') / 'kan.traineddata'
    if not td.exists():
        return None
    import subprocess, tempfile
    try:
        with tempfile.TemporaryDirectory() as tmp:
            prefix = str(Path(tmp) / 'kan.')
            subprocess.run(['combine_tessdata', '-u', str(td), prefix],
                           capture_output=True, check=True)
            uc = Path(prefix + 'lstm-unicharset')
            if not uc.exists():
                return None
            lines = uc.read_text(encoding='utf-8', errors='replace').split('\n')
            return {l.split(' ')[0] for l in lines[1:] if l.strip()} or None
    except Exception:
        return None

_UNITS = _load_units()
_MAXU  = max((len(u) for u in _UNITS), default=1) if _UNITS else 1
# Only whitespace is exempt. The virama is NOT: it always belongs to a cluster
# unit (್ನ, ್ಯ, or the half-form ್‌), so exempting it would make this check more
# permissive than Tesseract's encoder and let unencodable lines through.
_EXEMPT = set(' \t\n')

def apply_wordfinal_virama(text: str) -> str:
    """
    Append ZWNJ to a word-final virama, exactly as 02-make-lstmf.sh's _clean_gt()
    does before building the training pair.

    Without this, clean-corpus is STRICTER than the pipeline it feeds. A bare ್
    has no unicharset unit, so any line ending a word in a consonant + virama
    (ರಾವ್, ಕನ್, ಸ್ — very common in Kannada) failed the encodability check and
    was discarded here, even though the lstmf stage would have encoded it fine
    via the ್‌ half-form unit. That single mismatch accounted for 1,112 of the
    1,689 lines this script was dropping.

    The corpus is written WITHOUT the ZWNJ — the transform belongs to the
    training pair, not to the corpus text — but the CHECK has to model what the
    pipeline will actually do.
    """
    return re.sub(r'್(?![ಕ-ಹೞೠೡ‌])', '್‌', text)


def encodable(text: str) -> bool:
    """
    Unit matching is tried BEFORE the exemption: ್‌ (virama + ZWNJ) is a real
    two-character unit, and skipping the virama as exempt first meant it was
    never tried — the encoder then failed on the bare ZWNJ and rejected the line.
    """
    if not _UNITS:
        return True                      # unicharset unavailable — don't drop
    i, n = 0, len(text)
    while i < n:
        matched = False
        for size in range(min(_MAXU, n - i), 0, -1):
            if text[i:i + size] in _UNITS:
                i += size
                matched = True
                break
        if matched:
            continue
        if text[i] in _EXEMPT:           # fallback only when no unit matched
            i += 1
            continue
        return False
    return True

raw = INPUT.read_text(encoding='utf-8').splitlines()
out_lines, dropped, unencodable = [], 0, 0

for line in raw:
    cleaned = clean_line(line)
    if kan_count(cleaned) < MIN_KAN:
        if cleaned.strip():
            dropped += 1
        continue
    for part in split_at_boundary(cleaned, MAX_CHARS):
        if kan_count(part) < MIN_KAN:
            continue
        # Check what the PIPELINE will encode, not the raw text: the lstmf
        # stage appends ZWNJ to word-final viramas before encoding.
        if not encodable(apply_wordfinal_virama(part)):
            unencodable += 1
            continue
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
print(f"Unencodable:   {unencodable}  (rejected by unicharset check"
      f"{'' if _UNITS else ' — SKIPPED, unicharset unavailable'})")
print(f"Written to:    {OUTPUT}")
