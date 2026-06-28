#!/usr/bin/env python3
"""
generate-specimen.py

Generates a comprehensive Kannada training specimen corpus — a deliberately
designed text that guarantees every character, vowel-consonant combination,
and common conjunct appears in training images across all fonts.

This supplements (or replaces) the random Wikipedia/Wikisource corpus with
systematic glyph coverage. Feed the output directly into render-corpus.py.

Output: corpus/kan_specimen.txt  (can be used as-is or merged into kan_corpus.txt)

Usage:
    python3 corpus/generate-specimen.py
    python3 corpus/generate-specimen.py --merge      # append to kan_corpus.txt
    python3 corpus/generate-specimen.py --only       # replace kan_corpus.txt

Structure of the specimen:
    Block 1 — All vowels (standalone)
    Block 2 — All consonants (standalone)
    Block 3 — Every consonant × every vowel sign (matras)
    Block 4 — Common consonant clusters / conjuncts
    Block 5 — Kannada digits and punctuation
    Block 6 — Common historical words (mission press vocabulary)
    Block 7 — Real sentences — 19th-century style prose
    Block 8 — Digit strings and mixed number lines
"""

import argparse
import sys
from pathlib import Path

ROOT   = Path(__file__).parent.parent
OUTDIR = ROOT / "corpus"
OUTDIR.mkdir(exist_ok=True)
OUT    = OUTDIR / "kan_specimen.txt"
CORPUS = OUTDIR / "kan_corpus.txt"

# ── Kannada Unicode blocks ─────────────────────────────────────────────────

VOWELS = list("ಅಆಇಈಉಊಎಏಐಒಓಔಂಃ")

# All consonants (excluding ಋ ಙ ಝ ಞ ಱ — not in kan.traineddata unicharset)
CONSONANTS = list("ಕಖಗಘಚಛಜಟಠಡಢಣತಥದಧನಪಫಬಭಮಯರಲಳವಶಷಸಹ")

# Vowel signs (matras) — paired with consonants below
# U+0CBE..U+0CCC, virama U+0CCD
MATRAS = [
    'ಾ',  # ಾ  aa
    'ಿ',  # ಿ  i
    'ೀ',  # ೀ  ii
    'ು',  # ು  u
    'ೂ',  # ೂ  uu
    'ೆ',  # ೆ  e
    'ೇ',  # ೇ  ee
    'ೈ',  # ೈ  ai
    'ೊ',  # ೊ  o
    'ೋ',  # ೋ  oo
    'ೌ',  # ೌ  au
    '಼',  # ಼  nukta (rare but present)
    'ಾಂ',  # ಾಂ  aa + anusvara
]
VIRAMA = '್'  # ್

# Common conjuncts (consonant + virama + consonant) that appear in historical text
COMMON_CONJUNCTS = [
    'ಕ್ತ', 'ಕ್ಷ', 'ಗ್ನ', 'ಚ್ಚ', 'ಜ್ಞ', 'ತ್ತ', 'ತ್ರ', 'ದ್ದ',
    'ದ್ಧ', 'ನ್ನ', 'ನ್ತ', 'ನ್ದ', 'ಪ್ಪ', 'ಪ್ರ', 'ಬ್ಬ', 'ಭ್ರ',
    'ಮ್ಮ', 'ಯ್ಯ', 'ರ್ಕ', 'ರ್ತ', 'ರ್ಪ', 'ರ್ಮ', 'ಲ್ಲ', 'ವ್ವ',
    'ಸ್ತ', 'ಸ್ಥ', 'ಸ್ಪ', 'ಸ್ರ', 'ಸ್ಸ', 'ಹ್ನ', 'ಹ್ಮ', 'ಹ್ವ',
    'ಕ್ಕ', 'ಗ್ಗ', 'ಡ್ಡ', 'ಣ್ಣ', 'ತ್ನ', 'ದ್ಯ', 'ನ್ಯ', 'ರ್ವ',
    'ಶ್ರ', 'ಷ್ಟ', 'ಸ್ನ', 'ಳ್ಳ', 'ಕ್ರ', 'ಗ್ರ', 'ದ್ರ', 'ಭ್ನ',
    'ನ್ಮ', 'ರ್ನ', 'ವ್ರ', 'ಶ್ಚ', 'ಸ್ಕ', 'ಸ್ಮ', 'ಕ್ಲ', 'ಗ್ಲ',
    'ಪ್ಲ', 'ಬ್ಲ', 'ಫ್ಲ', 'ಮ್ಲ', 'ಹ್ಲ',
]

# Kannada digits
DIGITS = list('೦೧೨೩೪೫೬೭೮೯')

# Common words from 19th-century Kannada letterpress texts
# (German Mission Press, Wesleyan Mission Press, Basel Mission Press vocabulary)
HISTORICAL_WORDS = [
    # Religious / mission press vocabulary
    'ದೇವರು', 'ಮನುಷ್ಯನು', 'ಸತ್ಯವೇದ', 'ಪ್ರಭುವು', 'ಕ್ರಿಸ್ತನು',
    'ಪ್ರಾರ್ಥನೆ', 'ಸಭೆಯು', 'ಪವಿತ್ರ', 'ಆತ್ಮನು', 'ಜ್ಞಾನ',
    # Administrative / government
    'ರಾಜ್ಯ', 'ಸರ್ಕಾರ', 'ಪ್ರಜೆಗಳು', 'ನ್ಯಾಯ', 'ಕಾನೂನು',
    'ಅಧಿಕಾರ', 'ನ್ಯಾಯಾಲಯ', 'ತೆರಿಗೆ', 'ಗ್ರಾಮ', 'ಜಿಲ್ಲೆ',
    # Common nouns
    'ಮನೆ', 'ನೀರು', 'ಭೂಮಿ', 'ಆಕಾಶ', 'ಸೂರ್ಯ', 'ಚಂದ್ರ',
    'ಮಳೆ', 'ಗಾಳಿ', 'ಬೆಳಕು', 'ಕತ್ತಲು', 'ನದಿ', 'ಬೆಟ್ಟ',
    # Verbs / verbal nouns
    'ಹೋಗು', 'ಬಾ', 'ಮಾಡು', 'ಹೇಳು', 'ಕೇಳು', 'ನೋಡು',
    'ತಿಳಿ', 'ಕಲಿ', 'ಬರೆ', 'ಓದು', 'ತಿನ್ನು', 'ಕುಡಿ',
    # Kannada literature
    'ಕನ್ನಡ', 'ನಾಡು', 'ಭಾಷೆ', 'ಸಾಹಿತ್ಯ', 'ಕಾವ್ಯ', 'ಗ್ರಂಥ',
    'ಪದ್ಯ', 'ಗದ್ಯ', 'ಲೇಖಕ', 'ಪುಸ್ತಕ', 'ಮುದ್ರಣ', 'ಪ್ರಕಾಶ',
]

# Real sentences — 19th-century Kannada prose style
SENTENCES = [
    # Mission press / Bible style
    'ಆದಿಯಲ್ಲಿ ದೇವರು ಆಕಾಶವನ್ನೂ ಭೂಮಿಯನ್ನೂ ಸೃಷ್ಟಿಸಿದರು.',
    'ದೇವರು ಬೆಳಕನ್ನು ನೋಡಿ ಅದು ಒಳ್ಳೇದೆಂದು ಕಂಡರು.',
    'ಮನುಷ್ಯನು ದೇವರ ಸ್ವರೂಪದಲ್ಲಿ ಸೃಷ್ಟಿಸಲ್ಪಟ್ಟನು.',
    'ಸತ್ಯವೇದದ ಮಾತುಗಳು ಎಂದೆಂದಿಗೂ ನಿಲ್ಲುವವು.',
    'ಪ್ರಭುವಿನ ನಾಮವು ಸ್ತುತಿಸಲ್ಪಡಲಿ.',
    # Administrative / historical
    'ಮೈಸೂರು ರಾಜ್ಯದಲ್ಲಿ ಅನೇಕ ಜಿಲ್ಲೆಗಳಿವೆ.',
    'ಪ್ರಜೆಗಳು ರಾಜನಿಗೆ ತೆರಿಗೆ ಕೊಡಬೇಕು.',
    'ನ್ಯಾಯಾಲಯದಲ್ಲಿ ನ್ಯಾಯ ದೊರಕುವುದು.',
    'ಸರ್ಕಾರದ ಅಧಿಕಾರಿಗಳು ಗ್ರಾಮಗಳಿಗೆ ಬಂದರು.',
    'ಕಾನೂನನ್ನು ಎಲ್ಲರೂ ಪಾಲಿಸಬೇಕು.',
    # Nature / descriptive
    'ಆಕಾಶದಲ್ಲಿ ಸೂರ್ಯನು ಪ್ರಕಾಶಿಸುತ್ತಾನೆ.',
    'ನದಿಯ ನೀರು ಶುದ್ಧವಾಗಿ ಹರಿಯುತ್ತದೆ.',
    'ಮಳೆ ಬಿದ್ದ ನಂತರ ಗಾಳಿ ತಂಪಾಯಿತು.',
    'ಬೆಟ್ಟದ ಮೇಲೆ ದೊಡ್ಡ ಮರಗಳಿವೆ.',
    'ರಾತ್ರಿ ಚಂದ್ರ ಮತ್ತು ನಕ್ಷತ್ರಗಳು ಕಾಣಿಸುತ್ತವೆ.',
    # Kannada language / literature
    'ಕನ್ನಡ ಭಾಷೆ ಬಹಳ ಪ್ರಾಚೀನವಾದುದು.',
    'ಈ ಗ್ರಂಥವನ್ನು ಮಂಗಳೂರಿನ ಮುದ್ರಣಾಲಯದಲ್ಲಿ ಮುದ್ರಿಸಲಾಯಿತು.',
    'ಕವಿಗಳು ಕನ್ನಡ ಸಾಹಿತ್ಯವನ್ನು ಸಮೃದ್ಧಗೊಳಿಸಿದ್ದಾರೆ.',
    'ಈ ಪುಸ್ತಕವು ಶಾಲೆಯ ಮಕ್ಕಳಿಗಾಗಿ ಬರೆಯಲ್ಪಟ್ಟಿದೆ.',
    'ಕನ್ನಡ ನಾಡಿನ ಜನರು ಶ್ರದ್ಧಾಳುಗಳು.',
    # Mixed numbers and text (common in historical documents)
    'ಇಸ್ವಿ ೧೮೫೦ ನೇ ಸಾಲಿನಲ್ಲಿ ಈ ಗ್ರಂಥ ಪ್ರಕಟವಾಯಿತು.',
    '೧೨ ಜನ ಶಿಷ್ಯರು ಅಲ್ಲಿ ಇದ್ದರು.',
    'ಆ ಊರಿನಲ್ಲಿ ೫೦೦ ಮನೆಗಳಿದ್ದವು.',
    'ಅಧ್ಯಾಯ ೩ ರಿಂದ ೭ ರವರೆಗೆ ಓದಿರಿ.',
    # More historical sentences
    'ಜ್ಞಾನವು ದೇವರ ದಾನ.',
    'ಮನುಷ್ಯನ ಜೀವನ ಕ್ಷಣಿಕ.',
    'ದುಷ್ಟರ ಮಾರ್ಗವನ್ನು ಬಿಡಬೇಕು.',
    'ಸತ್ಯದ ಮಾರ್ಗದಲ್ಲಿ ನಡೆಯಿರಿ.',
    'ಧರ್ಮದ ಮಾರ್ಗ ಶ್ರೇಷ್ಠ.',
    'ಪ್ರೀತಿ ಸರ್ವಕ್ಕಿಂತ ದೊಡ್ಡದು.',
    'ವಿದ್ಯೆಯು ಎಲ್ಲ ಸಂಪತ್ತಿಗಿಂತ ಮಿಗಿಲು.',
    'ಕರ್ಮವೇ ಧರ್ಮ ಎಂದು ಹಿರಿಯರು ಹೇಳಿದ್ದಾರೆ.',
    'ಆ ದಿನಗಳಲ್ಲಿ ಕನ್ನಡ ನಾಡು ಸಮೃದ್ಧವಾಗಿತ್ತು.',
    'ಮಿಷನರಿಗಳು ಶಾಲೆಗಳನ್ನು ಸ್ಥಾಪಿಸಿದರು.',
    'ಈ ಪ್ರದೇಶದ ಭಾಷೆ ಕನ್ನಡ.',
    'ಅವರು ಬಹಳ ಪ್ರಯಾಸಪಟ್ಟು ಕೆಲಸ ಮಾಡಿದರು.',
    'ನಮ್ಮ ದೇಶ ಸಸ್ಯ ಸಮೃದ್ಧ.',
    'ಭೂಮಿ ಸುತ್ತಲೂ ಆಕಾಶ ಹರಡಿದೆ.',
]

# Longer paragraph-style lines (simulate full prose lines as seen in letterpress books)
PARAGRAPHS = [
    'ಈ ಪ್ರಪಂಚದಲ್ಲಿ ಅನೇಕ ಜಾತಿಯ ಮನುಷ್ಯರು ವಾಸಿಸುತ್ತಾರೆ.',
    'ಪ್ರತಿಯೊಬ್ಬ ಮನುಷ್ಯನೂ ದೇವರ ಸ್ವರೂಪದಲ್ಲಿ ಸೃಷ್ಟಿಸಲ್ಪಟ್ಟಿದ್ದಾನೆ.',
    'ಕನ್ನಡ ನಾಡಿನ ಇತಿಹಾಸ ಬಹಳ ಹಿಂದಕ್ಕೆ ಹೋಗುತ್ತದೆ.',
    'ಮಂಗಳೂರಿನ ಬಾಸಲ್ ಮಿಷನ್ ಮುದ್ರಣಾಲಯ ಕನ್ನಡ ಪುಸ್ತಕಗಳನ್ನು ಪ್ರಕಟಿಸಿತು.',
    'ಜರ್ಮನ್ ಮಿಷನ್ ಮುದ್ರಣಾಲಯ ಕ್ರಿಶ್ಚಿಯನ್ ಧಾರ್ಮಿಕ ಗ್ರಂಥಗಳನ್ನು ಮುದ್ರಿಸಿತು.',
    'ವೆಸ್ಲಿಯನ್ ಮಿಷನ್ ಪ್ರೆಸ್ ಬೆಂಗಳೂರಿನಲ್ಲಿ ಸ್ಥಾಪಿತವಾಯಿತು.',
    'ಫ಼ರ್ಡಿನಾಂಡ್ ಕಿಟ್ಟೆಲ್ ಕನ್ನಡ ಭಾಷೆಗೆ ಶ್ರೇಷ್ಠ ಸೇವೆ ಸಲ್ಲಿಸಿದರು.',
    'ಕಿಟ್ಟೆಲ್ ಅವರು ಕನ್ನಡ ಇಂಗ್ಲಿಷ್ ನಿಘಂಟನ್ನು ರಚಿಸಿದರು.',
    'ಆ ಕಾಲದ ಕನ್ನಡ ಮುದ್ರಣ ಇಂದಿಗೂ ಮಹತ್ವದ್ದಾಗಿದೆ.',
    'ಅಕ್ಷರ ಜ್ಞಾನ ಎಲ್ಲ ಜ್ಞಾನದ ತಳಹದಿ.',
    'ಶಿಕ್ಷಣ ಪ್ರತಿಯೊಬ್ಬರ ಹಕ್ಕು ಮತ್ತು ಕರ್ತವ್ಯ.',
    'ಪ್ರಾಚೀನ ಕನ್ನಡ ಲಿಪಿ ಕ್ರಮೇಣ ಬೆಳೆದು ಇಂದಿನ ರೂಪ ಪಡೆದಿದೆ.',
    'ಕರ್ನಾಟಕ ರಾಜ್ಯ ದಕ್ಷಿಣ ಭಾರತದಲ್ಲಿ ಇದೆ.',
    'ಕನ್ನಡ ಭಾಷೆಗೆ ಶಾಸ್ತ್ರೀಯ ಭಾಷೆಯ ಮನ್ನಣೆ ಲಭಿಸಿದೆ.',
    'ಹಿಂದಿನ ಕಾಲದ ಕನ್ನಡ ಸಾಹಿತ್ಯ ಅತ್ಯಂತ ಶ್ರೀಮಂತ.',
]


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def build_specimen():
    lines = []

    def section(title):
        lines.append(f"# {title}")

    # ── Block 1: Vowels ────────────────────────────────────────────
    section("Standalone vowels")
    lines.append(' '.join(VOWELS))
    # Groups of 4–5
    for chunk in chunks(VOWELS, 5):
        lines.append(' '.join(chunk))

    # ── Block 2: Consonants ────────────────────────────────────────
    section("Standalone consonants")
    lines.append(' '.join(CONSONANTS))
    for chunk in chunks(CONSONANTS, 8):
        lines.append(' '.join(chunk))

    # ── Block 3: Consonant × Matra grid ───────────────────────────
    section("Consonant + vowel sign combinations")
    # Each matra applied to all consonants — one line per matra
    for matra in MATRAS:
        row = ' '.join(c + matra for c in CONSONANTS)
        lines.append(row)
        # Also shorter chunks so lines aren't too long
        for chunk in chunks(CONSONANTS, 10):
            lines.append(' '.join(c + matra for c in chunk))

    # Each consonant with all matras — one line per consonant
    section("Each consonant with all matras")
    for c in CONSONANTS:
        row = c + '  ' + ' '.join(c + m for m in MATRAS[:8])
        lines.append(row)

    # ── Block 4: Virama / half-consonant forms ─────────────────────
    section("Virama (half-consonant) forms")
    for chunk in chunks(CONSONANTS, 8):
        lines.append(' '.join(c + VIRAMA for c in chunk))

    # ── Block 5: Conjuncts ─────────────────────────────────────────
    section("Common conjunct consonants")
    lines.append(' '.join(COMMON_CONJUNCTS[:20]))
    lines.append(' '.join(COMMON_CONJUNCTS[20:40]))
    lines.append(' '.join(COMMON_CONJUNCTS[40:]))
    # Conjuncts with vowel signs
    for matra in ['ಾ', 'ಿ', 'ು', 'ೆ', 'ೊ']:
        row = ' '.join(c + matra for c in COMMON_CONJUNCTS[:15])
        lines.append(row)
    # Conjuncts in short words
    section("Conjuncts in syllables")
    for c in COMMON_CONJUNCTS:
        lines.append(c + 'ನ ' + c + 'ರ ' + c + 'ಲ ' + c + 'ವ ' + c + 'ಕ')

    # ── Block 6: Digits ────────────────────────────────────────────
    section("Kannada digits")
    lines.append(' '.join(DIGITS))
    lines.append(''.join(DIGITS))
    # Digit sequences
    for i in range(0, 100, 11):
        lines.append(''.join(DIGITS[int(d)] for d in str(i).zfill(2)))
    lines.append('೧೦ ೨೦ ೩೦ ೪೦ ೫೦ ೬೦ ೭೦ ೮೦ ೯೦ ೧೦೦')
    lines.append('೧೮೩೦ ೧೮೫೦ ೧೮೭೦ ೧೮೯೦ ೧೯೦೦ ೧೯೧೦ ೧೯೨೦')
    lines.append('೧ ೨ ೩ ೪ ೫ ೬ ೭ ೮ ೯ ೧೦ ೧೧ ೧೨')

    # ── Block 7: Words ─────────────────────────────────────────────
    section("Historical vocabulary")
    for chunk in chunks(HISTORICAL_WORDS, 6):
        lines.append(' '.join(chunk))

    # ── Block 8: Sentences ─────────────────────────────────────────
    section("Sentences — 19th century style")
    lines.extend(SENTENCES)

    # ── Block 9: Paragraphs ────────────────────────────────────────
    section("Paragraph-length lines")
    lines.extend(PARAGRAPHS)

    # ── Block 10: Mixed consonant+matra dense lines ────────────────
    # These simulate the kind of dense text seen on a letterpress page
    section("Dense consonant+matra sequences")
    all_sylls = [c + m for c in CONSONANTS for m in MATRAS[:6]]
    import random
    random.seed(7)
    random.shuffle(all_sylls)
    for chunk in chunks(all_sylls, 16):
        lines.append(' '.join(chunk))

    return lines


def clean_lines(lines):
    """Strip comment lines and blank lines, normalise whitespace."""
    out = []
    UNSUPPORTED = set('ಋಙಝಞಱ')
    for line in lines:
        if line.startswith('#'):
            continue
        tokens = [t for t in line.split() if not (len(t) == 1 and t in UNSUPPORTED)]
        clean = ' '.join(tokens).strip()
        if clean:
            out.append(clean)
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate Kannada OCR specimen corpus")
    parser.add_argument('--merge', action='store_true',
                        help='Append specimen to existing kan_corpus.txt')
    parser.add_argument('--only',  action='store_true',
                        help='Write specimen as kan_corpus.txt (replaces existing)')
    args = parser.parse_args()

    raw   = build_specimen()
    lines = clean_lines(raw)

    # Write specimen file
    OUT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f"✓  Specimen written: {OUT}  ({len(lines)} lines)")

    if args.merge:
        existing = CORPUS.read_text(encoding='utf-8').splitlines() if CORPUS.exists() else []
        merged   = existing + lines
        # Deduplicate while preserving order
        seen, uniq = set(), []
        for l in merged:
            if l not in seen:
                seen.add(l)
                uniq.append(l)
        CORPUS.write_text('\n'.join(uniq) + '\n', encoding='utf-8')
        print(f"✓  Merged into {CORPUS}  ({len(uniq)} lines total)")

    elif args.only:
        CORPUS.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        print(f"✓  Written as {CORPUS}  ({len(lines)} lines)")

    else:
        print(f"\nTo use:")
        print(f"  Append to corpus:   python3 corpus/generate-specimen.py --merge")
        print(f"  Replace corpus:     python3 corpus/generate-specimen.py --only")
        print(f"  Then render:        python3 corpus/render-corpus.py")
        print(f"  Then make lstmf:    ./scripts/02-make-lstmf.sh")


if __name__ == '__main__':
    main()
