#!/usr/bin/env python3
"""
find-missing-clusters.py — find every grapheme cluster the unicharset cannot
                           encode, and emit real words containing them

WHY THIS EXISTS
    Tesseract's LSTM unicharset stores whole grapheme clusters as single units:
    ರ್ಘ, ಖ್ಯ, ಞ್ಝ — not a bare virama plus a consonant. A word containing a
    cluster with no unit cannot be encoded at all, and lstmtraining discards the
    entire line:

        Encoding of string failed! Failure bytes: e0 b3 8d e0 b2 96
        Can't encode transcription: 'ಘನಕರುಣೆ ನಿಮ್ಮ ಧ್ಯಾನವ ಮಾಡದವ ಮೂರ್ಖ'

    Adding units by hand does not work: you have to guess which clusters appear
    in the corpus, and a wrong guess is silent. Supplying ಮುಖ್ಯ to add ್ಖ
    produces the unit ಖ್ಯ instead — the virama binds to what FOLLOWS it, so the
    cluster ರ್ಖ in ಮೂರ್ಖ is still missing.

    This script inverts the problem: it reads the corpus, finds what actually
    fails, and emits real words as evidence.

USAGE
    python3 scripts/find-missing-clusters.py                  # report
    python3 scripts/find-missing-clusters.py --emit words.txt # write word list
    python3 scripts/find-missing-clusters.py --update-00c     # patch 00c script

    Then re-run:  ./scripts/00c-expand-unicharset.sh --force
"""

import argparse
import collections
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent


def load_units():
    """Grapheme-cluster units from the traineddata training will actually use."""
    mode = ROOT / 'output' / '.tessdata_mode'
    expanded = mode.exists() and 'expanded' in mode.read_text(errors='ignore')
    td = ROOT / ('tessdata_expanded' if expanded else 'tessdata_best') / 'kan.traineddata'
    if not td.exists():
        sys.exit(f"ERROR: {td} not found")
    with tempfile.TemporaryDirectory() as tmp:
        prefix = str(Path(tmp) / 'kan.')
        subprocess.run(['combine_tessdata', '-u', str(td), prefix],
                       capture_output=True, check=True)
        uc = Path(prefix + 'lstm-unicharset')
        if not uc.exists():
            sys.exit("ERROR: could not extract lstm-unicharset")
        lines = uc.read_text(encoding='utf-8', errors='replace').split('\n')
    return {l.split(' ')[0] for l in lines[1:] if l.strip()}, td


VIRAMA = '್'
ZWNJ   = '‌'


def clean(t):
    """The word-final virama rule 02-make-lstmf.sh applies before encoding."""
    return re.sub(r'್(?![ಕ-ಹೞೠೡ‌])', '್‌', t)


def is_kannada(c):
    return 0x0C80 <= ord(c) <= 0x0CFF


def first_failure(word, units, maxu):
    """
    Return the grapheme cluster that cannot be encoded, or None.

    Reports the FULL cluster (ರ್ಖ), not just the character the walk stopped on
    (್). The unicharset stores whole clusters, so ್ alone is not actionable —
    you cannot add it, and it does not tell you which word to supply.
    """
    i, n = 0, len(word)
    while i < n:
        matched = False
        for size in range(min(maxu, n - i), 0, -1):
            if word[i:i + size] in units:
                i += size
                matched = True
                break
        if matched:
            continue
        if word[i] in ' \t\n':
            i += 1
            continue

        # Walk back to the start of the cluster: a virama binds the consonant
        # before it to the one after, so ರ + ್ + ಖ is a single unit ರ್ಖ.
        start = i
        if start > 0 and word[start - 1] == VIRAMA:
            start -= 1
        if start > 0 and is_kannada(word[start - 1]):
            start -= 1
        # Extend forward over any following virama chains and vowel signs.
        end = i + 1
        while end < n and (word[end] == VIRAMA or
                           (end > 0 and word[end - 1] == VIRAMA) or
                           (is_kannada(word[end]) and not word[end].isspace()
                            and 0x0CBE <= ord(word[end]) <= 0x0CCC)):
            end += 1
        return word[start:min(end, n)]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--emit', help='write the sample words to this file')
    ap.add_argument('--update-00c', action='store_true',
                    help='append the words to scripts/00c-expand-unicharset.sh')
    ap.add_argument('--limit', type=int, default=40, help='max clusters to report')
    ap.add_argument('--kannada-only', action='store_true',
                    help='ignore clusters containing non-Kannada characters — those are '
                         'digitisation noise to be cleaned, not units to add')
    args = ap.parse_args()

    units, td = load_units()
    maxu = max(len(u) for u in units)
    print(f"Unicharset: {len(units)} units from {td.relative_to(ROOT)}\n")

    # (label, path, glob) — glob is None for a single file.
    #
    # Scan ground truth is mined too, and it matters more than its size
    # suggests. Real scans are transcribed by hand from historical print, so
    # they are the likeliest source of a conjunct no synthetic corpus contains.
    # If a cluster only ever appears in a scan and is not mined here, the
    # unicharset never learns it, and 02-make-lstmf.sh then drops that line at
    # the encodability guard — quietly, as one more 'failed' in a count of
    # thousands. Losing hand-transcribed lines that way is the most expensive
    # kind of silent failure this project has.
    #
    # scan-holdout/ is mined as well: the unicharset must be able to REPRESENT
    # held-out text even though the model never trains on it, or evaluation
    # scores a vocabulary gap as a recognition error.
    sources = [
        ('corpus',       ROOT / 'corpus' / 'kan_corpus.txt',   None),
        ('classical',    ROOT / 'classical-corpus-kannada',    '*/*.txt'),
        ('scan-input',   ROOT / 'scan-input',                  '*.gt.txt'),
        ('scan-holdout', ROOT / 'scan-holdout',                '*.gt.txt'),
        ('scan-lines',   ROOT / 'scan-lines',                  '*/*.gt.txt'),
    ]

    missing = collections.Counter()
    example = {}

    def scan_text(text):
        for word in text.split():
            w = clean(word)
            bad = first_failure(w, units, maxu)
            if bad:
                missing[bad] += 1
                example.setdefault(bad, word)

    for label, path, pattern in sources:
        if not path.exists():
            continue
        if path.is_file():
            scan_text(path.read_text(encoding='utf-8', errors='ignore'))
            print(f"  scanned {label}")
        else:
            files = sorted(path.glob(pattern))
            for txt in files:
                scan_text(txt.read_text(encoding='utf-8', errors='ignore'))
            print(f"  scanned {label}  ({len(files)} file(s))")

    if not missing:
        print("\n✓ Every cluster in the corpus is encodable — nothing to add.")
        return 0

    if args.kannada_only:
        before = len(missing)
        missing = collections.Counter({
            k: v for k, v in missing.items()
            if k and all(is_kannada(c) or c == VIRAMA for c in k)
        })
        print(f"\n  (filtered {before - len(missing)} non-Kannada cluster(s) — "
              f"digitisation noise for the corpus cleaner, not units to add)")

    if not missing:
        print("\n✓ No Kannada clusters missing.")
        return 0

    print(f"\n{len(missing)} cluster(s) cannot be encoded:\n")
    print(f"  {'cluster':10s} {'occurrences':>12s}   example word")
    words = []
    for cluster, count in missing.most_common(args.limit):
        w = example[cluster]
        words.append(w)
        print(f"  {cluster:10s} {count:>12d}   {w}")

    total = sum(missing.values())
    print(f"\n  {total} word occurrences affected — every LINE containing one is discarded.")

    if args.emit:
        Path(args.emit).write_text(' '.join(words) + '\n', encoding='utf-8')
        print(f"\n✓ words written to {args.emit}")

    if args.update_00c:
        script = ROOT / 'scripts' / '00c-expand-unicharset.sh'
        src = script.read_text(encoding='utf-8')
        marker = '\nEOF\n'
        idx = src.find(marker)
        if idx == -1:
            print("✗ could not find the corpus heredoc in 00c-expand-unicharset.sh")
            return 1
        # Words must appear in CONTEXT — unicharset_extractor rejects bare
        # combining sequences and drops the whole line with them.
        block = '\n'.join(' '.join(words[i:i + 8]) for i in range(0, len(words), 8))
        src = src[:idx] + '\n' + block + src[idx:]
        script.write_text(src, encoding='utf-8')
        print(f"\n✓ appended {len(words)} words to {script.relative_to(ROOT)}")
        print("  Now run:  ./scripts/00c-expand-unicharset.sh --force")

    return 0


if __name__ == '__main__':
    sys.exit(main())
