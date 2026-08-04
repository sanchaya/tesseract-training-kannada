#!/usr/bin/env python3
"""
clean-classical-corpus.py

Strips non-Kannada training noise from classical corpus source files
and the rendered a5-pages gt.txt files.

Cleans:
  • XML/HTML tags: <ch>ಐದನೆಯ ಸಂಧಿ</ch>, <p>…</p>, etc.
  • Devanagari dandas: ।  ॥  (U+0964/0965)
  • Devanagari digits: ०-९  (U+0966–U+096F)
  • Editorial annotations: (ಕ)  (ಚ)  single Kannada char in parens
  • Verse-divider +
  • Stray quotes: '  "  ʼ
  • Fixes ೦→ಂ  (Kannada digit-zero mis-keyed as anusvara)

Only writes files where actual tag/danda/annotation content was removed.
Whitespace-only differences are never counted as changes.

Two passes:
  1. Source corpus .txt files  (classical-corpus-kannada/<title>/*.txt)
  2. Rendered gt.txt files      (classical-corpus-kannada/a5-pages/**/*.gt.txt)

Usage:
  python3 corpus/clean-classical-corpus.py [--dry-run] [--source-only] [--gt-only]

Options:
  --dry-run      Show what would change without writing files
  --source-only  Only clean source .txt files
  --gt-only      Only clean a5-pages .gt.txt files
"""

import argparse
import re
import sys
from pathlib import Path

# ── Cleaning rules ─────────────────────────────────────────────────────────────
# Pattern that matches anything we want to remove (used to detect if cleaning needed)
_NOISE = re.compile(
    r'<[^>]+>'            # XML/HTML tags
    r'|[।॥]'              # Devanagari dandas U+0964/U+0965
    r'|[०-९]'    # Devanagari digits
    r'|\([ಀ-೿]\)'         # editorial (X) annotations
    r'|\+'                # verse-divider +
    r"|['\"￼ʼ]" # stray quotes / obj replacement char
    r'|೦(?=[ಂ-ೞ]|$)'    # ೦ only when it stands in for ಂ (followed by Kannada or EOL)
)

def _apply(text: str) -> str:
    t = text
    t = re.sub(r'೦\b', 'ಂ', t)           # digit-zero → anusvara (word boundary)
    t = t.replace('೦', 'ಂ')              # remaining digit-zeros
    t = re.sub(r'<[^>]+>', '', t)         # strip XML/HTML tags
    t = re.sub(r'[।॥]', '', t)           # Devanagari dandas
    t = re.sub(r'[०-९]', '', t)  # Devanagari digits
    t = re.sub(r'\([ಀ-೿]\)', '', t)      # editorial (X) annotations
    t = t.replace('+', '')                # verse-divider +
    t = re.sub(r'[\'\"ʼ]', '', t)        # stray quotes
    return t


def needs_cleaning(text: str) -> bool:
    """Return True only if text contains actual noise patterns."""
    return bool(re.search(
        r'<[^>]+>'           # XML/HTML tags
        r'|[।॥]'             # Devanagari dandas
        r'|[०-९]'   # Devanagari digits
        r'|\([ಀ-೿]\)'        # editorial annotations
        r'|\+'               # verse-divider
        r"|['\"￼]"      # stray quotes
        , text
    ))


def clean_source_file(raw: str) -> str:
    """
    Clean a multi-line source .txt file.
    Cleans content per-line, preserves blank lines and line structure.
    """
    lines = raw.splitlines(keepends=True)
    result = []
    for line in lines:
        eol = '\n' if line.endswith('\n') else ''
        cleaned = re.sub(r'[ \t]+', ' ', _apply(line.rstrip('\n'))).strip()
        result.append(cleaned + eol if cleaned else ('' + eol))
    return ''.join(result)


def clean_gt_file(raw: str) -> str:
    """
    Clean a single-line gt.txt (one page of text on one line).
    Preserves a single trailing newline.
    """
    t = _apply(raw.strip())
    t = re.sub(r'[ \t]+', ' ', t).strip()
    return t + '\n'


# ── File processing ────────────────────────────────────────────────────────────

def process_file(path: Path, cleaner, dry_run: bool) -> tuple[bool, str]:
    """
    Returns (changed, reason).
    Only marks changed if actual noise was removed, not whitespace differences.
    """
    try:
        original = path.read_text(encoding='utf-8')
    except Exception as e:
        return False, f"READ ERROR: {e}"

    if not needs_cleaning(original):
        return False, 'clean'

    cleaned = cleaner(original)
    if cleaned == original:
        return False, 'clean'

    if not dry_run:
        path.write_text(cleaned, encoding='utf-8')

    # Build a short diff for display
    orig_lines = original.splitlines()
    new_lines  = cleaned.splitlines()
    for i, (o, n) in enumerate(zip(orig_lines, new_lines)):
        if o != n:
            return True, (f"line {i+1}:\n"
                          f"    - {o[:100]}\n"
                          f"    + {n[:100]}")
    return True, f"({len(orig_lines)} lines cleaned)"


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dry-run', action='store_true',
                        help='Show changes without writing files')
    parser.add_argument('--source-only', action='store_true',
                        help='Only clean source corpus .txt files')
    parser.add_argument('--gt-only', action='store_true',
                        help='Only clean a5-pages .gt.txt files')
    args = parser.parse_args()

    root = Path(__file__).parent.parent / 'classical-corpus-kannada'
    if not root.exists():
        print(f"ERROR: {root} not found — run from project root or check path.")
        sys.exit(1)

    mode = '(DRY RUN) ' if args.dry_run else ''
    total_changed = 0

    # ── Pass 1: source .txt files ──────────────────────────────────────────────
    if not args.gt_only:
        print(f"\n{'━'*60}")
        print(f"  {mode}Pass 1 — source corpus .txt files")
        print(f"{'━'*60}")
        source_files = sorted(
            p for p in root.rglob('*.txt')
            if 'a5-pages' not in p.parts
        )
        changed = skipped = 0
        for f in source_files:
            ok, info = process_file(f, clean_source_file, args.dry_run)
            if ok:
                rel = f.relative_to(root)
                verb = '[would change]' if args.dry_run else '✓'
                print(f"  {verb} {rel}")
                print(f"    {info}")
                changed += 1
            else:
                skipped += 1
        print(f"\n  {changed} files {'would be ' if args.dry_run else ''}changed, "
              f"{skipped} already clean")
        total_changed += changed

    # ── Pass 2: a5-pages gt.txt files ─────────────────────────────────────────
    if not args.source_only:
        print(f"\n{'━'*60}")
        print(f"  {mode}Pass 2 — a5-pages gt.txt files")
        print(f"{'━'*60}")
        a5_dir = root / 'a5-pages'
        if not a5_dir.exists():
            print(f"  NOTE: {a5_dir} not found — run render-a5-pages.py first")
        else:
            gt_files = sorted(a5_dir.rglob('*.gt.txt'))
            changed = skipped = shown = 0
            for f in gt_files:
                ok, info = process_file(f, clean_gt_file, args.dry_run)
                if ok:
                    rel = f.relative_to(root / 'a5-pages')
                    verb = '[would change]' if args.dry_run else '✓'
                    if shown < 15 or changed % 200 == 0:
                        print(f"  {verb} {rel}")
                        print(f"    {info}")
                        shown += 1
                    changed += 1
                else:
                    skipped += 1
            if changed > shown:
                print(f"  … and {changed - shown} more")
            print(f"\n  {changed} files {'would be ' if args.dry_run else ''}changed, "
                  f"{skipped} already clean")
            total_changed += changed

    print(f"\n{'━'*60}")
    if args.dry_run:
        print(f"  DRY RUN complete — {total_changed} files would be changed.")
        print(f"  Re-run without --dry-run to apply.")
    else:
        print(f"  Done — {total_changed} files cleaned.")
        if total_changed > 0:
            print(f"\n  Next steps:")
            print(f"    rm -rf lstmf/classical/")
            print(f"    CLASSICAL_A5_DIR=classical-corpus-kannada/a5-pages \\")
            print(f"        ./scripts/02-make-lstmf.sh")


if __name__ == '__main__':
    main()
