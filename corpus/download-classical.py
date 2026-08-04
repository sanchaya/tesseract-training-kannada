#!/usr/bin/env python3
"""
download-classical.py
─────────────────────
Imports text from the classical Kannada corpus (Sanchaya
classical-corpus-kannada project) into corpus/raw_kannada.txt, making
it available alongside Wikipedia and Wikisource as a GT text source.

The classical corpus contains 16 pre-classical and classical Kannada
texts (Pampa, Ranna, Jaimini, etc.) — rich in archaic vocabulary and
conjunct forms absent from modern Wikipedia text.  This makes it
especially valuable for fine-tuning kan_hist (historical font training).

Usage
─────
    python3 corpus/download-classical.py \\
        --corpus-dir /path/to/classical-corpus-kannada

    # Dry-run (print stats without writing):
    python3 corpus/download-classical.py \\
        --corpus-dir /path/to/classical-corpus-kannada --dry-run

Output
──────
    corpus/raw_kannada.txt  (appended; safe to run alongside wiki/wikisource)
    Prints a per-title line count summary.

No external dependencies beyond Python stdlib.
"""

import argparse
import re
import sys
from pathlib import Path

CORPUS_DIR  = Path(__file__).parent
OUTPUT_FILE = CORPUS_DIR / "raw_kannada.txt"

# Minimum Kannada character ratio to accept a line
MIN_KAN_RATIO = 0.35

# Skip lines shorter than this many characters
MIN_LINE_LEN = 4


def is_kannada(c: str) -> bool:
    return 0x0C80 <= ord(c) <= 0x0CFF


def kan_ratio(text: str) -> float:
    if not text:
        return 0.0
    kan = sum(1 for c in text if is_kannada(c))
    return kan / len(text)


def clean_line(line: str) -> str:
    """
    Clean a single source line:
    - Strip whitespace
    - Collapse internal whitespace runs
    - Remove verse-number patterns like ।। ೧ ।।  or  || 1 ||
    """
    line = line.strip()
    # Remove common verse-number suffixes (classical Kannada poetry)
    line = re.sub(r'[।|]{2}[\s\d೦-೯]+[।|]{2}\s*$', '', line).rstrip()
    # Collapse multiple spaces
    line = re.sub(r'[ \t]+', ' ', line)
    return line


def load_title(txt_path: Path) -> list[str]:
    """
    Load and clean lines from one title's TXT file.
    Returns only lines that pass the Kannada-content filter.
    """
    raw = txt_path.read_text(encoding="utf-8", errors="replace")
    kept: list[str] = []
    for raw_line in raw.splitlines():
        line = clean_line(raw_line)
        if not line or len(line) < MIN_LINE_LEN:
            continue
        if kan_ratio(line) < MIN_KAN_RATIO:
            continue
        kept.append(line)
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import classical Kannada corpus text into raw_kannada.txt",
    )
    parser.add_argument(
        "--corpus-dir", required=True,
        help="Path to the classical-corpus-kannada folder",
    )
    parser.add_argument(
        "--output",
        help=f"Output file (default: {OUTPUT_FILE})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print stats only, do not write",
    )
    parser.add_argument(
        "--title", action="append", metavar="TITLE",
        help="Import only this title (repeatable; default: all)",
    )
    args = parser.parse_args()

    corpus_dir  = Path(args.corpus_dir).resolve()
    output_file = Path(args.output).resolve() if args.output else OUTPUT_FILE
    filter_titles = set(args.title) if args.title else None

    if not corpus_dir.is_dir():
        print(f"ERROR: corpus dir not found: {corpus_dir}")
        sys.exit(1)

    # ── Discover title folders ────────────────────────────────────────────
    titles: dict[str, Path] = {}
    for item in sorted(corpus_dir.iterdir()):
        if item.is_dir() and not item.name.startswith("."):
            if filter_titles and item.name not in filter_titles:
                continue
            txts = sorted(item.glob("*.txt"))
            if txts:
                titles[item.name] = txts[0]

    if not titles:
        print(f"ERROR: no title folders with .txt files found in {corpus_dir}")
        sys.exit(1)

    # ── Load and clean each title ─────────────────────────────────────────
    print("=" * 60)
    print("download-classical.py")
    print("=" * 60)
    print(f"Source  : {corpus_dir}  ({len(titles)} titles)")
    print(f"Output  : {output_file}")
    if args.dry_run:
        print("Mode    : DRY RUN — nothing will be written")
    print()

    all_lines: list[str] = []
    for title_name, txt_path in sorted(titles.items()):
        lines = load_title(txt_path)
        all_lines.extend(lines)
        size_kb = txt_path.stat().st_size // 1024
        print(f"  {title_name:<50}  {len(lines):>5} lines  ({size_kb} KB)")

    print()
    print(f"Total lines: {len(all_lines)}")
    total_chars = sum(len(l) for l in all_lines)
    print(f"Total chars: {total_chars:,}")

    if args.dry_run:
        print("\nDry run complete — nothing written.")
        return

    # ── Append to output file ─────────────────────────────────────────────
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "a", encoding="utf-8") as f:
        f.write("\n".join(all_lines))
        f.write("\n")

    existing_lines = sum(1 for _ in open(output_file, encoding="utf-8"))
    print(f"\nAppended to {output_file}")
    print(f"File now has {existing_lines:,} lines total.")
    print()
    print("Next: python3 corpus/clean-corpus.py  (dedup + shuffle)")


if __name__ == "__main__":
    main()
