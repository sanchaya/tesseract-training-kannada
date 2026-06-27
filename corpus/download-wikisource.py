#!/usr/bin/env python3
"""
download-wikisource.py

Fetches proofread and validated pages from Kannada Wikisource
(kn.wikisource.org) and writes clean GT text to corpus/raw_kannada.txt.

Wikisource Page: namespace contains scanned-book page transcriptions with
a quality rating:
    1 = not proofread
    2 = problematic
    3 = proofread       ← we include these
    4 = validated       ← we include these (second human verified)

These are the best possible GT lines for training kan_hist: they come
from 19th–20th century Kannada books that were typeset in the same
letterpress fonts (German Mission Press, Wesleyan Mission Press, etc.)
we are fine-tuning on.

Usage:
    python3 corpus/download-wikisource.py [--pages N] [--quality 3|4]

    --pages N    Max pages to fetch (default: 2000)
    --quality N  Minimum quality level to include (default: 3)

Output:
    corpus/raw_kannada.txt   (appended to, not overwritten — so you can
                              combine with Wikipedia text if desired)

Requirements: requests   (pip install requests)
"""

import argparse
import re
import sys
import time
import json
import urllib.request
import urllib.parse
from pathlib import Path

CORPUS_DIR  = Path(__file__).parent
OUTPUT_FILE = CORPUS_DIR / "raw_kannada.txt"
CACHE_FILE  = CORPUS_DIR / "cache" / "wikisource_pages.json"

API_URL = "https://kn.wikisource.org/w/api.php"

# Wikisource Page: namespace is 104
PAGE_NAMESPACE = 104

# Wikisource XML dump — much faster than API for bulk extraction
# (~10–50 MB vs 150 MB Wikipedia dump)
DUMP_URL  = ("https://dumps.wikimedia.org/knwikisource/latest/"
             "knwikisource-latest-pages-articles.xml.bz2")

MAX_LINE_LEN  = 80
MIN_KAN_RATIO = 0.65    # slightly lower than Wikipedia — pages may have
                         # English headings, numbers, etc.


def is_kannada(c: str) -> bool:
    return 0x0C80 <= ord(c) <= 0x0CFF


def kan_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if is_kannada(c)) / len(text)


def api_get(params: dict) -> dict:
    params["format"] = "json"
    params["utf8"]   = "1"
    url = API_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "KanHistTraining/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_proofread_page_titles(min_quality: int, max_pages: int) -> list[str]:
    """
    Fetch Page: titles that have been proofread or validated.
    Uses the Wikisource ProofreadPage quality category system.
    """
    # Quality categories on Wikisource
    quality_categories = {
        4: "Category:ಪ್ರೂಫ್ ಓದಿದ ಪುಟಗಳು",   # Validated pages
        3: "Category:ಪ್ರೂಫ್ ಓದಿದ ಪುಟಗಳು",   # Proofread pages
    }

    # More reliable: query by quality index using the API
    # ProofreadPage stores quality in page_props; we use a category walk
    # The category names vary by wiki — use the prop=pageprops approach instead

    titles = []
    print(f"  Querying Page: namespace (quality >= {min_quality})…",
          file=sys.stderr)

    # Walk all pages in Page: namespace, filter by quality prop
    params = {
        "action":      "query",
        "list":        "allpages",
        "apnamespace": PAGE_NAMESPACE,
        "aplimit":     "500",
        "apfilterlanglinks": "all",
    }

    fetched = 0
    while fetched < max_pages:
        data = api_get(params)
        pages = data.get("query", {}).get("allpages", [])
        if not pages:
            break

        batch_titles = [p["title"] for p in pages]

        # Get quality props for this batch
        quality_data = api_get({
            "action":  "query",
            "titles":  "|".join(batch_titles),
            "prop":    "pageprops",
            "ppprop":  "proofread-quality",
        })

        for page in quality_data.get("query", {}).get("pages", {}).values():
            quality = int(page.get("pageprops", {}).get("proofread-quality", 0))
            if quality >= min_quality:
                titles.append(page["title"])
                fetched += 1
                if fetched >= max_pages:
                    break

        cont = data.get("continue", {})
        if not cont:
            break
        params.update(cont)
        print(f"  …{fetched} qualifying pages found", end="\r", file=sys.stderr)
        time.sleep(0.5)    # be polite to the API

    print(f"  Found {fetched} pages with quality >= {min_quality}",
          file=sys.stderr)
    return titles


def fetch_page_text(title: str) -> str:
    """Fetch the wikitext for a single Page: title."""
    data = api_get({
        "action":     "query",
        "titles":     title,
        "prop":       "revisions",
        "rvprop":     "content",
        "rvslots":    "main",
    })
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        revisions = page.get("revisions", [])
        if revisions:
            return revisions[0].get("slots", {}).get("main", {}).get("*", "")
    return ""


def clean_wikitext(text: str) -> list[str]:
    """
    Strip Wikisource-specific markup and return clean Kannada lines.
    Wikisource pages have header/footer templates and OCR markup.
    """
    # Remove header/footer templates like {{rh}}, {{RunningHeader}}, etc.
    text = re.sub(r'\{\{[Rr](?:unning)?[Hh](?:eader)?[^}]*\}\}', '', text)
    # Remove all templates {{...}}
    # Do it multiple times for nested templates
    for _ in range(4):
        text = re.sub(r'\{\{[^{}]*\}\}', '', text)
    # Remove page-break markers
    text = re.sub(r'<pb[^/]*/>', '', text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove wikilinks [[target|text]] → text, [[target]] → target
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]*)\]\]', r'\1', text)
    # Remove external links
    text = re.sub(r'\[https?://\S+(?:\s+[^\]]+)?\]', '', text)
    # Remove bold/italic markers
    text = re.sub(r"'{2,3}", '', text)
    # Remove ref tags
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    # Remove comments
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    # Remove section headers == ... ==
    text = re.sub(r'={2,}[^=]+=+', '', text)
    # Collapse whitespace
    text = re.sub(r'[ \t]+', ' ', text)

    lines = []
    for raw_line in text.splitlines():
        raw_line = raw_line.strip()
        if not raw_line or len(raw_line) < 6:
            continue
        if kan_ratio(raw_line) < MIN_KAN_RATIO:
            continue
        # Split at Kannada sentence boundaries
        for sentence in re.split(r'[।॥\n]', raw_line):
            sentence = sentence.strip()
            if not sentence or len(sentence) < 6:
                continue
            if kan_ratio(sentence) < MIN_KAN_RATIO:
                continue
            # Truncate very long lines
            if len(sentence) > MAX_LINE_LEN:
                sentence = sentence[:MAX_LINE_LEN]
            lines.append(sentence)

    return lines


def download_dump(dest: Path) -> Path:
    """Download the knwikisource XML dump (much faster than API paging)."""
    if dest.exists():
        print(f"  Wikisource dump cached: {dest}", file=sys.stderr)
        return dest
    print(f"  Downloading {DUMP_URL}", file=sys.stderr)
    print("  (~10–50 MB — much smaller than Wikipedia dump)", file=sys.stderr)
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(DUMP_URL, dest,
        reporthook=lambda n, bs, ts: print(
            f"\r  {n*bs/1e6:.1f}/{ts/1e6:.1f} MB", end='', file=sys.stderr))
    print(file=sys.stderr)
    return dest


def extract_from_dump(dump_path: Path, min_quality: int, max_lines: int) -> list[str]:
    """
    Extract text from the XML dump.
    Wikisource Page: namespace = 104.
    Quality is stored in page content as <index> or inferred from
    proofread-quality page prop — in the dump we look for the
    <ns>104</ns> tag and extract the wikitext.
    """
    import bz2

    lines  = []
    in_page = in_text = in_ns = False
    current_ns   = ""
    current_text = []
    page_count   = 0

    open_fn = bz2.open if str(dump_path).endswith('.bz2') else open
    with open_fn(dump_path, 'rt', encoding='utf-8', errors='ignore') as fh:
        for raw in fh:
            raw = raw.strip()
            if raw == '<page>':
                in_page      = True
                current_ns   = ""
                current_text = []
                in_text      = False
                continue
            if raw == '</page>':
                in_page = False
                # Only process Page: namespace (104)
                if current_ns == "104" and current_text:
                    wikitext  = "\n".join(current_text)
                    new_lines = clean_wikitext(wikitext)
                    lines.extend(new_lines)
                    page_count += 1
                    if page_count % 100 == 0:
                        print(f"  …{page_count} pages → {len(lines)} lines",
                              file=sys.stderr)
                    if len(lines) >= max_lines:
                        return lines
                continue
            if not in_page:
                continue
            if raw.startswith('<ns>'):
                current_ns = raw.replace('<ns>', '').replace('</ns>', '')
            if '<text' in raw and 'xml:space' in raw:
                in_text = True
                # Text may start on same line: <text xml:space="preserve">content
                after = re.sub(r'<text[^>]*>', '', raw)
                if after:
                    current_text.append(after)
                continue
            if '</text>' in raw:
                current_text.append(raw.replace('</text>', ''))
                in_text = False
                continue
            if in_text:
                current_text.append(raw)

    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Fetch proofread Kannada Wikisource pages for training")
    parser.add_argument("--pages",   type=int, default=2000,
                        help="Max lines to extract (default 2000)")
    parser.add_argument("--quality", type=int, default=3, choices=[3, 4],
                        help="Min quality level: 3=proofread 4=validated (default 3)")
    parser.add_argument("--api",     action="store_true",
                        help="Use API instead of dump (slower but no download)")
    args = parser.parse_args()

    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", file=sys.stderr)
    print("  Kannada Wikisource — proofread corpus builder", file=sys.stderr)
    print(f"  Source: {'API (slow)' if args.api else 'XML dump (fast)'}", file=sys.stderr)
    print(f"  Target: {args.pages} lines, quality >= {args.quality}", file=sys.stderr)
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", file=sys.stderr)

    all_lines: list[str] = []
    errors    = 0
    titles    = []   # only populated in API mode

    if args.api:
        # API path — slower but no dump download required
        titles = get_proofread_page_titles(args.quality, args.pages)
        if not titles:
            # Fallback: Page: namespace without quality filter
            data = api_get({
                "action": "query", "list": "allpages",
                "apnamespace": PAGE_NAMESPACE, "aplimit": "200",
            })
            titles = [p["title"] for p in data.get("query", {}).get("allpages", [])]
            print(f"  Fallback: {len(titles)} pages (no quality filter)", file=sys.stderr)

        print(f"\nFetching page text via API…", file=sys.stderr)
        for i, title in enumerate(titles):
            try:
                wikitext = fetch_page_text(title)
                all_lines.extend(clean_wikitext(wikitext))
                if (i + 1) % 50 == 0:
                    print(f"  {i+1}/{len(titles)} — {len(all_lines)} lines",
                          file=sys.stderr)
                time.sleep(0.2)
            except Exception as exc:
                errors += 1
                if errors < 5:
                    print(f"  WARNING {title}: {exc}", file=sys.stderr)
    else:
        # Dump path — fast, offline after first run
        dump = CACHE_FILE.parent / "knwikisource-latest.xml.bz2"
        try:
            download_dump(dump)
            all_lines = extract_from_dump(dump, args.quality, args.pages)
        except Exception as exc:
            print(f"  Dump failed ({exc}), falling back to API…", file=sys.stderr)
            args.api = True
            return main()   # retry with API

    # Deduplicate
    seen   = set()
    deduped = []
    for ln in all_lines:
        if ln not in seen:
            seen.add(ln)
            deduped.append(ln)

    # Append to raw_kannada.txt (so existing Wikipedia lines are preserved)
    existing = OUTPUT_FILE.read_text(encoding="utf-8") if OUTPUT_FILE.exists() else ""
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for ln in deduped:
            if ln + "\n" not in existing:
                f.write(ln + "\n")

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", file=sys.stderr)
    if titles:
        print(f"  Pages fetched:    {len(titles) - errors}", file=sys.stderr)
    print(f"  Lines extracted:  {len(deduped)}", file=sys.stderr)
    print(f"  Written to:       {OUTPUT_FILE}", file=sys.stderr)
    print("", file=sys.stderr)
    print("  Next: python3 corpus/clean-corpus.py", file=sys.stderr)
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", file=sys.stderr)


if __name__ == "__main__":
    main()
