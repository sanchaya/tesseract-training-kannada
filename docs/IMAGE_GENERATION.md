# Image Generation — How every PNG in this project is produced

**Scope:** the four render paths, what each one feeds, and the shaping method they share.
**Companion docs:** [CONJUNCT_RENDERING.md](CONJUNCT_RENDERING.md) (why `aalt` matters), [TRAINING.md](TRAINING.md), [PORTAL.md](PORTAL.md)

---

## 1. Why shaping is the whole problem

Kannada is a complex script. A cluster like `ಕ್ಷ` is three codepoints (`ಕ` + virama + `ಷ`) that must be
substituted into **one** ligature glyph by the font's OpenType GSUB tables.

Pillow's `ImageDraw.text()` does not run GSUB. It maps each codepoint to its own glyph and places them
side by side, so `ಕ್ಷ` renders as three detached shapes. Any image produced that way is wrong — and if it
is used for training, Tesseract faithfully learns the broken shape.

Every render path in this project therefore runs a real shaping engine. There are two:

| Engine | Where | Why this one |
|---|---|---|
| **HarfBuzz + FreeType** (Python) | `corpus/shaping_render.py` | Full control over which OpenType features are enabled — required for per-font `aalt` |
| **Headless Chrome** | `corpus/browser_render.js` | Same text stack as fonts.sanchaya.net; handles page layout, line breaking and justification for A5 pages |

---

## 2. The four render paths

| Path | Script | Output | Engine | Degradation |
|---|---|---|---|---|
| Font gallery / OCR test | `scripts/gen-char-images.py` | `test-images/` | shaping_render | never — diagnostic must be clean |
| Character inventory | `corpus/generate-inventory.py` | `inventory/` | shaping_render | no |
| Synthetic corpus lines | `corpus/render-corpus.py` | `rendered/` | shaping_render | per `degrade:` flag |
| Classical A5 pages | `corpus/render-a5-pages.py` | `classical-corpus-kannada/a5-pages/` | headless Chrome | per `degrade:` flag |

Paths 2–4 produce **training data**. Path 1 is **diagnostic only** and never enters the training set —
its job is to show what the glyphs *should* look like, so it is always rendered clean.

---

## 3. The shared shaping method (`corpus/shaping_render.py`)

Used by three of the four paths. Two stages:

### Stage 1 — Shape with HarfBuzz

```python
hb_font.scale = (font_size * 64, font_size * 64)   # 26.6 fixed-point
buf = hb.Buffer(); buf.add_str(text)
buf.guess_segment_properties()                      # → script=Kannada, direction=LTR
hb.shape(hb_font, buf, features)
```

`guess_segment_properties()` detects the script so HarfBuzz selects the Indic shaper.

**`_KANNADA_FEATURES` is deliberately empty.** Do not add the Indic tags to it.

The Indic shaper applies `nukt akhn rphf pref blwf half pstf vatu cjct pres blws psts haln` itself,
**per glyph**, using internal masks — `blwf` only on the consonant that must take the below-base (ottu)
form, `half` only on the one taking the half form. Passing them in the feature dict enables them
*globally across the run*, so below-base substitution also hits the base consonant, the cluster
reorders, and the ottu is emitted before its base:

```
ತ್ತ   correct → uni0CA4 + kn_t_ottu     (base, then ottu)
     forced  → kn_t_ottu + uni0CA4     (ottu first — visually broken)
```

The only feature this project passes explicitly is `aalt`, per font (§4).

Output is a glyph-ID stream with per-glyph x/y offsets and advances. Note that 2 glyphs is the
**correct** result for most Kannada conjuncts — base + a separate ottu glyph drawn below-left within
its own small advance. Only akhand ligatures such as `ಕ್ಷ` in GTN/WMP collapse to a single glyph.
Glyph count alone therefore does not tell you whether shaping is right; glyph **order** does.

### Stage 2 — Rasterise with FreeType

Each glyph ID is loaded with `FT_LOAD_RENDER` and alpha-blended into a numpy canvas at its shaped
position.

**Font unit conversion.** HarfBuzz returns all positions in **26.6 fixed-point** — 1/64 of a pixel.
Because `hb_font.scale` is set to `font_size * 64`, converting to pixels is a divide by 64, done as a
bit-shift:

```python
x_off = pos.x_offset  >> 6     # pixels
y_off = pos.y_offset  >> 6
x_pen += pos.x_advance >> 6
```

Equivalent explicit form: `font_units * font_size / uPEM`, where uPEM is the font's units-per-em
(1000 for CFF/OTF, 2048 for TrueType). Both agree when the scale is set as above. Omitting the
conversion places every glyph 64× too far right — the symptom is text that vanishes off-canvas.

### Fallback

If `uharfbuzz`, `freetype-py` or `numpy` are missing, `SHAPING_AVAILABLE` is `False` and callers drop to
Pillow with a printed warning. **Conjuncts will be broken and the output is unfit for training.** Install:

```bash
pip install uharfbuzz freetype-py numpy --break-system-packages
```

---

## 4. Per-font `aalt` — read from fonts.yml, never hardcoded

GTN, GMP and WMP all store conjunct forms under the `aalt` ("access all alternates") GSUB feature, which
no shaper enables by default. But the correct setting is **not uniform**:

| Font | `aalt` | Reason |
|---|---|---|
| GTN | **on** | correct ottu forms live in `aalt` |
| WMP | **on** | same |
| GMP | **off** | correct forms are in the base features; `aalt` *breaks* them |
| Kittel | off | no `aalt` in its GSUB at all |

This is declared once in `fonts.yml` as `font_features: "'aalt' 1"` and read by every generator. Full
GSUB analysis in [CONJUNCT_RENDERING.md](CONJUNCT_RENDERING.md).

---

## 5. Path 1 — Font gallery (`scripts/gen-char-images.py`)

Drives the portal **Images** tab and the 1:1 OCR test.

```bash
python3 scripts/gen-char-images.py                     # all fonts
python3 scripts/gen-char-images.py --font-id kan_gmp   # one family
python3 scripts/gen-char-images.py --dpi 150 --size 48
```

Portal equivalent: **Regenerate all** → `POST /api/char-images/generate`.

**Output:** `test-images/<font_id>/<variant>/` — **98 images per variant** at 48 px / 150 DPI.

| Type | Count | Example |
|---|---|---|
| Vowels | 15 | `vowel_a.png` |
| Consonants | 35 | `consonant_ka.png` |
| Conjuncts | 24 | `conjunct_ksha.png` |
| Digits + danda | 12 | `digit_0.png` |
| Line strips | 4 | `line_conjuncts.png` |
| Sample sentences | 8 | `sentence_01.png` |

Each PNG is written with a matching `.gt.txt` containing the exact source text, plus one
`manifest.json` per variant (font path, size, DPI, count, errors, character list).

The `.gt.txt` pairing is the point — these are not just previews. The OCR test tab runs Tesseract.js over
every PNG and diffs the result against ground truth, which is how you find the specific characters the
model is failing on.

**Variant selection.** `scan_font_dir()` scopes its search to the `font_dir` declared in fonts.yml (plus
sibling `ttf/` and `otf/` dirs), so a family that ships extra width sets on disk does not explode the
variant count. Anek Kannada has 41 files across 5 widths; `font_dir: static/AnekKannada` narrows that to
its 8 default-width weights. Variable fonts (`[wght]`, `VariableFont`) and `webfonts/`, `Source/` dirs
are skipped. Where a stem exists as both TTF and OTF, both are kept as `<Stem>-ttf` / `<Stem>-otf` —
the two rasterise slightly differently (CFF vs quadratic outlines), which is free training diversity.

---

## 6. Path 2 — Character inventory (`corpus/generate-inventory.py`)

Produces the character-baseline set used by inventory-first training.

```bash
python3 corpus/generate-inventory.py              # fonts.yml-declared weights
python3 corpus/generate-inventory.py --all-fonts  # every .ttf/.otf on disk
```

**Output:** `inventory/<font_stem>/char_<id>.png` + `.gt.txt` — 98 combinations × 22 declared weights
= **2,156 images**.

Combinations: single vowels, single consonants, consonant + each vowel sign, consonant + virama +
consonant conjuncts, anusvara/visarga, and numerals.

Font discovery is recursive across `fonts/` and covers **both** `.ttf` and `.otf`, restricted by default
to the weights declared in fonts.yml so the inventory matches what the rest of the pipeline trains on.

> **History.** Until August 2026 this generator used bare Pillow with no shaping, no `aalt`, and a
> top-level `fonts/*.ttf` glob that found only 2 files. Since the inventory is almost entirely
> conjuncts, this was the worst place in the project to be unshaped. Kittel (OTF-only) had zero
> inventory coverage. See §9.

---

## 7. Path 3 — Synthetic corpus lines (`corpus/render-corpus.py`)

One image per corpus line, per font weight, rendered in parallel via `multiprocessing.Pool`.

**Output:** `rendered/<font_id>_<font_stem>_lineNNNN.png` + `.gt.txt`. Naming matters — the portal's
font registry buckets per-font image counts by the leading `<font_id>_` prefix.

Already-rendered files are skipped, so the script is safe to resume. To force a rebuild, clear
`rendered/` first (portal: **↺ Clear & re-render**).

### Degradation (`degrade: true`)

Applied to historical letterpress revivals only. Simulates the artefacts of metal type on paper so the
model does not only ever see pristine digital outlines:

| Effect | Value | Simulates |
|---|---|---|
| Gaussian blur | radius 0.6 px | ink bleed into paper fibre |
| Salt-and-pepper noise | 0.3% of pixels → pure black/white | paper grain, foxing, scan speckle |
| Rotation | ±0.8° | page skew on platen or scanner bed |

Seeded from `hash(tag, idx)`, so re-rendering reproduces byte-identical images rather than drifting.

In the portal's font registry the **Degraded** badge means exactly this flag — it is a rendering mode,
not a fault. Modern digital faces (GTN, Anek, Baloo) render **Clean** because they will be read from
clean digital sources.

---

## 8. Path 4 — Classical A5 pages (`corpus/render-a5-pages.py`)

Full A5 pages of real historical texts at 150 DPI, rendered through headless Chrome
(`browser_render.js`) — the same text stack as fonts.sanchaya.net. Chrome is used here rather than the
Python path because full-page rendering needs line breaking, justification and margin handling.

`font_features` from fonts.yml is injected as CSS `font-feature-settings`, so `aalt` behaves identically
to the Python path. Runs multiple Chrome processes with configurable per-process page concurrency.

### Always use `--lines` for training data

```bash
python3 corpus/render-a5-pages.py --lines        # LSTM-ready line images
python3 corpus/render-a5-pages.py                # page images — NOT trainable
```

**Page mode output is unusable for LSTM training.** Tesseract needs one image per text line. A full A5
page paired with the whole page's text cannot be aligned by CTC: the LSTM scales input to 48px height,
so an 875×1241 page becomes ~33 timesteps while the transcription needs ~700 labels. `lstmtraining`
then reports:

```
Compute CTC targets failed for <file>.lstmf!
```

Every one of the 28,534 pages rendered in page mode failed this way (sampled 200/200 infeasible), while
`rendered/` and `inventory/` line images passed 200/200.

**How `--lines` works.** After layout, `measureLinesInPage()` walks the text one character at a time
asking Chrome for each character's client rect, and groups characters sharing a baseline row (3px
tolerance) into a visual line. That yields the exact pixel box of every wrapped line *and* the text
that produced it, so the crop and its ground truth cannot drift apart — no OCR or heuristic
segmentation is involved. The page is screenshotted once and cropped with sharp.

Degradation is applied to the page *before* cropping, so line images keep realistic page-level artefacts.

**Output:** `<title>/<font_tag>/pageNNNN_lineNNN.png` + `.gt.txt`, typically ~15 lines per page at
875×55 each — about 760 timesteps for ~46 characters, roughly 16× the CTC minimum.

`02-make-lstmf.sh` carries a matching guard that skips any pair whose labels exceed the timestep
budget, so page-mode leftovers can never silently re-enter `list.txt`.

---

## 9. August 2026 audit — what was found

An audit of all render paths after the HarfBuzz unit-conversion documentation:

Two separate defects were found, in two passes.

**Pass 1 — the inventory path had no shaping at all.**

| Path | Verdict |
|---|---|
| **`generate-inventory.py`** | **bare Pillow, no shaping, no `aalt`, 2-font discovery** |
| others | shaping present |

Rebuilt 196 → 2,156 images. Kittel (OTF-only) had previously had *zero* inventory coverage.

**Pass 2 — forced Indic features corrupted every conjunct in GTN, GMP and WMP.**

Triggered by a report that ~21 conjuncts (`ತ್ತ ದ್ದ ನ್ನ ಮ್ಮ ಲ್ಲ ಸ್ತ ಪ್ರ …`, plus `ಕ್ಷ` in GMP) rendered
wrong in three fonts but were fine in Kittel. Shaping each cluster under four feature settings showed
the ottu emitted *before* its base whenever the Indic tags were forced (§3). Kittel was immune — its
GSUB carries only `blwf/blws/haln/psts` and no reordering triggers — which is precisely why it looked
correct and masked the bug.

**The fonts were never at fault.** `_KANNADA_FEATURES` was emptied; all 21 clusters × 4 fonts now shape
in the correct order.

Everything rendered through the Python path was affected and was regenerated:
`test-images/` (3,038), `inventory/` (2,156), `rendered/` (9,300, now including the two new families).
The classical A5 set (28,534 pages) was **not** affected — it renders through headless Chrome, which
never forced features.

### Verifying a render path yourself

Glyph **order** is the reliable test — count is not, since 2 glyphs is correct for most conjuncts:

```python
import uharfbuzz as hb
from fontTools.ttLib import TTFont
order = TTFont(font_path, lazy=True).getGlyphOrder()
face = hb.Face(open(font_path, 'rb').read()); font = hb.Font(face)
font.scale = (48*64, 48*64); hb.ot_font_set_funcs(font)
buf = hb.Buffer(); buf.add_str('ತ್ತ'); buf.guess_segment_properties()
hb.shape(font, buf, {})                      # {} or {'aalt': True} only
[order[g.codepoint] for g in buf.glyph_infos]
# ['uni0CA4', 'kn_t_ottu']  ✓ base then ottu
# ['kn_t_ottu', 'uni0CA4']  ✗ reversed — a feature is being forced
```

---

## 10. Adding a font

1. Place files under `fonts/<id>/` — **the directory name must equal the `id` in fonts.yml.**
   Every generator resolves fonts at `fonts/<id>/`; a mismatched folder name makes the font invisible
   to the gallery, the OCR test and training alike.
2. Add the entry to `fonts.yml`: `id`, `name`, `font_dir`, `font_files`, `degrade`, `max_pages`, and
   `font_features: "'aalt' 1"` if its conjuncts live in `aalt`.
3. Regenerate: **Prep base** → **Render images**, then `gen-char-images.py` for the gallery.

Fonts installed by download rather than `git clone` (e.g. Google Fonts) are fully supported — presence
is detected by scanning for font files, not by looking for a `.git` directory.
