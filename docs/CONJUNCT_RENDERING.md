# Conjunct (ottu) Rendering Investigation — Historical Karnata Fonts

**Project:** `kan_hist.traineddata` — fine-tuned Tesseract 5 LSTM model for 19th-century Kannada letterpress typefaces  
**Date:** August 2026  
**Status:** Root cause identified; fix **implemented** — per-font `aalt` is declared in `fonts.yml`
(`font_features: "'aalt' 1"`) and honoured by every render path. See
[IMAGE_GENERATION.md](IMAGE_GENERATION.md) for the pipeline as it stands today.  
**Fonts affected:** Karnata GTN (6 weights), Karnata German Mission Press (GMP), Karnata Wesleyan Mission Press (WMP) — the Sanchaya historical revivals  
**Reference (correct) font:** Karnata F Kittel

---

## 1. Symptom

Virama / ottu conjuncts (e.g. `ರ್ಕ`, `ಕರ್ಣ`, `ರ್ಕ`, `ಸ್ಥ`, `ದ್ದ`) render **badly** in the GTN, GMP, and WMP fonts when compared with Kittel. The user observed "conjuncts with virama not rendering well for all fonts other than kittel."

Because the historical Sanchaya fonts are used to render the OCR training pages (`classical-corpus-kannada/a5-pages/…`), broken conjuncts are baked into the training images, and Tesseract learns to recognise the *broken* shapes.

---

## 2. Rendering pipeline under investigation

Three render paths exist; the **browser path is authoritative** for the training corpus:

| Path | File | Used for |
|---|---|---|
| Headless Chrome | `corpus/browser_render.js` | A5 training pages (`render-a5-pages.py`), live preview |
| Python HarfBuzz + FreeType | `corpus/shaping_render.py` | `render-corpus.py`-style batch rendering |
| Portal live preview | `server.js` `tryBrowser()`, `public/index.html` `@font-face` | User-facing Unicode reference |

Key facts:

- `corpus/browser_render.js` sets **no** `font-feature-settings` — rendering relies entirely on Chrome's default Indic shaping.
- `corpus/shaping_render.py` passes an explicit `_KANNADA_FEATURES` dict to HarfBuzz (`nukt akhn rphf pref blwf half pstf vatu cjct pres abvs blws psts haln dist abvm blwm …`) that does **not** include `aalt`.
- The Sanchaya reference site (`fonts.sanchaya.net/css/fonts.css`) is only `@font-face` declarations — no `font-feature-settings` anywhere. So the reference site relies on browser defaults too, and would show the same broken conjuncts.

---

## 3. Investigation method

The `data:text/html` Puppeteer tests from earlier sessions **silently fell back to the system font** (opaque origin blocked the `file://` fonts), invalidating all their "identical" measurements. The reliable harnesses are:

1. **Glyph-level shaping** via `uharfbuzz` (the same shaping engine Chrome uses) — shows exactly which GIDs are produced per input string. `tmp/shapecheck.py`, `tmp/gidmap.py`.
2. **Headless Chrome + connected-component analysis** — counts ink components and merges x-overlapping clusters to distinguish "fused" (one box) from "detached" (two boxes). `tmp/cc_analyze.js`.
3. **ASCII-art pixel rendering** — renders glyphs as `#` blocks so shapes can be compared without an image viewer. `tmp/ascii_render.js`.
4. **Ink-width / blob-count measurement** on real browser output. `test_lang2.js`, `test_aalt6.js`.

All browser measurements use an HTML file written to disk and loaded via `page.goto(file://…)` (never `data:`), with the `@font-face` referencing the font via `file://` URL.

---

## 4. GSUB table analysis

All four fonts have GSUB tables. Feature sets and script coverage:

| Font | GSUB features | Scripts |
|---|---|---|
| GTN (all 6 weights) | `aalt, akhn, blwf, blws, haln, psts, rphf` | `DFLT, knd2, knda, latn` |
| GMP | `aalt, akhn, blwf, blws, haln, psts, rphf` | `DFLT, knd2, knda, latn` |
| WMP | `aalt, akhn, blwf, blws, haln, psts, rphf` | `DFLT, knd2, knda, latn` |
| Kittel | `blwf, blws, haln, psts` only | `knda` |

**Critical observation:** the Sanchaya fonts (GTN/GMP/WMP) store their conjunct/ottu substitution under the **`aalt`** ("access all alternates") GSUB feature. Chrome's default shaping does **not** enable `aalt`, so the intended conjunct forms are never selected.

The `aalt` subtable is structurally identical across GTN/GMP/WMP (35 single substitutions):

```
kn_arka_ottu → kn_r_ottu          # the ra-ottu fix (GTN/WMP naming)
uni0C95      → kn_k_v             # all base consonants → *_v / *-kannada forms
uni0C96      → kn_kh_v
uni0C97      → kn_g_v
… (through uni0CB9 → kn_h_v)
```

(GMP uses `-kannada` suffix naming, e.g. `k-kannada`, `reph-kannada → r-kannada`.)

---

## 5. Root cause

1. Chrome's Indic shaper (HarfBuzz) enables the *base* conjunct features (`blwf`, `blws`, `haln`, `psts`, …) but **not** `aalt`.
2. For **GTN and WMP**, the correct conjunct forms live only in `aalt`. Without it, the default base features produce a **detached ottu**: `ರ್ಕ` renders as two separated full-width ink boxes instead of a fused `ಕ` with the `ರ್` ottu below it.
3. For **GMP**, the correct forms live in the **base** features. GMP renders `ರ್ಕ` correctly by default — and enabling `aalt` *breaks* it (produces two boxes).
4. **Kittel** has no `aalt` at all; its conjuncts are direct `blwf`/`haln` substitutions already enabled by default, which is why Kittel is correct.

The three Sanchaya fonts contradict each other: GTN/WMP store correct forms in `aalt`; GMP stores correct forms in the base Indic features.

---

## 6. Measurement evidence

### 6.1 HarfBuzz glyph sequences (default vs. `aalt`)

For GTN `ರ್ಕ` (input: ರ + ್ + ಕ):

| Shaping | Glyph sequence | Result |
|---|---|---|
| Default (no features) | `uni0C95` + `kn_arka_ottu` | detached / broken |
| `aalt` 1 | `kn_k_v` + `kn_r_ottu` | fused / correct |

For GMP `ರ್ಕ`:

| Shaping | Glyph sequence | Result |
|---|---|---|
| Default (no features) | `uni0C95` + `reph-kannada` | fused / correct |
| `aalt` 1 | `k-kannada` + `r-kannada` | detached / broken |

### 6.2 Browser connected-components (Kittel = reference: single fused box)

| Font | `ರ್ಕ` default | `ರ್ಕ` with `aalt` 1 |
|---|---|---|
| GTN | 2 separate boxes (`w48/w47`) — **broken** | 1 fused box (`w65`) — **fixed** |
| WMP | 2 separate boxes (`w46/w51`) — **broken** | 1 fused box (`w64`) — **fixed** |
| GMP | 1 fused box (`w30`+ottu `w20` below) — **correct** | 2 boxes — **broken** |
| Kittel | 1 fused box (`w51`) — correct | n/a (no `aalt` in font) |

### 6.3 Why earlier "aalt breaks everything" tests were wrong

An early test applied a **full 24-feature string** (`kern liga calt clig akhn rphf pref blwf half pstf vatu cjct pres abvs blws psts haln dist abvm blwm aalt nukt`). This overrides Chrome's *native* Indic shaping pipeline and indeed broke `ಶ್ರ ತ್ರ ಪ್ರ ನ್ನ ಸ್ಥ ದ್ದ` (blobs 1→2). The correct fix is `aalt` **alone**, which:

- fixes `ರ್ಕ` / `ಕರ್ಣ` for GTN and WMP (fused), and
- leaves other conjuncts (`ಕ್ಷ ಶ್ರ ತ್ರ ಪ್ರ ನ್ನ ಸ್ಥ ದ್ದ ಜ್ಞ`) and plain Kannada text structurally unchanged.

Verified with both connected-component counts and ASCII rendering on `ಕನ್ನಡ`, `ಸಂಸ್ಕೃತಿ`, and the full conjunct set. Note: WMP's `aalt` does produce slightly more components on some plain text (its `*_v` base-letter alternates differ more from the base glyphs than GTN's do) — worth a visual spot-check before shipping.

### 6.4 Non-factors ruled out

- `lang="kn"` on `<html>`/text: **no effect** on shaping (tested).
- Changing Tesseract's compiled box buffer size: rejected — the fix for long GT box files is chunking in `02-make-lstmf.sh` (documented separately in `TRAINING.md`).
- `data:` HTML pages: invalid for measurements (silent system-font fallback).

---

## 7. Recommended fix

Apply `aalt` **per font**, only to GTN and WMP:

1. **`fonts.yml`** — add a per-font flag, e.g. `features: "aalt" 1` on the `kan_gtn` and `kan_wmp` entries (leave `kan_gmp` and `kan_kittel` without it).
2. **`corpus/render-a5-pages.py`** — carry the flag into each job dict.
3. **`corpus/browser_render.js`** — `buildHtml()` accepts an optional `featureSettings` and injects `font-feature-settings: <value>;` into the `#t` style when present.
4. **`corpus/shaping_render.py`** — append `"aalt": True` to `_KANNADA_FEATURES` only for GTN/WMP (or make it a parameter).
5. **Portal preview** (`server.js` `tryBrowser()` and `public/index.html` live view) — same treatment so the user-facing Unicode reference matches the training images.
6. **Re-render** affected font pages → regenerate `.lstmf` → resume training.

### Correct feature string

Use `font-feature-settings: "aalt" 1;` **alone**. Do **not** use the full 24-feature string — it overrides Chrome's native Indic shaping and breaks other conjuncts.

---

## 8. Status / next actions

- [x] Root cause identified (Chrome doesn't enable `aalt`; GTN/WMP need it, GMP must not get it)
- [x] Fix validated at component level and via ASCII rendering
- [x] Implement per-font feature plumbing (fonts.yml → render-a5-pages.py → browser_render.js)
- [x] Add `aalt` to `shaping_render.py` / portal preview for consistency
- [x] Re-render pages, regenerate lstmf, resume training
- [x] Visual spot-check WMP plain-text with `aalt` (see §6.3 caveat)

---

## 9. Second root cause — forced Indic features in the Python path (Aug 2026)

`aalt` was necessary but **not sufficient**. After the per-font `aalt` plumbing landed, ~21 conjuncts
were still rendering wrong in GTN, GMP and WMP — `ತ್ತ ದ್ದ ನ್ನ ಮ್ಮ ಲ್ಲ ಸ್ತ ಪ್ರ ಗ್ರ ತ್ರ ಶ್ರ ಸ್ವ ರ್ಕ ಕ್ಕ ಬ್ಬ ಷ್ಟ ಕ್ತ ನ್ಮ ಧ್ವ ಸ್ಪ ನ್ದ`,
plus `ಕ್ಷ` in GMP — while Kittel remained correct.

### The finding

§7 of this document already stated the rule for Chrome:

> Use `font-feature-settings: "aalt" 1;` **alone**. Do **not** use the full 24-feature string — it
> overrides Chrome's native Indic shaping and breaks other conjuncts.

That rule was applied to `browser_render.js` but **never to `corpus/shaping_render.py`**, which kept
passing a 23-tag `_KANNADA_FEATURES` dict to `hb.shape()` — including the Indic tags `blwf half pref
pstf vatu cjct rphf`.

HarfBuzz's Indic shaper applies those features itself, per glyph, using internal masks: `blwf` only on
the consonant that must take the below-base (ottu) form, `half` only on the one taking the half form.
Passing them in the feature dict enables them globally across the run, so below-base substitution also
hits the base consonant. The cluster then reorders and the ottu is emitted **before** its base.

### Evidence — `ತ್ತ` under four feature settings

| Setting | GTN | Verdict |
|---|---|---|
| A — no features | `uni0CA4 + kn_t_ottu` | ✅ base, then ottu |
| B — all Indic tags (was production) | `kn_t_ottu + uni0CA4` | ❌ reversed |
| C — all Indic tags + `aalt` | `kn_t_ottu + kn_t_v` | ❌ reversed |
| D — `aalt` only | `kn_t_v + kn_t_ottu` | ✅ base, then ottu |

GMP under setting B shaped `ಕ್ಷ` to `uni0C95.below + uni0CB7` — below-form of *ka* plus full *ssa*,
both wrong — instead of the correct `uni0C95 + uni0CB7.below`.

**Kittel was immune under all four settings** (byte-identical output). Its GSUB carries only
`blwf/blws/haln/psts` with no reordering triggers, so forcing tags changes nothing. This is why it read
as "the font that works" and masked the defect in every earlier comparison.

### Why glyph count is a false test

An earlier verification used glyph count (`3 = unshaped`). That is wrong for Kannada: **2 glyphs is the
correct result** for most conjuncts — base plus a separate ottu glyph drawn below-left within its own
small advance. All four fonts returned 2 for `ತ್ತ` both before and after the fix. Only akhand ligatures
such as `ಕ್ಷ` in GTN/WMP collapse to 1 glyph.

**Glyph order is the reliable test, not count.**

### Fix

`_KANNADA_FEATURES = {}` in `corpus/shaping_render.py`. `aalt` remains the only explicitly-passed
feature, still per font from `fonts.yml`. Verified across 21 clusters × 4 fonts: zero reversed.

None of the fonts required changes. The GSUB tables were correct throughout — the shaping *call* was
not.

### Blast radius

Every image produced through the Python path was affected and has been regenerated: `test-images/`
(3,038), `inventory/` (2,156), `rendered/` (9,300). The classical A5 corpus renders through Chrome,
which never forced features, and was unaffected.

---

*Relevant files: `fonts.yml`, `corpus/browser_render.js`, `corpus/render-a5-pages.py`, `corpus/shaping_render.py`, `server.js`, `public/index.html`.*  
*Measurement tools: `test_lang2.js`, `test_aalt6.js`, `tmp/cc_analyze.js`, `tmp/ascii_render.js`, `tmp/shapecheck.py`, `tmp/gidmap.py`.*
