# Research Notes — Training Tesseract 5 for Historical Kannada OCR

**Project:** `kan_hist.traineddata` — fine-tuned Tesseract 5 LSTM model for 19th-century Kannada letterpress typefaces  
**Organisation:** Sanchaya / Sanchi Foundation  
**Fonts:** Karnata font family (digital revivals of German Mission Press, Wesleyan Mission Press, Basel Mission Press, G.T. Narayana Rao)  
**Base model:** `kan.traineddata` from Tesseract tessdata_best (Apache 2.0)  
**Tesseract version:** 5.5.2

These notes document technical findings, problems encountered, and decisions made during the development of the training pipeline and the `kan_hist` model. They are intended as a resource for a future research paper on OCR for low-resource historical Indic scripts.

---

## 1. Problem statement

The Tesseract 5 `kan` model was trained on modern digital Kannada fonts. Historical Kannada letterpress typefaces from the 1830s–1900s (German Mission Press, Wesleyan Mission Press, Basel Mission Press) differ from modern digital fonts in several systematic ways:

- **Stroke weight variation** — letterpress inking is uneven; strokes swell and thin in ways no digital font reproduces
- **Ink spread** — ink bleeds into paper fibres, rounding sharp corners and merging closely-spaced elements
- **Baseline instability** — type was hand-set; characters sit at slightly different heights along a line
- **Conjunct variation** — some conjunct forms (stacked consonants) were rendered differently in 19th-century presses than in modern Unicode fonts
- **Paper texture** — foxing, show-through, and irregular grain add visual noise

The result: the base `kan` model achieves an estimated CER of ~25% on historical letterpress scans, compared to <5% on modern typeset text.

---

## 2. Training approach

### 2.1 Fine-tuning vs. training from scratch

We chose fine-tuning from `kan.lstm` (the raw LSTM weights extracted from `kan.traineddata`) rather than training from scratch. Rationale:

- The base model already encodes Kannada language structure, conjunct recognition, and virama handling learned from a large corpus of modern Kannada text
- Fine-tuning requires orders of magnitude fewer training examples (thousands vs. millions)
- Training from scratch is impractical without GPU infrastructure for a community project
- The unicharset (set of characters the model can output) is inherited from the base model — the 4 characters absent from it (`ಋ ಙ ಝ ಱ`) are rare in our target documents

**Tradeoff:** Fine-tuning is constrained by the base unicharset. Characters absent from `kan.traineddata`'s unicharset must be added via `combine_lang_model` before they can be learned. This is possible without retraining from scratch — see §3.6.

### 2.2 Synthetic training data with simulated degradation

Because we do not have large volumes of ground-truth scans, the primary training data is synthetic: corpus lines rendered with Pillow at 150 DPI in each Karnata font variant, with degradation applied to historical typefaces:

- **Gaussian blur** σ = 0.6 — simulates ink spread
- **Salt-and-pepper noise** at 1–2% pixel level — simulates paper grain
- **Rotation** ±0.8° — simulates hand-set type baseline variation

The degradation parameters were chosen empirically by visual comparison with real German Mission Press and Wesleyan Mission Press scans.

### 2.3 Corpus selection

**Primary: Kannada Wikisource** (`kn.wikisource.org`)  
Human-proofread transcriptions of scanned Kannada books. Many Wikisource books were typeset in the exact same presses (German Mission Press, Wesleyan Mission Press) that the Karnata fonts revive. This gives the training text authentic letter co-occurrence statistics for historical letterpress material. Pages are quality-rated; we use quality ≥ 3 (reviewed by at least one human proofreader).

**Supplement: Kannada Wikipedia**  
Modern Kannada prose — adds Unicode coverage and modern vocabulary. Less relevant to historical typeface recognition, but ensures the model does not forget modern Kannada during fine-tuning.

**Key insight:** corpus relevance to the target typeface matters more than corpus size. A smaller corpus of 19th-century Wikisource text produces better results on historical scans than a larger Wikipedia corpus, because the character co-occurrence patterns in 19th-century prose are different from modern Kannada.

### 2.4 Training format: WordStr box files

Tesseract 5 uses a line-level training format. Each training example is a PNG image of a text line plus a `.box` file in WordStr format:

```
WordStr 0 0 <width> <height> 0 #<ground truth text>

```

The blank second line is mandatory. This differs from Tesseract 4, which used character-level box files (`char x1 y1 x2 y2 page`). Using Tesseract 4 box format with Tesseract 5 produces silent failures or nonsense output.

---

## 3. Unicharset analysis and the missing-character problem

### 3.1 Discovery

During lstmf generation (`make lstmf` step), the following error appeared:

```
Can't encode transcription: 'ಅ ಆ ಇ ಈ ಉ ಊ ಋ ಎ ಏ ಐ ಒ ಓ ಔ ಂ ಃ' in language ''
Encoding of string failed:
 Char 0xe0 0xb2 0x8b is 'ಋ' ...
 [followed by every remaining byte on the line]
```

Initial hypothesis (incorrect): double spaces in the ground-truth file caused the failure. Ground truth was normalised, but the error persisted.

### 3.2 Root cause

Extracted the `kan.traineddata` unicharset:

```bash
combine_tessdata -u tessdata_best/kan.traineddata /tmp/kan_base
```

The unicharset has **140 entries** — a subset of the Kannada Unicode block (U+0C80–U+0CFF). Most "missing" codepoints in that range are unassigned or extremely rare. After careful verification, exactly **four** Kannada characters relevant to real Kannada text are absent as standalone entries:

| Character | Unicode | Name | Occurrence |
|---|---|---|---|
| `ಋ` | U+0C8B | Kannada Letter Vocalic R | Loanwords, Vedic proper nouns (ಋಷಿ, ಋಗ್ವೇದ) |
| `ಙ` | U+0C99 | Kannada Letter NGA | Nasal in ಅಂಗ, ಪಂಚಾಂಗ (via anusvara in practice) |
| `ಝ` | U+0C9D | Kannada Letter JHA | Loanwords (ಝರ, ಝಲ) |
| `ಱ` | U+0CB1 | Kannada Letter RRA | Archaic retroflex, rare in modern text |

These characters were absent from the Wikipedia corpus that `tessdata_best/kan.traineddata` was built on.

### 3.3 ಞ (U+0C9E) — NOT missing (corrected)

Early analysis incorrectly listed `ಞ` (NYA, U+0C9E) as missing. It is **present** in the unicharset. This error arose because the "Can't encode" cascade (see §3.4) made it appear absent.

`ಞ` as a standalone character is confirmed present. The common conjunct `ಜ್ಞ` (jña, as in ಜ್ಞಾನ — knowledge) also encodes correctly: Tesseract matches it as `ಜ` + `್ಞ` (virama cluster), which is in the unicharset as a composite entry.

### 3.4 The error cascade effect

When Tesseract hits the first unsupported character in a line, it aborts encoding for the entire line and reports every subsequent byte as a separate failure. A single unsupported character at position 7 in a 30-character line produces an error listing bytes 7–30 — making it appear as though many characters are unsupported when only one is.

**Diagnostic rule:** the number of characters listed in the error ≠ number of characters missing from the unicharset. To verify: extract the unicharset and search directly.

```bash
combine_tessdata -u tessdata_best/kan.traineddata /tmp/kan_base
python3 -c "
data = open('/tmp/kan_base.lstm-unicharset').read()
for ch in 'ಋಙಝಞಱ':
    print(ch, 'PRESENT' if ch in data else 'MISSING')
"
```

### 3.5 Workaround — filter missing characters from training data

The immediate fix to unblock training: strip the 4 missing characters from all ground-truth files before box/lstmf generation. Applied at two layers:

**`02-make-lstmf.sh`** (Python block inside):
```python
_UNSUPPORTED = set('ಋಙಝಱ')  # ಞ is NOT missing — do not include
_tokens = [t for t in _raw.split(' ') if not (len(t)==1 and t in _UNSUPPORTED)]
gt_text = re.sub(r'\s+', ' ', ' '.join(_tokens)).strip()
```

This allows training to proceed, but the resulting `kan_hist.traineddata` cannot recognise ಋ ಙ ಝ ಱ.

### 3.6 Proper fix — expand the unicharset

Adding characters to an existing Tesseract 5 LSTM model requires:

1. **Merge the new codepoints into the unicharset** via `unicharset_extractor` + `merge_unicharsets`
2. **Rebuild `kan.traineddata`** with the expanded unicharset + updated recoder via `combine_lang_model`
3. **Continue training from the existing checkpoint** — Tesseract auto-expands the LSTM output layer for new chars, preserving all existing weights; new-char output nodes start with random weights and learn from training examples

Script: `scripts/00c-expand-unicharset.sh`  
Output: `tessdata_expanded/kan.traineddata` (140 → 144 unicharset entries)

After running the expansion:
- `03-train.sh` auto-detects `tessdata_expanded/` and uses it
- `02-make-lstmf.sh` auto-removes the filter when `tessdata_expanded/kan.traineddata` exists
- The first ~5,000 iterations after expansion may show slightly elevated BCER as new output nodes initialise

**Key dependency:** `combine_lang_model` requires Kannada langdata from the `tessdata-langdata_lstm` repository (downloaded automatically by the script on first run).

---

## 4. Training pipeline findings

### 4.1 `--max_iterations` is absolute

`lstmtraining --max_iterations N` counts from iteration 0, not from the current checkpoint's iteration. This is not clearly documented.

**Consequence:** resuming from a checkpoint at iteration 183,976 with `--max_iterations 100,000` causes lstmtraining to exit at the first step (183,976 > 100,000). The training log shows the startup message but no iteration output — this looks like the process is stuck or produced no output, not like it exited.

**Fix:** set `MAX_ITERATIONS` higher than any checkpoint the pipeline will ever resume from. Default changed to 400,000. Exposed as an environment variable for easy override.

### 4.2 Rolling checkpoint vs. named checkpoints

`lstmtraining` saves two types of checkpoint:

- **Rolling checkpoint** (`kan_hist_checkpoint`) — updated every save. Contains the full model state including optimizer momentum (Adam/Adagrad history). Always reflects the most recent training state.
- **Named checkpoints** (`kan_hist_<BCER>_<iter>.checkpoint`) — snapshots at specific iterations. Useful for rolling back to an earlier state if the model overfits.

**Important:** the rolling checkpoint is the correct file to resume from. Named checkpoints are useful for selecting a specific earlier iteration (e.g. if BCER plateaued and then degraded).

### 4.3 Checkpoint sort bug

To find the most recent named checkpoint without the rolling checkpoint:

```bash
# WRONG — sorts by BCER field (k3), picks the worst checkpoint
ls output/kan_hist_*.checkpoint | sort -t_ -k3 -n | tail -1

# CORRECT — sorts by iteration field (k4)
ls output/kan_hist_*.checkpoint | grep -v '_checkpoint$' | sort -t_ -k4 -n | tail -1
```

The filename pattern is `kan_hist_<BCER>_<iter>.checkpoint`. The `_` delimiter is shared between the model name components (`kan`, `hist`) and the numeric fields. With `-t_`, field 1 = `kan`, 2 = `hist`, 3 = BCER, 4 = iteration. Sorting on k3 (BCER) selected the checkpoint with the *highest error rate*, not the most recent one — the exact opposite of the intended behaviour.

### 4.4 UnicodeDecodeError on real scan images

Tesseract's stderr output can contain non-UTF-8 bytes when processing real scanned images (binary content from failed page-segmentation metadata embedded in error output). Using `text=True` in Python's `subprocess.run()` causes a `UnicodeDecodeError` on these bytes.

Fix: use `encoding='utf-8', errors='replace'` instead of `text=True`:

```python
result = subprocess.run(
    [...],
    capture_output=True, encoding='utf-8', errors='replace'
)
```

This replaces non-decodable bytes with `�` (Unicode replacement character) and continues, rather than crashing the entire lstmf generation run.

---

## 5. Font and variant findings

### 5.1 TTF vs. OTF rasterisation

Where a font ships both `.ttf` and `.otf` variants (GMP, WMP, GTN), both are used for training. The two formats are rasterised slightly differently by FreeType: OTF uses PostScript outlines (CFF) while TTF uses TrueType quadratic curves. At 150 DPI these differences are subtle but real — sub-pixel hinting differs, and edge sharpness varies slightly. Using both formats effectively doubles the training diversity for free.

### 5.2 Font families

| Family | Code | Historical source | Degraded? | Variants |
|---|---|---|---|---|
| Karnata German Mission Press | `kan_gmp` | German Mission Press, Mangaluru | Yes | TTF + OTF |
| Karnata Wesleyan Mission Press | `kan_wmp` | Wesleyan Mission Press, Bengaluru | Yes | TTF + OTF |
| Karnata F. Kittel | `kan_kittel` | Basel Mission Press, 1830–1900 | Yes | OTF only |
| Karnata GTN | `kan_gtn` | G.T. Narayana Rao revival | No | TTF + OTF + 6 weights |

Karnata Bandipur is excluded from training (decorative face not suited to body text OCR).

### 5.3 Degradation parameters

Historical fonts (`degrade: true`) receive:
- Gaussian blur σ = 0.6 px
- Salt-and-pepper noise at 1.5% of pixels
- Random rotation ±0.8°

These were calibrated empirically against real German Mission Press scans from the Sanchaya collection. The goal is not to perfectly simulate scans but to span the range of visual variability the model will encounter.

### 5.4 1:1 character test limitations

A naive quality test ran Tesseract on individual single-character images (one PNG per Kannada codepoint per font variant) and measured character accuracy. This systematically underestimates model quality because:

1. Tesseract is a line-level LSTM model — it uses left-to-right context to disambiguate characters. Isolated characters lack this context.
2. The model's confusion patterns on isolated glyphs do not predict its confusion on real text.
3. Some Kannada characters are visually similar in isolation but unambiguous in context (e.g. ಳ vs ಲ preceded by a vowel mark).

**Better approach:** OCR a held-out line from each font variant and compute CER against the known ground truth. This is what the portal's per-font OCR quality test now does.

---

## 6. Pipeline architecture decisions

### 6.1 SSE (Server-Sent Events) for log streaming

The portal streams training log output to the browser using HTML5 Server-Sent Events rather than WebSockets. Rationale:
- SSE is one-directional (server → client), which matches the use case perfectly
- No extra library needed on the client (`new EventSource(url)`)
- Automatic reconnection is built into the SSE protocol
- SSE works through HTTP/1.1 (no upgrade handshake), which simplifies proxy/Docker setups

### 6.2 Skip-if-complete for idempotent steps

The corpus download step is idempotent: re-downloading the Wikisource dump when it is already cached is wasteful (~80 MB). The portal now checks for existing corpus content before running the download step and returns a `{ skipped: true, reason: "..." }` response instead of re-running. A `?force=1` query parameter overrides the skip.

This pattern should be applied to any step where re-running from scratch wastes significant time without producing different output.

### 6.3 BCER measurement — training vs. test

The BCER reported in training logs is measured on training data, not held-out data. Tesseract's training does not have a separate validation set by default. This means:
- BCER will always decrease (or plateau) — it cannot increase unless you restart training with a different dataset
- It cannot detect overfitting on its own
- Real-world performance must be validated externally

The portal's OCR quality tab addresses this by running the packaged model against held-out rendered test images and computing CER via Levenshtein distance — providing a true held-out accuracy estimate.

---

## 7. Corpus quality observations

### 7.1 Wikisource quality levels

Quality level 3 (proofread by one human) is sufficient for training. Quality level 4 (validated by two humans) is preferable for building a small high-quality evaluation set. In practice, the difference in OCR model quality between training on level-3 vs. level-4 data appears small — the primary benefit of level-4 data is reduced noise in the evaluation metrics, not improved model training.

### 7.2 Character frequency in historical text

19th-century Kannada letterpress text has different character frequency distributions than modern Kannada:
- Archaic verb forms and case suffixes occur more frequently
- Some conjunct forms are used that are rare in modern text
- Sanskrit loanwords are more common, bringing in less common consonant clusters

This suggests that a model trained purely on modern Kannada Wikipedia text will underperform on historical documents even if the fonts match, because the language model component (n-gram statistics) will not generalise well to archaic text.

---

## 8. Open questions for future research

1. **Quantitative comparison of synthetic vs. real scan training data.** We know from qualitative observation that real scan GT pairs improve accuracy significantly, but we have not measured this systematically. A controlled experiment (model A: synthetic only, model B: synthetic + N real scans, varying N) would quantify the breakeven point.

2. **Cross-press generalisation.** Does a model trained on German Mission Press typefaces generalise to Wesleyan Mission Press, or do the two presses need separate models? The Karnata font families are visually distinct; it would be worth measuring inter-press CER.

3. **Optimal degradation parameters.** The degradation (blur/noise/rotation) parameters were chosen by eye. A systematic search over degradation intensity vs. held-out CER on real scans could identify better parameters. There is likely a degradation level that is too mild (model not robust enough) and one that is too severe (model can no longer read clean text).

4. **Fine-tuning vs. training from scratch for archaic characters.** The five missing characters (`ಋ ಙ ಝ ಞ ಱ`) were filtered out because they are absent from the base unicharset. A parallel experiment training from scratch with an extended unicharset would show whether including them is worth the additional complexity, and whether the base model's existing language knowledge survives a full retrain.

5. **Evaluation on held-out Wikisource pages.** We have access to proofread Wikisource pages typeset in the target presses. Using a properly held-out subset as the evaluation corpus (never seen during training) would give a more reliable accuracy estimate than the per-font test images.

6. **Language model component.** Tesseract 5 uses both a neural LSTM and a language model (word n-grams, dawg word list). The language model in `kan.traineddata` was built from modern Kannada. Rebuilding it from a 19th-century corpus might further improve historical OCR by reducing false corrections of archaic spellings.

---

## 9. Key references

- Ray Smith, "An Overview of the Tesseract OCR Engine," ICDAR 2007
- Ray Smith, "Improving the accuracy of Tesseract for low resolution images," DAS 2009  
- Tesseract 5 training documentation: https://tesseract-ocr.github.io/tessdoc/TrainingTesseract-5.html
- Karnata font family: https://fonts.sanchaya.net
- Kannada Wikisource: https://kn.wikisource.org
- tessdata_best repository: https://github.com/tesseract-ocr/tessdata_best

---

## 10. Technical environment

| Component | Version / detail |
|---|---|
| Tesseract | 5.5.2 |
| `lstmtraining` | from `tesseract-training-tools` (Homebrew) |
| Python | 3.11 |
| Pillow | 10.x |
| Node.js portal | Express 4.x |
| Training hardware | Apple Silicon (M-series), CPU only |
| Base model | `kan.traineddata` from tessdata_best (2023) |

Training runs on CPU (Apple Silicon M-series). A single training run to 400,000 iterations takes approximately 8–16 hours depending on the number of lstmf files and the iteration. `caffeinate -i` is used to prevent macOS sleep during long training runs.

---

*Last updated: June 2026*
