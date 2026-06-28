# Training reference — kan_hist

Deep-dive on training parameters, quality targets, and how to tune the model.

## How fine-tuning works

`kan_hist` is built by fine-tuning, not training from scratch. The starting point is `kan.lstm` extracted from Google's `tessdata_best/kan.traineddata`. Fine-tuning teaches the existing LSTM to recognise the distinctive letterforms of historical Karnata typefaces while retaining the language-level knowledge it already has (conjunct rules, character ordering, word shapes).

The training data is synthetic: the Karnata fonts are rendered against a real Kannada text corpus, with degradation applied to simulate actual letterpress printing conditions.

## Key parameters in `03-train.sh`

| Parameter | Value | Notes |
|---|---|---|
| `--learning_rate` | `0.001` | Standard starting point for fine-tuning. Reduce to `0.0001` if BCER oscillates or does not converge. |
| `--max_iterations` | `500,000` | Upper bound — absolute, not relative to checkpoint. In practice, convergence occurs between 20,000–60,000 iterations. Set high so resuming from a late checkpoint does not exit immediately. |
| `--target_error_rate` | `-1` | Disabled — we stop manually when BCER plateaus. |

### When to stop training

Stop when the BCER (best-so-far) does not improve for ~10,000 iterations. Continuing beyond that overfits the model to the synthetic training images and typically degrades performance on real scans.

Practical targets for a well-functioning model:

| BCER | Assessment |
|---|---|
| > 5% | Poor — likely a corpus or rendering issue |
| 2–5% | Adequate — usable for lightly degraded material |
| 1–2% | Good — suitable for typical mission press scans |
| < 1% | Excellent — test carefully on real held-out scans |

**Observed result (June 2026, ~5,400 synthetic Karnata images):** BCER reached **0.092%** at iteration 9,589. This is well under the excellent threshold and was reached early — a consequence of the high rendering consistency of the Karnata fonts. Verify on real historical scans before treating this as a real-world accuracy figure.

BCER (Byte Character Error Rate) measures how many characters the model gets wrong in training data, not on real documents. A BCER of 1.5% on synthetic images might correspond to 5–10% error on actual historical scans, depending on their condition.

## Corpus design

### Character coverage lines

`download-wiki.py` generates one synthetic line per Unicode codepoint in the Kannada block (U+0C80–U+0CFF). These lines look like:
```
ಕ ಖ ಗ ಘ ಙ ಚ ಛ ಜ ಝ ಞ
```
They ensure the model sees every glyph at least once, regardless of corpus frequency.

> **Unicharset note:** Four characters — `ಋ ಙ ಝ ಱ` — are absent from `tessdata_best/kan.traineddata`'s unicharset. Coverage lines containing them are silently filtered by `02-make-lstmf.sh` unless you first run `scripts/00c-expand-unicharset.sh` to produce `tessdata_expanded/kan.traineddata`. After expansion, the filter is automatically lifted. `ಞ` is present in the unicharset and does not need special handling.

### Specimen corpus

`corpus/generate-specimen.py` produces a systematically designed corpus that guarantees coverage of every vowel, every consonant×matra combination, common conjuncts, and historical vocabulary. Run with `--merge` to append to the existing corpus rather than replace it. This complements Wikipedia prose, which may have gaps in rare character combinations.

### Wikipedia prose

After coverage lines, the corpus uses Kannada Wikipedia prose for natural word and sentence context. Prose lines train the model on real word shapes, spacing, and conjunct frequency — things coverage lines cannot provide.

### Corpus quality rules (applied by `clean-corpus.py`)

- Minimum 8 Kannada characters per line (short lines produce unusable training images)
- Lines over 80 characters are split at word boundaries
- Non-Kannada content stripped (except spaces, digits, and common punctuation: `.,;:()-—'"`)
- Lines with fewer than 60% Kannada characters dropped

## Rendering parameters (`render-corpus.py`)

| Parameter | Value | Rationale |
|---|---|---|
| `FONT_SIZE` | 36 px | Matches typical letterpress point size at 150 DPI |
| `DPI` | 150 | Standard for Tesseract training (not 300 — Tesseract's internal resolution) |
| `PAD_X / PAD_Y` | 20 / 12 px | Enough margin to avoid clipping descenders |
| `MIN_H` | 60 px | Tesseract minimum accepted image height |

### Degradation applied to historical fonts

Applied only when `degrade: true` in `fonts.yml`:

1. **Gaussian blur, radius 0.6** — simulates ink spread into paper fibres
2. **Salt-and-pepper noise, 0.3% of pixels** — simulates paper grain and foxing
3. **Random rotation ±0.8°** — simulates print misalignment

These values are conservative by design. Over-degrading makes the model brittle on clean scans.

## Training dataset size

Approximate rendered images per font at default `max_pages` settings:

| Font | Styles | Max pages | ~Images |
|---|---|---|---|
| Karnata GTN | 4 | 800 each | 3,200 |
| Karnata German Mission Press | 1 | 600 | 600 |
| Karnata Wesleyan Mission Press | 1 | 600 | 600 |
| Karnata F Kittel | 1 | 600 | 600 |
| **Total** | | | **~5,000** |

With a 5,000-line corpus, the total is ~25,000–30,000 training images. Adding real scanned pages from `scan-input/` increases coverage of real-world noise.

### Adjusting dataset size

To reduce training time: lower `max_pages` in `fonts.yml` (e.g. 200–400).
To increase coverage: raise `max_pages` or add more corpus lines to `download-wiki.py --lines`.

## Resuming and branching

Training checkpoints are named `kan_hist_<BCER>_<iter>.checkpoint`. The `03-train.sh` script auto-selects the most recent one. To branch from a specific point:

```bash
CONTINUE_FROM=output/kan_hist_2.5_20000.checkpoint ./scripts/03-train.sh
```

This is useful for trying different learning rates from the same starting point.

### Expanding the unicharset mid-training

If you run `scripts/00c-expand-unicharset.sh` after training has already started, `03-train.sh` will detect `tessdata_expanded/` on the next run and use it as `--traineddata`. Tesseract automatically resizes the LSTM output layer to cover the new characters, preserving all existing weights. New-character output nodes start with random weights and require training examples containing those characters to become useful. Expect BCER to rise slightly for ~5,000 iterations, then recover.

## Adding real scanned training data

Synthetic training (Pillow-rendered fonts) is a baseline. Real scanned pages in `scan-input/` dramatically improve accuracy on actual document scans.

### How to create good scan+GT pairs

1. Scan at 300 DPI or higher, grayscale or bitonal.
2. Crop to a single text block or paragraph (avoid mixed columns).
3. Transcribe the text exactly as it appears — including archaic spelling, missing punctuation, and ligature forms.
4. Save as `scan-input/page001.png` + `scan-input/page001.gt.txt`.

The portal's Scan upload tab handles this via drag-and-drop.

### How many scans?

Even 20–50 high-quality scan+GT pairs noticeably improve a model trained purely on synthetic data. The improvement in generalisation is usually more valuable than adding thousands more synthetic lines.

## Evaluating quality beyond BCER

BCER is a training-time metric. To assess real performance:

1. Place representative historical scans in `test-images/`.
2. Run `./scripts/05-test.sh test-images/scan.tif` to compare `kan` vs `kan_hist` side by side.
3. Manually compute CER on 3–5 pages against a reference transcription to get a real-world error rate.
4. Use the portal OCR test tab for quick visual comparison of individual images.

Target real-world CER (on typical 19th-century Kannada letterpress scans in good condition): under 5%. Heavily damaged or faded material will have higher error rates regardless of model quality.
