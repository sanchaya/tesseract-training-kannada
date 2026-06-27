# TrainOCR — Tesseract training for historical Kannada

[![License](https://img.shields.io/badge/license-Apache%202.0-5B21B6.svg)](LICENSE)
[![Fonts](https://img.shields.io/badge/fonts-SIL%20OFL%201.1-7C3AED.svg)](https://fonts.sanchaya.net)
[![Model](https://img.shields.io/badge/model-kan__hist-059669.svg)](best/)
[![Portal](https://img.shields.io/badge/portal-trainocr.sanchaya.net-F59E0B.svg)](https://trainocr.sanchaya.net)

`kan_hist.traineddata` is a fine-tuned Tesseract 5 LSTM model for OCR of Kannada text printed in 19th-century letterpress typefaces. It is trained on the [Karnata font family](https://fonts.sanchaya.net) — digital revivals of historical Kannada printing types developed by [Sanchaya](https://sanchaya.org).

The standard `kan` model was trained on modern digital fonts and struggles with the distinctive stroke shapes, ink spread, and conjunct forms of historical letterpress material. `kan_hist` fills this gap.

**TrainOCR** wraps the entire pipeline in a web portal — making Tesseract training approachable for librarians, archivists, and language communities, not just ML engineers. Try it at [trainocr.sanchaya.net](https://trainocr.sanchaya.net).

---

## Quick start

### Option A — Docker (recommended)

```bash
git clone https://github.com/sanchaya/tesseract-training-kannada.git
cd tesseract-training-kannada
docker compose up --build
# → http://localhost:3000
```

### Option B — Native Node.js

```bash
git clone https://github.com/sanchaya/tesseract-training-kannada.git
cd tesseract-training-kannada

# Install system dependencies (macOS)
brew install tesseract tesseract-training-tools python git node

# Install Python + Node dependencies
pip install pillow pyyaml --break-system-packages
npm install

# Start the portal
node server.js
# → http://localhost:3000
```

### Option C — Python / Flask (python-portal branch)

```bash
git clone -b python-portal https://github.com/sanchaya/tesseract-training-kannada.git
cd tesseract-training-kannada
pip install -r requirements-portal.txt
python portal.py
# → http://localhost:5000
```

From the portal Dashboard, click each step button in order. Or run the scripts directly (see [Training workflow](#training-workflow) below).

---

## Fonts used

All fonts are from the Sanchaya Karnata family (SIL Open Font License 1.1). Karnata Bandipur is excluded.

| Font | Historical source | Styles | Rendering |
|---|---|---|---|
| [Karnata GTN](https://fonts.sanchaya.net/family/KarnataGTN) | G.T. Narayana Rao handwriting revival | 4 (Regular, Medium, Bold, Black) | Clean |
| [Karnata German Mission Press](https://fonts.sanchaya.net/family/Karnata-German-Mission-Press) | German Mission Press, Mangaluru (19th c.) | 1 | Degraded |
| [Karnata Wesleyan Mission Press](https://fonts.sanchaya.net/family/Karnata-Wesleyan-Mission-Press) | Wesleyan Mission Press, Bengaluru (19th c.) | 1 | Degraded |
| [Karnata F Kittel](https://fonts.sanchaya.net/family/Karnata-F-Kittel-Font) | Basel Mission Press, Mangalore (1830–1900) | 1 | Degraded |

**Clean** = rendered faithfully. **Degraded** = Gaussian blur + salt-and-pepper noise + ±0.8° rotation, simulating real letterpress ink spread and paper texture.

---

## Repository layout

```
kan_hist/
├── best/
│   └── kan_hist.traineddata      ← finished model (output of step 6)
│
├── corpus/
│   ├── download-wikisource.py    ← PREFERRED: proofread pages from kn.wikisource.org
│   ├── download-wiki.py          supplement: Kannada Wikipedia prose
│   ├── clean-corpus.py           clean raw text → kan_corpus.txt
│   ├── render-corpus.py          render corpus → PNG+gt.txt pairs
│   ├── cache/                    downloaded dumps (gitignored — large files)
│   └── kan_corpus.txt            cleaned training text (generated)
│
├── fonts/                        font repos cloned by 01-prep-base.sh
├── rendered/                     PNG+gt.txt training pairs (generated)
├── scan-input/                   optional: real scanned pages + .gt.txt
├── lstmf/                        .lstmf files (generated)
├── output/                       training checkpoints (generated)
├── tessdata_best/                 kan.traineddata base model (downloaded)
├── test-images/                  sample scans for evaluation
│
├── fonts.yml                     ← font registry — single source of truth
├── portal.py                     web portal (Flask)
├── start-portal.sh               one-command portal launcher
│
└── scripts/
    ├── 01-prep-base.sh           clone fonts, download base model
    ├── 02-make-lstmf.sh          PNG+gt.txt → .lstmf training files
    ├── 03-train.sh               fine-tune LSTM
    ├── 04-package.sh             export kan_hist.traineddata
    └── 05-test.sh                compare kan vs kan_hist on an image
```

---

## Dependencies

| Tool | macOS | Ubuntu/Debian |
|---|---|---|
| Tesseract 5 + training tools | `brew install tesseract tesseract-training-tools` | `apt install tesseract-ocr tesseract-ocr-kan libtesseract-dev` |
| Python 3.9+ | `brew install python` | `apt install python3 python3-pip` |
| Pillow | `pip install pillow` | `pip install pillow` |
| PyYAML | `pip install pyyaml` | `pip install pyyaml` |
| Flask (portal only) | `pip install flask` | `pip install flask` |
| Git | `brew install git` | `apt install git` |

---

## Training workflow

### Step 1 — Prep base model and fonts

```bash
./scripts/01-prep-base.sh
```

Downloads `kan.traineddata` from tessdata_best, extracts `kan.lstm`, and clones all font repositories listed in `fonts.yml` into `fonts/`.

**Output:** `tessdata_best/kan.traineddata`, `output/kan.lstm`, `fonts/<id>/` for each font.

---

### Step 2 — Build the corpus

The corpus is Kannada GT text that is rendered into training images. Two sources are available; Wikisource is strongly preferred for `kan_hist`.

#### Preferred: Kannada Wikisource (proofread pages)

[kn.wikisource.org](https://kn.wikisource.org) contains human-proofread transcriptions of scanned Kannada books — many of them 19th-century texts typeset in the same letterpress fonts (German Mission Press, Wesleyan Mission Press, Basel Mission Press) that `kan_hist` is trained on. This is the highest-quality GT source for historical Kannada OCR.

Wikisource pages carry a quality rating:
| Level | Meaning |
|---|---|
| 1 | Not proofread |
| 2 | Problematic |
| **3** | **Proofread** — reviewed by one human ✓ |
| **4** | **Validated** — reviewed by two humans ✓✓ |

```bash
# Download proofread + validated pages (quality ≥ 3, dumps ~80 MB, cached)
python3 corpus/download-wikisource.py --pages 3000

# For validated-only (highest quality):
python3 corpus/download-wikisource.py --pages 3000 --quality 4
```

The script downloads the knwikisource XML dump (~80 MB, cached in `corpus/cache/`), extracts `Page:` namespace entries, strips all Wikisource templates and wiki markup, and appends clean lines to `corpus/raw_kannada.txt`.

#### Supplement: Kannada Wikipedia (modern prose)

Wikipedia provides modern Kannada prose — useful for Unicode coverage but less relevant to historical typography. Use it to supplement, not replace, Wikisource.

```bash
# Download modern Kannada text (~150 MB dump, cached)
python3 corpus/download-wiki.py --lines 5000
```

`download-wiki.py` also generates character coverage lines — one line per Kannada Unicode codepoint (U+0C80–U+0CFF) — to guarantee complete glyph coverage regardless of corpus content.

#### Clean and prepare

```bash
python3 corpus/clean-corpus.py
```

The cleaner keeps lines with ≥ 8 Kannada characters, strips markdown/wiki artifacts (`*`, `#`, `==`), and drops lines longer than 80 characters. Output is `corpus/kan_corpus.txt`.

You can also supply your own corpus — write lines to `corpus/raw_kannada.txt` and run `clean-corpus.py`.

**Output:** `corpus/kan_corpus.txt` (~5,000+ lines)

---

### Step 3 — Render training images

```bash
python3 corpus/render-corpus.py
```

For each corpus line × each font style, renders a 150 DPI PNG image with matching `.gt.txt` ground-truth file. Historical/letterpress fonts get degradation applied automatically based on the `degrade: true` flag in `fonts.yml`.

**Output:** `rendered/<font_id>_<style>_line<N>.png` + matching `.gt.txt`

**To include real scanned pages:** place `page.png` (or `.tif`) + `page.gt.txt` in `scan-input/`. They will be picked up in the next step.

---

### Step 4 — Generate lstmf files

```bash
./scripts/02-make-lstmf.sh
```

Converts all PNG+gt.txt pairs in `rendered/` and `scan-input/` into Tesseract `.lstmf` binary training files. Writes the file list to `lstmf/list.txt`.

**Output:** `lstmf/*.lstmf`, `lstmf/list.txt`

---

### Step 5 — Train

```bash
# Recommended: background process with caffeinate (macOS)
caffeinate -i ./scripts/03-train.sh > training.log 2>&1 &
tail -f training.log
```

Fine-tunes `kan.lstm` for up to 100,000 iterations at learning rate 0.001. Saves a checkpoint every 100 iterations. Checkpoint filenames encode BCER and iteration: `kan_hist_<BCER>_<iter>.checkpoint`.

**What to expect:**

| Iteration range | Typical BCER | Notes |
|---|---|---|
| 0–1,000 | 5–15% | Model adjusting to new letterforms |
| 1,000–5,000 | Dropping steadily | Watch for consistent improvement |
| 5,000–20,000 | 1–5% | Rate of improvement slows |
| 20,000+ | Plateau | Stop when no new best for ~10,000 iterations |

**Safe to stop at any time** with Ctrl+C. Resume by re-running `03-train.sh` — it automatically picks up from the most recent checkpoint.

To resume from a specific checkpoint:
```bash
CONTINUE_FROM=output/kan_hist_2.1_50000.checkpoint ./scripts/03-train.sh
```

**Output:** `output/kan_hist_<BCER>_<iter>.checkpoint`, `training.log`

---

### Step 6 — Package

```bash
./scripts/04-package.sh
```

Selects the checkpoint with the lowest BCER, exports it using `lstmtraining --stop_training`, combines with the Kannada wordlist and dawg files, and writes `best/kan_hist.traineddata`.

**Output:** `best/kan_hist.traineddata`

---

### Step 7 — Test

```bash
./scripts/05-test.sh test-images/your_scan.tif
```

Runs both `kan` (base) and `kan_hist` on the same image and prints the outputs side by side for comparison.

---

## Training portal

The portal provides a browser UI for running the pipeline, monitoring progress, and testing the finished model.

```bash
bash start-portal.sh
# → http://localhost:5000
```

### Tabs

**Dashboard** — pipeline status for each step, run buttons, live training indicator, and summary statistics (images rendered, corpus lines, best BCER).

**OCR test** — upload any Kannada scan and run both `kan` and `kan_hist` through Tesseract.js entirely in the browser. Results appear side by side. Paste ground-truth text to get CER (character error rate) and WER (word error rate) computed live. Includes preprocessing toggles (binarize, denoise, enhance contrast) applied before OCR. Requires the packaging step to be complete for `kan_hist`.

**Scan upload** — drag-and-drop real scanned pages and paste ground-truth text. Files are saved to `scan-input/` and included automatically in the next make-lstmf run.

**Fonts** — font registry from `fonts.yml`, showing clone status and rendered image count per font.

**Training** — BCER chart (Chart.js), corpus statistics (Unicode coverage, character frequency), and checkpoint browser. Any checkpoint can be packaged directly from this tab.

**Live log** — streaming tail of `training.log` with colour-coded BCER, checkpoint, and error lines.

**Images** — sampled gallery of rendered training images.

**Shareable report** — click "Report" in the header to download a self-contained HTML file with the BCER curve, pipeline status, and sample images, suitable for sharing without running the server.

---

## Adding a new font

1. Add an entry to `fonts.yml`:

```yaml
- id: kan_newpress
  name: "Karnata New Press"
  description: "Description of the historical source"
  repo: https://github.com/sanchaya/karnata-new-press-typeface
  font_dir: fonts
  font_files:
    - KarnataNewPress.ttf
  degrade: true          # true for historical/letterpress, false for modern
  max_pages: 600
```

2. Re-run the affected steps:

```bash
./scripts/01-prep-base.sh      # clones the new repo
python3 corpus/render-corpus.py
./scripts/02-make-lstmf.sh
./scripts/03-train.sh          # continues from last checkpoint
./scripts/04-package.sh
```

The new font's images are merged with the existing dataset. Training resumes from the latest checkpoint, so previous training is not wasted.

---

## Using the model

```bash
# Install
cp best/kan_hist.traineddata /opt/homebrew/share/tessdata/         # macOS Homebrew
cp best/kan_hist.traineddata /usr/share/tesseract-ocr/5/tessdata/  # Linux

# Run
tesseract historical_document.tif output -l kan_hist

# Side-by-side comparison
tesseract scan.tif out_base -l kan
tesseract scan.tif out_hist -l kan_hist
diff out_base.txt out_hist.txt
```

Use `kan_hist` for scanned historical Kannada material printed with letterpress typefaces — mission press books, newspapers, government records (roughly pre-1960).

Use `kan` for modern digital or typeset Kannada text.

---

## Troubleshooting

**`lstmtraining: command not found`**
Tesseract training tools are not installed. On macOS: `brew install tesseract-training-tools`. On Ubuntu: `apt install libtesseract-dev`.

**Tesseract renders blank or crashes on a font**
The font may not contain the required Unicode codepoints. Check: `fc-query fonts/<id>/font.ttf | grep -i kannada`. If coverage is missing, set `max_pages` lower or remove that font file from `font_files` in `fonts.yml`.

**Training BCER stays above 10% after 5,000 iterations**
Likely causes: corpus too short, all images look identical, or the base `kan.lstm` is not the right starting point. Try reducing `--learning_rate` to `0.0001` in `03-train.sh` and checking that `lstmf/list.txt` has at least 1,000 entries.

**Portal: `kan_hist` OCR returns "Model not available yet"**
Complete the packaging step (Step 6 / Dashboard → ⑦ Package) first. The portal serves `kan_hist.traineddata` from `best/` over HTTP for Tesseract.js.

**`Operation not permitted` when deleting old script files**
macOS system integrity protection can prevent deleting files in certain directories. Use `sudo rm` or move them instead.

---

## Project structure in depth

See [`docs/TRAINING.md`](docs/TRAINING.md) for detail on training parameters, BCER interpretation, and how to tune the model. See [`docs/PORTAL.md`](docs/PORTAL.md) for the portal's REST API reference.

---

## Branches

| Branch | Contents | Start command |
|--------|----------|---------------|
| `master` | Node.js / Express portal + Docker | `node server.js` or `docker compose up` |
| `python-portal` | Flask / Python portal | `python portal.py` |
| `static` | Standalone HTML frontend (no backend) | open `public/index.html` |

---

## License

| Component | License |
|---|---|
| Code — scripts, server, portal, corpus tools | **[Apache 2.0](LICENSE)** |
| Karnata fonts | [SIL Open Font License 1.1](https://fonts.sanchaya.net) |
| Base model (`kan.traineddata`) | Apache 2.0 (Google / Tesseract project) |
| Trained model (`kan_hist.traineddata`) | Apache 2.0 |

Copyright 2025 [Sanchaya / Sanchi Foundation](https://sanchaya.org)
