# Portal reference — kan_hist Training Portal

The portal is a Node.js / Express app (`server.js`) that provides a browser UI for the full `kan_hist` training pipeline.

## Starting the portal

```bash
node server.js
# → http://localhost:3000

# Custom port
PORT=8080 node server.js

# Dev mode (auto-restarts on file change)
node --watch server.js
```

## REST API

All API endpoints return JSON unless noted.

### Status

#### `GET /api/status`

Returns completion state of each pipeline step plus whether `lstmtraining` is currently running.

```json
{
  "00_unichar": { "label": "Expand unicharset", "done": false, "detail": "Not done — ಋ ಙ ಝ ಱ missing" },
  "01_prep":    { "label": "1. Prep base",      "done": true,  "detail": "kan.traineddata ✓  fonts 4/4" },
  "02_corpus":  { "label": "2. Corpus",         "done": true,  "detail": "5,200 lines" },
  "03_render":  { "label": "3. Render images",  "done": true,  "detail": "31,200 PNG images" },
  "04_lstmf":   { "label": "4. Make lstmf",     "done": true,  "detail": "31,200 .lstmf files" },
  "05_train":   { "label": "5. Train",          "done": true,  "detail": "47 checkpoints" },
  "06_package": { "label": "6. Package",        "done": false, "detail": "Not done" },
  "_training": false,
  "_runningStep": null,
  "_completedSteps": {}
}
```

#### `GET /api/fonts`

Returns font registry from `fonts.yml` with runtime status.

```json
[
  {
    "id": "kan_gtn",
    "name": "Karnata GTN",
    "repo": "https://github.com/sanchaya/karnata-gtn-typeface",
    "styles": 4,
    "degrade": false,
    "cloned": true,
    "rendered": 3200
  }
]
```

#### `GET /api/bcer`

Returns BCER (character error rate) history parsed from checkpoint filenames, or from `training.log` if no named checkpoints are found.

```json
[
  { "iter": 100,   "bcer": 12.34 },
  { "iter": 200,   "bcer": 9.87  },
  { "iter": 50000, "bcer": 1.23  }
]
```

#### `GET /api/checkpoints`

Returns detailed checkpoint list with the current best identified.

```json
{
  "checkpoints": [
    { "file": "kan_hist_1.234_50000.checkpoint", "bcer": 1.234, "iter": 50000, "size": 15728640 }
  ],
  "best": { "file": "kan_hist_1.234_50000.checkpoint", "bcer": 1.234, "iter": 50000, "size": 15728640 }
}
```

#### `GET /api/corpus/stats`

Returns character-level statistics for the built corpus.

```json
{
  "total_lines": 5200,
  "total_chars": 418000,
  "kan_chars": 395000,
  "unique_kan": 121,
  "coverage_pct": 94.5,
  "top_chars": [
    { "ch": "ಕ", "count": 12400 }
  ]
}
```

#### `GET /api/images?n=<count>`

Returns up to `n` base64-encoded sample images from `rendered/` (default: 16).

```json
[
  { "name": "kan_gtn_regular_line001", "b64": "iVBOR...", "gt": "ಕನ್ನಡ ಭಾಷೆ" }
]
```

### Running steps

#### `POST /api/run/<step>`

Starts a pipeline step in a background thread. Output is appended to `training.log`.

Valid step values: `expandunichar`, `prep`, `wiki`, `clean`, `specimen`, `render`, `lstmf`, `train`, `package`

| Step | Script | Notes |
|---|---|---|
| `expandunichar` | `scripts/00c-expand-unicharset.sh` | Adds ಋ ಙ ಝ ಱ to unicharset → `tessdata_expanded/` |
| `prep` | `scripts/01-prep-base.sh` | Downloads base model, clones fonts |
| `wiki` | `corpus/download-wiki.py` | Download Kannada Wikipedia corpus |
| `clean` | `corpus/clean-corpus.py` | Clean raw corpus |
| `specimen` | `corpus/generate-specimen.py --merge` | Generate systematic glyph-coverage corpus |
| `render` | `corpus/render-corpus.py` | Render PNG+gt.txt pairs |
| `lstmf` | `scripts/02-make-lstmf.sh` | PNG+gt.txt → .lstmf files |
| `train` | `scripts/03-train.sh` | Fine-tune LSTM |
| `package` | `scripts/04-package.sh` | Export `kan_hist.traineddata` |

```bash
curl -X POST http://localhost:3000/api/run/expandunichar
curl -X POST http://localhost:3000/api/run/train
```

```json
{ "ok": true, "step": "train" }
```

### Checkpoints

#### `POST /api/checkpoints/package`

Packages a specific checkpoint instead of the auto-selected best.

```bash
curl -X POST http://localhost:5000/api/checkpoints/package \
  -H "Content-Type: application/json" \
  -d '{"checkpoint": "kan_hist_1.234_50000.checkpoint"}'
```

```json
{ "ok": true, "checkpoint": "kan_hist_1.234_50000.checkpoint" }
```

### Scan upload

#### `GET /api/scans`

Lists files in `scan-input/`.

```json
[
  { "name": "page001.png", "has_gt": true, "gt_preview": "ಕನ್ನಡ ಭಾಷ...", "size": 204800 }
]
```

#### `POST /api/scans/upload`

Accepts multipart form data: `image` (file) + `gt` (text). Saves to `scan-input/`.

```bash
curl -X POST http://localhost:5000/api/scans/upload \
  -F "image=@page001.png" \
  -F "gt=ಕನ್ನಡ ಭಾಷೆ"
```

```json
{ "ok": true, "saved": "page001.png", "has_gt": true }
```

#### `POST /api/scans/delete`

Deletes a scan and its `.gt.txt` file.

```bash
curl -X POST http://localhost:5000/api/scans/delete \
  -H "Content-Type: application/json" \
  -d '{"name": "page001.png"}'
```

### Image preprocessing

#### `POST /api/preprocess`

Applies server-side image preprocessing (Pillow) and returns the result as a base64 PNG.

```bash
curl -X POST http://localhost:5000/api/preprocess \
  -H "Content-Type: application/json" \
  -d '{"image_b64": "data:image/png;base64,...", "ops": ["binarize", "denoise"]}'
```

Available `ops`: `binarize`, `denoise`, `enhance_contrast`

```json
{ "image_b64": "data:image/png;base64,..." }
```

### Tesseract.js model serving

#### `GET /traineddata/<lang>.traineddata[.gz]`

Serves traineddata files for Tesseract.js. The Live OCR tab fetches models from this endpoint automatically.

- `kan_hist.traineddata` → served from `best/`
- `kan_hist.traineddata.gz` → gzip-compressed on-the-fly (Tesseract.js v5 always requests `.gz`)

Tesseract.js v5 requests the `.gz` form first. The server compresses the file on-the-fly using Node's built-in `zlib` — no pre-compressed copy needed. Only `.traineddata` files are served; other extensions return 403.

#### `DELETE /api/rendered`

Deletes all files in `rendered/` (PNGs, `.gt.txt`, and subdirectories like `font-test/`). Used by the "↺ Clear & re-render" button.

#### `DELETE /api/test-images[?fontId=<id>]`

Deletes character test images. Without `fontId`, clears all fonts. With `fontId`, clears only that font's directory.

### Log streaming

#### `GET /api/log/stream`

Server-Sent Events (SSE) stream of `training.log`. Each event carries one line as a JSON string.

```javascript
const src = new EventSource('/api/log/stream');
src.onmessage = e => console.log(JSON.parse(e.data));
```

#### `GET /api/log/tail?lines=<n>`

Returns the last `n` lines of `training.log` (default: 100).

```json
{ "lines": ["At iteration 50000 ..."], "exists": true, "total": 4200 }
```

### Report

#### `GET /report`

Returns a self-contained HTML report as a file download. Includes BCER chart, pipeline status table, font list, and sample images — no server required to view.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `3000` | Port to listen on |
| `BEST_CHECKPOINT` | (auto) | Override checkpoint for packaging step |

## Directory assumptions

The portal expects the following relative to `server.js`:

| Path | Purpose |
|---|---|
| `fonts.yml` | Font registry |
| `corpus/kan_corpus.txt` | Cleaned training corpus |
| `lstmf/list.txt` | lstmf file list |
| `output/` | Training checkpoints |
| `rendered/` | PNG+gt.txt pairs |
| `scan-input/` | Real scans (created automatically) |
| `best/` | Packaged `kan_hist.traineddata` |
| `tessdata_best/` | Base `kan.traineddata` |
| `tessdata_expanded/` | Expanded unicharset traineddata (created by `00c-expand-unicharset.sh`) |
| `tmp/langdata_lstm/` | Cached langdata from GitHub (created by `00c-expand-unicharset.sh`) |
| `training.log` | Training log (streamed live) |
