# Portal reference — kan_hist Training Portal

The portal is a single-file Flask app (`portal.py`) that provides a browser UI for the full `kan_hist` training pipeline.

## Starting the portal

```bash
bash start-portal.sh          # installs dependencies and starts
# or
python3 portal.py
# → http://localhost:5000

# Custom port
PORT=8080 python3 portal.py
```

## REST API

All API endpoints return JSON unless noted.

### Status

#### `GET /api/status`

Returns completion state of each pipeline step plus whether `lstmtraining` is currently running.

```json
{
  "01_prep":   { "label": "1. Prep base model", "done": true,  "detail": "kan.traineddata ✓  fonts 4/4" },
  "02_corpus": { "label": "2. Build corpus",    "done": true,  "detail": "5,200 lines" },
  "03_render": { "label": "3. Render images",   "done": true,  "detail": "31,200 PNG images" },
  "04_lstmf":  { "label": "4. Generate lstmf",  "done": true,  "detail": "31,200 .lstmf files" },
  "05_train":  { "label": "5. Train",           "done": true,  "detail": "47 checkpoints" },
  "06_package":{ "label": "6. Package",         "done": false, "detail": "Not done" },
  "_training": false
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

Valid step values: `prep`, `wiki`, `clean`, `render`, `lstmf`, `train`, `package`

```bash
curl -X POST http://localhost:5000/api/run/train
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

#### `GET /tessdata/<lang>.traineddata`

Serves traineddata files for Tesseract.js. The OCR test tab fetches models from this endpoint automatically using the same origin (no CORS configuration needed).

- `kan.traineddata` → served from `tessdata_best/`
- `kan_hist.traineddata` → served from `best/`

Only `.traineddata` files are allowed.

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
| `PORT` | `5000` | Port to listen on |
| `BEST_CHECKPOINT` | (auto) | Override checkpoint for packaging step |

## Directory assumptions

The portal expects the following relative to `portal.py`:

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
| `training.log` | Training log (streamed live) |
