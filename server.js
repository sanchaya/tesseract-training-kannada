/**
 * server.js — TrainOCR by Sanchaya
 *
 * Node.js / Express backend for the kan_hist Tesseract training portal.
 * Serves the frontend from public/ and exposes a REST API for all pipeline
 * operations. Tesseract.js OCR runs in the browser — this server just
 * serves the traineddata files and the static UI.
 *
 * Usage:
 *   npm install
 *   node server.js          # production
 *   node --watch server.js  # dev with auto-restart
 *
 * Deploy:
 *   pm2 start ecosystem.config.js
 *   # Reverse-proxy with nginx → trainocr.sanchaya.net
 */

"use strict";

const fs        = require("fs");
const path      = require("path");
const { execFile, spawn } = require("child_process");
const { promisify } = require("util");

const express   = require("express");
const yaml      = require("js-yaml");
const multer    = require("multer");

const execFileAsync = promisify(execFile);

const app  = express();
const PORT = process.env.PORT || 3000;
const ROOT = __dirname;

// ── Paths ──────────────────────────────────────────────────────────────────
const P = {
  fontsYml:    path.join(ROOT, "fonts.yml"),
  corpusTxt:   path.join(ROOT, "corpus", "kan_corpus.txt"),
  lstmfList:   path.join(ROOT, "lstmf", "list.txt"),
  outputDir:   path.join(ROOT, "output"),
  rendered:    path.join(ROOT, "rendered"),
  scanDir:     path.join(ROOT, "scan-input"),
  logFile:     path.join(ROOT, "training.log"),
  bestDir:     path.join(ROOT, "best"),
  tessdataDir: path.join(ROOT, "tessdata_best"),
  scripts:     path.join(ROOT, "scripts"),
  corpus:      path.join(ROOT, "corpus"),
  public:      path.join(ROOT, "public"),
  testImages:  path.join(ROOT, "test-images"),
};

fs.mkdirSync(P.scanDir,     { recursive: true });
fs.mkdirSync(P.public,      { recursive: true });
fs.mkdirSync(P.testImages,  { recursive: true });

// ── Middleware ─────────────────────────────────────────────────────────────
app.use(express.json({ limit: "50mb" }));
app.use(express.static(P.public));
app.use("/test-images", express.static(path.join(ROOT, "test-images")));
app.use("/fonts",       express.static(path.join(ROOT, "fonts"),
  { setHeaders: (res) => res.setHeader("Access-Control-Allow-Origin", "*") }));

// ── Serve traineddata for Tesseract.js in-browser testing ─────────────────
// Tesseract.js v5 always fetches <lang>.traineddata.gz first — serve gzipped on the fly
app.get("/traineddata/:filename", (req, res) => {
  const fn      = req.params.filename;
  const wantsGz = fn.endsWith('.traineddata.gz');
  const baseName = wantsGz ? fn.slice(0, -3) : fn;
  if (!baseName.endsWith('.traineddata')) return res.status(403).end();

  const candidates = [
    path.join(ROOT, "best",          baseName),
    path.join(ROOT, "tessdata_best", baseName),
    path.join(ROOT, baseName),
  ];
  const f = candidates.find(c => fs.existsSync(c));
  if (!f) return res.status(404).json({ error: `${baseName} not found` });

  res.setHeader("Access-Control-Allow-Origin", "*");
  if (wantsGz) {
    res.setHeader("Content-Type", "application/gzip");
    const zlib = require("zlib");
    fs.createReadStream(f).pipe(zlib.createGzip()).pipe(res);
  } else {
    res.setHeader("Content-Type", "application/octet-stream");
    res.sendFile(f);
  }
});

const upload = multer({ dest: path.join(ROOT, "tmp") });

// ── Helpers ────────────────────────────────────────────────────────────────
function loadFonts() {
  if (!fs.existsSync(P.fontsYml)) return [];
  return yaml.load(fs.readFileSync(P.fontsYml, "utf8")).fonts || [];
}

function isTrainingRunning() {
  try {
    const result = require("child_process")
      .spawnSync("pgrep", ["-x", "lstmtraining"]);
    return result.status === 0;
  } catch { return false; }
}

function globCount(dir, ext) {
  if (!fs.existsSync(dir)) return 0;
  return fs.readdirSync(dir).filter(f => f.endsWith(ext)).length;
}

function lineCount(file) {
  if (!fs.existsSync(file)) return 0;
  return fs.readFileSync(file, "utf8").split("\n").filter(Boolean).length;
}

function getCheckpoints() {
  if (!fs.existsSync(P.outputDir)) return [];
  return fs.readdirSync(P.outputDir)
    // format: kan_hist_<BCER>_<ITER>_<SAMPLES>.checkpoint
    .filter(f => /^kan_hist_[\d.]+_\d+_\d+\.checkpoint$/.test(f))
    .map(f => {
      const parts = f.replace(".checkpoint", "").split("_");
      // parts: ["kan","hist",<bcer>,<iter>,<samples>]
      const stat  = fs.statSync(path.join(P.outputDir, f));
      return {
        file:    f,
        bcer:    parseFloat(parts[2]),
        iter:    parseInt(parts[3]),
        samples: parseInt(parts[4]),
        size:    stat.size,
      };
    })
    .sort((a, b) => a.iter - b.iter);
}

function getBcerHistory() {
  const pts = getCheckpoints();
  if (pts.length) return pts.map(p => ({ iter: p.iter, bcer: p.bcer }));
  if (!fs.existsSync(P.logFile)) return [];
  const log = fs.readFileSync(P.logFile, "utf8");
  const re  = /At iteration\s+(\d+).*?BCER train=([\d.]+)%/g;
  const out = [];
  let m;
  while ((m = re.exec(log)) !== null) out.push({ iter: +m[1], bcer: +m[2] });
  return out;
}

function corpusStats() {
  if (!fs.existsSync(P.corpusTxt)) return null;
  const text   = fs.readFileSync(P.corpusTxt, "utf8");
  const kanRe  = /[ಀ-೿]/g;
  const kanAll = text.match(kanRe) || [];
  const unique = new Set(kanAll);
  const freq   = {};
  kanAll.forEach(c => { freq[c] = (freq[c] || 0) + 1; });
  const top = Object.entries(freq)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 15)
    .map(([ch, count]) => ({ ch, count }));
  return {
    total_lines:  text.split("\n").filter(Boolean).length,
    total_chars:  text.length,
    kan_chars:    kanAll.length,
    unique_kan:   unique.size,
    coverage_pct: +(unique.size / 128 * 100).toFixed(1),
    top_chars:    top,
  };
}

function sampleImages(n = 16) {
  if (!fs.existsSync(P.rendered)) return [];
  const pngs = fs.readdirSync(P.rendered).filter(f => f.endsWith(".png")).sort();
  const step = Math.max(1, Math.floor(pngs.length / n));
  return pngs.filter((_, i) => i % step === 0).slice(0, n).map(f => {
    const stem  = f.replace(".png", "");
    const gtF   = path.join(P.rendered, stem + ".gt.txt");
    const gt    = fs.existsSync(gtF)
      ? fs.readFileSync(gtF, "utf8").trim().slice(0, 70)
      : "";
    return {
      name: stem,
      b64:  fs.readFileSync(path.join(P.rendered, f)).toString("base64"),
      gt,
    };
  });
}

function runBg(cmd, args, stepId, opts = {}) {
  runningStep = stepId;
  const log  = fs.openSync(P.logFile, "a");
  const sep  = "=".repeat(55);
  fs.writeSync(log, `\n${sep}\n[trainocr] ${stepId}: ${cmd} ${args.join(" ")}\n${new Date().toISOString()}\n${sep}\n\n`);
  fs.closeSync(log);

  const child = spawn(cmd, args, {
    cwd:   ROOT,
    stdio: ["ignore", fs.openSync(P.logFile, "a"), fs.openSync(P.logFile, "a")],
    detached: false,
    env: { ...process.env, ...(opts.env || {}) },
  });
  child.on("exit", code => {
    runningStep = null;
    completedSteps[stepId] = { code, ts: Date.now(), ok: code === 0 };
    const l = fs.openSync(P.logFile, "a");
    fs.writeSync(l, `\n[trainocr] ${stepId} ${code === 0 ? "✓ done" : `✗ failed (exit ${code})`}\n`);
    fs.closeSync(l);
  });
}

/// ── API: corpus sources ──────────────────────────────────────────────────────
app.get("/api/corpus/sources", (req, res) => {
  const cacheDir   = path.join(__dirname, "corpus", "cache");
  const corpusFile = path.join(__dirname, "corpus", "kan_corpus.txt");
  const rawFile    = path.join(__dirname, "corpus", "raw_kannada.txt");

  const dumpSizes = {
    wikisource: path.join(cacheDir, "knwikisource-latest.xml.bz2"),
    wikipedia:  path.join(cacheDir, "knwiki-latest.xml.bz2"),
  };

  function fileInfo(p) {
    try { const s = fs.statSync(p); return { exists: true, size: s.size, mtime: s.mtimeMs }; }
    catch { return { exists: false }; }
  }
  function countLines(p) {
    try {
      const txt = fs.readFileSync(p, "utf8");
      return txt.split("\n").filter(l => l.trim()).length;
    } catch { return 0; }
  }

  const ws  = fileInfo(dumpSizes.wikisource);
  const wk  = fileInfo(dumpSizes.wikipedia);
  const raw = fileInfo(rawFile);
  const rawLines = raw.exists ? countLines(rawFile) : 0;

  res.json({
    sources: [
      {
        key:    "wikisource",
        name:   "Kannada Wikisource (preferred)",
        detail: ws.exists
          ? `Dump cached — ${(ws.size/1e6).toFixed(1)} MB — last updated ${new Date(ws.mtime).toLocaleDateString()}`
          : "Not downloaded yet — proofread historical Kannada pages",
        ready:       ws.exists,
        downloading: false,
        lines:       ws.exists ? rawLines : 0,
        cmd: "python3 corpus/download-wikisource.py --pages 3000",
      },
      {
        key:    "wikipedia",
        name:   "Kannada Wikipedia (supplement)",
        detail: wk.exists
          ? `Dump cached — ${(wk.size/1e6).toFixed(1)} MB — last updated ${new Date(wk.mtime).toLocaleDateString()}`
          : "Not downloaded yet — modern Kannada prose + glyph coverage",
        ready:       wk.exists,
        downloading: false,
        lines:       0,
        cmd: "python3 corpus/download-wiki.py --lines 5000",
      },
    ],
    corpus_lines: countLines(corpusFile),
  });
});

app.post("/api/corpus/download/:key", (req, res) => {
  const key = req.params.key;
  const scripts = {
    wikisource: "python3 corpus/download-wikisource.py --pages 3000",
    wikipedia:  "python3 corpus/download-wiki.py --lines 5000",
  };
  const cmd = scripts[key];
  if (!cmd) return res.status(400).json({ error: "unknown key" });
  const logFile = path.join(__dirname, "logs", `download-${key}.log`);
  fs.mkdirSync(path.join(__dirname, "logs"), { recursive: true });
  const { spawn } = require("child_process");
  const child = spawn("bash", ["-c", cmd], {
    cwd: __dirname,
    detached: true,
    stdio: ["ignore", fs.openSync(logFile, "a"), fs.openSync(logFile, "a")],
  });
  child.unref();
  res.json({ started: true, log: logFile });
});

// ── API: system check ─────────────────────────────────────────────────────
app.get("/api/syscheck", async (req, res) => {
  const { execFile } = require("child_process");
  const util = require("util");
  const execP = util.promisify(execFile);

  async function ver(cmd, args, re) {
    try {
      const { stdout, stderr } = await execP(cmd, args, { timeout: 5000 });
      const m = (stdout + stderr).match(re);
      return m ? m[0] : "found";
    } catch { return null; }
  }

  const [tessVer, lstmVer, combineVer, py3Ver, node] = await Promise.all([
    ver("tesseract",      ["--version"],  /tesseract\s+[\d.]+/i),
    ver("lstmtraining",   ["--version"],  /[\d.]+/),
    ver("combine_tessdata",["--version"], /[\d.]+/),
    ver("python3",        ["--version"],  /[\d.]+/),
    ver("node",           ["--version"],  /[\d.]+/),
  ]);

  // Pillow
  let pillowVer = null;
  try {
    const { stdout } = await execP("python3", ["-c", "import PIL; print(PIL.__version__)"], { timeout: 5000 });
    pillowVer = stdout.trim();
  } catch { pillowVer = null; }

  const kanBase   = fs.existsSync(path.join(P.tessdataDir, "kan.traineddata"));
  const diskBytes = fs.statfsSync ? null : null; // skip if unavailable

  res.json({
    tools: [
      { name: "Tesseract OCR",       key: "tesseract",       ver: tessVer,
        ok: !!tessVer && /tesseract\s+5\./i.test(tessVer||""),
        warn: !!tessVer && /tesseract\s+4\./i.test(tessVer||""),
        need: "5.x required — macOS: brew install tesseract  |  Ubuntu: ppa:alex-p/tesseract-ocr5" },
      { name: "lstmtraining",        key: "lstmtraining",    ver: lstmVer,    ok: !!lstmVer,    need: "bundled with Tesseract" },
      { name: "combine_tessdata",    key: "combine_tessdata",ver: combineVer, ok: !!combineVer, need: "bundled with Tesseract" },
      { name: "Python 3",            key: "python3",         ver: py3Ver,     ok: !!py3Ver,     need: "3.8+" },
      { name: "Pillow (PIL)",        key: "pillow",          ver: pillowVer,  ok: !!pillowVer,  need: "pip install Pillow" },
      { name: "Node.js",             key: "node",            ver: node,       ok: !!node,       need: "18+" },
    ],
    data: [
      { name: "Base model (kan.traineddata)", key: "kan_base",  ok: kanBase,
        detail: kanBase ? "tessdata_best/kan.traineddata" : "Run step ① to download" },
      { name: "Rendered training images",     key: "rendered",  ok: globCount(P.rendered, ".png") > 0,
        detail: `${globCount(P.rendered, ".png").toLocaleString()} PNG files in rendered/` },
      { name: "lstmf training files",         key: "lstmf",     ok: lineCount(P.lstmfList) > 0,
        detail: `${lineCount(P.lstmfList).toLocaleString()} files in lstmf/list.txt` },
    ],
  });
});

// ── API: status ────────────────────────────────────────────────────────────
app.get("/api/status", (req, res) => {
  const fonts    = loadFonts();
  const cloned   = fonts.filter(f =>
    fs.existsSync(path.join(ROOT, "fonts", f.id, ".git"))).length;
  const rendered = globCount(P.rendered, ".png");
  const lstmfN   = lineCount(P.lstmfList);
  const cps      = getCheckpoints();
  const corpN    = lineCount(P.corpusTxt);
  const bestOk   = fs.existsSync(path.join(P.bestDir, "kan_hist.traineddata"));

  res.json({
    "00_unichar": { label: "Expand unicharset", done: fs.existsSync(path.join(ROOT, "tessdata_expanded", "kan.traineddata")), detail: fs.existsSync(path.join(ROOT, "tessdata_expanded", "kan.traineddata")) ? "Expanded (ಋ ಙ ಝ ಱ added)" : "Not done — ಋ ಙ ಝ ಱ missing" },
    "01_prep":   { label: "1. Prep base",      done: fs.existsSync(path.join(P.tessdataDir, "kan.traineddata")) && cloned === fonts.length, detail: `kan.traineddata ${fs.existsSync(path.join(P.tessdataDir,"kan.traineddata"))?"✓":"✗"}  fonts ${cloned}/${fonts.length}` },
    "02_corpus": { label: "2. Corpus",         done: corpN > 0,     detail: `${corpN.toLocaleString()} lines` },
    "03_render": { label: "3. Render images",  done: rendered > 0,  detail: `${rendered.toLocaleString()} PNG images` },
    "04_lstmf":  { label: "4. Make lstmf",     done: lstmfN > 0,    detail: `${lstmfN.toLocaleString()} .lstmf files` },
    "05_train":  { label: "5. Train",          done: cps.length > 0, detail: cps.length ? `${cps.length} checkpoints` : "Not started" },
    "06_package":{ label: "6. Package",        done: bestOk,        detail: bestOk ? "kan_hist.traineddata ready" : "Not done" },
    _training:       isTrainingRunning(),
    _runningStep:    runningStep,
    _completedSteps: completedSteps,
  });
});

// ── API: fonts ─────────────────────────────────────────────────────────────
app.get("/api/fonts", (req, res) => {
  const fonts = loadFonts();
  res.json(fonts.map(f => ({
    id:       f.id,
    name:     f.name,
    repo:     f.repo,
    styles:   (f.font_files || []).length,
    degrade:  !!f.degrade,
    cloned:   fs.existsSync(path.join(ROOT, "fonts", f.id, ".git")),
    rendered: globCount(P.rendered, ".png"),
  })));
});

// ── API: BCER ──────────────────────────────────────────────────────────────
app.get("/api/bcer", (req, res) => res.json(getBcerHistory()));

// ── API: checkpoints ───────────────────────────────────────────────────────
app.get("/api/checkpoints", (req, res) => {
  const cps  = getCheckpoints();
  const best = cps.length ? cps.reduce((b, c) => c.bcer < b.bcer ? c : b) : null;
  res.json({ checkpoints: cps, best });
});

app.post("/api/checkpoints/package", (req, res) => {
  const { checkpoint } = req.body;
  if (!checkpoint) return res.status(400).json({ error: "No checkpoint specified" });
  const cpPath = path.join(P.outputDir, checkpoint);
  if (!fs.existsSync(cpPath)) return res.status(404).json({ error: "Checkpoint not found" });
  const env = { ...process.env, BEST_CHECKPOINT: cpPath };
  runBg("bash", [path.join(P.scripts, "04-package.sh")], `package:${checkpoint}`);
  res.json({ ok: true, checkpoint });
});

// ── API: corpus stats ──────────────────────────────────────────────────────
app.get("/api/corpus/stats", (req, res) => res.json(corpusStats()));

// ── API: images ────────────────────────────────────────────────────────────
app.get("/api/images", (req, res) => {
  const n = parseInt(req.query.n) || 16;
  res.json(sampleImages(n));
});

// ── API: scans ─────────────────────────────────────────────────────────────
app.get("/api/scans", (req, res) => {
  const files = fs.existsSync(P.scanDir)
    ? fs.readdirSync(P.scanDir).filter(f => /\.(png|tiff?|jpg)$/i.test(f))
    : [];
  res.json(files.map(f => {
    const stem = f.replace(/\.[^.]+$/, "");
    const gtF  = path.join(P.scanDir, stem + ".gt.txt");
    const stat = fs.statSync(path.join(P.scanDir, f));
    return {
      name:       f,
      has_gt:     fs.existsSync(gtF),
      gt_preview: fs.existsSync(gtF) ? fs.readFileSync(gtF, "utf8").trim().slice(0, 60) : "",
      size:       stat.size,
    };
  }));
});

app.post("/api/scans/upload", upload.single("image"), (req, res) => {
  if (!req.file) return res.status(400).json({ error: "No image file" });
  const gt  = (req.body.gt || "").trim();
  const ext = path.extname(req.file.originalname).toLowerCase() || '.png';
  // Sanitize filename — replace spaces/special chars, keep ASCII+extension
  const rawStem = path.basename(req.file.originalname, path.extname(req.file.originalname));
  const stem    = rawStem.replace(/[^\w\-]/g, '_').replace(/_+/g, '_').slice(0, 80);
  const name    = stem + ext;
  fs.renameSync(req.file.path, path.join(P.scanDir, name));
  if (gt) fs.writeFileSync(path.join(P.scanDir, stem + ".gt.txt"), gt, "utf8");
  res.json({ ok: true, saved: name, has_gt: !!gt });
});

app.post("/api/scans/delete", (req, res) => {
  const { name } = req.body;
  if (!name) return res.status(400).json({ error: "No name" });
  const stem = name.replace(/\.[^.]+$/, "");
  [name, stem + ".gt.txt"].forEach(f => {
    const p = path.join(P.scanDir, f);
    if (fs.existsSync(p)) fs.unlinkSync(p);
  });
  res.json({ ok: true });
});

// ── API: image preprocessing ───────────────────────────────────────────────
app.post("/api/preprocess", async (req, res) => {
  const { image_b64, ops = [] } = req.body;
  if (!image_b64) return res.status(400).json({ error: "No image" });
  try {
    const sharp  = require("sharp");
    const raw    = Buffer.from(image_b64.replace(/^data:[^;]+;base64,/, ""), "base64");
    let img      = sharp(raw).grayscale();
    if (ops.includes("enhance_contrast")) img = img.normalise();
    if (ops.includes("binarize"))  img = img.threshold(128);
    if (ops.includes("denoise"))   img = img.median(3);
    const out = await img.png().toBuffer();
    res.json({ image_b64: "data:image/png;base64," + out.toString("base64") });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// Strip absolute ROOT path from log lines so user paths are not exposed in the UI.
// e.g. /Users/alice/Projects/tesseract-training-kannada/tmp/… → tmp/…
const _rootPrefix = ROOT.endsWith("/") ? ROOT : ROOT + "/";
function sanitizeLogLine(line) {
  return line.split(_rootPrefix).join("");
}

// ── API: log tail ──────────────────────────────────────────────────────────
app.get("/api/log/tail", (req, res) => {
  const n = parseInt(req.query.lines) || 100;
  if (!fs.existsSync(P.logFile)) return res.json({ lines: [], exists: false });
  const lines = fs.readFileSync(P.logFile, "utf8").split("\n");
  res.json({ lines: lines.slice(-n).map(sanitizeLogLine), exists: true, total: lines.length });
});

// ── API: log SSE stream ────────────────────────────────────────────────────
app.get("/api/log/stream", (req, res) => {
  res.setHeader("Content-Type",  "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("X-Accel-Buffering", "no");
  res.flushHeaders();

  let pos = fs.existsSync(P.logFile) ? fs.statSync(P.logFile).size : 0;

  const interval = setInterval(() => {
    if (!fs.existsSync(P.logFile)) return;
    const size = fs.statSync(P.logFile).size;
    if (size <= pos) return;
    const fd   = fs.openSync(P.logFile, "r");
    const buf  = Buffer.alloc(size - pos);
    fs.readSync(fd, buf, 0, buf.length, pos);
    fs.closeSync(fd);
    pos = size;
    buf.toString("utf8").split("\n").forEach(line => {
      if (line) res.write(`data: ${JSON.stringify(sanitizeLogLine(line))}\n\n`);
    });
  }, 1000);

  req.on("close", () => clearInterval(interval));
});

// ── API: run step ──────────────────────────────────────────────────────────
app.post("/api/run/:step", (req, res) => {
  const cmds = {
    prep:          ["bash",    [path.join(P.scripts, "01-prep-base.sh")]],
    wiki:          ["python3", [path.join(P.corpus,  "download-wiki.py")]],
    clean:         ["python3", [path.join(P.corpus,  "clean-corpus.py")]],
    specimen:      ["python3", [path.join(P.corpus,  "generate-specimen.py"), "--merge"]],
    render:        ["python3", [path.join(P.corpus,  "render-corpus.py")]],
    lstmf:         ["bash",    [path.join(P.scripts, "02-make-lstmf.sh")]],
    train:         ["bash",    [path.join(P.scripts, "03-train.sh")]],
    package:       ["bash",    [path.join(P.scripts, "04-package.sh")]],
    expandunichar: ["bash",    [path.join(P.scripts, "00c-expand-unicharset.sh")]],
  };
  const { step } = req.params;
  const force = req.query.force === '1';
  if (!cmds[step]) return res.status(400).json({ error: `Unknown step: ${step}` });
  const [cmd, args] = cmds[step];
  if (!fs.existsSync(args[0])) return res.status(404).json({ error: `Script not found: ${args[0]}` });

  // Skip wiki download if corpus already exists (unless ?force=1)
  if (step === 'wiki' && !force) {
    const stats = corpusStats();
    if (stats && stats.total_lines > 0) {
      return res.json({ ok: true, step, skipped: true,
        reason: `Corpus already downloaded — ${stats.total_lines.toLocaleString()} lines. Use ?force=1 to re-download.` });
    }
  }

  // Pass --force to expand script when requested
  const runArgs = [...args];
  if (step === 'expandunichar' && force) runArgs.push('--force');

  // Support TRAIN_MODE=fresh for training with expanded unicharset from base weights
  const runOpts = {};
  if (step === 'train' && req.query.mode === 'fresh') {
    runOpts.env = { TRAIN_MODE: 'fresh' };
  }

  runBg(cmd, runArgs, step, runOpts);
  res.json({ ok: true, step });
});

// ── Font image comparison ─────────────────────────────────────────────────
app.get("/api/font-images/compare", (req, res) => {
  const testDir  = path.join(ROOT, "test-images");
  const families = ["kan_gmp", "kan_gtn", "kan_kittel", "kan_wmp"];
  const TESTS    = ["line_vowels","line_consonants","line_conjuncts","line_digits",
                    "sentence_01","sentence_02","sentence_03","sentence_04"];

  function pickVariant(fontId) {
    const fdir = path.join(testDir, fontId);
    if (!fs.existsSync(fdir)) return null;
    const variants = fs.readdirSync(fdir).filter(v => {
      const vp = path.join(fdir, v);
      return fs.statSync(vp).isDirectory() && fs.existsSync(path.join(vp, "line_vowels.png"));
    });
    // prefer non-otf, non-ttf; then otf; then ttf
    const base = variants.find(v => !v.endsWith("-otf") && !v.endsWith("-ttf"));
    return base || variants.find(v => v.endsWith("-otf")) || variants[0] || null;
  }

  const result = families.map(fontId => {
    const variant = pickVariant(fontId);
    if (!variant) return null;
    const varDir = path.join(testDir, fontId, variant);
    const images = TESTS.map(t => {
      const img = path.join(varDir, t + ".png");
      const gt  = path.join(varDir, t + ".gt.txt");
      if (!fs.existsSync(img)) return null;
      return {
        name: t,
        b64:  fs.readFileSync(img).toString("base64"),
        gt:   fs.existsSync(gt) ? fs.readFileSync(gt, "utf8").trim() : "",
      };
    }).filter(Boolean);
    return { id: fontId, variant, images };
  }).filter(Boolean);

  res.json(result);
});

// ── OCR quality check ─────────────────────────────────────────────────────
app.get("/api/ocr-quality", async (req, res) => {
  const testDir  = path.join(ROOT, "test-images");
  const tessdata = path.join(ROOT, "tessdata_best");
  const families = ["kan_gmp", "kan_gtn", "kan_kittel", "kan_wmp"];
  const TESTS    = ["line_vowels","line_consonants","line_conjuncts","line_digits","sentence_01","sentence_02"];
  const model    = req.query.model || "kan_hist";

  // Check model exists
  const modelPath = path.join(tessdata, model + ".traineddata");
  if (!fs.existsSync(modelPath)) {
    const alt = path.join(ROOT, "output", model + ".traineddata");
    if (fs.existsSync(alt)) fs.copyFileSync(alt, modelPath);
    else return res.json({ error: `Model ${model}.traineddata not found`, results: [] });
  }

  function cer(gt, pred) {
    const g = [...gt.replace(/\s+/g,' ').trim()];
    const p = [...pred.replace(/\s+/g,' ').trim()];
    if (!g.length) return 0;
    // Levenshtein distance
    const dp = Array.from({length: g.length+1}, () => new Array(p.length+1).fill(0));
    for (let i=1;i<=g.length;i++) dp[i][0]=i;
    for (let j=1;j<=p.length;j++) dp[0][j]=j;
    for (let i=1;i<=g.length;i++)
      for (let j=1;j<=p.length;j++)
        dp[i][j] = g[i-1]===p[j-1] ? dp[i-1][j-1]
          : 1 + Math.min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]);
    return +(dp[g.length][p.length] / g.length * 100).toFixed(1);
  }

  function runTess(imgPath) {
    return new Promise(resolve => {
      execFile("tesseract", [imgPath, "stdout", "--tessdata-dir", tessdata,
        "--dpi", "150", "--psm", "6", "-l", model], (err, stdout) => {
        resolve((stdout || "").replace(/\n/g, " ").trim());
      });
    });
  }

  function pickVariant(fontId) {
    const fdir = path.join(testDir, fontId);
    if (!fs.existsSync(fdir)) return null;
    const variants = fs.readdirSync(fdir).filter(v => {
      const vp = path.join(fdir, v);
      return fs.statSync(vp).isDirectory() && fs.existsSync(path.join(vp, "line_vowels.png"));
    });
    const base = variants.find(v => !v.endsWith("-otf") && !v.endsWith("-ttf"));
    return base || variants.find(v => v.endsWith("-otf")) || variants[0] || null;
  }

  const results = [];
  for (const fontId of families) {
    const variant = pickVariant(fontId);
    if (!variant) continue;
    const varDir = path.join(testDir, fontId, variant);
    const fontResults = [];
    for (const t of TESTS) {
      const img = path.join(varDir, t + ".png");
      const gt  = path.join(varDir, t + ".gt.txt");
      if (!fs.existsSync(img) || !fs.existsSync(gt)) continue;
      const gtText   = fs.readFileSync(gt, "utf8").trim();
      const predText = await runTess(img);
      fontResults.push({ test: t, gt: gtText, pred: predText, cer: cer(gtText, predText) });
    }
    const avgCer = fontResults.length
      ? +(fontResults.reduce((s,r)=>s+r.cer,0) / fontResults.length).toFixed(1) : 999;
    results.push({ fontId, variant, avgCer, tests: fontResults });
  }

  // Overall
  const overall = results.length
    ? +(results.reduce((s,r)=>s+r.avgCer,0)/results.length).toFixed(1) : 999;

  res.json({ model, overall, results });
});

// ── Font registry: scans actual files + merges manifest data ──────────────
function scanFontDir(fontId) {
  // Mirror of Python scan_font_dir — finds all TTF+OTF, both formats when both exist
  const fontRoot = path.join(ROOT, "fonts", fontId);
  if (!fs.existsSync(fontRoot)) return {};
  const SKIP_DIRS  = new Set(["webfonts", "Source", "source"]);
  const byStem     = {};

  function walk(dir) {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        if (!SKIP_DIRS.has(entry.name)) walk(fullPath);
      } else {
        if (/\[|\]/.test(entry.name)) continue;
        const ext = path.extname(entry.name).toLowerCase();
        if (ext !== ".ttf" && ext !== ".otf") continue;
        const stem = path.basename(entry.name, ext);
        const fmt  = ext.slice(1); // 'ttf' or 'otf'
        if (!byStem[stem]) byStem[stem] = {};
        if (!byStem[stem][fmt]) byStem[stem][fmt] = fullPath; // first found wins
      }
    }
  }
  walk(fontRoot);

  const result = {};
  for (const [stem, fmts] of Object.entries(byStem).sort()) {
    if (fmts.ttf && fmts.otf) {
      result[`${stem}-ttf`] = { path: fmts.ttf, url: "/" + fmts.ttf.replace(ROOT + "/", ""), fmt: "TTF" };
      result[`${stem}-otf`] = { path: fmts.otf, url: "/" + fmts.otf.replace(ROOT + "/", ""), fmt: "OTF" };
    } else if (fmts.ttf) {
      result[stem] = { path: fmts.ttf, url: "/" + fmts.ttf.replace(ROOT + "/", ""), fmt: "TTF" };
    } else {
      result[stem] = { path: fmts.otf, url: "/" + fmts.otf.replace(ROOT + "/", ""), fmt: "OTF" };
    }
  }
  return result;
}

app.get("/api/fonts-registry", (req, res) => {
  const yaml = require("js-yaml");
  let fonts = [];
  try {
    const doc  = yaml.load(fs.readFileSync(P.fontsYml, "utf8"));
    const tiDir = path.join(ROOT, "test-images");

    fonts = (doc.fonts || []).map(f => {
      const scanned  = scanFontDir(f.id);
      const seenNames = new Set();
      const variants  = [];

      // Follow font_files order then add extras from scan
      for (const filename of (f.font_files || [])) {
        const stem = filename.replace(/\.[^.]+$/, "");
        for (const suffix of [`${stem}-ttf`, `${stem}-otf`, stem]) {
          if (scanned[suffix] && !seenNames.has(suffix)) {
            const imagesDir = path.join(tiDir, f.id, suffix);
            variants.push({
              name:        suffix,
              displayName: suffix.replace(/^Karnata(GTN|GermanMissionPressTypeface|WesleyanMissionPress|FKittel)?-?/,"") || suffix,
              fmt:         scanned[suffix].fmt,
              fontUrl:     scanned[suffix].url,
              hasImages:   fs.existsSync(imagesDir),
              imagesUrl:   `/test-images/${f.id}/${encodeURIComponent(suffix)}`,
            });
            seenNames.add(suffix);
          }
        }
      }
      // Append remaining scanned (extra weights etc.)
      for (const [name, info] of Object.entries(scanned).sort()) {
        if (!seenNames.has(name)) {
          const imagesDir = path.join(tiDir, f.id, name);
          variants.push({
            name,
            displayName: name,
            fmt:         info.fmt,
            fontUrl:     info.url,
            hasImages:   fs.existsSync(imagesDir),
            imagesUrl:   `/test-images/${f.id}/${encodeURIComponent(name)}`,
          });
          seenNames.add(name);
        }
      }
      return { id: f.id, name: f.name, description: f.description, degrade: !!f.degrade, variants };
    });
  } catch(e) {
    return res.status(500).json({ error: e.message });
  }
  res.json({ fonts });
});

// ── List PNGs for a specific font/variant ─────────────────────────────────
app.get("/api/test-images/:fontId/:variant", (req, res) => {
  const dir = path.join(ROOT, "test-images", req.params.fontId,
                        decodeURIComponent(req.params.variant));
  if (!fs.existsSync(dir)) return res.json({ files: [] });
  const manifest = path.join(dir, "manifest.json");
  if (fs.existsSync(manifest)) {
    try {
      const m = JSON.parse(fs.readFileSync(manifest, "utf8"));
      return res.json({
        files: (m.characters || []).map(c => ({
          ...c,
          url: `/test-images/${req.params.fontId}/${req.params.variant}/${c.file}`,
        }))
      });
    } catch(_) {}
  }
  // Fallback: raw directory listing
  const files = fs.readdirSync(dir)
    .filter(f => f.endsWith(".png"))
    .sort()
    .map(f => ({ file: f, url: `/test-images/${req.params.fontId}/${req.params.variant}/${f}` }));
  res.json({ files });
});

// ── Serve tessdata for Tesseract.js ────────────────────────────────────────
app.get("/tessdata/:file", (req, res) => {
  const { file } = req.params;
  if (!file.endsWith(".traineddata")) return res.status(403).send("Forbidden");
  for (const dir of [P.bestDir, P.tessdataDir]) {
    const p = path.join(dir, file);
    if (fs.existsSync(p)) return res.sendFile(p);
  }
  res.status(404).send("Not found");
});

// ── Report download ────────────────────────────────────────────────────────
app.get("/report", (req, res) => {
  const fonts  = loadFonts();
  const bcer   = getBcerHistory();
  const images = sampleImages(6);
  const stats  = corpusStats();
  const now    = new Date().toISOString().replace("T", " ").slice(0, 16);

  // Minimal self-contained HTML report
  const bestBcer = bcer.length ? Math.min(...bcer.map(p => p.bcer)).toFixed(3) : null;
  const stepsHtml = Object.values((() => {
    const fonts  = loadFonts();
    const cloned = fonts.filter(f => fs.existsSync(path.join(ROOT,"fonts",f.id,".git"))).length;
    const cps    = getCheckpoints();
    return {
      "01_prep":   { label:"1. Prep base model",  done:fs.existsSync(path.join(P.tessdataDir,"kan.traineddata")) && cloned===fonts.length },
      "02_corpus": { label:"2. Build corpus",     done:lineCount(P.corpusTxt)>0 },
      "03_render": { label:"3. Render images",    done:globCount(P.rendered,".png")>0 },
      "04_lstmf":  { label:"4. Generate lstmf",   done:lineCount(P.lstmfList)>0 },
      "05_train":  { label:"5. Train",            done:cps.length>0 },
      "06_package":{ label:"6. Package",          done:fs.existsSync(path.join(P.bestDir,"kan_hist.traineddata")) },
    };
  })()).map(s => `<tr><td>${s.done?"✅":"⏳"}</td><td>${s.label}</td></tr>`).join("");

  const html = `<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>TrainOCR Report — ${now}</title>
<style>body{font-family:system-ui,sans-serif;max-width:900px;margin:40px auto;padding:0 20px;color:#1e1b4b}
h1{color:#5b21b6;display:flex;align-items:center;gap:12px}
h1 img{height:36px}
h2{color:#4c1d95;border-bottom:2px solid #ede9fe;padding-bottom:6px;margin-top:2rem}
table{width:100%;border-collapse:collapse}th,td{padding:8px 12px;border:1px solid #e2e8f0;text-align:left}
th{background:#f5f3ff}.imgs{display:flex;flex-wrap:wrap;gap:10px}
.img-c{border:1px solid #e2e8f0;border-radius:8px;overflow:hidden;max-width:180px}
.img-c img{width:100%}.gt{font-size:.7em;color:#6b7280;padding:4px}</style>
</head><body>
<h1><img src="https://pada.sanchaya.net/images/sanchaya-logo.png" alt="Sanchaya">TrainOCR — kan_hist Report</h1>
<p>Generated ${now}${bestBcer ? ` &nbsp;|&nbsp; Best BCER: <strong>${bestBcer}%</strong>` : ""}</p>
<h2>Pipeline status</h2>
<table><tr><th></th><th>Step</th></tr>${stepsHtml}</table>
${stats ? `<p>Corpus: ${stats.total_lines.toLocaleString()} lines &nbsp;|&nbsp; Kannada chars: ${stats.kan_chars.toLocaleString()} &nbsp;|&nbsp; Unicode coverage: ${stats.coverage_pct}%</p>` : ""}
<h2>Fonts</h2>
<table><tr><th>Font</th><th>Styles</th><th>Rendering</th></tr>
${fonts.map(f=>`<tr><td>${f.name}</td><td>${(f.font_files||[]).length}</td><td>${f.degrade?"Degraded":"Clean"}</td></tr>`).join("")}
</table>
${images.length ? `<h2>Sample images</h2><div class="imgs">${images.map(i=>`<div class="img-c"><img src="data:image/png;base64,${i.b64}"><div class="gt">${i.gt}</div></div>`).join("")}</div>` : ""}
${bcer.length ? `<h2>BCER chart</h2><canvas id="c" height="60"></canvas>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script>new Chart(document.getElementById("c"),{type:"line",data:{labels:${JSON.stringify(bcer.map(p=>p.iter))},datasets:[{label:"BCER %",data:${JSON.stringify(bcer.map(p=>p.bcer))},borderColor:"#7c3aed",backgroundColor:"#ede9fe",tension:.3}]},options:{plugins:{legend:{display:false}}}})</script>` : ""}
</body></html>`;

  res.setHeader("Content-Disposition", `attachment; filename="trainocr_report_${now.slice(0,10)}.html"`);
  res.setHeader("Content-Type", "text/html");
  res.send(html);
});

// ── Generate Kannada test images (per-font or all) ────────────────────────
app.post("/api/generate-test-images", async (req, res) => {
  const script = path.join(ROOT, "scripts", "gen-char-images.py");
  const outDir = path.join(ROOT, "test-images");
  if (!fs.existsSync(script)) {
    return res.status(404).json({ error: "gen-char-images.py not found" });
  }
  const { spawn } = require("child_process");
  const fontId = req.body && req.body.fontId;
  const args   = ["python3", script, "--outdir", outDir, "--dpi", "150"];
  if (fontId) { args.push("--font-id"); args.push(fontId); }
  const proc = spawn(args[0], args.slice(1), { cwd: ROOT });
  let stdout = "", stderr = "";
  proc.stdout.on("data", d => { stdout += d; });
  proc.stderr.on("data", d => { stderr += d; });
  proc.on("close", code => {
    if (code !== 0) {
      return res.status(500).json({ error: stderr || "Script failed", code });
    }
    // Count generated files
    let count = 0;
    let manifest = {};
    const manifestPath = path.join(outDir, "manifest.json");
    if (fs.existsSync(manifestPath)) {
      try { manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8")); count = manifest.count || 0; }
      catch (_) {}
    } else {
      count = fs.existsSync(outDir)
        ? fs.readdirSync(outDir).filter(f => f.endsWith(".png")).length
        : 0;
    }
    res.json({ ok: true, count, outDir });
  });
});

// ── Clear rendered/ training images ──────────────────────────────────────
app.delete("/api/rendered", (req, res) => {
  const dir = path.join(ROOT, "rendered");
  if (!fs.existsSync(dir)) return res.json({ ok: true, pngs: 0, txts: 0 });
  let pngs = 0, txts = 0;
  for (const entry of fs.readdirSync(dir)) {
    const full = path.join(dir, entry);
    const stat = fs.statSync(full);
    if (stat.isDirectory()) {
      // Subdirs like font-test/ — wipe entirely
      fs.rmSync(full, { recursive: true, force: true });
    } else if (entry.endsWith(".png"))    { fs.unlinkSync(full); pngs++; }
      else if (entry.endsWith(".gt.txt")) { fs.unlinkSync(full); txts++; }
  }
  res.json({ ok: true, pngs, txts });
});

// ── Clear test images (font or all) ──────────────────────────────────────
app.delete("/api/test-images", (req, res) => {
  const outDir = path.join(ROOT, "test-images");
  const fontId = req.query.fontId;   // optional: clear one font; omit = clear all
  if (!fs.existsSync(outDir)) return res.json({ ok: true, deleted: 0 });

  let deleted = 0;
  if (fontId) {
    // Clear just the one font's directory
    const fontDir = path.join(outDir, fontId);
    if (fs.existsSync(fontDir)) {
      fs.rmSync(fontDir, { recursive: true, force: true });
      deleted++;
    }
  } else {
    // Clear everything under test-images/
    for (const entry of fs.readdirSync(outDir)) {
      const full = path.join(outDir, entry);
      fs.rmSync(full, { recursive: true, force: true });
      deleted++;
    }
  }
  res.json({ ok: true, deleted, fontId: fontId || null });
});

// ── List test images ──────────────────────────────────────────────────────
app.get("/api/test-images", (req, res) => {
  const outDir = path.join(ROOT, "test-images");
  if (!fs.existsSync(outDir)) return res.json({ files: [] });
  const files = fs.readdirSync(outDir)
    .filter(f => f.endsWith(".png"))
    .sort()
    .map(f => ({ name: f, url: `/test-images/${f}` }));
  res.json({ files });
});

// ── Char-train: save/load OCR baseline results ────────────────────────────
app.post("/api/char-train/baseline", express.json(), (req, res) => {
  const { fontId, variant, results } = req.body || {};
  if (!fontId || !variant || !results) return res.status(400).json({ error: "fontId, variant, results required" });
  const varDir = path.join(P.testImages, fontId, variant);
  if (!fs.existsSync(varDir)) return res.status(404).json({ error: "Variant dir not found" });
  const histFile = path.join(varDir, "ocr_history.json");
  let history = [];
  if (fs.existsSync(histFile)) {
    try { history = JSON.parse(fs.readFileSync(histFile, "utf8")); } catch (_) {}
  }
  const entry = { ts: new Date().toISOString(), label: req.body.label || "baseline", ...results };
  history.push(entry);
  fs.writeFileSync(histFile, JSON.stringify(history, null, 2));
  res.json({ ok: true, history });
});

app.get("/api/char-train/history/:fontId/:variant", (req, res) => {
  const varDir = path.join(P.testImages, req.params.fontId,
                           decodeURIComponent(req.params.variant));
  const histFile = path.join(varDir, "ocr_history.json");
  if (!fs.existsSync(histFile)) return res.json({ history: [] });
  try { res.json({ history: JSON.parse(fs.readFileSync(histFile, "utf8")) }); }
  catch (_) { res.json({ history: [] }); }
});

// ── Char-train: generate lstmf + fine-tune from test images ───────────────
// State tracking for the long-running train job
let charTrainJob  = null;
let runningStep   = null;   // currently active pipeline step
let completedSteps = {};    // stepId → { code, ts }

app.post("/api/char-train/start", express.json(), (req, res) => {
  if (charTrainJob && charTrainJob.running) {
    return res.status(409).json({ error: "Training already running", jobId: charTrainJob.id });
  }
  const { fontId, variant, iterations = 500 } = req.body || {};
  if (!fontId || !variant) return res.status(400).json({ error: "fontId and variant required" });

  const varDir   = path.join(P.testImages, fontId, variant);
  const lstmfDir = path.join(ROOT, "lstmf", "char-train", fontId, variant);
  const tessdata = path.join(ROOT, "tessdata_best");
  const outputDir = path.join(ROOT, "output");

  if (!fs.existsSync(varDir)) return res.status(404).json({ error: "No test images for this variant. Generate images first." });
  if (!fs.existsSync(path.join(tessdata, "kan.traineddata"))) {
    return res.status(412).json({ error: "kan.traineddata not found in tessdata_best/. Run Step 1 (Prep) first." });
  }

  fs.mkdirSync(lstmfDir, { recursive: true });

  // Ensure tessdata configs dir exists (required for lstm.train mode)
  const configsDir = path.join(tessdata, "configs");
  if (!fs.existsSync(configsDir)) {
    const { execSync } = require("child_process");
    try {
      const sysConfigs = execSync(
        "find /usr/local/share /opt/homebrew/share /usr/share -name 'configs' -path '*/tessdata/*' 2>/dev/null | head -1"
      ).toString().trim();
      if (sysConfigs) {
        fs.symlinkSync(sysConfigs, configsDir);
      } else {
        // Create minimal configs dir with lstm.train file
        fs.mkdirSync(configsDir, { recursive: true });
        fs.writeFileSync(path.join(configsDir, "lstm.train"), "lstm_train_mode 1\n");
      }
    } catch(_) {
      fs.mkdirSync(configsDir, { recursive: true });
      fs.writeFileSync(path.join(configsDir, "lstm.train"), "lstm_train_mode 1\n");
    }
  }

  const jobId = Date.now().toString();
  charTrainJob = { id: jobId, running: true, phase: "lstmf", log: [], error: null, done: false, fontId, variant };

  // Run async — client polls /api/char-train/status
  (async () => {
    const addLog = msg => { charTrainJob.log.push(msg); };
    try {
      // ── Phase 1: generate lstmf from line + sentence images ──
      addLog("Phase 1/2 — Generating .lstmf training files…");
      const pngs = fs.readdirSync(varDir)
        .filter(f => /^(line_|sentence_)/.test(f) && f.endsWith(".png"))
        .sort();
      if (pngs.length === 0) throw new Error("No line or sentence images found. Generate images first.");
      addLog(`Found ${pngs.length} training images (line_ + sentence_ files)`);

      const lstmfFiles = [];
      for (const png of pngs) {
        const base    = png.replace(".png", "");
        const srcPng  = path.join(varDir, png);
        const gtTxt   = path.join(varDir, base + ".gt.txt");
        const outBase = path.join(lstmfDir, base);
        const dstPng  = outBase + ".png";
        const boxFile = outBase + ".box";
        const lstmf   = outBase + ".lstmf";
        if (!fs.existsSync(gtTxt)) { addLog(`  skip ${png} (no .gt.txt)`); continue; }

        // Copy image into lstmf work dir
        fs.copyFileSync(srcPng, dstPng);

        // Build WordStr box file (required for Tesseract 5 line-level training)
        // Collapse spaces; strip chars absent from kan.traineddata unicharset (ಋ ಙ ಝ ಞ ಱ)
        const UNSUPPORTED = new Set([...'ಋಙಝಞಱ']);
        const gt = fs.readFileSync(gtTxt, "utf8").trim()
          .split(' ').filter(tok => !(tok.length === 1 && UNSUPPORTED.has(tok))).join(' ')
          .replace(/\s+/g, " ").trim();
        let w = 300, h = 100;
        try {
          const dimOut = await new Promise((res, rej) =>
            execFile("python3", ["-c",
              `from PIL import Image; im=Image.open(${JSON.stringify(dstPng)}); print(im.width,im.height)`
            ], (err, stdout) => err ? rej(err) : res(stdout))
          );
          const [pw, ph] = dimOut.trim().split(" ").map(Number);
          if (pw > 0 && ph > 0) { w = pw; h = ph; }
        } catch(_) {}
        fs.writeFileSync(boxFile, `WordStr 0 0 ${w} ${h} 0 #${gt}\n\n`);

        await new Promise((resolve) => {
          const proc = spawn("tesseract", [
            dstPng, outBase,
            "--tessdata-dir", tessdata,
            "--dpi", "150", "--psm", "6",
            "-l", "kan", "lstm.train"
          ], { cwd: ROOT });
          let err = "";
          proc.stderr.on("data", d => { err += d.toString(); });
          proc.stdout.on("data", d => { /* swallow */ });
          proc.on("close", code => {
            if (code === 0 && fs.existsSync(lstmf)) {
              lstmfFiles.push(lstmf);
              addLog(`  ✓ ${base}.lstmf`);
            } else {
              const lastErr = err.trim().split("\n").pop() || `exit ${code}`;
              addLog(`  ✗ ${base}: ${lastErr}`);
            }
            resolve();
          });
        });
      }

      if (lstmfFiles.length === 0) throw new Error("No .lstmf files produced. Check Tesseract 5 is installed and kan.traineddata is present.");

      // Write list.txt for this char-train job
      const listTxt = path.join(lstmfDir, "list.txt");
      fs.writeFileSync(listTxt, lstmfFiles.join("\n") + "\n");
      addLog(`\n${lstmfFiles.length} lstmf files ready → starting fine-tune…`);

      // ── Phase 2: lstmtraining fine-tune ──
      charTrainJob.phase = "train";
      const bestCheckpoint = fs.existsSync(outputDir)
        ? fs.readdirSync(outputDir)
            .filter(f => /^kan_hist_[\d.]+_\d+_\d+\.checkpoint$/.test(f))
            .sort((a, b) => {
              const bcer = f => parseFloat(f.split("_")[2]);
              return bcer(a) - bcer(b);
            })[0]
        : null;
      const startModel = bestCheckpoint
        ? path.join(outputDir, bestCheckpoint)
        : path.join(outputDir, "kan.lstm");
      if (!fs.existsSync(startModel)) throw new Error(`Starting model not found: ${startModel}. Run Step 1 (Prep) first.`);

      addLog(`Starting from: ${path.basename(startModel)}`);
      addLog(`Fine-tuning for ${iterations} iterations…\n`);

      await new Promise((resolve, reject) => {
        const proc = spawn("lstmtraining", [
          "--traineddata",    path.join(tessdata, "kan.traineddata"),
          "--model_output",   path.join(outputDir, `kan_hist_chartrain_${fontId}`),
          "--continue_from",  startModel,
          "--train_listfile", listTxt,
          "--max_iterations", String(iterations),
          "--target_error_rate", "0.01",
          "--debug_interval", "0"
        ], { cwd: ROOT });

        const bcerRe = /At iteration\s+(\d+).*?BCER train=([\d.]+)%/;
        let lastBcer = null;
        const onData = d => {
          const lines = d.toString().split("\n");
          lines.forEach(line => {
            if (!line.trim()) return;
            addLog(line);
            const m = bcerRe.exec(line);
            if (m) { charTrainJob.bcer = parseFloat(m[2]); charTrainJob.iter = parseInt(m[1]); }
          });
        };
        proc.stdout.on("data", onData);
        proc.stderr.on("data", onData);
        proc.on("close", code => {
          if (code === 0 || code === null) resolve();
          else reject(new Error(`lstmtraining exited with code ${code}`));
        });
      });

      addLog("\n✓ Fine-tuning complete!");
      charTrainJob.running = false;
      charTrainJob.done    = true;
    } catch (err) {
      addLog(`\n✗ Error: ${err.message}`);
      charTrainJob.error   = err.message;
      charTrainJob.running = false;
    }
  })();

  res.json({ ok: true, jobId });
});

app.get("/api/char-train/status", (req, res) => {
  if (!charTrainJob) return res.json({ running: false, done: false, log: [] });
  res.json({
    running: charTrainJob.running,
    done:    charTrainJob.done,
    phase:   charTrainJob.phase,
    bcer:    charTrainJob.bcer,
    iter:    charTrainJob.iter,
    error:   charTrainJob.error,
    log:     charTrainJob.log.slice(-80), // last 80 lines
    fontId:  charTrainJob.fontId,
    variant: charTrainJob.variant
  });
});

// ── Per-font image summary ─────────────────────────────────────────────────
app.get("/api/char-images/summary", (req, res) => {
  const tiDir = path.join(ROOT, "test-images");
  let fonts = [];
  try {
    const doc = require("js-yaml").load(fs.readFileSync(P.fontsYml, "utf8"));
    fonts = (doc.fonts || []).map(f => {
      const scanned = scanFontDir(f.id);
      const variants = Object.entries(scanned).map(([varName, info]) => {
        const varDir = path.join(tiDir, f.id, varName);
        const exists = fs.existsSync(varDir);
        let line = 0, sent = 0, chars = 0;
        if (exists) {
          for (const fn of fs.readdirSync(varDir)) {
            if (!fn.endsWith(".png")) continue;
            if (fn.startsWith("line_"))     line++;
            else if (fn.startsWith("sentence_")) sent++;
            else if (fn.startsWith("char_"))    chars++;
          }
        }
        return { name: varName, fmt: info.fmt, line, sent, chars,
                 hasTrainImages: line + sent > 0, hasImages: line + sent + chars > 0 };
      });
      const ready = variants.filter(v => v.hasTrainImages).length;
      return { id: f.id, name: f.name, variants, ready, total: variants.length };
    });
  } catch(e) { return res.status(500).json({ error: e.message }); }
  res.json({ fonts });
});

// ── Sync font test images into rendered/font-test/ for main pipeline ───────
app.post("/api/sync-test-images", async (req, res) => {
  const tiDir   = path.join(ROOT, "test-images");
  const syncDir = path.join(ROOT, "rendered", "font-test");
  fs.mkdirSync(syncDir, { recursive: true });
  let copied = 0, skipped = 0, errors = [];
  if (!fs.existsSync(tiDir)) return res.json({ ok: true, copied: 0, skipped: 0, errors: [] });

  for (const fontId of fs.readdirSync(tiDir)) {
    const fontPath = path.join(tiDir, fontId);
    if (!fs.statSync(fontPath).isDirectory()) continue;
    for (const variant of fs.readdirSync(fontPath)) {
      const varPath = path.join(fontPath, variant);
      if (!fs.statSync(varPath).isDirectory()) continue;
      const pngs = fs.readdirSync(varPath)
        .filter(f => (f.startsWith("line_") || f.startsWith("sentence_")) && f.endsWith(".png"));
      for (const png of pngs) {
        const gt = path.join(varPath, png.replace(".png", ".gt.txt"));
        if (!fs.existsSync(gt)) { skipped++; continue; }
        const stem = `${fontId}__${variant}__${png.replace(".png","")}`;
        try {
          fs.copyFileSync(path.join(varPath, png), path.join(syncDir, stem + ".png"));
          fs.copyFileSync(gt, path.join(syncDir, stem + ".gt.txt"));
          copied++;
        } catch(e) { errors.push(`${stem}: ${e.message}`); }
      }
    }
  }
  res.json({ ok: true, copied, skipped, errors, syncDir });
});

// ── Pre-flight check before training pipeline ──────────────────────────────
app.get("/api/preflight", (req, res) => {
  const tiDir    = path.join(ROOT, "test-images");
  const syncDir  = path.join(ROOT, "rendered", "font-test");
  const rendDir  = path.join(ROOT, "rendered");

  let totalVariants = 0, readyVariants = 0, trainImages = 0;
  const fontStatus = [];

  if (fs.existsSync(tiDir)) {
    for (const fontId of fs.readdirSync(tiDir)) {
      const fp = path.join(tiDir, fontId);
      if (!fs.statSync(fp).isDirectory()) continue;
      const varResults = [];
      for (const variant of fs.readdirSync(fp)) {
        const vp = path.join(fp, variant);
        if (!fs.statSync(vp).isDirectory()) continue;
        totalVariants++;
        const files = fs.readdirSync(vp);
        const line = files.filter(f => (f.startsWith("line_") || f.startsWith("sentence_")) && f.endsWith(".png")).length;
        const chars = files.filter(f => f.startsWith("char_") && f.endsWith(".png")).length;
        if (line > 0) { readyVariants++; trainImages += line; }
        varResults.push({ variant, line, chars, ready: line > 0 });
      }
      fontStatus.push({ fontId, variants: varResults });
    }
  }

  const syncedCount = fs.existsSync(syncDir)
    ? fs.readdirSync(syncDir).filter(f => f.endsWith(".png")).length : 0;
  const renderedCount = fs.existsSync(rendDir)
    ? fs.readdirSync(rendDir).filter(f => f.endsWith(".png")).length : 0;
  const kanExists = fs.existsSync(path.join(ROOT, "tessdata_best", "kan.traineddata"));

  res.json({
    fonts: fontStatus,
    fontVariants:  { ready: readyVariants, total: totalVariants, trainImages },
    synced:        { count: syncedCount, upToDate: syncedCount >= trainImages },
    rendered:      { count: renderedCount },
    kanExists,
    goForTraining: readyVariants > 0 && kanExists,
  });
});

// ── SPA fallback ───────────────────────────────────────────────────────────
app.get("*", (req, res) => {
  res.sendFile(path.join(P.public, "index.html"));
});

// ── Start ──────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`\n${"━".repeat(52)}`);
  console.log(`  TrainOCR by Sanchaya`);
  console.log(`  http://localhost:${PORT}`);
  console.log(`${"━".repeat(52)}\n`);
});
