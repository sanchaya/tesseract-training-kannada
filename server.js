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
  lstmfDir:    path.join(ROOT, "lstmf"),
  outputDir:   path.join(ROOT, "output"),
  rendered:    path.join(ROOT, "rendered"),
  classicalA5: path.join(ROOT, "classical-corpus-kannada", "a5-pages"),
  scanDir:     path.join(ROOT, "scan-input"),
  logsDir:     path.join(ROOT, "logs"),
  logFile:     path.join(ROOT, "logs", "training.log"),
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
fs.mkdirSync(P.logsDir,     { recursive: true });

// Migrate legacy log from repo root if it exists and new location is empty/absent
const _legacyLog = path.join(ROOT, "training.log");
if (fs.existsSync(_legacyLog) && !fs.existsSync(P.logFile)) {
  fs.renameSync(_legacyLog, P.logFile);
}

// ── Log rotation ───────────────────────────────────────────────────────────
// Called before each runBg() to prevent training.log growing unbounded.
// Keeps up to LOG_KEEP archives: training.log.1, .2, .3 (oldest deleted).
const LOG_MAX_BYTES = 50 * 1024 * 1024;  // 50 MB per file
const LOG_KEEP      = 3;                  // number of rotated archives to retain

function rotateLogIfNeeded() {
  if (!fs.existsSync(P.logFile)) return;
  const { size } = fs.statSync(P.logFile);
  if (size < LOG_MAX_BYTES) return;

  // Shift existing archives:  .3 → deleted, .2 → .3, .1 → .2, active → .1
  for (let i = LOG_KEEP; i >= 1; i--) {
    const older = `${P.logFile}.${i}`;
    const newer = i === 1 ? P.logFile : `${P.logFile}.${i - 1}`;
    if (fs.existsSync(older)) {
      if (i === LOG_KEEP) fs.unlinkSync(older);
      else                fs.renameSync(older, `${P.logFile}.${i}`);
    }
    if (i === 1 && fs.existsSync(newer)) fs.renameSync(newer, older);
  }
  // Start fresh active log
  fs.writeFileSync(P.logFile, `[trainocr] Log rotated at ${new Date().toISOString()}\n`);
}

// ── Middleware ─────────────────────────────────────────────────────────────
app.use(express.json({ limit: "50mb" }));
app.use(express.static(P.public));
app.use("/test-images", express.static(path.join(ROOT, "test-images")));
app.use("/fonts",           express.static(path.join(ROOT, "fonts"),
  { setHeaders: (res) => res.setHeader("Access-Control-Allow-Origin", "*") }));
app.use("/classical-pages", express.static(P.classicalA5,
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

// Font family ids come from fonts.yml — never hardcode a family list, or newly
// registered fonts silently vanish from the gallery / OCR test pages.
function fontFamilyIds() {
  return loadFonts().map(f => f.id).filter(Boolean);
}

// Returns the list of LIVE lstmtraining PIDs (zombies excluded).
// A finished-but-unreaped (defunct) process still matches `pgrep`, which used to
// leave the portal showing "Training is running in the background" forever.
function trainingPids() {
  try {
    const cp  = require("child_process");
    const out = cp.spawnSync("pgrep", ["-x", "lstmtraining"], { encoding: "utf8" });
    if (out.status !== 0 || !out.stdout) return [];

    const pids = out.stdout.split("\n").map(s => s.trim()).filter(Boolean);
    return pids.filter(pid => {
      // ps STAT: 'Z' (Linux) / 'Z+' (macOS) means the process is defunct.
      const st = cp.spawnSync("ps", ["-o", "stat=", "-p", pid], { encoding: "utf8" });
      const stat = (st.stdout || "").trim();
      return stat && !stat.startsWith("Z");
    });
  } catch { return []; }
}

function isTrainingRunning() {
  return trainingPids().length > 0;
}

// True when fonts/<id>/ contains at least one .ttf/.otf at any depth.
function hasFontFiles(fontId) {
  const base = path.join(ROOT, "fonts", fontId);
  if (!fs.existsSync(base)) return false;
  const stack = [base];
  while (stack.length) {
    const dir = stack.pop();
    let entries;
    try { entries = fs.readdirSync(dir, { withFileTypes: true }); }
    catch { continue; }
    for (const e of entries) {
      if (e.isDirectory()) {
        if (e.name !== ".git") stack.push(path.join(dir, e.name));
      } else if (/\.(ttf|otf)$/i.test(e.name)) {
        return true;
      }
    }
  }
  return false;
}

function globCount(dir, ext) {
  if (!fs.existsSync(dir)) return 0;
  return fs.readdirSync(dir).filter(f => f.endsWith(ext)).length;
}

function lineCount(file) {
  if (!fs.existsSync(file)) return 0;
  return fs.readFileSync(file, "utf8").split("\n").filter(Boolean).length;
}

// ── Classical corpus helpers ────────────────────────────────────────────────
// Count PNG files recursively in a directory tree (cached 60 s).
let _classicalCache = { count: 0, ts: 0 };
function classicalImgCount() {
  const now = Date.now();
  if (now - _classicalCache.ts < 60_000) return _classicalCache.count;
  let count = 0;
  function walk(dir) {
    if (!fs.existsSync(dir)) return;
    for (const entry of fs.readdirSync(dir)) {
      const full = path.join(dir, entry);
      try {
        const st = fs.statSync(full);
        if (st.isDirectory()) walk(full);
        else if (entry.endsWith(".png")) count++;
      } catch { /* skip permission errors */ }
    }
  }
  walk(P.classicalA5);
  _classicalCache = { count, ts: now };
  return count;
}

// Break lstmf/list.txt into source buckets: rendered / classical / font-test / scan / other.
function lstmfBreakdown() {
  if (!fs.existsSync(P.lstmfList)) return { total: 0 };
  const lines = fs.readFileSync(P.lstmfList, "utf8").split("\n").filter(Boolean);
  const buckets = { rendered: 0, classical: 0, "font-test": 0, scan: 0, other: 0 };
  for (const l of lines) {
    const m = l.match(/lstmf\/([^/]+)\//);
    const key = m ? m[1] : "other";
    if (key in buckets) buckets[key]++;
    else buckets.other++;
  }
  return { total: lines.length, ...buckets };
}

// Sample a handful of classical A5 page image paths (not base64 — served via /classical-pages/).
function sampleClassicalImages(n = 6) {
  if (!fs.existsSync(P.classicalA5)) return [];
  const results = [];
  const titles  = fs.readdirSync(P.classicalA5).filter(e =>
    fs.statSync(path.join(P.classicalA5, e)).isDirectory()
  );
  for (const title of titles) {
    if (results.length >= n) break;
    const titleDir = path.join(P.classicalA5, title);
    const fonts    = fs.readdirSync(titleDir).filter(e =>
      fs.statSync(path.join(titleDir, e)).isDirectory()
    );
    const fontDir = fonts[0] ? path.join(titleDir, fonts[0]) : null;
    if (!fontDir) continue;
    const pngs = fs.readdirSync(fontDir).filter(f => f.endsWith(".png")).sort();
    if (!pngs.length) continue;
    // Pick a page from the middle of the document for a representative sample
    const pick = pngs[Math.floor(pngs.length / 2)];
    const stem = pick.replace(".png", "");
    const gtF  = path.join(fontDir, stem + ".gt.txt");
    const gt   = fs.existsSync(gtF)
      ? fs.readFileSync(gtF, "utf8").trim().slice(0, 120)
      : "";
    results.push({
      title,
      font:  fonts[0],
      name:  stem,
      url:   `/classical-pages/${encodeURIComponent(title)}/${encodeURIComponent(fonts[0])}/${pick}`,
      gt,
    });
  }
  return results;
}

function getCheckpoints() {
  if (!fs.existsSync(P.outputDir)) return [];
  return fs.readdirSync(P.outputDir)
    // format: kan_hist_<BCER>_<ITER>_<SAMPLES>.checkpoint
    .filter(f => /^kan_hist_[\d.]+_\d+_\d+\.checkpoint$/.test(f))
    .map(f => {
      const parts = f.replace(".checkpoint", "").split("_");
      // Filename format: kan_hist_<BCER>_<EXAMPLES_SEEN>_<ITERATION>.checkpoint
      // parts: ["kan","hist", <bcer>, <examples_seen>, <iteration>]
      // NOTE: parts[3] is examples_seen (e.g. 8822), parts[4] is the actual
      //       training iteration step (e.g. 396100). These were previously swapped.
      const stat  = fs.statSync(path.join(P.outputDir, f));
      return {
        file:    f,
        bcer:    parseFloat(parts[2]),
        iter:    parseInt(parts[4]),   // training iteration (last numeric field)
        samples: parseInt(parts[3]),   // examples seen in this checkpoint cycle
        size:    stat.size,
      };
    })
    .sort((a, b) => a.iter - b.iter);
}

function getBcerHistory() {
  const pts = getCheckpoints();
  if (pts.length) return pts.map(p => ({ iter: p.iter, bcer: p.bcer }));
  if (!fs.existsSync(P.logFile)) return [];
  // Training logs can grow very large (hundreds of MB of "Can't encode" spam).
  // Read only the last 4 MB to avoid Node.js string-length errors.
  const MAX_TAIL = 4 * 1024 * 1024; // 4 MB
  let log;
  try {
    const stat = fs.statSync(P.logFile);
    if (stat.size > MAX_TAIL) {
      const buf = Buffer.alloc(MAX_TAIL);
      const fd  = fs.openSync(P.logFile, 'r');
      fs.readSync(fd, buf, 0, MAX_TAIL, stat.size - MAX_TAIL);
      fs.closeSync(fd);
      log = buf.toString('utf8');
    } else {
      log = fs.readFileSync(P.logFile, 'utf8');
    }
  } catch (e) {
    return [];
  }
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

  const modernLines = text.split("\n").filter(Boolean).length;

  // ── Classical corpus source files ─────────────────────────────────────────
  // Scan classical-corpus-kannada/<title>/<title>.txt (one .txt per title,
  // not inside a5-pages/).
  const classicalRoot = path.join(ROOT, "classical-corpus-kannada");
  const classicalTitles = [];
  let classicalLines = 0;
  if (fs.existsSync(classicalRoot)) {
    for (const entry of fs.readdirSync(classicalRoot)) {
      const titleDir = path.join(classicalRoot, entry);
      try {
        if (!fs.statSync(titleDir).isDirectory()) continue;
        if (entry === "a5-pages") continue;
        // Find .txt files directly inside this title directory
        const txts = fs.readdirSync(titleDir).filter(f => f.endsWith(".txt"));
        for (const txt of txts) {
          const lines = lineCount(path.join(titleDir, txt));
          classicalLines += lines;
          classicalTitles.push({ title: entry, file: txt, lines });
        }
      } catch { /* skip unreadable entries */ }
    }
  }

  // Classical rendered page count (ground truth for training)
  const classicalPages = classicalImgCount();

  return {
    total_lines:      modernLines + classicalLines,   // grand total
    modern_lines:     modernLines,
    classical_lines:  classicalLines,
    classical_pages:  classicalPages,
    classical_titles: classicalTitles.sort((a, b) => b.lines - a.lines),
    total_chars:      text.length,
    kan_chars:        kanAll.length,
    unique_kan:       unique.size,
    coverage_pct:     +(unique.size / 128 * 100).toFixed(1),
    top_chars:        top,
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
  rotateLogIfNeeded();
  runningStep = stepId;
  const log  = fs.openSync(P.logFile, "a");
  const sep  = "=".repeat(55);
  fs.writeSync(log, `\n${sep}\n[trainocr] ${stepId}: ${cmd} ${args.join(" ")}\n${new Date().toISOString()}\n${sep}\n\n`);
  fs.closeSync(log);

  const child = spawn(cmd, args, {
    cwd:   ROOT,
    stdio: ["ignore", fs.openSync(P.logFile, "a"), fs.openSync(P.logFile, "a")],
    detached: false,
    // TRAINOCR_NO_TEE: scripts self-log to logs/training.log when run on a
    // terminal. Here stdout is already that file, so tee would duplicate lines.
    env: { ...process.env, TRAINOCR_NO_TEE: "1", ...(opts.env || {}) },
  });
  const clear = (code, why) => {
    if (runningStep === stepId) runningStep = null;
    completedSteps[stepId] = { code, ts: Date.now(), ok: code === 0 };
    const l = fs.openSync(P.logFile, "a");
    fs.writeSync(l, `\n[trainocr] ${stepId} ${code === 0 ? "✓ done" : `✗ ${why} (exit ${code})`}\n`);
    fs.closeSync(l);
  };
  child.on("exit",  code => clear(code, "failed"));
  // 'exit' does not fire if the spawn itself fails (ENOENT, EACCES …). Without
  // this, runningStep stays set forever and the portal shows a phantom
  // "Training is running in the background" banner.
  child.on("error", err => {
    console.error(`  [${stepId}] spawn failed:`, err.message);
    clear(-1, `spawn failed: ${err.message}`);
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

  // uharfbuzz (required for correct Kannada conjunct rendering)
  let uharfbuzzVer = null;
  try {
    const { stdout } = await execP("python3", ["-c",
      "import uharfbuzz as hb; print(getattr(hb,'__version__',None) or 'found')"], { timeout: 5000 });
    uharfbuzzVer = stdout.trim() || null;
  } catch { uharfbuzzVer = null; }

  // freetype-py (required for glyph rasterisation in shaped rendering)
  let freetypeVer = null;
  try {
    // freetype-py exposes version via FT_Library_Version(), not __version__
    const { stdout } = await execP("python3", ["-c",
      "import freetype; v=getattr(freetype,'__version__',None); print(v if v else 'found')"], { timeout: 5000 });
    freetypeVer = stdout.trim() || null;
  } catch { freetypeVer = null; }

  const kanBase   = fs.existsSync(path.join(P.tessdataDir, "kan.traineddata"));

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
      { name: "uharfbuzz",           key: "uharfbuzz",       ver: uharfbuzzVer, ok: !!uharfbuzzVer,
        need: "pip install uharfbuzz — required for correct Kannada conjunct rendering" },
      { name: "freetype-py",         key: "freetype",        ver: freetypeVer,  ok: !!freetypeVer,
        need: "pip install freetype-py — required for glyph rasterisation" },
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
  const fonts      = loadFonts();
  // Count families that actually have font files on disk. (Previously this
  // tested for a .git dir, which broke for fonts installed without cloning —
  // e.g. Google Fonts downloads — and after the fonts/ folder was cleaned.)
  const cloned     = fonts.filter(f => hasFontFiles(f.id)).length;
  const rendered   = globCount(P.rendered, ".png");
  const classical  = classicalImgCount();
  const breakdown  = lstmfBreakdown();
  const lstmfN     = breakdown.total;
  const cps        = getCheckpoints();
  const corpN      = lineCount(P.corpusTxt);
  const bestOk     = fs.existsSync(path.join(P.bestDir, "kan_hist.traineddata"));

  // ── Reconcile the training flag against reality ──────────────────────────
  // The portal showed a phantom "Training is running" banner because the UI
  // trusts (_training || _runningStep === 'train'), and runningStep could get
  // stuck: training is launched detached via caffeinate, so if the wrapper is
  // killed — or the portal is restarted while a run is in flight — the 'exit'
  // handler that clears it may never fire. The OS process table is the source
  // of truth, so clear the stale flag whenever no live lstmtraining exists.
  const livePids  = trainingPids();
  const trainLive = livePids.length > 0;
  if (!trainLive && runningStep === "train") runningStep = null;

  // Build a human-readable lstmf breakdown string, e.g. "11.2K classical · 5.2K rendered"
  const bdParts = [
    breakdown.classical  ? `${(breakdown.classical/1000).toFixed(1)}K classical`    : null,
    breakdown.rendered   ? `${(breakdown.rendered/1000).toFixed(1)}K rendered`      : null,
    breakdown["font-test"]? `${breakdown["font-test"]} font-test`                   : null,
    breakdown.scan       ? `${breakdown.scan} scan`                                 : null,
  ].filter(Boolean);
  const bdStr = bdParts.length ? bdParts.join(" · ") : `${lstmfN.toLocaleString()} files`;

  const renderDetail = [
    rendered  ? `${rendered.toLocaleString()} rendered lines` : null,
    classical ? `${classical.toLocaleString()} classical pages` : null,
  ].filter(Boolean).join(" · ") || "No images yet";

  res.json({
    "00_unichar": { label: "Expand unicharset", done: fs.existsSync(path.join(ROOT, "tessdata_expanded", "kan.traineddata")), detail: fs.existsSync(path.join(ROOT, "tessdata_expanded", "kan.traineddata")) ? "Expanded (ಋ ಙ ಝ ಱ added)" : "Not done — ಋ ಙ ಝ ಱ missing" },
    "01_prep":   { label: "1. Prep base",      done: fs.existsSync(path.join(P.tessdataDir, "kan.traineddata")) && cloned === fonts.length, detail: `kan.traineddata ${fs.existsSync(path.join(P.tessdataDir,"kan.traineddata"))?"✓":"✗"}  fonts ${cloned}/${fonts.length}` },
    "02_corpus": { label: "2. Corpus",         done: corpN > 0,      detail: `${corpN.toLocaleString()} lines` },
    "03_render": { label: "3. Render images",  done: rendered > 0 || classical > 0, detail: renderDetail },
    "04_lstmf":  { label: "4. Make lstmf",     done: lstmfN > 0,     detail: `${lstmfN.toLocaleString()} .lstmf files (${bdStr})` },
    "05_train":  { label: "5. Train",          done: cps.length > 0, detail: cps.length ? `${cps.length} checkpoints` : "Not started" },
    "06_package":{ label: "6. Package",        done: bestOk,         detail: bestOk ? "kan_hist.traineddata ready" : "Not done" },
    _training:       trainLive,
    _trainingPids:   livePids,
    _runningStep:    runningStep,
    _completedSteps: completedSteps,
    // Image counts for dashboard cards
    _rendered:   rendered,
    _classical:  classical,
    _lstmfTotal: lstmfN,
    _lstmfBreakdown: breakdown,
  });
});

// ── API: fonts ─────────────────────────────────────────────────────────────
app.get("/api/fonts", (req, res) => {
  const fonts = loadFonts();

  // Per-font rendered counts. Rendered files are named "<id>_<fontstem>_lineNNNN.png",
  // so bucket by the leading font id. (This column previously reported
  // globCount(rendered/) — the folder total — so every font showed the same
  // number regardless of how many images it actually contributed.)
  const counts = Object.create(null);
  if (fs.existsSync(P.rendered)) {
    for (const fn of fs.readdirSync(P.rendered)) {
      if (!fn.endsWith(".png")) continue;
      for (const f of fonts) {
        if (fn.startsWith(f.id + "_")) { counts[f.id] = (counts[f.id] || 0) + 1; break; }
      }
    }
  }

  res.json(fonts.map(f => ({
    id:       f.id,
    name:     f.name,
    repo:     f.repo,
    styles:   (f.font_files || []).length,
    degrade:  !!f.degrade,
    // Presence of font files, not a .git dir — fonts installed by download
    // (Google Fonts) are legitimately present without ever being cloned.
    cloned:   hasFontFiles(f.id),
    rendered: counts[f.id] || 0,
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

/// ── API: images ────────────────────────────────────────────────────────────
app.get("/api/images", (req, res) => {
  const n = parseInt(req.query.n) || 16;
  res.json(sampleImages(n));
});

// ── API: classical corpus samples ─────────────────────────────────────────
// Returns URL references (not base64) — images served via /classical-pages/
app.get("/api/classical-samples", (req, res) => {
  const n = parseInt(req.query.n) || 6;
  res.json(sampleClassicalImages(n));
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
// Use `tail -n N` to avoid loading the entire (potentially huge) log into memory.
app.get("/api/log/tail", (req, res) => {
  const n = parseInt(req.query.lines) || 100;
  if (!fs.existsSync(P.logFile)) return res.json({ lines: [], exists: false });
  try {
    const { execFileSync } = require("child_process");
    const out   = execFileSync("tail", ["-n", String(n), P.logFile], { encoding: "utf8", maxBuffer: 10 * 1024 * 1024 });
    const lines = out.split("\n").filter((_, i, a) => i < a.length - 1 || a[i] !== "");
    // Total line count via wc -l (cheap — just reads inode metadata path)
    let total = 0;
    try {
      const wc = execFileSync("wc", ["-l", P.logFile], { encoding: "utf8" });
      total = parseInt(wc.trim().split(/\s+/)[0]) || 0;
    } catch { /* non-critical */ }
    res.json({ lines: lines.map(sanitizeLogLine), exists: true, total });
  } catch (e) {
    res.status(500).json({ error: String(e.message) });
  }
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
    inventory:     ["python3", [path.join(P.corpus,  "generate-inventory.py")]],
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

  // render-corpus.py skips images that already exist. --force overwrites them
  // in place, which is what you want after a shaping or font change — no need
  // to delete rendered/ first.
  if (step === 'render' && force) runArgs.push('--force');

  // generate-inventory.py defaults to the weights declared in fonts.yml;
  // ?all_fonts=1 widens it to every .ttf/.otf on disk.
  if (step === 'inventory' && req.query.all_fonts === '1') runArgs.push('--all-fonts');

  // Support TRAIN_MODE variants for the training step
  const runOpts = {};
  if (step === 'train') {
    if (req.query.mode === 'fresh')  runOpts.env = { TRAIN_MODE: 'fresh'  };
    if (req.query.mode === 'expand') runOpts.env = { TRAIN_MODE: 'expand' };
  }

  // Auto-pass CLASSICAL_A5_DIR to 02-make-lstmf.sh so classical corpus pages
  // are included without manual env setup.  Priority:
  //   1. Last corpus dir used in the A5 render panel  (most specific)
  //   2. classical-corpus-kannada/a5-pages next to project root  (convention)
  //   3. Any <name>/a5-pages dir found directly under ROOT  (fallback scan)
  if (step === 'lstmf' && !runOpts.env?.CLASSICAL_A5_DIR) {
    const candidates = [];
    if (_a5CorpusPath) candidates.push(path.join(_a5CorpusPath, 'a5-pages'));
    candidates.push(path.join(ROOT, 'classical-corpus-kannada', 'a5-pages'));
    try {
      for (const entry of fs.readdirSync(ROOT)) {
        const p = path.join(ROOT, entry, 'a5-pages');
        if (!candidates.includes(p)) candidates.push(p);
      }
    } catch (_) {}
    const found = candidates.find(p => { try { return fs.statSync(p).isDirectory(); } catch { return false; } });
    if (found) {
      runOpts.env = { ...runOpts.env, CLASSICAL_A5_DIR: found };
      console.log(`  [lstmf] CLASSICAL_A5_DIR auto-detected: ${found}`);
    } else {
      console.log(`  [lstmf] No a5-pages directory found — classical corpus will be skipped`);
    }
  }

  // Same treatment for the character inventory. 02-make-lstmf.sh only includes
  // inventory/ when INVENTORY_DIR is set, so without this the portal silently
  // built a training set with NO character baselines — the inventory-first
  // strategy simply didn't happen for portal-driven runs.
  if (step === 'lstmf') {
    const invDir = path.join(ROOT, 'inventory');
    // withFileTypes so a stray file (.DS_Store) can't make readdirSync throw and
    // mask a perfectly good inventory. Accept PNGs at the top level too.
    const hasInv = (() => {
      try {
        const entries = fs.readdirSync(invDir, { withFileTypes: true });
        if (entries.some(e => e.isFile() && e.name.endsWith('.png'))) return true;
        return entries.some(e => {
          if (!e.isDirectory()) return false;
          try { return fs.readdirSync(path.join(invDir, e.name)).some(f => f.endsWith('.png')); }
          catch { return false; }
        });
      } catch { return false; }
    })();
    if (hasInv) {
      runOpts.env = { ...runOpts.env, INVENTORY_DIR: invDir };
      console.log(`  [lstmf] INVENTORY_DIR set: ${invDir}`);
    } else {
      console.log(`  [lstmf] inventory/ empty or missing — run the Inventory step first`);
    }
  }

  runBg(cmd, runArgs, step, runOpts);
  res.json({ ok: true, step });
});

/// ── A5 render process tracking ────────────────────────────────────────────
let _a5Proc       = null;   // currently running child process
let _a5Stopped    = false;  // true if explicitly stopped by user (vs completed)
let _a5CorpusPath = '';     // last corpus dir used — passed to 02-make-lstmf.sh

// ── API: render A5 corpus pages (browser rendering) ───────────────────────
// Runs corpus/render-a5-pages.py which uses browser_render.js --batch to
// produce correctly-shaped training images for historical fonts.
app.post("/api/render-a5-pages", express.json(), (req, res) => {
  if (_a5Proc) return res.status(409).json({ error: "Already running — stop it first" });

  // `lines` defaults to TRUE: page-mode output cannot be used for LSTM training
  // (a full page paired with the whole page's text is CTC-infeasible — see
  // docs/IMAGE_GENERATION.md §8). Pass lines:false only to produce page images
  // for visual inspection.
  const { corpus_dir, font_size = 32, workers = 1, concurrency = 2,
          lines = true } = req.body || {};
  if (!corpus_dir) return res.status(400).json({ error: "corpus_dir required" });

  // Resolve path: expand ~, resolve relative to ROOT, follow symlinks
  const rawPath   = corpus_dir.replace(/^~/, require("os").homedir());
  const candidate = path.isAbsolute(rawPath) ? rawPath : path.join(ROOT, rawPath);
  let corpusPath  = candidate;
  try { corpusPath = fs.realpathSync(candidate); } catch (_) {
    // realpathSync fails on broken symlinks — try statSync on the candidate anyway
  }
  // Skip validation entirely: let Python report the error with full context.
  // Just log what we're passing so the server terminal shows it.
  console.log(`  [a5] corpus_dir input: "${corpus_dir}"  →  "${corpusPath}"  (ROOT=${ROOT})`);
  // Soft check only — warn but don't block
  try {
    const st = fs.statSync(corpusPath);
    if (!st.isDirectory()) console.warn(`  [a5] WARNING: ${corpusPath} is not a directory`);
  } catch (e) {
    console.warn(`  [a5] WARNING: statSync failed: ${e.message} — passing path to Python anyway`);
  }

  const script = path.join(ROOT, "corpus", "render-a5-pages.py");
  if (!fs.existsSync(script)) {
    return res.status(404).json({ error: "corpus/render-a5-pages.py not found" });
  }

  const args = [
    script,
    "--corpus-dir", corpusPath,
    "--font-size",  String(Math.max(16, Math.min(72, parseInt(font_size) || 32))),
    "--workers",    String(Math.max(1,  Math.min(9,  parseInt(workers)    || 1))),
    "--concurrency",String(Math.max(1,  Math.min(8,  parseInt(concurrency)|| 2))),
  ];
  if (lines) args.push("--lines");
  console.log(`  [a5] mode: ${lines ? "LINE images (LSTM-ready)" : "PAGE images (not trainable)"}`);

  rotateLogIfNeeded();
  const logFd = fs.openSync(P.logFile, "a");
  const sep = "=".repeat(55);
  fs.writeSync(logFd, `\n${sep}\n[trainocr] render-a5-pages: python3 ${args.join(" ")}\n${new Date().toISOString()}\n${sep}\n\n`);
  fs.closeSync(logFd);

  _a5Stopped    = false;
  _a5CorpusPath = corpusPath;   // remember for lstmf step
  _a5Proc = spawn("python3", args, {
    cwd:      ROOT,
    // detached: true gives the child its own process group so we can kill
    // the entire tree (Python + its node sub-workers) with process.kill(-pid).
    detached: true,
    stdio:    ["ignore", fs.openSync(P.logFile, "a"), fs.openSync(P.logFile, "a")],
    env:      { ...process.env },
  });
  runningStep = "render-a5-pages";

  _a5Proc.on("exit", code => {
    _a5Proc    = null;
    runningStep = null;
    const l = fs.openSync(P.logFile, "a");
    fs.writeSync(l, `\n[trainocr] render-a5-pages ${
      _a5Stopped ? "⏹ stopped by user" : code === 0 ? "✓ done" : `✗ failed (exit ${code})`
    }\n`);
    fs.closeSync(l);
  });

  res.json({ ok: true, step: "render-a5-pages", corpus_dir: corpusPath });
});

// Stop the running A5 render (safe — resume picks up on next Start)
app.post("/api/render-a5-pages/stop", (req, res) => {
  if (!_a5Proc) return res.json({ ok: true, message: "Not running" });
  _a5Stopped = true;
  try {
    // Kill entire process group (Python + all node browser_render workers)
    process.kill(-_a5Proc.pid, "SIGTERM");
  } catch (_) {
    try { _a5Proc.kill("SIGTERM"); } catch (_2) {}
  }
  res.json({ ok: true });
});

// Status: count how many A5 PNGs have been rendered so far
app.get("/api/render-a5-pages/progress", (req, res) => {
  const { corpus_dir } = req.query;
  if (!corpus_dir) return res.json({ count: 0, running: !!_a5Proc, stopped: _a5Stopped });
  const rawPath = corpus_dir.replace(/^~/, require("os").homedir());
  const resolved = path.isAbsolute(rawPath) ? rawPath : path.join(ROOT, rawPath);
  const a5Dir = path.join(resolved, "a5-pages");
  let count = 0;
  try {
    const { execSync } = require("child_process");
    count = parseInt(execSync(`find "${a5Dir}" -name "*.png" 2>/dev/null | wc -l`).toString().trim(), 10) || 0;
  } catch (_) {}
  res.json({ count, a5_dir: a5Dir, running: !!_a5Proc, stopped: _a5Stopped });
});

// ── Font image comparison ─────────────────────────────────────────────────
app.get("/api/font-images/compare", (req, res) => {
  const testDir  = path.join(ROOT, "test-images");
  const families = fontFamilyIds();
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
  const families = fontFamilyIds();
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
      return { id: f.id, name: f.name, description: f.description, degrade: !!f.degrade,
               font_features: f.font_features || "", variants };
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
  const force  = !!(req.body && req.body.force);

  // Force-clear: remove existing PNGs so they are regenerated fresh
  if (force) {
    const clearDir = fontId
      ? path.join(outDir, fontId)   // clear only this font's subfolder
      : outDir;                      // clear everything
    if (fs.existsSync(clearDir)) {
      // Node.js 25 + macOS: rmSync({ recursive }) can throw ENOTEMPTY
      // when Finder or .DS_Store locks a file.  Shell rm -rf is reliable.
      try {
        fs.rmSync(clearDir, { recursive: true, force: true });
      } catch (_) {
        require("child_process").execSync(`rm -rf "${clearDir}"`);
      }
      fs.mkdirSync(clearDir, { recursive: true });
    }
  }

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
  const outputDir = path.join(ROOT, "output");

  // Use expanded unicharset base if available (checkpoints trained with expansion
  // have code range 140; using the original 116-char base causes "Code range changed" fatal error)
  const tessdataExp  = path.join(ROOT, "tessdata_expanded");
  const tessdataBest = path.join(ROOT, "tessdata_best");
  const tessdata = fs.existsSync(path.join(tessdataExp, "kan.traineddata")) ? tessdataExp : tessdataBest;

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

  // A5 classical pages — check multiple candidate locations.
  // Priority: last-used session path → conventional name → scan all ROOT subdirs.
  // Also check lstmf/classical/ which is the definitive indicator that the
  // lstmf step already processed classical pages (survives server restarts).
  function isDir(p) { try { return fs.statSync(p).isDirectory(); } catch { return false; } }

  const a5Candidates = [];
  if (_a5CorpusPath) a5Candidates.push(path.join(_a5CorpusPath, 'a5-pages'));
  a5Candidates.push(path.join(ROOT, 'classical-corpus-kannada', 'a5-pages'));
  // Scan every first-level directory/symlink under ROOT for an a5-pages subdir
  try {
    for (const entry of fs.readdirSync(ROOT, { withFileTypes: true })) {
      if (entry.isDirectory() || entry.isSymbolicLink()) {
        const p = path.join(ROOT, entry.name, 'a5-pages');
        if (!a5Candidates.includes(p)) a5Candidates.push(p);
      }
    }
  } catch (_) {}
  const a5Dir = a5Candidates.find(isDir) || null;

  // Count rendered PNGs (or fall back to lstmf/classical count × some factor)
  let a5Count = 0;
  const lstmfClassicalDir = path.join(ROOT, 'lstmf', 'classical');
  const lstmfClassicalExists = isDir(lstmfClassicalDir);
  if (a5Dir) {
    try {
      const { execSync } = require("child_process");
      a5Count = parseInt(execSync(`find "${a5Dir}" -name "*.png" 2>/dev/null | wc -l`).toString().trim(), 10) || 0;
    } catch (_) {}
  }
  // If we couldn't find the a5-pages dir but lstmf/classical/ exists, the
  // pages were rendered and converted — report via the lstmf count instead.
  if (!a5Count && lstmfClassicalExists) {
    try {
      const { execSync } = require("child_process");
      a5Count = parseInt(execSync(`find "${lstmfClassicalDir}" -name "*.lstmf" 2>/dev/null | wc -l`).toString().trim(), 10) || 0;
    } catch (_) {}
  }

  res.json({
    fonts: fontStatus,
    fontVariants:  { ready: readyVariants, total: totalVariants, trainImages },
    synced:        { count: syncedCount, upToDate: syncedCount >= trainImages },
    rendered:      { count: renderedCount },
    kanExists,
    a5Pages:       { dir: a5Dir || lstmfClassicalDir, count: a5Count, lstmfReady: lstmfClassicalExists },
    goForTraining: readyVariants > 0 && kanExists,
  });
});

// ── API: renderer diagnostics ─────────────────────────────────────────────
app.get("/api/renderer-info", (req, res) => {
  const { execFileSync } = require("child_process");
  let chromePath = null;
  const candidates = [
    process.env.PUPPETEER_EXECUTABLE_PATH,
    process.env.CHROME_PATH,
    "/usr/bin/chromium-browser",
    "/usr/bin/chromium",
    "/usr/bin/google-chrome",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
  ].filter(Boolean);
  for (const c of candidates) {
    try { require("fs").accessSync(c, require("fs").constants.X_OK); chromePath = c; break; } catch (_) {}
  }
  // Also check puppeteer cache
  const home = require("os").homedir();
  const cacheRoots = [
    path.join(home, ".cache", "puppeteer", "chrome"),
    path.join(home, "Library", "Caches", "puppeteer", "chrome"),
  ];
  if (!chromePath) {
    for (const root of cacheRoots) {
      try {
        for (const ver of fs.readdirSync(root)) {
          const candidates2 = [
            path.join(root, ver, "chrome-linux64", "chrome"),
            path.join(root, ver, "chrome-mac-arm64", "Google Chrome for Testing.app", "Contents", "MacOS", "Google Chrome for Testing"),
            path.join(root, ver, "chrome-mac-x64",  "Google Chrome for Testing.app", "Contents", "MacOS", "Google Chrome for Testing"),
          ];
          for (const c of candidates2) {
            try { fs.accessSync(c, fs.constants.X_OK); chromePath = c; break; } catch (_) {}
          }
          if (chromePath) break;
        }
      } catch (_) {}
      if (chromePath) break;
    }
  }
  let puppeteerVer = "unknown";
  try { puppeteerVer = require("./node_modules/puppeteer/package.json").version; } catch (_) {}
  res.json({
    platform:      process.platform,
    node:          process.version,
    puppeteer:     puppeteerVer,
    chromePath,
    cacheRootsChecked: cacheRoots,
    browserRenderScript: fs.existsSync(path.join(ROOT, "corpus", "browser_render.js")),
  });
});

// ── Browser (Puppeteer) instance — shared, lazy-initialised ───────────────
// We keep one browser open so successive preview requests don't pay Chrome
// startup cost every time.  The browser is launched on the first preview
// request and reused until the server exits.
let _browser = null;
let _browserErr = null;
async function getBrowser() {
  if (_browser) return _browser;
  const puppeteer = require("puppeteer");
  const os2 = require("os");

  // Find Chrome / Chromium
  function findChrome() {
    const home = os2.homedir();
    const isMac = process.platform === "darwin";
    // Priority order:
    //   1. Explicit env overrides (always respected)
    //   2. System Chrome (usually the most up-to-date; avoids Puppeteer cache version mismatches)
    //   3. Puppeteer cache  (populated by: npx puppeteer browsers install chrome)
    //      Sort versions descending so the newest cached Chrome wins.
    //   4. Other system browsers
    function puppeteerCachePaths() {
      const roots = [
        path.join(home, ".cache", "puppeteer", "chrome"),
        path.join(home, "Library", "Caches", "puppeteer", "chrome"),
      ];
      const found = [];
      for (const root of roots) {
        try {
          const vers = fs.readdirSync(root).sort().reverse(); // newest first
          for (const ver of vers) {
            found.push(
              path.join(root, ver, "chrome-mac-arm64", "Google Chrome for Testing.app", "Contents", "MacOS", "Google Chrome for Testing"),
              path.join(root, ver, "chrome-mac-x64",  "Google Chrome for Testing.app", "Contents", "MacOS", "Google Chrome for Testing"),
              path.join(root, ver, "chrome-linux64", "chrome"),
            );
          }
        } catch (_) {}
      }
      return found;
    }

    const candidates = [
      process.env.PUPPETEER_EXECUTABLE_PATH,
      process.env.CHROME_PATH,
      // System Chrome first — avoids stale Puppeteer-cache versions (e.g. Chrome 119 with Puppeteer v25)
      "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
      "/Applications/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing",
      "/usr/bin/google-chrome-stable",
      "/usr/bin/google-chrome",
      // Then Puppeteer cache (newest version first)
      ...puppeteerCachePaths(),
      // Other browsers
      "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
      "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
      "/usr/bin/chromium-browser", "/usr/bin/chromium",
    ].filter(Boolean);
    return candidates.find(p => { try { fs.accessSync(p, fs.constants.X_OK); return true; } catch { return false; } }) || null;
  }

  const chromePath = findChrome();
  const launchArgs = ["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage", "--font-render-hinting=none"];
  if (process.platform !== "darwin") launchArgs.push("--disable-gpu");
  // protocolTimeout: time allowed for individual CDP commands (screenshot, evaluate, etc.)
  // Default in Puppeteer v25 may be as low as 30 s; raise it to avoid spurious timeouts.
  const opts = { headless: true, args: launchArgs, protocolTimeout: 60000 };
  if (chromePath) opts.executablePath = chromePath;

  _browser = await puppeteer.launch(opts);
  console.log(`  [preview] Browser launched: ${chromePath || "bundled"}`);
  return _browser;
}

// ── SPA fallback ───────────────────────────────────────────────────────────
// ── API: live font render preview ──────────────────────────────────────────
// Renders using in-process Puppeteer.  The font is loaded via HTTP from the
// local server (not base64-embedded) so Chrome never stalls on a huge data URI.
// Falls back to Python HarfBuzz if Puppeteer fails.
app.post("/api/render-preview", express.json(), async (req, res) => {
  const { text, font_path, font_size = 48 } = req.body || {};
  if (!text || !font_path) return res.status(400).json({ error: "text and font_path required" });

  const abs = path.resolve(ROOT, font_path);
  if (!abs.startsWith(path.join(ROOT, "fonts"))) return res.status(403).json({ error: "Invalid font path" });
  if (!fs.existsSync(abs)) return res.status(404).json({ error: "Font not found: " + font_path });

  const size = Math.min(200, Math.max(8, parseInt(font_size) || 48));

  // Per-font OpenType feature settings (e.g. "'aalt' 1" for GTN/WMP, whose
  // correct conjunct forms live in the aalt GSUB feature — Chrome does not
  // enable aalt by default). See docs/CONJUNCT_RENDERING.md.
  const fontId = String(font_path).split("/")[1] || "";
  const fonts  = loadFonts();
  const font   = fonts.find(f => f.id === fontId);
  const features = (font && font.font_features) || "";

  // ── Try browser renderer ────────────────────────────────────────────────
  async function tryBrowser() {
    const browser = await getBrowser();
    const page    = await browser.newPage();
    page.setDefaultTimeout(30000);

    // Write HTML to a temp file and navigate via file://.
    // Avoids setContent()'s about:blank origin which blocks HTTP font loading.
    // Font referenced via file:// URL — Chrome loads it from disk directly,
    // caches it, and applies the full OS text-shaping stack (CoreText on macOS).
    const ext     = path.extname(abs).toLowerCase();
    const fmt     = ext === ".otf" ? "opentype" : "truetype";
    const fontUrl = `file://${abs}`;
    const escaped = text.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
    const html = `<!DOCTYPE html><html><head><meta charset="UTF-8"><style>
      @font-face { font-family:'KF'; src:url('${fontUrl}') format('${fmt}'); font-weight:normal; font-style:normal; }
      *{margin:0;padding:0;box-sizing:border-box;}
      body{background:#fff;display:inline-block;padding:16px 20px;white-space:nowrap;}
      #t{font-family:'KF',serif;font-size:${size}px;color:#000;line-height:1.4;white-space:pre-wrap;${features ? `font-feature-settings:${features};` : ""}}
    </style></head><body><div id="t">${escaped}</div></body></html>`;

    const tmpFile = require("os").tmpdir() + `/kanpreview_${Date.now()}.html`;
    fs.writeFileSync(tmpFile, html, "utf8");

    try {
      await page.setViewport({ width: 2400, height: 600, deviceScaleFactor: 1 });
      await page.goto(`file://${tmpFile}`, { waitUntil: "load" });
      // Explicitly trigger font load and wait — fonts are lazy in @font-face
      await page.evaluate(() => Promise.race([
        document.fonts.ready,
        new Promise(r => setTimeout(r, 8000)),
      ]));
      await page.evaluate(() => { void document.body.offsetHeight; }); // force layout

      const box = await page.evaluate(() => {
        const b = document.body.getBoundingClientRect();
        return { w: Math.ceil(Math.max(b.width, 60)), h: Math.ceil(Math.max(b.height, 40)) };
      });
      await page.setViewport({ width: box.w, height: box.h, deviceScaleFactor: 1 });
      const buf = await page.screenshot({ type: "png", clip: { x:0, y:0, width:box.w, height:box.h } });
      return buf.toString("base64");
    } finally {
      await page.close();
      try { fs.unlinkSync(tmpFile); } catch (_) {}
    }
  }

  // ── Try Python HarfBuzz fallback ────────────────────────────────────────
  function tryPython() {
    return new Promise((resolve, reject) => {
      const { spawn } = require("child_process");
      const pyArgs = [path.join(ROOT, "corpus", "render_preview.py"), abs, String(size)];
      if (features) pyArgs.push("--aalt");
      const py = spawn("python3", pyArgs, { cwd: ROOT });
      let out = "", err = "";
      py.stdout.on("data", d => out += d);
      py.stderr.on("data", d => err += d);
      const t = setTimeout(() => { py.kill(); reject(new Error("Python timeout")); }, 12000);
      py.on("close", code => {
        clearTimeout(t);
        if (code === 0 && out.trim()) resolve(out.trim());
        else reject(new Error(err.trim() || "Python render failed"));
      });
      py.stdin.write(text);
      py.stdin.end();
    });
  }

  try {
    const png = await tryBrowser();
    res.json({ png, shaped: true, renderer: "browser" });
  } catch (browserErr) {
    console.error("  [preview] Browser failed:", browserErr.message, "— falling back to Python");
    try {
      const png = await tryPython();
      res.json({ png, shaped: true, renderer: "python", warning: browserErr.message });
    } catch (pyErr) {
      res.status(500).json({ error: pyErr.message || browserErr.message || "Both renderers failed" });
    }
  }
});

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
