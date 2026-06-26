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
};

fs.mkdirSync(P.scanDir, { recursive: true });
fs.mkdirSync(P.public,  { recursive: true });

// ── Middleware ─────────────────────────────────────────────────────────────
app.use(express.json({ limit: "50mb" }));
app.use(express.static(P.public));

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
    .filter(f => /^kan_hist_[\d.]+_\d+\.checkpoint$/.test(f))
    .map(f => {
      const parts = f.replace(".checkpoint", "").split("_");
      const stat  = fs.statSync(path.join(P.outputDir, f));
      return { file: f, bcer: parseFloat(parts[2]), iter: parseInt(parts[3]), size: stat.size };
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

function runBg(cmd, args, stepId) {
  const log  = fs.openSync(P.logFile, "a");
  const sep  = "=".repeat(55);
  fs.writeSync(log, `\n${sep}\n[trainocr] ${stepId}: ${cmd} ${args.join(" ")}\n${new Date().toISOString()}\n${sep}\n\n`);
  fs.closeSync(log);

  const child = spawn(cmd, args, {
    cwd:   ROOT,
    stdio: ["ignore", fs.openSync(P.logFile, "a"), fs.openSync(P.logFile, "a")],
    detached: false,
  });
  child.on("exit", code => {
    const l = fs.openSync(P.logFile, "a");
    fs.writeSync(l, `\n[trainocr] ${stepId} exit ${code}\n`);
    fs.closeSync(l);
  });
}

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
    "01_prep":   { label: "1. Prep base model",  done: fs.existsSync(path.join(P.tessdataDir, "kan.traineddata")) && cloned === fonts.length, detail: `kan.traineddata ${fs.existsSync(path.join(P.tessdataDir,"kan.traineddata"))?"✓":"✗"}  fonts ${cloned}/${fonts.length}` },
    "02_corpus": { label: "2. Build corpus",     done: corpN > 0,     detail: `${corpN.toLocaleString()} lines` },
    "03_render": { label: "3. Render images",    done: rendered > 0,  detail: `${rendered.toLocaleString()} PNG images` },
    "04_lstmf":  { label: "4. Generate lstmf",   done: lstmfN > 0,    detail: `${lstmfN.toLocaleString()} .lstmf files` },
    "05_train":  { label: "5. Train",            done: cps.length > 0, detail: cps.length ? `${cps.length} checkpoints` : "Not started" },
    "06_package":{ label: "6. Package",          done: bestOk,        detail: bestOk ? "kan_hist.traineddata ready" : "Not done" },
    _training:   isTrainingRunning(),
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
  const gt   = (req.body.gt || "").trim();
  const name = req.file.originalname;
  const stem = name.replace(/\.[^.]+$/, "");
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

// ── API: log tail ──────────────────────────────────────────────────────────
app.get("/api/log/tail", (req, res) => {
  const n = parseInt(req.query.lines) || 100;
  if (!fs.existsSync(P.logFile)) return res.json({ lines: [], exists: false });
  const lines = fs.readFileSync(P.logFile, "utf8").split("\n");
  res.json({ lines: lines.slice(-n), exists: true, total: lines.length });
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
      if (line) res.write(`data: ${JSON.stringify(line)}\n\n`);
    });
  }, 1000);

  req.on("close", () => clearInterval(interval));
});

// ── API: run step ──────────────────────────────────────────────────────────
app.post("/api/run/:step", (req, res) => {
  const cmds = {
    prep:    ["bash", [path.join(P.scripts, "01-prep-base.sh")]],
    wiki:    ["python3", [path.join(P.corpus, "download-wiki.py")]],
    clean:   ["python3", [path.join(P.corpus, "clean-corpus.py")]],
    render:  ["python3", [path.join(P.corpus, "render-corpus.py")]],
    lstmf:   ["bash", [path.join(P.scripts, "02-make-lstmf.sh")]],
    train:   ["bash", [path.join(P.scripts, "03-train.sh")]],
    package: ["bash", [path.join(P.scripts, "04-package.sh")]],
  };
  const { step } = req.params;
  if (!cmds[step]) return res.status(400).json({ error: `Unknown step: ${step}` });
  const [cmd, args] = cmds[step];
  if (!fs.existsSync(args[0])) return res.status(404).json({ error: `Script not found: ${args[0]}` });
  runBg(cmd, args, step);
  res.json({ ok: true, step });
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
