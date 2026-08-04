#!/usr/bin/env node
/**
 * browser_render.js
 * ─────────────────
 * Renders Kannada text using a headless browser (Puppeteer / system Chromium).
 * The browser applies the full OS text-rendering stack — the same pipeline as
 * fonts.sanchaya.net — so conjuncts and complex characters form correctly even
 * for fonts whose OpenType GSUB tables are incomplete.
 *
 * Two modes
 * ─────────
 * 1. Single render (for live preview / gen-char-images):
 *      node corpus/browser_render.js --font <path> --text <text> [--size 48]
 *      Writes base64 PNG to stdout.
 *
 * 2. Batch render (for render-corpus replacement):
 *      node corpus/browser_render.js --batch <jobs.json>
 *      jobs.json = [{ font, text, out, size, degrade, seed }, ...]
 *      Renders all jobs in a single browser session and writes PNG files.
 *      Writes a JSON summary to stdout: { ok, skip, fail, total }
 *
 * Why browser rendering?
 * ──────────────────────
 * Historical Sanchaya fonts (GMP, WMP, GTN TTFs) rely on the OS text stack
 * for conjunct shaping. Their OpenType GSUB tables are incomplete or absent.
 * Python HarfBuzz + FreeType renders each glyph in isolation → broken conjuncts.
 * The browser (CoreText on macOS, HarfBuzz+Pango on Linux) fills the gap.
 */

'use strict';

const path    = require('path');
const fs      = require('fs');
const os      = require('os');

// ── Puppeteer / browser detection ────────────────────────────────────────────
let puppeteer;
try {
  puppeteer = require('puppeteer');
} catch (_) {
  try {
    puppeteer = require('puppeteer-core');
  } catch (_2) {
    console.error('ERROR: puppeteer not installed. Run: npm install puppeteer');
    process.exit(1);
  }
}

// Chromium/Chrome binary candidates
function findChromium() {
  const home = os.homedir();

  // Puppeteer cache: populated by `npx puppeteer browsers install chrome`
  // Structure: ~/.cache/puppeteer/chrome/<platform>-<version>/chrome-linux64/chrome
  //        or: ~/Library/Caches/puppeteer/chrome/mac_arm-<ver>/…/Google Chrome for Testing
  function findInPuppeteerCache() {
    const cacheRoots = [
      path.join(home, '.cache', 'puppeteer', 'chrome'),
      path.join(home, 'Library', 'Caches', 'puppeteer', 'chrome'),
    ];
    for (const root of cacheRoots) {
      if (!fs.existsSync(root)) continue;
      try {
        for (const ver of fs.readdirSync(root).sort().reverse()) { // newest first
          const verDir = path.join(root, ver);
          // Linux: .../chrome-linux64/chrome
          const linuxBin = path.join(verDir, 'chrome-linux64', 'chrome');
          if (isExec(linuxBin)) return linuxBin;
          // macOS arm: .../chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing
          const macArm = path.join(verDir, 'chrome-mac-arm64',
            'Google Chrome for Testing.app', 'Contents', 'MacOS', 'Google Chrome for Testing');
          if (isExec(macArm)) return macArm;
          // macOS x86: .../chrome-mac-x64/...
          const macX86 = path.join(verDir, 'chrome-mac-x64',
            'Google Chrome for Testing.app', 'Contents', 'MacOS', 'Google Chrome for Testing');
          if (isExec(macX86)) return macX86;
        }
      } catch (_) {}
    }
    return null;
  }

  function isExec(p) {
    try { fs.accessSync(p, fs.constants.X_OK); return true; } catch { return false; }
  }

  // Priority: env overrides → system Chrome (most up-to-date) → Puppeteer cache → other browsers.
  // System Chrome first avoids stale cache versions (e.g. Chrome 119 vs Puppeteer v25 expecting 131+).
  const candidates = [
    process.env.PUPPETEER_EXECUTABLE_PATH,
    process.env.CHROME_PATH,
    // macOS system Chrome
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/Applications/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing',
    // Linux system Chrome
    '/usr/bin/google-chrome-stable',
    '/usr/bin/google-chrome',
    '/usr/bin/chromium-browser',
    '/usr/bin/chromium',
    '/snap/bin/chromium',
    // Puppeteer cache (newest first via sort)
    findInPuppeteerCache(),
    // Other browsers
    '/Applications/Chromium.app/Contents/MacOS/Chromium',
    '/Applications/Brave Browser.app/Contents/MacOS/Brave Browser',
    '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge',
    '/Applications/Arc.app/Contents/MacOS/Arc',
  ].filter(Boolean);
  return candidates.find(isExec) || null;
}

// Puppeteer v21+ executablePath() is async — check it separately
async function findChromiumAsync() {
  const sync = findChromium();
  if (sync) return sync;
  // Fallback: ask puppeteer where its bundled Chrome would be
  try {
    const ep = await puppeteer.executablePath();
    if (ep && typeof ep === 'string') {
      try { fs.accessSync(ep, fs.constants.X_OK); return ep; } catch (_) {}
    }
  } catch (_) {}
  return null;
}

// ── HTML template ─────────────────────────────────────────────────────────────
// Renders text exactly as fonts.sanchaya.net does: CSS @font-face + div.
//
// Font loading strategy:
//   We reference the font via file:// URL (or pass an http:// URL directly).
//   This keeps every HTML page tiny (~500 B vs 1-3 MB of base64), lets Chrome
//   load the font from disk at its own speed, and — crucially — Chrome caches
//   the decoded font face across all pages that share the same file:// URL,
//   so subsequent pages in the same worker session load near-instantly.
//   DO NOT embed base64 here: 1-3 MB of inline data stalls page.setContent()
//   and makes even temp-file navigation slow enough to trigger job timeouts.
//
// Page mode (pageW + pageH > 0): fixed A5-size canvas, text wraps at page
//   width.  Overflow is clipped.  Screenshot is exactly pageW × pageH px.
// Line mode (default): viewport auto-sized to content (existing behaviour).
function buildHtml(fontPath, text, fontSize,
                   bgColor = '#ffffff', inkColor = '#000000',
                   pageW = 0, pageH = 0, marginX = 50, marginY = 60,
                   features = '') {
  // Accept http/https/data URIs as-is; convert local paths to file:// URLs.
  const fontUrl = (fontPath.startsWith('http') || fontPath.startsWith('data:'))
    ? fontPath
    : `file://${path.resolve(fontPath)}`;

  const ext    = path.extname(fontPath).toLowerCase();
  const format = ext === '.otf' ? 'opentype' : 'truetype';

  const escaped  = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');

  const isPage = pageW > 0 && pageH > 0;

  const bodyStyle = isPage
    ? `width:${pageW}px; height:${pageH}px; overflow:hidden; background:${bgColor}; padding:${marginY}px ${marginX}px; box-sizing:border-box;`
    : `background:${bgColor}; display:inline-block; padding:16px 20px; white-space:nowrap;`;

  // Optional per-font OpenType feature settings (e.g. "'aalt' 1" for GTN/WMP,
  // whose correct conjunct forms live in the aalt GSUB feature). Empty string
  // = browser defaults, which is correct for GMP/Kittel. See docs/CONJUNCT_RENDERING.md.
  const featureCss = features ? ` font-feature-settings:${features};` : '';

  const textStyle = isPage
    ? `font-family:'KanFont',serif; font-size:${fontSize}px; color:${inkColor}; line-height:1.6; white-space:pre-wrap; word-break:break-word; width:100%;${featureCss}`
    : `font-family:'KanFont',serif; font-size:${fontSize}px; color:${inkColor}; line-height:1.4; white-space:pre-wrap; word-break:break-word;${featureCss}`;

  return `<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  @font-face {
    font-family: 'KanFont';
    src: url('${fontUrl}') format('${format}');
    font-weight: normal;
    font-style: normal;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { ${bodyStyle} }
  #t { ${textStyle} }
</style>
</head>
<body><div id="t">${escaped}</div></body>
</html>`;
}

// ── Per-line box measurement (runs inside the page) ──────────────────────────
// Walks the rendered text one character at a time, asking Chrome for each
// character's client rect, and groups characters that share a baseline row into
// a visual line. This gives the exact pixel box of every wrapped line together
// with the text that produced it — so a cropped line image and its ground truth
// can never drift apart.
//
// Serialised to the browser by page.evaluate(), so it must be self-contained.
function measureLinesInPage() {
  const el = document.getElementById('t');
  if (!el || !el.firstChild) return [];
  const node = el.firstChild;
  const text = node.textContent || '';
  const range = document.createRange();

  const lines = [];
  let cur = null;
  const ROW_TOLERANCE = 3;   // px — same line if tops differ by less than this

  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (ch === '\n') { cur = null; continue; }   // explicit break starts a line

    range.setStart(node, i);
    range.setEnd(node, i + 1);
    const r = range.getBoundingClientRect();

    // Zero-area rects are collapsed whitespace — keep the character in the
    // transcription (word spacing matters) but don't let it define the box.
    if (!r || (r.width === 0 && r.height === 0)) {
      if (cur) cur.text += ch;
      continue;
    }

    // Group by VERTICAL OVERLAP, not by matching rect tops.
    //
    // Comparing r.top to cur.top was wrong for Kannada. Glyph rects on one
    // visual line have wildly different tops: a bare consonant, one carrying an
    // ascender matra, and a below-base ottu all start at different heights.
    // Any difference over the tolerance began a NEW line group, so a single
    // visual line was split into several — each holding part of the text but
    // only a sliver of the height. The result was 386x10 crops with the full
    // 16-character transcription and 0% ink: the glyphs were outside the box.
    //
    // Characters belonging to the same line overlap vertically — but ANY
    // overlap is far too loose a test. Consecutive lines touch: a descender or
    // below-base ottu on one line reaches into the ascender zone of the next.
    // With "> 0" the groups chain transitively (line 1 touches 2, 2 touches 3…)
    // and the whole page collapses into one block. That produced crops carrying
    // 240+ characters, which are CTC-infeasible for the same reason a full page
    // is: far more labels than the timestep budget.
    //
    // Require the overlap to cover most of the SHORTER box instead. Two glyphs
    // on one line overlap almost completely relative to the shorter of the two;
    // glyphs on adjacent lines graze each other by a few pixels out of ~40.
    const OVERLAP_FRAC = 0.5;
    const overlaps = (a, b) => {
      const inter = Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top);
      if (inter <= 0) return false;
      const shorter = Math.min(a.bottom - a.top, b.bottom - b.top);
      return shorter > 0 && inter / shorter >= OVERLAP_FRAC;
    };

    if (cur && (overlaps(cur, r) || Math.abs(r.top - cur.top) <= ROW_TOLERANCE)) {
      cur.left   = Math.min(cur.left,   r.left);
      cur.right  = Math.max(cur.right,  r.right);
      cur.top    = Math.min(cur.top,    r.top);
      cur.bottom = Math.max(cur.bottom, r.bottom);
      cur.text  += ch;
    } else {
      cur = { top: r.top, left: r.left, right: r.right, bottom: r.bottom, text: ch };
      lines.push(cur);
    }
  }

  // Merge fragments of the same visual line that reading order left separate
  // (a below-base mark measured after the following base consonant). Uses the
  // same majority-overlap rule — a plain "> 0" test here would re-chain the
  // whole page into a single block.
  const merged = [];
  for (const l of lines.sort((a, b) => a.top - b.top)) {
    const prev = merged[merged.length - 1];
    const inter = prev ? Math.min(prev.bottom, l.bottom) - Math.max(prev.top, l.top) : 0;
    const shorter = prev ? Math.min(prev.bottom - prev.top, l.bottom - l.top) : 0;
    if (prev && shorter > 0 && inter / shorter >= 0.5) {
      prev.left   = Math.min(prev.left,   l.left);
      prev.right  = Math.max(prev.right,  l.right);
      prev.top    = Math.min(prev.top,    l.top);
      prev.bottom = Math.max(prev.bottom, l.bottom);
      prev.text  += l.text;
    } else {
      merged.push({ ...l });
    }
  }

  return merged
    .map(l => ({
      text: l.text.trim(),
      x: l.left, y: l.top,
      w: l.right - l.left,
      h: l.bottom - l.top,
    }))
    // A usable line must be tall enough to contain glyphs. Slivers indicate a
    // measurement failure, not a real line — never emit them as training data.
    .filter(l => l.text.length > 0 && l.w > 0 && l.h >= 16);
}

// ── Noise / blur degradation (mimics historical print) ───────────────────────
function deterministicRng(seed) {
  let s = seed >>> 0;
  return () => {
    s ^= s << 13; s ^= s >>> 17; s ^= s << 5;
    return (s >>> 0) / 4294967296;
  };
}

async function degradePng(page, seed) {
  // Apply mild Gaussian-like noise via canvas manipulation
  await page.evaluate((seedVal) => {
    const img = document.querySelector('body');
    const canvas = document.createElement('canvas');
    const rect   = img.getBoundingClientRect();
    canvas.width  = rect.width;
    canvas.height = rect.height;
    // Noise is applied via CSS filter — simpler than pixel manipulation
    img.style.filter = 'blur(0.4px)';
  }, seed);
}

// ── Shared page render helper ─────────────────────────────────────────────────
async function renderPage(browser, html, { degrade = false, seed = 0 } = {}) {
  // Write HTML to a temp file — page.goto(file://) is far faster than
  // page.setContent() when the HTML embeds a 1-2 MB base64 font string.
  const tmpFile = path.join(os.tmpdir(),
    `kanfont_${Date.now()}_${Math.random().toString(36).slice(2)}.html`);
  fs.writeFileSync(tmpFile, html, 'utf8');

  const page = await browser.newPage();
  page.setDefaultNavigationTimeout(30000);
  page.setDefaultTimeout(30000);
  try {
    // Start with a wide viewport so inline-block body expands freely
    await page.setViewport({ width: 2400, height: 800, deviceScaleFactor: 1 });
    await page.goto(`file://${tmpFile}`, { waitUntil: 'load' });

    // Wait for font with in-browser timeout — avoids dangling async evaluate
    // that would block subsequent CDP calls (e.g. screenshot).
    await page.evaluate(() => Promise.race([
      document.fonts.ready,
      new Promise(r => setTimeout(r, 10_000)),
    ]));
    // Force synchronous layout so Chrome has composited before screenshot
    await page.evaluate(() => { void document.body.offsetHeight; });

    // Measure the actual rendered content size
    const box = await page.evaluate(() => {
      const el = document.getElementById('t') || document.body;
      const r  = el.getBoundingClientRect();
      const bs = document.body.getBoundingClientRect();
      return {
        x:      Math.floor(bs.x),
        y:      Math.floor(bs.y),
        width:  Math.ceil(bs.width  || r.width  + 40),
        height: Math.ceil(bs.height || r.height + 32),
      };
    });

    const W = Math.max(box.width,  60);
    const H = Math.max(box.height, 40);

    if (degrade) {
      const rng   = deterministicRng(seed);
      const blur  = (0.4 + rng() * 0.4).toFixed(2);
      const angle = ((rng() - 0.5) * 1.6).toFixed(3);
      await page.evaluate((b, a) => {
        document.body.style.filter    = `blur(${b}px)`;
        document.body.style.transform = `rotate(${a}deg)`;
      }, blur, angle);
      await new Promise(r => setTimeout(r, 30));
    }

    // Resize viewport to exact content size, then screenshot
    await page.setViewport({ width: W, height: H, deviceScaleFactor: 1 });
    const buf = await page.screenshot({
      type: 'png',
      clip: { x: 0, y: 0, width: W, height: H },
    });
    return buf;
  } finally {
    await page.close();
    try { fs.unlinkSync(tmpFile); } catch (_) {}
  }
}

// ── Single render ─────────────────────────────────────────────────────────────
async function renderSingle(browser, fontPath, text, fontSize, features = '') {
  const html = buildHtml(fontPath, text, fontSize, '#ffffff', '#000000', 0, 0, 0, 0, features);
  return renderPage(browser, html);
}

// ── Batch render ──────────────────────────────────────────────────────────────
async function renderBatch(browser, jobs, label = '') {
  let ok = 0, skip = 0, fail = 0;
  const CONCURRENCY = parseInt(process.env.BROWSER_CONCURRENCY || '4', 10);
  const prefix = label ? `[${label}] ` : '';

  // Per-job timeout: if Chrome hangs on a page, don't block the entire batch.
  // With file:// font URLs a page typically renders in 1-5 s on modern Chrome.
  // 45 s is generous; failures surface quickly so we can bail/report early.
  const JOB_TIMEOUT_MS = 45_000;

  async function processJob(job) {
    const {
      font, text, out, size = 36, degrade = false, seed = 0, features = '',
      // A5 page-mode fields (optional)
      page_w = 0, page_h = 0, margin_x = 50, margin_y = 60,
      // Line mode: emit one cropped image per visual text line instead of one
      // page image. Required for LSTM training — see the note in renderBatch.
      lines: lineMode = false, line_pad: linePad = 6,
      // force: re-render even when output exists. Required after a shaping or
      // font-feature change — otherwise the resume check below preserves every
      // image rendered under the old settings, and a "re-render" silently does
      // nothing. This is how aalt-rendered GTN/WMP pages survived a rebuild.
      force = false,
    } = job;
    const isPage = page_w > 0 && page_h > 0;

    // Resume: skip already rendered.
    // In line mode the page itself is never written, so completion is judged
    // by the first line's output instead.
    const gtPath = out.replace(/\.png$/, '.gt.txt');
    if (force) {
      // fall through to render
    } else if (lineMode) {
      const probe = out.replace(/\.png$/, '_line000.png');
      if (fs.existsSync(probe) && fs.existsSync(probe.replace(/\.png$/, '.gt.txt'))) {
        return 'skip';
      }
    } else if (fs.existsSync(out) && fs.existsSync(gtPath)) {
      return 'skip';
    }

    const page = await browser.newPage();
    // Cap every individual CDP call (evaluate, screenshot, etc.) at 30 s.
    // Without this, page.screenshot() can hang indefinitely on macOS when
    // a dangling async evaluate is still running in the tab.
    page.setDefaultTimeout(30_000);
    page.setDefaultNavigationTimeout(60_000);

    try {
      let W, H;

      // Helper: write HTML to a temp file and navigate via file://.
      // Avoids page.setContent() stalls for large HTML (1-2 MB base64 font).
      async function gotoHtml(htmlStr) {
        const tmp = path.join(os.tmpdir(),
          `kanfont_${Date.now()}_${Math.random().toString(36).slice(2)}.html`);
        fs.writeFileSync(tmp, htmlStr, 'utf8');
        try {
          await page.goto(`file://${tmp}`, { waitUntil: 'load' });
        } finally {
          try { fs.unlinkSync(tmp); } catch (_) {}
        }
      }

      // Wait for the KanFont face with a deadline entirely inside the browser.
      // Keeping the timeout in-browser means the evaluate() always resolves
      // within FONT_WAIT_MS — no dangling async evaluate to block screenshot.
      const FONT_WAIT_MS = 10_000;
      async function waitForFont() {
        await page.evaluate((ms) => Promise.race([
          document.fonts.ready,
          new Promise(r => setTimeout(r, ms)),
        ]), FONT_WAIT_MS);
        // Force a synchronous layout so Chrome has composited the page
        // before we call page.screenshot().
        await page.evaluate(() => { void document.body.offsetHeight; });
      }

      if (isPage) {
        // ── A5 fixed-dimension mode ──────────────────────────────────
        W = page_w; H = page_h;
        await page.setViewport({ width: W, height: H, deviceScaleFactor: 1 });
        const html = buildHtml(font, text, size, '#ffffff', '#000000', W, H, margin_x, margin_y, features);
        await gotoHtml(html);
        await waitForFont();
      } else {
        // ── Auto-size line mode ──────────────────────────────────────
        const html = buildHtml(font, text, size, '#ffffff', '#000000', 0, 0, 0, 0, features);
        await page.setViewport({ width: 2400, height: 800, deviceScaleFactor: 1 });
        await gotoHtml(html);
        await waitForFont();

        const box = await page.evaluate(() => {
          const bs = document.body.getBoundingClientRect();
          return { width: Math.ceil(bs.width || 200), height: Math.ceil(bs.height || 60) };
        });
        W = Math.max(box.width,  60);
        H = Math.max(box.height, 40);
        await page.setViewport({ width: W, height: H, deviceScaleFactor: 1 });
      }

      // Degrade for historical fonts: blur via CSS
      if (degrade) {
        const rng = deterministicRng(seed);
        const blur    = (0.4 + rng() * 0.4).toFixed(2);
        const rotate  = ((rng() - 0.5) * 1.6).toFixed(3);
        await page.evaluate((b, r) => {
          document.body.style.filter    = `blur(${b}px)`;
          document.body.style.transform = `rotate(${r}deg)`;
        }, blur, rotate);
      }

      const pngBuf = await page.screenshot({
        type: 'png',
        clip: { x: 0, y: 0, width: W, height: H }
      });

      fs.mkdirSync(path.dirname(out), { recursive: true });

      // ── Line mode ────────────────────────────────────────────────
      // Tesseract LSTM training needs ONE IMAGE PER TEXT LINE. A full page
      // paired with the whole page's text cannot be aligned by CTC: the image
      // scales to ~33 timesteps at 48px height while the transcription needs
      // hundreds, and lstmtraining reports "Compute CTC targets failed".
      //
      // Chrome already knows where every line landed after layout, so we ask
      // it for the per-line boxes and crop the page screenshot accordingly.
      // The crop is exact and the ground truth is the text of that line —
      // no OCR or heuristic segmentation involved.
      if (lineMode) {
        const boxes = await page.evaluate(measureLinesInPage);
        if (!boxes.length) return 'fail:no lines measured';

        const sharp = require('sharp');
        const stem  = out.replace(/\.png$/, '');
        let written = 0;

        for (let i = 0; i < boxes.length; i++) {
          const b = boxes[i];
          if (!b.text) continue;
          // Pad, then clamp to the page so the crop is always in bounds.
          const x = Math.max(0, Math.floor(b.x - linePad));
          const y = Math.max(0, Math.floor(b.y - linePad));
          const wantW = Math.ceil(b.w + linePad * 2);
          const wantH = Math.ceil(b.h + linePad * 2);
          const w = Math.min(W - x, wantW);
          const h = Math.min(H - y, wantH);

          // Drop lines cut off by the page fold.
          //
          // The page is `overflow:hidden`, so the last line of a chunk is often
          // only partly rendered. Its measured box still reports the FULL line
          // height, and clamping to the page then yields a sliver — typically
          // 10px — while the ground truth still describes the whole line. That
          // is a mismatched pair: the transcription names text the image does
          // not show. One per page, ~5% of all crops, all at the highest line
          // index (line020/line021 in a 21-line page).
          //
          // Losing the tail of a page costs almost nothing; training on an image
          // whose GT does not match it is actively harmful.
          if (h < wantH * 0.7 || h < 16 || w < 8) continue;

          const lineOut = `${stem}_line${String(i).padStart(3, '0')}.png`;
          await sharp(pngBuf)
            .extract({ left: x, top: y, width: w, height: h })
            .png()
            .toFile(lineOut);
          fs.writeFileSync(lineOut.replace(/\.png$/, '.gt.txt'), b.text, 'utf8');
          written++;
        }
        return written ? 'ok' : 'fail:no usable lines';
      }

      fs.writeFileSync(out, pngBuf);
      fs.writeFileSync(gtPath, text, 'utf8');
      return 'ok';
    } catch (e) {
      return `fail:${e.message}`;
    } finally {
      await page.close();
    }
  }

  // Wrap each job with a timeout so a hung page doesn't stall the pipeline.
  function processJobSafe(job) {
    return Promise.race([
      processJob(job),
      new Promise(resolve =>
        setTimeout(() => resolve('fail:timeout'), JOB_TIMEOUT_MS)
      ),
    ]);
  }

  // Process jobs in sliding batches of CONCURRENCY.
  // Print progress every batch (not just every 200) so the terminal stays alive.
  const total = jobs.length;
  let done = 0;
  const REPORT_EVERY = Math.max(CONCURRENCY, 20);   // at most one line per 20 jobs
  let lastReport = 0;

  // Track unique error messages so we can surface them (max 5 distinct reasons).
  const errorSeen = new Map(); // message → count
  const MAX_ERR_TYPES = 5;

  // Bail out early only if a substantial run of jobs ALL fail with no successes
  // AND no skips.  Skip-only results (resume mode) are fine — they mean prior
  // output already exists.  We check after 5 % of jobs (min 20, max 200).
  const EARLY_BAIL_THRESHOLD = Math.min(Math.max(Math.round(total * 0.05), 20), 200);
  let earlyBailChecked = false;

  for (let i = 0; i < total; i += CONCURRENCY) {
    const chunk   = jobs.slice(i, i + CONCURRENCY);
    const results = await Promise.all(chunk.map(processJobSafe));
    results.forEach(r => {
      if (r === 'ok')        ok++;
      else if (r === 'skip') skip++;
      else {
        fail++;
        // r is 'fail:<message>' or 'fail:timeout'
        const msg = r.startsWith('fail:') ? r.slice(5) : r;
        errorSeen.set(msg, (errorSeen.get(msg) || 0) + 1);
      }
    });
    done += chunk.length;

    // Early bail: stop only if we've seen zero successes AND zero skips, but
    // have failures — meaning something is fundamentally broken (bad Chrome,
    // missing fonts, etc.).  Skips alone are fine (resume mode).
    if (!earlyBailChecked && done >= EARLY_BAIL_THRESHOLD && ok === 0 && skip === 0 && fail > 0) {
      earlyBailChecked = true;
      process.stderr.write(`\n${prefix}⚠  Early-bail check: ${fail} failures, 0 successes after ${done} jobs.\n`);
      process.stderr.write(`${prefix}   Top errors seen:\n`);
      for (const [msg, count] of [...errorSeen.entries()].slice(0, MAX_ERR_TYPES)) {
        process.stderr.write(`${prefix}     [${count}×] ${msg}\n`);
      }
      process.stderr.write(`${prefix}   Stopping — fix the error above before re-running.\n\n`);
      break;
    }

    if (done - lastReport >= REPORT_EVERY || done === total) {
      lastReport = done;
      process.stderr.write(
        `${prefix}${done}/${total} (${Math.round(done * 100 / total)}%)` +
        `  ok=${ok}  skip=${skip}  fail=${fail}\n`
      );
    }
  }

  // Always print a summary of errors if any occurred.
  if (fail > 0 && errorSeen.size > 0) {
    process.stderr.write(`\n${prefix}Error summary (${fail} total):\n`);
    for (const [msg, count] of [...errorSeen.entries()].slice(0, MAX_ERR_TYPES)) {
      process.stderr.write(`${prefix}  [${count}×] ${msg}\n`);
    }
    process.stderr.write('\n');
  }

  return { ok, skip, fail, total };
}

// ── CLI ───────────────────────────────────────────────────────────────────────
async function main() {
  const args = process.argv.slice(2);
  const get  = (flag, def) => {
    const i = args.indexOf(flag);
    return i >= 0 ? args[i + 1] : def;
  };

  const isBatch = args.includes('--batch');
  // Optional label for this worker (used when multiple processes run in parallel)
  const LABEL = get('--label', '') ? `[${get('--label', '')}] ` : '';

  // Launch browser
  // headless: true  — 'new' was deprecated in Puppeteer v22+ and breaks on some versions.
  // --disable-gpu is Linux-only; on macOS it causes blank screenshots, so omit it.
  const isMac = process.platform === 'darwin';
  const launchArgs = [
    '--no-sandbox',
    '--disable-setuid-sandbox',
    '--disable-dev-shm-usage',
    '--font-render-hinting=none',
  ];
  if (!isMac) launchArgs.push('--disable-gpu');

  const launchOpts = { headless: true, args: launchArgs };

  const log = msg => process.stderr.write(`${LABEL}${msg}\n`);

  log('Finding Chrome…');
  const chromium = await findChromiumAsync();
  if (chromium) {
    launchOpts.executablePath = chromium;
    log(`Chrome: ${chromium}`);
  } else {
    const platform = process.platform;
    const hint = platform === 'darwin'
      ? 'Install Chrome (https://www.google.com/chrome) OR run: npx puppeteer browsers install chrome'
      : platform === 'linux'
      ? 'Run: apt-get install -y chromium-browser  OR: npx puppeteer browsers install chrome'
      : 'Install Chrome from https://www.google.com/chrome';
    throw new Error(`No Chrome/Chromium found.\n${hint}`);
  }

  log('Launching browser…');
  const browser = await puppeteer.launch(launchOpts);
  log('Browser ready.');

  try {
    if (isBatch) {
      const jobsFile = get('--batch', null);
      if (!jobsFile || !fs.existsSync(jobsFile)) {
        console.error('ERROR: --batch <jobs.json> required');
        process.exit(1);
      }
      log('Reading jobs file…');
      const jobs = JSON.parse(fs.readFileSync(jobsFile, 'utf8'));
      log(`${jobs.length} jobs loaded.`);

      // List fonts (sizes from disk — no base64 pre-caching needed; Chrome loads
      // fonts via file:// URL directly and caches them across tabs in the session).
      const fontPaths = [...new Set(jobs.map(j => j.font))];
      log(`Fonts (${fontPaths.length}):`);
      for (const fp of fontPaths) {
        try {
          const kb = Math.round(fs.statSync(fp).size / 1024);
          log(`  ${path.basename(fp)}  (${kb} KB)`);
        } catch (_) {
          log(`  ${path.basename(fp)}  (not found!)`);
        }
      }

      // Warm-up: render one simple page to JIT-compile Chrome's shaping engine.
      // HTML is now tiny (~500 B — font loaded via file:// URL, no base64),
      // so setContent() is safe to use here.
      log('Warm-up render…');
      const warmPage = await browser.newPage();
      try {
        const warmFont = fontPaths[0];
        const warmHtml = buildHtml(warmFont, 'ಕನ್ನಡ ಸಂಚಯ', 32);
        await warmPage.setViewport({ width: 400, height: 100, deviceScaleFactor: 1 });
        await warmPage.setContent(warmHtml, { waitUntil: 'load' });
        await warmPage.evaluate(() => Promise.race([
          document.fonts.ready,
          new Promise(r => setTimeout(r, 5000)),
        ]));
      } finally {
        await warmPage.close();
      }
      log('Warm-up done. Starting batch…');

      const summary = await renderBatch(browser, jobs, get('--label', ''));
      process.stdout.write(JSON.stringify(summary) + '\n');

    } else {
      // Single render → base64 PNG to stdout
      const fontPath = get('--font', null);
      const text     = get('--text', 'ಕನ್ನಡ');
      const size     = parseInt(get('--size', '48'), 10);
      const features = get('--features', '');

      if (!fontPath) { console.error('ERROR: --font <path> required'); process.exit(1); }
      if (!fs.existsSync(fontPath)) { console.error('ERROR: font not found: ' + fontPath); process.exit(1); }

      const pngBuf = await renderSingle(browser, fontPath, text, size, features);
      process.stdout.write(pngBuf.toString('base64'));
    }
  } finally {
    await browser.close();
  }
}

main().catch(e => { console.error('ERROR:', e.message); process.exit(1); });
