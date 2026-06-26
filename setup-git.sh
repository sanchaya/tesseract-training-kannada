#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════
#  TrainOCR — Git repository setup
#
#  Run once from the project root after cloning / first setup:
#    chmod +x setup-git.sh
#    ./setup-git.sh
#    # or override the remote:
#    ./setup-git.sh https://github.com/sanchaya/tesseract-training-kannada.git
#
#  This script:
#    1. Initialises git (if not already done)
#    2. Commits the Node.js app on master
#    3. Creates the python-portal branch (Flask app)
#    4. Creates the static branch (standalone HTML)
#    5. Pushes all three branches to GitHub
#
#  Safe to re-run — uses -B to force-recreate branches.
#  Prerequisites: git, GitHub private repo already created
# ══════════════════════════════════════════════════════════════════
set -euo pipefail

REMOTE_URL="${1:-}"

if [[ -z "$REMOTE_URL" ]]; then
  REMOTE_URL="https://github.com/sanchaya/tesseract-training-kannada.git"
  echo "Using default remote: $REMOTE_URL"
fi

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "╔══════════════════════════════════════════════╗"
echo "║   TrainOCR — Git setup                      ║"
echo "╚══════════════════════════════════════════════╝"

# ── Init if needed ────────────────────────────────────────────────
if [[ ! -d .git ]]; then
  git init
  echo "✓ git init"
fi

# ── Configure git ─────────────────────────────────────────────────
git config user.email "omshivaprakash@gmail.com"
git config user.name  "Om Shivaprakash"

# Ensure _private/ is never tracked
grep -q '_private' .gitignore 2>/dev/null || echo "_private/" >> .gitignore
echo "✓ .gitignore excludes _private/"

# Create empty placeholder dirs so git tracks them
for d in rendered lstmf output best scan-input logs fonts tessdata_best test-images; do
  mkdir -p "$d"
  touch "$d/.gitkeep"
done

# ════════════════════════════════════════════════════════════════
#  MASTER — Node.js / Express app  (portal.py also committed here
#           so it survives branch switches without being wiped)
# ════════════════════════════════════════════════════════════════
echo ""
echo "── Branch: master (Node.js) ──────────────────────────────"

# Force checkout master (safe on re-runs)
git checkout -B master 2>/dev/null || git checkout -f master

# Stage everything git knows about (respects .gitignore — excludes node_modules, _private, etc.)
git add -A

git commit -m "feat: TrainOCR Node.js portal — initial release

- Express server with 6-step Tesseract training pipeline
- In-browser OCR testing via Tesseract.js (kan + kan_hist)
- Real-time BCER training chart and checkpoint browser
- Live log streaming via SSE
- Scan upload + GT ingestion
- Interactive on-screen guide with 8-step tour
- Docker support (Dockerfile + docker-compose)
- PM2 + nginx deploy config for trainocr.sanchaya.net
- Sanchaya branding and purple design system

Fonts: Karnata GTN, German Mission Press, Wesleyan Mission Press, F Kittel
Target model: kan_hist — historical Kannada Tesseract model"

echo "✓ master committed"

# ════════════════════════════════════════════════════════════════
#  python-portal — Flask / Python app  (orphan: no shared history)
# ════════════════════════════════════════════════════════════════
echo ""
echo "── Branch: python-portal (Flask) ─────────────────────────"

git checkout --orphan python-portal
# Clear the index (files stay on disk)
git rm -rf --cached . 2>/dev/null || true

cat > requirements-portal.txt << 'PYREQ'
Flask>=3.0.0
Pillow>=10.0.0
PyYAML>=6.0
requests>=2.31.0
PYREQ

cat > .python-portal-note.md << 'NOTE'
# python-portal branch

This branch contains the **Flask / Python** version of the TrainOCR portal.

## Quick start

```bash
pip install -r requirements-portal.txt
python portal.py
# open http://localhost:5000
```

For the Node.js version see the `master` branch.
For a standalone static HTML version see the `static` branch.
NOTE

# Stage everything on disk, then strip Node.js-only files from the index
git add -A

git rm --cached \
  server.js \
  package.json package-lock.json \
  ecosystem.config.js \
  Dockerfile docker-compose.yml docker-compose.prod.yml .dockerignore \
  2>/dev/null || true

git commit -m "feat: python-portal — Flask/Python training portal

Alternative backend using Python/Flask + Pillow.
Includes corpus/, scripts/, public/ — same assets as master.

Quick start:
  pip install -r requirements-portal.txt
  python portal.py"

echo "✓ python-portal committed"

# ════════════════════════════════════════════════════════════════
#  static — standalone HTML only  (orphan: no shared history)
# ════════════════════════════════════════════════════════════════
echo ""
echo "── Branch: static (standalone HTML) ──────────────────────"

git checkout --orphan static
git rm -rf --cached . 2>/dev/null || true

cat > .static-branch-note.md << 'NOTE'
# static branch

This branch contains only the **standalone HTML frontend** — a single
`public/index.html` that can be served from any web server or even
opened directly in a browser.

Without a backend:
- The OCR test tab uses Tesseract.js entirely in-browser
- All pipeline step buttons will show "no server" errors (expected)
- The guide, glossary, and interactive tour work fully offline

## Serve locally

```bash
# Python
python3 -m http.server 8080 --directory public

# Node.js
npx serve public
```

For a full training pipeline use the `master` (Node.js) or
`python-portal` (Flask) branch.
NOTE

# Stage everything, then remove backend files from disk AND index
# so `git checkout master` can restore them cleanly without conflicts
git add -A

git rm -f \
  server.js portal.py \
  package.json package-lock.json ecosystem.config.js \
  requirements.txt requirements-portal.txt \
  Dockerfile docker-compose.yml docker-compose.prod.yml .dockerignore \
  start-portal.sh fonts.yml \
  .python-portal-note.md \
  2>/dev/null || true
git rm -rf corpus/ scripts/ deploy/ docs/ 2>/dev/null || true

git commit -m "feat: static — standalone HTML frontend

Single-page public/index.html with:
- In-browser OCR testing (Tesseract.js, no server required)
- Interactive 8-step tour and on-screen guide
- Glossary and historical document tips

No backend, no build step — open in any browser."

echo "✓ static committed"

# ════════════════════════════════════════════════════════════════
#  Leave repo on master
# ════════════════════════════════════════════════════════════════
git checkout master

# ════════════════════════════════════════════════════════════════
#  Push all branches to GitHub
# ════════════════════════════════════════════════════════════════
echo ""
echo "── Pushing to GitHub ─────────────────────────────────────"

git remote remove origin 2>/dev/null || true
git remote add origin "$REMOTE_URL"

git push -u origin master --force
git push -u origin python-portal --force
git push -u origin static --force

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  All done!                                   ║"
echo "╠══════════════════════════════════════════════╣"
echo "║  master        → Node.js / Express app       ║"
echo "║  python-portal → Flask / Python app          ║"
echo "║  static        → Standalone HTML frontend    ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "Repo: https://github.com/sanchaya/tesseract-training-kannada"
echo ""
echo "Default branch on GitHub → Settings → General → Default branch"
