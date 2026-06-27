#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════
#  push-changes.sh
#
#  Pushes the latest changes to all three GitHub branches.
#  Run whenever you've made local edits and want to sync to GitHub.
#
#  Usage:
#    chmod +x push-changes.sh
#    ./push-changes.sh "your commit message"
#    ./push-changes.sh              # uses a default message
# ══════════════════════════════════════════════════════════════════
set -euo pipefail

MSG="${1:-chore: sync portal, scripts, and checklist updates}"
REMOTE="https://github.com/sanchaya/tesseract-training-kannada.git"

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "╔══════════════════════════════════════════════╗"
echo "║   TrainOCR — push changes                   ║"
echo "╚══════════════════════════════════════════════╝"
echo "  Message: $MSG"
echo ""

# ── Ensure we're on master before touching other branches ─────────
CURRENT=$(git branch --show-current 2>/dev/null || echo "")
if [[ "$CURRENT" != "master" ]]; then
  echo "Switching to master..."
  git checkout master
fi

# ════════════════════════════════════════════════════════════════
#  MASTER — full Node.js codebase
# ════════════════════════════════════════════════════════════════
echo "── master ────────────────────────────────────────────────"
git add -A
if git diff --cached --quiet; then
  echo "  No changes to commit."
else
  git commit -m "$MSG"
  echo "  ✓ committed"
fi
git push origin master --force
echo "  ✓ pushed"

# ════════════════════════════════════════════════════════════════
#  PYTHON-PORTAL — Flask app (portal.py + shared assets)
#  Excludes Node.js-only files (server.js, package.json, Docker…)
# ════════════════════════════════════════════════════════════════
echo ""
echo "── python-portal ─────────────────────────────────────────"
git checkout python-portal

# Bring in all files from master, then remove Node.js-only ones
git checkout master -- \
  portal.py \
  requirements.txt \
  public/ \
  scripts/ \
  corpus/ \
  fonts.yml \
  start-portal.sh \
  .gitignore \
  2>/dev/null || true

# Remove Node.js-only files from index (they may not be on disk in this branch)
git rm --cached \
  server.js package.json package-lock.json ecosystem.config.js \
  Dockerfile docker-compose.yml docker-compose.prod.yml .dockerignore \
  2>/dev/null || true

if git diff --cached --quiet; then
  echo "  No changes to commit."
else
  git commit -m "$MSG"
  echo "  ✓ committed"
fi
git push origin python-portal --force
echo "  ✓ pushed"

# ════════════════════════════════════════════════════════════════
#  STATIC — standalone HTML only
# ════════════════════════════════════════════════════════════════
echo ""
echo "── static ────────────────────────────────────────────────"
git checkout static

# Bring in the frontend from master
git checkout master -- public/ .gitignore 2>/dev/null || true

# Ensure no backend files linger in index
git rm --cached \
  server.js portal.py \
  package.json package-lock.json ecosystem.config.js \
  requirements.txt requirements-portal.txt \
  Dockerfile docker-compose.yml docker-compose.prod.yml .dockerignore \
  start-portal.sh fonts.yml \
  2>/dev/null || true
git rm -rf --cached corpus/ scripts/ deploy/ docs/ 2>/dev/null || true

if git diff --cached --quiet; then
  echo "  No changes to commit."
else
  git commit -m "$MSG"
  echo "  ✓ committed"
fi
git push origin static --force
echo "  ✓ pushed"

# ── Back to master ────────────────────────────────────────────────
git checkout master

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  Done! All three branches pushed.           ║"
echo "╠══════════════════════════════════════════════╣"
echo "║  master        → Node.js app + scripts      ║"
echo "║  python-portal → Flask app + scripts        ║"
echo "║  static        → public/index.html only     ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "  https://github.com/sanchaya/tesseract-training-kannada"
echo ""
