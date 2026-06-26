#!/bin/bash
# start-portal.sh — Install deps and launch the kan_hist Training Portal
set -e

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  kan_hist Training Portal"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check Python
python3 --version &>/dev/null || { echo "ERROR: Python 3 not found."; exit 1; }

# Install dependencies quietly
echo "→ Checking Python dependencies..."
python3 -m pip install flask pyyaml pillow --quiet --break-system-packages 2>/dev/null \
  || python3 -m pip install flask pyyaml pillow --quiet

echo "→ Starting portal at http://localhost:5000"
echo "   Press Ctrl+C to stop."
echo ""

cd "$(dirname "$0")"
python3 portal.py
