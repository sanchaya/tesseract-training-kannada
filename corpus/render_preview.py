#!/usr/bin/env python3
"""
render_preview.py
─────────────────
Live font preview helper.  Renders text with a given font using full
OpenType shaping (HarfBuzz + FreeType) and writes a base64-encoded PNG
to stdout.

Usage:
    echo "ಕನ್ನಡ" | python3 corpus/render_preview.py <font_path> [font_size]

Called by the portal's /api/render-preview endpoint.
"""
import sys, base64, io
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    from shaping_render import render_text, SHAPING_AVAILABLE
except ImportError as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)

if len(sys.argv) < 2:
    print("Usage: echo text | render_preview.py <font_path> [size]", file=sys.stderr)
    sys.exit(1)

font_path = sys.argv[1]
font_size = max(8, min(200, int(sys.argv[2]) if len(sys.argv) > 2 else 48))
aalt      = '--aalt' in sys.argv
text      = sys.stdin.read().strip()

if not text:
    sys.exit(0)

if not Path(font_path).exists():
    print(f"ERROR: font not found: {font_path}", file=sys.stderr)
    sys.exit(1)

try:
    img = render_text(
        font_path, text,
        font_size=font_size,
        padding_x=24, padding_y=16,
        min_height=font_size + 20,
        bg_color=255, ink_color=0,
        aalt=aalt,
    )
    buf = io.BytesIO()
    img.save(buf, format='PNG', dpi=(150, 150))
    sys.stdout.write(base64.b64encode(buf.getvalue()).decode())
    sys.stdout.flush()
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
