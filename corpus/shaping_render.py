#!/usr/bin/env python3
"""
shaping_render.py
─────────────────
Shared text-rendering module that applies full OpenType shaping (GSUB/GPOS)
for Kannada and other complex scripts, replacing bare Pillow rendering.

Pillow's ImageDraw.text() renders each Unicode codepoint as an isolated glyph.
It never applies the font's OpenType GSUB tables, so conjuncts like ಕ್ಷ render
as three separate glyphs (ಕ + ್ + ಷ) instead of the correct ligature form.
The visual result is split/broken characters — exactly what you see in the
font comparison view.

This module uses uharfbuzz (HarfBuzz Python binding) for text shaping and
freetype-py for glyph rasterisation.  The output is a PIL Image that shows
correctly-formed conjuncts.

Font Unit Conversion (documented fix)
─────────────────────────────────────
HarfBuzz returns glyph positions in 1/64 pixel units (26.6 fixed-point format).
After setting hb_font.scale = (font_size * 64, font_size * 64), all positions
are in scaled coordinates. Converting to pixels requires dividing by 64 (>> 6).

The conversion formula: font_units * font_size / uPEM (where uPEM = 1000 or 2048)
is equivalent when HarfBuzz scale is set correctly. See render_text() for details.

Install once:
    pip install uharfbuzz freetype-py --break-system-packages

If either library is missing, SHAPING_AVAILABLE is False and callers should
fall back to Pillow with a warning (conjuncts will be broken).

Public API
──────────
    SHAPING_AVAILABLE : bool
        True when uharfbuzz + freetype-py + numpy are importable.

    render_text(font_path, text, font_size=36, padding_x=20, padding_y=12,
                min_height=60) -> PIL.Image.Image
        Render *text* from *font_path* with full OpenType shaping.
        Returns a greyscale ('L') PIL Image.

    render_char_with_label(char, label, font_path, font_size=48, label_size=12,
                           padding=16, dpi=150) -> PIL.Image.Image
        Render a single Kannada *char* (shaped) with an ASCII *label* below it
        (Pillow, no shaping needed for ASCII).
        Returns an RGB PIL Image (white background, dark ink).
"""

from pathlib import Path

# ── Availability check ─────────────────────────────────────────────────────

try:
    import uharfbuzz as hb
    import freetype
    import numpy as np
    from PIL import Image as _PilImage
    SHAPING_AVAILABLE = True
except ImportError:
    SHAPING_AVAILABLE = False

# ── OpenType features for Kannada complex script ──────────────────────────
# These are the GSUB/GPOS feature tags required for Indic scripts.
# Without them Pillow would render every codepoint separately.
_KANNADA_FEATURES = {
    # Standard ligatures & contextual alternates
    "liga": True, "calt": True, "clig": True,
    # Kerning & mark positioning
    "kern": True, "mark": True, "mkmk": True,
    # Indic-specific substitution stages (OpenType Indic spec)
    "nukt": True,   # Nukta forms
    "akhn": True,   # Akhand (pre-shaping ligatures, e.g. ಕ್ಷ, ಜ್ಞ)
    "rphf": True,   # Reph form
    "pref": True,   # Pre-base reordering
    "blwf": True,   # Below-base forms
    "half": True,   # Half forms
    "pstf": True,   # Post-base forms
    "vatu": True,   # Vattu (below-base consonant + reph)
    "cjct": True,   # Conjunct forms
    "pres": True,   # Pre-base substitutions
    "abvs": True,   # Above-base substitutions
    "blws": True,   # Below-base substitutions
    "psts": True,   # Post-base substitutions
    "haln": True,   # Halant (virama) forms
    "dist": True,   # Distances
    "abvm": True,   # Above-base mark positioning
    "blwm": True,   # Below-base mark positioning
}


# ── Core rendering ─────────────────────────────────────────────────────────

def render_text(
    font_path,
    text,
    font_size=36,
    padding_x=20,
    padding_y=12,
    min_height=60,
    bg_color=255,
    ink_color=0,
):
    """
    Render *text* from *font_path* with full OpenType shaping.

    Parameters
    ----------
    font_path : str or Path
        Path to a TTF or OTF font file.
    text : str
        Unicode text to render (may contain Kannada conjuncts).
    font_size : int
        Font size in pixels (not points).
    padding_x, padding_y : int
        Horizontal and vertical padding in pixels.
    min_height : int
        Minimum image height.
    bg_color : int (0–255)
        Background grey value (255 = white).
    ink_color : int (0–255)
        Foreground grey value (0 = black).

    Returns
    -------
    PIL.Image.Image
        Greyscale ('L') image.

    Raises
    ------
    ImportError
        If uharfbuzz, freetype-py, or numpy are not installed.
    """
    if not SHAPING_AVAILABLE:
        raise ImportError(
            "uharfbuzz, freetype-py and numpy are required for shaped rendering.\n"
            "Install with: pip install uharfbuzz freetype-py numpy --break-system-packages"
        )

    font_path = str(font_path)

    # ── HarfBuzz: shape the text ──────────────────────────────────
    # hb.Face() accepts raw bytes directly in uharfbuzz >= 0.30
    font_data = Path(font_path).read_bytes()
    hb_face   = hb.Face(font_data)
    hb_font   = hb.Font(hb_face)

    # HarfBuzz scale is in 1/64 pixel units (26.6 fixed-point)
    scale = font_size * 64
    hb_font.scale = (scale, scale)

    # Connect HarfBuzz to the font's own outline functions
    try:
        hb.ot_font_set_funcs(hb_font)
    except AttributeError:
        pass  # older uharfbuzz versions don't expose this

    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()  # sets script=Kannada, direction=LTR, language

    hb.shape(hb_font, buf, _KANNADA_FEATURES)

    glyph_infos     = buf.glyph_infos
    glyph_positions = buf.glyph_positions

    # ── FreeType: rasterise glyphs ────────────────────────────────
    ft_face = freetype.Face(font_path)
    ft_face.set_pixel_sizes(0, font_size)

    ascender  = ft_face.size.ascender  >> 6
    descender = abs(ft_face.size.descender >> 6)

    # Font unit conversion (HarfBuzz → pixels)
    # ──────────────────────────────────────────────────────────────
    # HarfBuzz returns glyph positions in 1/64 pixel units (26.6 fixed-point).
    # Since we set hb_font.scale = (font_size * 64, font_size * 64) above,
    # HarfBuzz positions are already in scaled coordinates. Converting to pixels
    # requires dividing by 64, which is done via bit-shift (>> 6).
    #
    # Alternative (explicit) conversion: font_units * font_size / uPEM
    # where uPEM is the font's units-per-em (typically 1000 or 2048).
    # Both methods are equivalent when scale is set correctly.

    # Total horizontal advance (sum of x_advance in 26.6, converted to pixels)
    total_adv = sum(pos.x_advance for pos in glyph_positions) >> 6

    W = max(total_adv + padding_x * 2, 1)
    H = max(ascender + descender + padding_y * 2, min_height)

    img = np.full((H, W), bg_color, dtype=np.uint8)

    x_pen      = padding_x
    y_baseline = padding_y + ascender

    for info, pos in zip(glyph_infos, glyph_positions):
        glyph_id = info.codepoint  # after shaping, .codepoint is the glyph ID

        # HarfBuzz glyph positioning: offset and advance in 1/64 pixel units
        # Convert to pixels via >> 6 (divide by 64).
        # These offsets handle diacritics, combining marks, and ligature positioning.
        x_off = pos.x_offset >> 6  # horizontal offset in pixels
        y_off = pos.y_offset >> 6  # vertical offset in pixels

        try:
            ft_face.load_glyph(glyph_id, freetype.FT_LOAD_RENDER)
        except freetype.FT_Exception:
            x_pen += pos.x_advance >> 6
            continue

        bm = ft_face.glyph.bitmap
        bw, bh = bm.width, bm.rows

        if bw > 0 and bh > 0:
            gx = x_pen + x_off + ft_face.glyph.bitmap_left
            gy = y_baseline - y_off - ft_face.glyph.bitmap_top

            # Source slice (into bitmap)
            sx1 = max(0, -gx);         sx2 = min(bw, W - gx)
            sy1 = max(0, -gy);         sy2 = min(bh, H - gy)
            # Destination slice (into img)
            dx1 = max(0, gx);          dx2 = min(W, gx + bw)
            dy1 = max(0, gy);          dy2 = min(H, gy + bh)

            if dx2 > dx1 and dy2 > dy1:
                pitch  = bm.pitch
                buf_bytes = bytes(bm.buffer)
                glyph_np  = np.frombuffer(buf_bytes, dtype=np.uint8)
                glyph_np  = glyph_np.reshape(bh, pitch)[:, :bw]

                alpha  = glyph_np[sy1:sy2, sx1:sx2].astype(np.float32) / 255.0
                region = img[dy1:dy2, dx1:dx2].astype(np.float32)
                blended = region * (1.0 - alpha) + float(ink_color) * alpha
                img[dy1:dy2, dx1:dx2] = blended.astype(np.uint8)

        x_pen += pos.x_advance >> 6

    return _PilImage.fromarray(img, mode='L')


# ── Convenience wrapper ────────────────────────────────────────────────────

def render_char_with_label(
    char,
    label,
    font_path,
    font_size=48,
    label_size=11,
    padding=16,
    dpi=150,
    bg=(255, 255, 255),
    ink=(20, 20, 60),
    label_ink=(120, 130, 150),
):
    """
    Render a single Kannada character (with full shaping) and an ASCII label
    underneath it.  Returns an RGB PIL Image.

    Parameters
    ----------
    char : str
        The Kannada character or cluster to render (e.g. 'ಕ್ಷ').
    label : str
        Short ASCII label shown below the character (e.g. 'U+0C95').
    font_path : str or Path
        TTF/OTF font file for the Kannada character.
    font_size : int
        Size of the main character in pixels.
    label_size : int
        Size of the label text in pixels.
    padding : int
        Padding around the content.
    dpi : int
        DPI stored in the PNG metadata.
    bg, ink, label_ink : tuple(int,int,int)
        RGB colours.
    """
    from PIL import Image, ImageDraw, ImageFont

    # ── Render main character (shaped) ───────────────────────────
    if SHAPING_AVAILABLE:
        char_grey = render_text(
            font_path, char,
            font_size=font_size,
            padding_x=padding,
            padding_y=padding // 2,
            min_height=font_size + padding,
            bg_color=255,
            ink_color=0,
        )
        cw, ch = char_grey.size
    else:
        # Pillow fallback (broken conjuncts, but better than crashing)
        try:
            pil_font = ImageFont.truetype(str(font_path), size=font_size)
        except Exception:
            pil_font = ImageFont.load_default()
        dummy = Image.new('L', (1, 1))
        bb = ImageDraw.Draw(dummy).textbbox((0, 0), char, font=pil_font)
        cw = bb[2] - bb[0] + padding * 2
        ch = max(bb[3] - bb[1] + padding, font_size + padding)
        char_grey = Image.new('L', (cw, ch), 255)
        ImageDraw.Draw(char_grey).text(
            (padding - bb[0], padding // 2 - bb[1]), char,
            font=pil_font, fill=0,
        )

    # ── Render ASCII label (Pillow is fine for ASCII) ────────────
    try:
        lbl_font = ImageFont.truetype(str(font_path), size=label_size)
    except Exception:
        lbl_font = ImageFont.load_default()

    dummy = Image.new('L', (1, 1))
    lbb   = ImageDraw.Draw(dummy).textbbox((0, 0), label, font=lbl_font)
    lw    = lbb[2] - lbb[0]
    lh    = lbb[3] - lbb[1]

    # ── Compose onto RGB canvas ───────────────────────────────────
    W = max(cw, lw + padding * 2)
    H = ch + lh + padding

    out = Image.new('RGB', (W, H), bg)

    # Paste the grey character image as RGB
    char_rgb = Image.merge('RGB', [char_grey] * 3)
    # Tint to ink colour: dark pixels → ink, white → bg
    char_tinted = _tint_grey(char_grey, bg, ink)
    cx_off = (W - cw) // 2
    out.paste(char_tinted, (cx_off, 0))

    # Draw label
    lx = (W - lw) // 2 - lbb[0]
    ly = ch + padding // 2 - lbb[1]
    ImageDraw.Draw(out).text((lx, ly), label, font=lbl_font, fill=label_ink)

    return out


def _tint_grey(grey_img, bg_rgb, ink_rgb):
    """Convert a greyscale image to RGB, mapping 255→bg and 0→ink."""
    import numpy as np
    from PIL import Image as _PI
    arr  = np.array(grey_img, dtype=np.float32) / 255.0  # 0=ink,1=bg
    r = (arr * bg_rgb[0] + (1 - arr) * ink_rgb[0]).astype(np.uint8)
    g = (arr * bg_rgb[1] + (1 - arr) * ink_rgb[1]).astype(np.uint8)
    b = (arr * bg_rgb[2] + (1 - arr) * ink_rgb[2]).astype(np.uint8)
    return _PI.fromarray(np.stack([r, g, b], axis=2), mode='RGB')


# ── Install helper ─────────────────────────────────────────────────────────

def check_and_warn():
    """Print a one-time warning if shaping libraries are not installed."""
    if not SHAPING_AVAILABLE:
        print(
            "\n  ⚠  WARNING: uharfbuzz and/or freetype-py are not installed.\n"
            "     Falling back to Pillow — Kannada conjuncts will render BROKEN.\n"
            "     Fix with:\n"
            "       pip install uharfbuzz freetype-py numpy --break-system-packages\n"
        )
