#!/usr/bin/env python3
# ============================================================
# چهره‌یاب (Chehrahyab) — Smart Face Recognition System
# Professional GUI · Faculty of Computer Science — Ghorjistan University
#
# Design: "Aurora Neon" 3.0 — obsidian-indigo canvas, electric cyan →
#         violet → hot-pink gradients, glowing glass cards, gradient
#         buttons, vector icons, Face-ID style scanning overlays.
#         Everything is drawn in code (PIL / Tk canvas) — no asset files.
#
# Run:  python src/gui.py            (normal)
#       python src/gui.py --smoketest  (build all screens then exit)
# ============================================================

import argparse
import math
import sys
import time
import tkinter as tk
import tkinter.font as tkfont
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox

import cv2
import numpy as np

import customtkinter as ctk
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageTk

from db import Database, PROFILES_DIR
from face_engine import FaceEngine, COSINE_THRESHOLD, models_present, cosine_sim

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_IMAGE = PROJECT_ROOT / "data" / "images" / "test.jpg"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# ---------- AURORA NEON palette ----------
BG_PAGE   = "#060713"   # app background  (obsidian indigo)
SIDEBAR   = "#0a0b1f"   # sidebar background
SIDEBAR_2 = "#171a42"   # sidebar hover / active
CARD      = "#10122c"   # card background
CARD_2    = "#181b3f"   # elevated card / inputs
CARD_3    = "#20245a"   # highest elevation / hover
BORDER    = "#2a2f6b"   # card border
BORDER_HI = "#4a4fb0"   # highlighted border
TEXT      = "#f5f3ff"   # primary text
MUTED     = "#9aa0d6"   # secondary text
DIM       = "#646aa8"   # tertiary text
CYAN      = "#2de2ff"   # primary accent (electric cyan)
CYAN_HI   = "#8af1ff"
VIOLET    = "#8b5cff"   # secondary accent
VIOLET_HI = "#b093ff"
MAGENTA   = "#ff4fd8"   # highlight accent (hot pink)
PINK      = "#ff7ab8"
GOOD      = "#2ee6a6"   # mint
BAD       = "#ff5c7a"   # coral
WARN      = "#ffc145"   # amber
ON_CYAN   = "#03212b"   # dark text on bright fills
ON_GOOD   = "#04150e"
ON_BAD    = "#2b0a12"
# soft tinted chip backgrounds
CYAN_BG, VIOLET_BG, GOOD_BG = "#0c3550", "#2a2070", "#0b3a35"
BAD_BG, WARN_BG, PINK_BG = "#431627", "#40320c", "#47173f"

GRAD_MAIN = (CYAN, VIOLET)
GRAD_HOT  = (VIOLET, MAGENTA)
GRAD_FULL = (CYAN, VIOLET, MAGENTA)
GRAD_GOOD = (GOOD, CYAN)
GRAD_BAD  = (BAD, MAGENTA)
GRAD_GOLD = (WARN, PINK)

FONT = "Tahoma"
FONT_CANDIDATES = ("Vazirmatn", "Vazir", "IRANSansX", "IRANSans", "Segoe UI", "Tahoma")

ZONE_ORDER = ["center", "slight_left", "slight_right", "far_left", "far_right", "up", "down"]
ZONE_TARGETS = {"center": 4, "slight_left": 4, "slight_right": 4,
                "far_left": 3, "far_right": 3, "up": 3, "down": 3}
ZONE_LABELS = {
    "center": "●  مستقیم رو به دوربین",
    "slight_left": "←  کمی به چپ",
    "slight_right": "→  کمی به راست",
    "far_left":    "⟲  صورت کامل به چپ",
    "far_right":   "⟳  صورت کامل به راست",
    "up":          "↑  سر کمی بالا",
    "down":        "↓  سر کمی پایین",
}
MISSING_PROMPT = {
    "center": "لطفاً مستقیم رو به دوربین نگاه کنید",
    "slight_left": "سرتان را کمی به چپ بچرخانید ←",
    "slight_right": "سرتان را کمی به راست بچرخانید →",
    "far_left": "صورت خود را کامل‌تر به چپ بچرخانید",
    "far_right": "صورت خود را کامل‌تر به راست بچرخانید",
    "up": "سرتان را کمی بالا کنید ↑",
    "down": "سرتان را کمی پایین کنید ↓",
}
QUALITY_REASON_TEXT = {
    "too_small": "چهره خیلی کوچک است — نزدیک‌تر شوید",
    "too_dark": "نور کم است — محیط را روشن‌تر کنید",
    "too_bright": "نور خیلی زیاد است",
    "low_contrast": "کنتراست چهره کم است",
    "blurry": "تصویر تار است — ثابت بمانید",
}


def pick_font(root):
    """Choose the best installed UI font (Persian-capable)."""
    global FONT
    try:
        have = {f.lower(): f for f in tkfont.families(root)}
        for name in FONT_CANDIDATES:
            if name.lower() in have:
                FONT = have[name.lower()]
                break
    except Exception:
        pass
    return FONT


def load_font(size, bold=False):
    return ctk.CTkFont(family=FONT, size=size, weight="bold" if bold else "normal")


def ui_scale(widget):
    """Approximate DPI scale (plain Tk canvases are not auto-scaled by CTk)."""
    try:
        return max(1.0, min(3.0, widget.winfo_fpixels("1i") / 96.0))
    except Exception:
        return 1.0


def tk_font(widget, size, bold=False):
    """Pixel-sized plain Tk font for canvas text (scaled for HiDPI)."""
    return tkfont.Font(family=FONT, size=-int(round(size * ui_scale(widget))),
                       weight="bold" if bold else "normal")


# =====================================================
# PIL artwork — core helpers
# =====================================================

def hex2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def mix(c1, c2, t):
    """Blend two hex colours → hex."""
    a, b = hex2rgb(c1), hex2rgb(c2)
    return "#%02x%02x%02x" % tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def lighten(c, t=0.25):
    return mix(c, "#ffffff", t)


def darken(c, t=0.25):
    return mix(c, "#000000", t)


def gradient_image(w, h, stops, direction="h"):
    """Multi-stop gradient (RGB). direction: h, v or d (diagonal)."""
    w, h = max(1, int(w)), max(1, int(h))
    cols = np.array([hex2rgb(s) if isinstance(s, str) else s for s in stops], np.float32)
    pos = np.linspace(0, 1, len(cols))
    x = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    t = {"h": x + 0 * y, "v": y + 0 * x, "d": (x + y) / 2.0}[direction]
    out = np.stack([np.interp(t, pos, cols[:, i]) for i in range(3)], -1)
    return Image.fromarray(out.astype(np.uint8), "RGB")


def rounded_mask(size, radius, ss=3):
    """Anti-aliased rounded-rectangle alpha mask."""
    w, h = size
    m = Image.new("L", (w * ss, h * ss), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, w * ss - 1, h * ss - 1],
                                        radius=radius * ss, fill=255)
    return m.resize((w, h), Image.LANCZOS)


def aurora_image(w, h, base=BG_PAGE, blobs=(), dots=True, seed=7):
    """Dark base + blurred colour blobs (+ faint dot grid) — the 'aurora'."""
    img = Image.new("RGBA", (w, h), hex2rgb(base) + (255,))
    glow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    for fx, fy, fr, col, a in blobs:           # fractions of width / height
        r = fr * max(w, h)
        gd.ellipse([fx * w - r, fy * h - r * 0.75, fx * w + r, fy * h + r * 0.75],
                   fill=hex2rgb(col) + (a,))
    glow = glow.filter(ImageFilter.GaussianBlur(max(w, h) * 0.09))
    img.alpha_composite(glow)
    if dots:                                    # faint dot grid (blended, not replaced)
        layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        step = 22
        for gx in range(step // 2, w, step):
            for gy in range(step // 2, h, step):
                d.point((gx, gy), fill=(255, 255, 255, 20))
        img.alpha_composite(layer)
    return img.convert("RGB")


def compose_on(img_rgba, bg_hex):
    """Flatten an RGBA image onto a solid colour (Tk-safe, no alpha)."""
    base = Image.new("RGBA", img_rgba.size, hex2rgb(bg_hex) + (255,))
    base.alpha_composite(img_rgba)
    return base.convert("RGB")


def gradient_pill(w, h, stops, bg, radius=None, sheen=True, border=None, ss=3):
    """Rounded gradient plate flattened on `bg` — used by buttons & chips."""
    w, h = max(2, int(w)), max(2, int(h))
    r = radius if radius is not None else h // 2
    g = gradient_image(w * ss, h * ss, stops, "h").convert("RGBA")
    if sheen:                                    # glossy top highlight
        sh = np.zeros((h * ss, w * ss, 4), np.uint8)
        sh[..., :3] = 255
        ramp = np.clip(1 - np.linspace(0, 1, h * ss) * 2.0, 0, 1)
        sh[..., 3] = (ramp[:, None] * 62).astype(np.uint8)
        g.alpha_composite(Image.fromarray(sh, "RGBA"))
    mask = rounded_mask((w * ss, h * ss), r * ss, ss=1)
    g.putalpha(mask)
    out = Image.new("RGBA", (w * ss, h * ss), hex2rgb(bg) + (255,))
    out.alpha_composite(g)
    if border:
        ImageDraw.Draw(out).rounded_rectangle(
            [0, 0, w * ss - 1, h * ss - 1], radius=r * ss, outline=hex2rgb(border) + (255,),
            width=ss)
    return out.resize((w, h), Image.LANCZOS).convert("RGB")


def solid_pill(w, h, fill, bg, radius=None, border=None, ss=3):
    w, h = max(2, int(w)), max(2, int(h))
    r = radius if radius is not None else h // 2
    out = Image.new("RGBA", (w * ss, h * ss), hex2rgb(bg) + (255,))
    ImageDraw.Draw(out).rounded_rectangle(
        [0, 0, w * ss - 1, h * ss - 1], radius=r * ss, fill=hex2rgb(fill) + (255,),
        outline=(hex2rgb(border) + (255,)) if border else None, width=ss)
    return out.resize((w, h), Image.LANCZOS).convert("RGB")


_FONT_CACHE = {}


def pil_font(size, bold=False):
    """Best-effort TrueType font for Latin text drawn with PIL."""
    key = (size, bold)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    names = (["C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/tahomabd.ttf",
              "DejaVuSans-Bold.ttf", "arialbd.ttf"] if bold else
             ["C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/tahoma.ttf",
              "DejaVuSans.ttf", "arial.ttf"])
    font = None
    for n in names:
        try:
            font = ImageFont.truetype(n, size)
            break
        except OSError:
            continue
    _FONT_CACHE[key] = font or ImageFont.load_default()
    return _FONT_CACHE[key]


# =====================================================
# PIL artwork — vector icons, logo, avatars, banners
# =====================================================

def _icon_canvas(size, ss=4):
    S = size * ss
    return Image.new("RGBA", (S, S), (0, 0, 0, 0)), S


def _draw_icon(name, S, d, c, w):
    """Draw icon `name` on an S×S canvas. c = RGBA colour, w = stroke width."""
    p = lambda f: S * f
    if name == "dashboard":
        for x, y, ww, hh in ((.16, .16, .30, .38), (.54, .16, .30, .22),
                             (.16, .62, .30, .22), (.54, .46, .30, .38)):
            d.rounded_rectangle([p(x), p(y), p(x + ww), p(y + hh)], radius=p(.07),
                                outline=c, width=w)
    elif name == "scan":
        draw_face_id_glyph_on(d, S, c, w)
    elif name == "people":
        d.ellipse([p(.30), p(.14), p(.54), p(.38)], outline=c, width=w)
        d.arc([p(.14), p(.46), p(.70), p(1.02)], 180, 360, fill=c, width=w)
        d.ellipse([p(.58), p(.22), p(.78), p(.42)], outline=c, width=w)
        d.arc([p(.58), p(.50), p(.92), p(.88)], 200, 345, fill=c, width=w)
    elif name == "report":
        d.rounded_rectangle([p(.20), p(.12), p(.80), p(.88)], radius=p(.09), outline=c, width=w)
        for i, y in enumerate((.34, .50, .66)):
            d.line([(p(.34), p(y)), (p(.66 if i < 2 else .54), p(y))], fill=c, width=w)
    elif name == "settings":
        for y, k in ((.28, .62), (.50, .34), (.72, .70)):
            d.line([(p(.14), p(y)), (p(.86), p(y))], fill=c, width=w)
            r = p(.085)
            d.ellipse([p(k) - r, p(y) - r, p(k) + r, p(y) + r], fill=(10, 11, 31, 255),
                      outline=c, width=w)
    elif name == "check":
        d.ellipse([p(.12), p(.12), p(.88), p(.88)], outline=c, width=w)
        d.line([(p(.31), p(.52)), (p(.45), p(.66)), (p(.70), p(.36))], fill=c, width=w, joint="curve")
    elif name == "cpu":
        d.rounded_rectangle([p(.26), p(.26), p(.74), p(.74)], radius=p(.07), outline=c, width=w)
        d.rounded_rectangle([p(.40), p(.40), p(.60), p(.60)], radius=p(.03), fill=c)
        for t in (.38, .50, .62):
            d.line([(p(t), p(.12)), (p(t), p(.26))], fill=c, width=w)
            d.line([(p(t), p(.74)), (p(t), p(.88))], fill=c, width=w)
            d.line([(p(.12), p(t)), (p(.26), p(t))], fill=c, width=w)
            d.line([(p(.74), p(t)), (p(.88), p(t))], fill=c, width=w)
    elif name == "play":
        d.polygon([(p(.30), p(.18)), (p(.82), p(.50)), (p(.30), p(.82))], fill=c)
    elif name == "stop":
        d.rounded_rectangle([p(.22), p(.22), p(.78), p(.78)], radius=p(.12), fill=c)
    elif name == "search":
        d.ellipse([p(.14), p(.14), p(.62), p(.62)], outline=c, width=w)
        d.line([(p(.55), p(.55)), (p(.86), p(.86))], fill=c, width=w)
    elif name == "trash":
        d.line([(p(.18), p(.27)), (p(.82), p(.27))], fill=c, width=w)
        d.line([(p(.38), p(.27)), (p(.38), p(.15)), (p(.62), p(.15)), (p(.62), p(.27))], fill=c, width=w)
        d.polygon([(p(.26), p(.27)), (p(.74), p(.27)), (p(.68), p(.88)), (p(.32), p(.88))], outline=c, width=w)
    elif name == "download":
        d.line([(p(.5), p(.12)), (p(.5), p(.62))], fill=c, width=w)
        d.line([(p(.30), p(.44)), (p(.5), p(.64)), (p(.70), p(.44))], fill=c, width=w, joint="curve")
        d.line([(p(.16), p(.74)), (p(.16), p(.88)), (p(.84), p(.88)), (p(.84), p(.74))], fill=c, width=w)
    elif name == "plus":
        d.line([(p(.5), p(.18)), (p(.5), p(.82))], fill=c, width=w)
        d.line([(p(.18), p(.5)), (p(.82), p(.5))], fill=c, width=w)
    elif name == "user_plus":
        d.ellipse([p(.22), p(.14), p(.52), p(.44)], outline=c, width=w)
        d.arc([p(.08), p(.52), p(.66), p(1.08)], 180, 360, fill=c, width=w)
        d.line([(p(.80), p(.34)), (p(.80), p(.64))], fill=c, width=w)
        d.line([(p(.65), p(.49)), (p(.95), p(.49))], fill=c, width=w)
    elif name == "pulse":
        d.line([(p(.08), p(.54)), (p(.30), p(.54)), (p(.40), p(.22)), (p(.54), p(.82)),
                (p(.64), p(.46)), (p(.72), p(.54)), (p(.92), p(.54))], fill=c, width=w, joint="curve")
    elif name == "db":
        for y in (.20, .50, .80):
            d.arc([p(.18), p(y - .12), p(.82), p(y + .12)], 0, 180, fill=c, width=w)
        d.ellipse([p(.18), p(.10), p(.82), p(.34)], outline=c, width=w)
        d.line([(p(.18), p(.22)), (p(.18), p(.78))], fill=c, width=w)
        d.line([(p(.82), p(.22)), (p(.82), p(.78))], fill=c, width=w)
    elif name == "bolt":
        d.polygon([(p(.56), p(.08)), (p(.22), p(.54)), (p(.46), p(.54)),
                   (p(.40), p(.92)), (p(.78), p(.42)), (p(.54), p(.42))], fill=c)
    elif name == "info":
        d.ellipse([p(.12), p(.12), p(.88), p(.88)], outline=c, width=w)
        d.line([(p(.5), p(.44)), (p(.5), p(.70))], fill=c, width=w)
        d.ellipse([p(.46), p(.27), p(.54), p(.35)], fill=c)
    elif name == "camera":
        d.rounded_rectangle([p(.10), p(.28), p(.90), p(.80)], radius=p(.10), outline=c, width=w)
        d.ellipse([p(.34), p(.38), p(.66), p(.70)], outline=c, width=w)
        d.line([(p(.30), p(.28)), (p(.38), p(.18)), (p(.62), p(.18)), (p(.70), p(.28))], fill=c, width=w)
    elif name == "inbox":
        d.polygon([(p(.14), p(.60)), (p(.28), p(.20)), (p(.72), p(.20)), (p(.86), p(.60)),
                   (p(.86), p(.84)), (p(.14), p(.84))], outline=c, width=w)
        d.line([(p(.14), p(.60)), (p(.36), p(.60)), (p(.42), p(.70)), (p(.58), p(.70)),
                (p(.64), p(.60)), (p(.86), p(.60))], fill=c, width=w)


_ICON_CACHE = {}


def icon_pil(name, size=22, color=TEXT):
    """Crisp supersampled vector icon (RGBA)."""
    key = (name, size, color)
    if key not in _ICON_CACHE:
        img, S = _icon_canvas(size)
        _draw_icon(name, S, ImageDraw.Draw(img), hex2rgb(color) + (255,), max(3, int(S * 0.075)))
        _ICON_CACHE[key] = img.resize((size, size), Image.LANCZOS)
    return _ICON_CACHE[key]


def icon_ctk(name, size=22, color=TEXT):
    img = icon_pil(name, size, color)
    return ctk.CTkImage(light_image=img, dark_image=img, size=(size, size))


def draw_face_id_glyph_on(d, S, color, w):
    """Face-ID style glyph (corner brackets + face) on an existing draw ctx."""
    m, arm = S * 0.14, S * 0.20

    def bracket(x, y, dx, dy):
        d.line([(x, y + dy * arm), (x, y), (x + dx * arm, y)], fill=color, width=w, joint="curve")

    bracket(m, m, +1, +1)
    bracket(S - m, m, -1, +1)
    bracket(m, S - m, +1, -1)
    bracket(S - m, S - m, -1, -1)
    c = S / 2
    for ex in (c - S * 0.105, c + S * 0.105):
        d.line([(ex, c - S * 0.20), (ex, c - S * 0.10)], fill=color, width=w)
    d.line([(c, c - S * 0.07), (c, c + S * 0.07)], fill=color, width=w)
    d.arc([c - S * 0.17, c - S * 0.02, c + S * 0.17, c + S * 0.24], 25, 155, fill=color, width=w)


def draw_face_id_glyph(size, color=(255, 255, 255, 255), width=None):
    """Original face-ID-style glyph on transparency."""
    ss = 3
    img = Image.new("RGBA", (size * ss, size * ss), (0, 0, 0, 0))
    draw_face_id_glyph_on(ImageDraw.Draw(img), size * ss, color, (width or max(3, size // 15)) * ss)
    return img.resize((size, size), Image.LANCZOS)


def make_app_logo(size=96, bg=None):
    """Gradient tile + glyph; with `bg` it gets a soft neon glow, flattened on bg."""
    ss = 4
    S = size * ss
    tile = gradient_image(S, S, GRAD_FULL, "d").convert("RGBA")
    tile.putalpha(rounded_mask((S, S), int(S * 0.27), ss=1))
    glyph = draw_face_id_glyph(int(S * 0.74), (255, 255, 255, 255), int(S * 0.028))
    tile.alpha_composite(glyph, ((S - glyph.width) // 2, (S - glyph.height) // 2))
    tile = tile.resize((size, size), Image.LANCZOS)
    if bg is None:
        return tile
    pad = int(size * 0.28)
    canvas = Image.new("RGBA", (size + 2 * pad, size + 2 * pad), (0, 0, 0, 0))
    halo = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ImageDraw.Draw(halo).rounded_rectangle([pad, pad + 6, pad + size, pad + size + 6],
                                           radius=int(size * .27), fill=hex2rgb(VIOLET) + (150,))
    canvas.alpha_composite(halo.filter(ImageFilter.GaussianBlur(pad * 0.55)))
    canvas.alpha_composite(tile, (pad, pad))
    return compose_on(canvas, bg)


def make_glow_glyph(size=170, alpha=90, bg=CARD):
    """Soft neon face-ID glyph flattened on `bg`."""
    pad = size // 3
    S = size + 2 * pad
    base = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    glow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse([pad * .6, pad * .6, S - pad * .6, S - pad * .6],
                                 fill=hex2rgb(VIOLET) + (int(alpha * .55),))
    base.alpha_composite(glow.filter(ImageFilter.GaussianBlur(S * 0.16)))
    g = draw_face_id_glyph(size, hex2rgb(CYAN) + (min(255, alpha + 120),), max(4, size // 22))
    base.alpha_composite(g, (pad, pad))
    return compose_on(base, bg)


def _initials(name):
    parts = [p for p in name.split() if p]
    return (parts[0][0] + (parts[-1][0] if len(parts) > 1 else "")).upper() if parts else "؟"


def _round_photo(img, size, bg, ring=True, ss=3):
    """Circular crop of `img` with a neon gradient ring, flattened on bg."""
    S = size * ss
    out = Image.new("RGBA", (S, S), hex2rgb(bg) + (255,))
    if ring:
        g = gradient_image(S, S, GRAD_FULL, "d").convert("RGBA")
        m = Image.new("L", (S, S), 0)
        ImageDraw.Draw(m).ellipse([0, 0, S - 1, S - 1], fill=255)
        g.putalpha(m)
        out.alpha_composite(g)
    inset = int(S * 0.055) if ring else 0
    inner = S - 2 * inset
    face = img.convert("RGBA").resize((inner, inner), Image.LANCZOS)
    m2 = Image.new("L", (inner, inner), 0)
    ImageDraw.Draw(m2).ellipse([0, 0, inner - 1, inner - 1], fill=255)
    face.putalpha(m2)
    out.alpha_composite(face, (inset, inset))
    return out.resize((size, size), Image.LANCZOS).convert("RGB")


def make_avatar(name, size=96, ring=True, bg=CARD_2):
    """Gradient-ring initials avatar."""
    ss = 3
    S = size * ss
    plate = Image.new("RGBA", (S, S), hex2rgb("#1d2058") + (255,))
    tint = gradient_image(S, S, ("#262a78", "#3a1f6e"), "d").convert("RGBA")
    plate.alpha_composite(tint)
    d = ImageDraw.Draw(plate)
    text = _initials(name)
    font = pil_font(int(S * 0.38), True)
    bb = d.textbbox((0, 0), text, font=font)
    d.text(((S - (bb[2] - bb[0])) / 2 - bb[0], (S - (bb[3] - bb[1])) / 2 - bb[1]), text,
           fill=(245, 243, 255, 255), font=font)
    return _round_photo(plate.resize((size * 2, size * 2), Image.LANCZOS), size, bg, ring)


def photo_or_avatar(person, size=96, bg=CARD_2):
    photo = person.get("photo_path")
    if photo and Path(photo).exists():
        try:
            im = Image.open(photo).convert("RGB")
            side = min(im.size)
            l, t = (im.width - side) // 2, (im.height - side) // 2
            return _round_photo(im.crop((l, t, l + side, t + side)), size, bg, True)
        except OSError:
            pass
    return make_avatar(person.get("full_name", "؟"), size, True, bg)


def make_hero_art(width=940, height=124, bg=BG_PAGE):
    """Wide aurora banner (art only — the Persian text is real Tk text on top)."""
    art = aurora_image(width, height, "#0c0e2e", [
        (0.08, 0.70, 0.26, CYAN, 95), (0.50, 0.00, 0.28, VIOLET, 120),
        (0.95, 0.90, 0.26, MAGENTA, 100)]).convert("RGBA")
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))      # everything blended
    d = ImageDraw.Draw(layer)
    cx, cy = int(height * 0.95), height // 2        # orbit rings behind the glyph
    for i, r in enumerate((int(height * .46), int(height * .70), int(height * 1.0))):
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 255, 255, 34 - i * 9), width=1)
    rng = np.random.default_rng(11)               # sparkles
    for _ in range(28):
        x, y, a = rng.integers(0, width), rng.integers(0, height), int(rng.integers(40, 150))
        d.ellipse([x, y, x + 2, y + 2], fill=(255, 255, 255, a))
    art.alpha_composite(layer)
    glyph = draw_face_id_glyph(int(height * .62), hex2rgb(CYAN_HI) + (235,), max(3, height // 24))
    art.alpha_composite(glyph, (cx - glyph.width // 2, cy - glyph.height // 2))
    art.putalpha(rounded_mask((width, height), 26))
    return compose_on(art, bg)


def make_idle_art(width, height, bg=CARD):
    """Calm 'camera ready' scene: soft rings, glow, tiny grid."""
    art = aurora_image(width, height, "#0d0f2b", [
        (0.50, 0.50, 0.34, VIOLET, 105), (0.15, 0.95, 0.22, CYAN, 70),
        (0.90, 0.05, 0.20, MAGENTA, 55)]).convert("RGBA")
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    cx, cy = width // 2, int(height * 0.42)
    for i, r in enumerate((90, 140, 200, 270)):
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 255, 255, 36 - i * 8), width=1)
    art.alpha_composite(layer)
    g = draw_face_id_glyph(150, hex2rgb(CYAN) + (230,), 7)
    art.alpha_composite(g, (cx - 75, cy - 75))
    art.putalpha(rounded_mask((width, height), 18))
    return compose_on(art, bg)


def make_ctk_image(pil_img, size=None):
    if size:
        pil_img = pil_img.resize(size, Image.LANCZOS)
    return ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=pil_img.size)


# =====================================================
# CUSTOM CANVAS WIDGETS (gradient buttons, bars, gauges)
# =====================================================

class GradientButton(tk.Canvas):
    """Pill button with a glossy gradient, hover glow and optional vector icon.

    Accepts a CTkButton-like API: configure(text=, text_color=, fg_color=,
    state=, command=) so it can replace CTkButton one-to-one.
    """

    def __init__(self, master, text="", command=None, colors=GRAD_MAIN,
                 text_color=ON_CYAN, width=160, height=46, radius=None,
                 bg=CARD, size=14, bold=True, icon=None, border=None):
        sc = ui_scale(master)
        super().__init__(master, width=int(width * sc), height=int(height * sc), bg=bg,
                         highlightthickness=0, bd=0, cursor="hand2")
        self._text, self._cmd, self._tc = text, command, text_color
        self._stops = self._norm(colors)
        self._bg, self._r, self._icon, self._border = bg, radius, icon, border
        self._state, self._hover, self._down = "normal", False, False
        self._font = tk_font(self, size, bold)
        self._size = size
        self._cache = {}
        self._img_id = self.create_image(0, 0, anchor="nw")
        self._ico_id = self.create_image(0, 0, anchor="center")
        self._txt_id = self.create_text(0, 0, text=text, font=self._font, fill=text_color,
                                        anchor="center")
        self.bind("<Configure>", lambda e: self._render())
        self.bind("<Enter>", lambda e: self._set_hover(True))
        self.bind("<Leave>", lambda e: self._set_hover(False))
        self.bind("<ButtonPress-1>", self._on_down)
        self.bind("<ButtonRelease-1>", self._on_up)

    @staticmethod
    def _norm(c):
        return (c, c) if isinstance(c, str) else tuple(c)

    # ---- CTkButton-compatible configure ----
    def configure(self, cnf=None, **kw):
        redraw = False
        for key in ("text", "text_color", "fg_color", "colors", "state", "command",
                    "icon", "hover_color", "border_color"):
            if key not in kw:
                continue
            val = kw.pop(key)
            if key == "text":
                self._text = val; redraw = True
            elif key == "text_color":
                self._tc = val; redraw = True
            elif key in ("fg_color", "colors"):
                self._stops = self._norm(val); redraw = True
            elif key == "state":
                self._state = val; redraw = True
                super().configure(cursor="hand2" if val == "normal" else "arrow")
            elif key == "command":
                self._cmd = val
            elif key == "icon":
                self._icon = val; redraw = True
            elif key == "border_color":
                self._border = val; redraw = True
        if cnf or kw:
            super().configure(cnf or {}, **kw)
        if redraw:
            self._render()

    config = configure

    def _set_hover(self, on):
        self._hover = on and self._state == "normal"
        self._render()

    def _on_down(self, _e):
        if self._state == "normal":
            self._down = True
            self._render()

    def _on_up(self, e):
        was_down, self._down = self._down, False
        self._render()
        inside = 0 <= e.x <= self.winfo_width() and 0 <= e.y <= self.winfo_height()
        if was_down and inside and self._cmd:
            self._cmd()

    def invoke(self):
        if self._state == "normal" and self._cmd:
            self._cmd()

    def _plate(self, w, h):
        disabled = self._state != "normal"
        stops = self._stops
        if disabled:
            key = (w, h, "off")
        else:
            amt = -0.18 if self._down else (0.16 if self._hover else 0.0)
            stops = tuple(lighten(s, amt) if amt > 0 else darken(s, -amt) for s in stops)
            key = (w, h, stops, self._border)
        if key not in self._cache:
            if disabled:
                self._cache[key] = solid_pill(w, h, CARD_3, self._bg, self._r, border=BORDER)
            else:
                self._cache[key] = ImageTk.PhotoImage(
                    gradient_pill(w, h, stops, self._bg, self._r, sheen=True, border=self._border))
            if disabled:
                self._cache[key] = ImageTk.PhotoImage(self._cache[key])
        return self._cache[key]

    def _render(self):
        w, h = self.winfo_width(), self.winfo_height()
        if w < 8 or h < 8:
            return
        self.itemconfigure(self._img_id, image=self._plate(w, h))
        color = MUTED if self._state != "normal" else self._tc
        self.itemconfigure(self._txt_id, text=self._text, fill=color, font=self._font)
        tw = self._font.measure(self._text)
        isz = int(self._size * 1.25 * ui_scale(self))
        if self._icon:
            self._ico = ImageTk.PhotoImage(icon_pil(self._icon, isz, color))
            gap = int(8 * ui_scale(self))
            total = tw + gap + isz
            left = (w - total) / 2
            self.coords(self._txt_id, left + tw / 2, h / 2 + 1)      # RTL: text left,
            self.coords(self._ico_id, left + tw + gap + isz / 2, h / 2)  # icon right
            self.itemconfigure(self._ico_id, image=self._ico, state="normal")
        else:
            self.coords(self._txt_id, w / 2, h / 2 + 1)
            self.itemconfigure(self._ico_id, state="hidden")


class AccentLine(tk.Canvas):
    """Thin glowing gradient divider that fades out at both ends."""

    def __init__(self, master, bg=CARD, stops=GRAD_FULL, height=3, fade=True):
        height = max(2, int(height * ui_scale(master)))
        super().__init__(master, width=2, height=height, bg=bg, highlightthickness=0, bd=0)
        self._bg, self._stops, self._fade, self._h = bg, stops, fade, height
        self._id = self.create_image(0, 0, anchor="nw")
        self._cw = 0
        self.bind("<Configure>", self._render)

    def _render(self, e=None):
        w = self.winfo_width()
        if w < 4 or w == self._cw:
            return
        self._cw = w
        g = gradient_image(w, self._h, self._stops, "h").convert("RGBA")
        if self._fade:
            a = np.clip(np.minimum(np.linspace(0, 1, w), np.linspace(1, 0, w)) * 6, 0, 1)
            g.putalpha(Image.fromarray((np.tile(a, (self._h, 1)) * 255).astype(np.uint8)))
        self._ph = ImageTk.PhotoImage(compose_on(g, self._bg))
        self.itemconfigure(self._id, image=self._ph)


class GradientBar(tk.Canvas):
    """Rounded progress bar with gradient fill (API: .set(0..1))."""

    def __init__(self, master, bg=CARD, stops=GRAD_MAIN, height=12, track=CARD_3):
        height = int(height * ui_scale(master))
        super().__init__(master, width=2, height=height, bg=bg, highlightthickness=0, bd=0)
        self._bg, self._stops, self._track, self._h = bg, stops, track, height
        self._v, self._cache = 0.0, {}
        self._id = self.create_image(0, 0, anchor="nw")
        self.bind("<Configure>", lambda e: self._render())

    def set(self, v, stops=None):
        self._v = max(0.0, min(1.0, float(v)))
        if stops:
            self._stops = stops
        self._render()

    def _render(self):
        w, h = self.winfo_width(), self._h
        if w < 8:
            return
        q = round(self._v * 150) / 150
        key = (w, q, self._stops)
        if key not in self._cache:
            if len(self._cache) > 400:
                self._cache.clear()
            base = solid_pill(w, h, self._track, self._bg).convert("RGBA")
            if q > 0:
                fw = max(h, int(w * q))
                base.alpha_composite(gradient_pill(fw, h, self._stops, self._track,
                                                   sheen=True).convert("RGBA"))
            self._cache[key] = ImageTk.PhotoImage(compose_on(base, self._bg))
        self.itemconfigure(self._id, image=self._cache[key])


class RingGauge(tk.Canvas):
    """Donut gauge with gradient arc and centred big value (API: .set)."""

    def __init__(self, master, size=150, bg=CARD, stops=GRAD_MAIN):
        sz = int(size * ui_scale(master))
        super().__init__(master, width=sz, height=sz, bg=bg, highlightthickness=0, bd=0)
        self._sz, self._bg, self._stops, self._cache = sz, bg, stops, {}
        self._img = self.create_image(sz // 2, sz // 2)
        self._big = self.create_text(sz // 2, sz // 2 - 6, text="—", fill=TEXT,
                                     font=tk_font(self, 30, True))
        self._small = self.create_text(sz // 2, sz // 2 + 24, text="", fill=MUTED,
                                       font=tk_font(self, 11))
        self.set(0)

    def set(self, v, stops=None, big=None, small=None):
        v = max(0.0, min(1.0, float(v)))
        stops = stops or self._stops
        key = (round(v * 100), stops)
        if key not in self._cache:
            S, th = self._sz * 3, int(self._sz * 0.085) * 3
            track = Image.new("L", (S, S), 0)
            ImageDraw.Draw(track).ellipse([th / 2, th / 2, S - th / 2, S - th / 2],
                                          outline=255, width=th)
            base = Image.new("RGBA", (S, S), hex2rgb(self._bg) + (255,))
            base.paste(hex2rgb(CARD_3), mask=track)
            if v > 0:
                arc = Image.new("L", (S, S), 0)
                ImageDraw.Draw(arc).arc([th / 2, th / 2, S - th / 2, S - th / 2], -90,
                                        -90 + 360 * v, fill=255, width=th)
                base.paste(gradient_image(S, S, stops, "d"), mask=arc)
            self._cache[key] = ImageTk.PhotoImage(base.resize((self._sz, self._sz), Image.LANCZOS)
                                                  .convert("RGB"))
        self.itemconfigure(self._img, image=self._cache[key])
        if big is not None:
            self.itemconfigure(self._big, text=big)
        if small is not None:
            self.itemconfigure(self._small, text=small)


def make_chip(icon, stops, size=40, bg=CARD, radius=12):
    """Gradient rounded-square icon chip (RGB, flattened on bg)."""
    ss = 3
    plate = gradient_pill(size * ss, size * ss, stops, bg, radius * ss, sheen=True).convert("RGBA")
    ic = icon_pil(icon, int(size * ss * 0.56), "#ffffff")
    plate.alpha_composite(ic, ((size * ss - ic.width) // 2, (size * ss - ic.height) // 2))
    # returned at 2× so it stays crisp on HiDPI — always wrap with an explicit CTkImage size
    return plate.resize((size * 2, size * 2), Image.LANCZOS).convert("RGB")


class HeroBanner(tk.Canvas):
    """Responsive aurora banner: pre-rendered art + real Tk text on top."""

    def __init__(self, master, title, subtitle, height=112, bg=BG_PAGE):
        height = int(height * ui_scale(master))
        super().__init__(master, width=2, height=height, bg=bg, highlightthickness=0, bd=0)
        self._h, self._bg, self._cw, self._cache = height, bg, 0, {}
        self._img = self.create_image(0, 0, anchor="nw")
        self._bar = self.create_image(0, 0, anchor="ne")
        self._t = self.create_text(0, 0, text=title, anchor="e", fill=TEXT,
                                   font=tk_font(self, 25, True))
        self._s = self.create_text(0, 0, text=subtitle, anchor="e", fill=CYAN_HI,
                                   font=tk_font(self, 12))
        self._timer = None
        self.bind("<Configure>", self._on_conf)

    def _on_conf(self, e):
        if self._timer:
            self.after_cancel(self._timer)
        self._timer = self.after(40, self._render)

    def _render(self):
        w, h = self.winfo_width(), self._h
        if w < 100 or w == self._cw:
            return
        self._cw = w
        if w not in self._cache:
            self._cache.clear()
            art = ImageTk.PhotoImage(make_hero_art(w, h, self._bg))
            bar = ImageTk.PhotoImage(gradient_pill(int(70 * ui_scale(self)), int(5 * ui_scale(self)), GRAD_FULL, "#0e1030"))
            self._cache[w] = (art, bar)
        art, bar = self._cache[w]
        self.itemconfigure(self._img, image=art)
        self.itemconfigure(self._bar, image=bar)
        s = ui_scale(self)
        pad = int(34 * s)
        self.coords(self._t, w - pad, h * 0.34)
        self.coords(self._bar, w - pad, h * 0.54)
        self.coords(self._s, w - pad, h * 0.74)


class CameraView(tk.Canvas):
    """Live-video canvas: rounded video, HUD chips, corner-bracket face boxes
    and the Face-ID style enrolment overlay. Persian labels are real Tk text."""

    def __init__(self, master, idle_title="", idle_sub="", bg=CARD, radius=18):
        super().__init__(master, width=2, height=2, bg=bg, highlightthickness=0, bd=0)
        self._bg, self._radius = bg, radius
        self._idle_title, self._idle_sub = idle_title, idle_sub
        self._mode, self._cache, self._timer = "idle", {}, None
        self._img = self.create_image(0, 0, anchor="center")
        self._f_tag = tk_font(self, 12, True)
        self._f_hud = tk_font(self, 11, True)
        self._f_t, self._f_s = tk_font(self, 17, True), tk_font(self, 12)
        self.bind("<Configure>", self._on_conf)

    # ---------- idle scene ----------
    def _on_conf(self, e):
        if self._mode == "idle":
            if self._timer:
                self.after_cancel(self._timer)
            self._timer = self.after(40, self.show_idle)

    def show_idle(self, title=None, sub=None):
        if title is not None:
            self._idle_title = title
        if sub is not None:
            self._idle_sub = sub
        self._mode = "idle"
        w, h = self.winfo_width(), self.winfo_height()
        if w < 60 or h < 60:
            return
        key = ("idle", w, h)
        if key not in self._cache:
            for k in [k for k in self._cache if k[0] == "idle"]:
                del self._cache[k]
            self._cache[key] = ImageTk.PhotoImage(make_idle_art(w, h, self._bg))
        self._photo = self._cache[key]
        self.coords(self._img, w / 2, h / 2)
        self.itemconfigure(self._img, image=self._photo)
        self.delete("ov")
        cy = h * 0.42
        self.create_text(w / 2, cy + 112, text=self._idle_title, fill=TEXT, font=self._f_t, tags="ov")
        self.create_text(w / 2, cy + 142, text=self._idle_sub, fill=MUTED, font=self._f_s, tags="ov")

    # ---------- helpers ----------
    def _mask(self, dw, dh):
        key = ("m", dw, dh)
        if key not in self._cache:
            self._cache[key] = rounded_mask((dw, dh), self._radius)
        return self._cache[key]

    def _vignette(self, dw, dh, cx, cy, rx, ry):
        key = ("v", dw, dh)
        if key not in self._cache:
            yy, xx = np.mgrid[0:dh, 0:dw].astype(np.float32)
            dist = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
            f = 0.38 + 0.62 * np.clip((1.10 - dist) / 0.16, 0, 1)
            self._cache[key] = f[..., None].astype(np.float32)
        return self._cache[key]

    @staticmethod
    def _brackets(d, x0, y0, x1, y1, color, k, arm, th):
        for (cx, cy, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
            pts = [(cx, cy + dy * arm), (cx, cy), (cx + dx * arm, cy)]
            d.line(pts, fill=color[:3] + (70,), width=th * 3, joint="curve")   # glow
            d.line(pts, fill=color, width=th, joint="curve")

    # ---------- live frame ----------
    def show_frame(self, frame, boxes=(), hud=(), scan=None):
        """boxes: (x, y, w, h, label, hex_color) in frame coords.
        hud: (text, dot_hex) chips top-left.  scan: dict(progress, angle)."""
        w, h = self.winfo_width(), self.winfo_height()
        if w < 60 or h < 60:
            return
        self._mode = "live"
        fh, fw = frame.shape[:2]
        s = min(w / fw, h / fh)
        dw, dh = max(8, int(fw * s)), max(8, int(fh * s))
        rgb = cv2.cvtColor(cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA
                                      if s < 1 else cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        K = 2
        ov = Image.new("RGBA", (dw * K, dh * K), (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        if scan is not None:
            cx, cy, rx, ry = dw / 2, dh / 2, dw * 0.24, dh * 0.40
            rgb = (rgb * self._vignette(dw, dh, cx, cy, rx, ry)).astype(np.uint8)
            self._draw_scan(d, cx * K, cy * K, rx * K, ry * K, scan, K)
        labels = []
        for (x, y, bw, bh, label, col) in boxes:
            rgba = hex2rgb(col) + (255,)
            x0, y0, x1, y1 = x * s * K, y * s * K, (x + bw) * s * K, (y + bh) * s * K
            d.rounded_rectangle([x0, y0, x1, y1], radius=10 * K, outline=rgba[:3] + (55,), width=K)
            self._brackets(d, x0, y0, x1, y1, rgba, K, max(16 * K, min(x1 - x0, y1 - y0) * 0.24), 3 * K)
            if label:
                tw = self._f_tag.measure(label) + 22
                ph = self._f_tag.metrics("linespace") + 8
                px = max(4, min(x * s, dw - tw - 4))
                py = y * s - ph - 8
                if py < 4:
                    py = (y + bh) * s + 8
                d.rounded_rectangle([px * K, py * K, (px + tw) * K, (py + ph) * K], radius=ph * K / 2,
                                    fill=rgba[:3] + (235,))
                labels.append((px + tw / 2, py + ph / 2, label))
        hx, chips = 12, []
        for text, dot in hud:
            cw = self._f_hud.measure(text) + 34
            d.rounded_rectangle([hx * K, 12 * K, (hx + cw) * K, 40 * K], radius=14 * K,
                                fill=(6, 8, 26, 205), outline=(255, 255, 255, 40), width=K)
            d.ellipse([(hx + 11) * K, 22 * K, (hx + 19) * K, 30 * K], fill=hex2rgb(dot) + (255,))
            chips.append((hx + 25, 26, text))
            hx += cw + 8
        im = Image.fromarray(rgb).convert("RGBA")
        im.alpha_composite(ov.resize((dw, dh), Image.BILINEAR))
        im.putalpha(self._mask(dw, dh))
        self._photo = ImageTk.PhotoImage(compose_on(im, self._bg))
        ox, oy = (w - dw) / 2, (h - dh) / 2
        self.coords(self._img, w / 2, h / 2)
        self.itemconfigure(self._img, image=self._photo)
        self.delete("ov")
        for (cx_, cy_, text) in labels:
            self.create_text(ox + cx_, oy + cy_, text=text, fill="#06101f", font=self._f_tag, tags="ov")
        for (cx_, cy_, text) in chips:
            self.create_text(ox + cx_, oy + cy_, text=text, anchor="w", fill=TEXT,
                             font=self._f_hud, tags="ov")

    def _draw_scan(self, d, cx, cy, rx, ry, scan, K):
        """Face-ID ring: radial ticks (lit by progress) + rotating comet arc."""
        N, prog = 90, float(scan.get("progress", 0.0))
        box = [cx - rx, cy - ry, cx + rx, cy + ry]
        d.ellipse(box, outline=(255, 255, 255, 38), width=2 * K)
        for i in range(N):
            t = i / N
            th = 2 * math.pi * t - math.pi / 2
            lit = t < prog
            ln = (16 if lit else 8) * K
            r0 = 12 * K
            p0 = (cx + (rx + r0) * math.cos(th), cy + (ry + r0) * math.sin(th))
            p1 = (cx + (rx + r0 + ln) * math.cos(th), cy + (ry + r0 + ln) * math.sin(th))
            col = hex2rgb(mix(CYAN, MAGENTA, t)) + (255,) if lit else (255, 255, 255, 46)
            d.line([p0, p1], fill=col, width=(4 if lit else 2) * K)
        a = float(scan.get("angle", 0.0))
        d.arc(box, a, a + 70, fill=hex2rgb(CYAN) + (80,), width=14 * K)
        d.arc(box, a, a + 70, fill=hex2rgb(CYAN_HI) + (255,), width=5 * K)


class NavButton(ctk.CTkFrame):
    """Sidebar item: glowing selection bar + vector icon + label."""

    def __init__(self, master, icon, label, command):
        super().__init__(master, fg_color="transparent", height=50)
        self._icon = icon
        self.grid_columnconfigure(0, weight=1)
        self.grid_propagate(False)
        self.button = ctk.CTkButton(self, text=label, font=load_font(14), height=46,
                                    image=icon_ctk(icon, 22, MUTED), compound="right",
                                    anchor="e", fg_color="transparent",
                                    hover_color=SIDEBAR_2, text_color=MUTED,
                                    corner_radius=14, command=command)
        self.button.grid(row=0, column=0, sticky="nsew", padx=(0, 4), pady=2)
        self.bar = ctk.CTkFrame(self, width=4, height=28, corner_radius=2, fg_color="transparent")
        self.bar.place(relx=1.0, rely=0.5, anchor="e")

    def set_active(self, active):
        self.bar.configure(fg_color=(CYAN if active else "transparent"))
        self.button.configure(fg_color=(SIDEBAR_2 if active else "transparent"),
                              text_color=(TEXT if active else MUTED),
                              image=icon_ctk(self._icon, 22, CYAN if active else MUTED))


class StatCard(ctk.CTkFrame):
    """KPI tile: gradient icon chip, big accent value, caption, glow edge."""

    def __init__(self, master, icon, title, accent, stops=GRAD_MAIN):
        super().__init__(master, fg_color=CARD, border_color=BORDER, border_width=1,
                         corner_radius=18)
        self.grid_columnconfigure(0, weight=1)
        chip = make_chip(icon, stops, 40, CARD)
        ctk.CTkLabel(self, image=ctk.CTkImage(chip, chip, (40, 40)), text="").grid(
            row=0, column=0, sticky="e", padx=14, pady=(14, 4))
        self.value = ctk.CTkLabel(self, text="0", font=load_font(26, True), text_color=accent)
        self.value.grid(row=1, column=0, sticky="e", padx=16)
        ctk.CTkLabel(self, text=title, font=load_font(10), text_color=MUTED).grid(
            row=2, column=0, sticky="e", padx=16, pady=(0, 8))
        AccentLine(self, bg=CARD, stops=stops, height=3).grid(
            row=3, column=0, sticky="ew", padx=18, pady=(0, 12))


# =====================================================
# MAIN APP
# =====================================================

class FaceRecognitionApp(ctk.CTk):
    VERSION = "3.0.0"

    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        pick_font(self)

        self.title("چهره‌یاب | Face Recognition — پوهنتون غرجستان")
        self.geometry("1260x780")
        self.minsize(1100, 700)
        self.configure(fg_color=BG_PAGE)

        self.db = Database()
        self.engine = None
        self._mean_emb = {}
        self._photo_cache = {}
        self._att_cache = {}
        self._screens = {}
        self._dash_cam = None
        self._dash_running = False
        self._enroll_cam = None
        self._enroll_state = None
        self._fps_history = []
        self._scan_angle = 0.0
        self._dot_on = True

        self._build_layout()
        self._refresh_mean_embeddings()
        self.show_screen("dashboard")
        self._tick_clock()
        self._pulse_dot()

    # =====================================================
    # LAYOUT — sidebar + content + statusbar
    # =====================================================

    def _build_layout(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ---------- sidebar ----------
        sidebar = ctk.CTkFrame(self, width=262, fg_color=SIDEBAR, corner_radius=0)
        sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew")
        sidebar.grid_propagate(False)
        sidebar.grid_rowconfigure(9, weight=1)
        sidebar.grid_columnconfigure(0, weight=1)

        logo_row = ctk.CTkFrame(sidebar, fg_color="transparent")
        logo_row.grid(row=0, column=0, sticky="ew", padx=10, pady=(14, 0))
        logo_img = make_ctk_image(make_app_logo(60, bg=SIDEBAR))
        ctk.CTkLabel(logo_row, image=logo_img, text="").pack(side="right")
        word = ctk.CTkFrame(logo_row, fg_color="transparent")
        word.pack(side="right", padx=(0, 4))
        ctk.CTkLabel(word, text="چهره‌یاب", font=load_font(26, True),
                     text_color=TEXT).pack(anchor="e")
        ctk.CTkLabel(word, text="F A C E   R E C O G N I T I O N", font=load_font(8, True),
                     text_color=CYAN).pack(anchor="e")
        AccentLine(sidebar, bg=SIDEBAR, height=2).grid(
            row=1, column=0, sticky="ew", padx=22, pady=(4, 14))

        nav = [
            ("dashboard", "dashboard", "داشبورد زنده"),
            ("enroll",    "scan",      "ثبت فرد جدید"),
            ("people",    "people",    "افراد ثبت‌شده"),
            ("report",    "report",    "گزارش حضور"),
            ("settings",  "settings",  "تنظیمات و سلامت"),
        ]
        self._nav_buttons = []
        for i, (key, icon, label) in enumerate(nav):
            btn = NavButton(sidebar, icon, label, command=lambda k=key: self.show_screen(k))
            btn.grid(row=2 + i, column=0, sticky="ew", padx=12, pady=2)
            self._nav_buttons.append((key, btn))

        # engine card (bottom)
        engine_card = ctk.CTkFrame(sidebar, fg_color=CARD, corner_radius=18,
                                   border_color=BORDER, border_width=1)
        engine_card.grid(row=10, column=0, sticky="ew", padx=16, pady=(6, 10))
        row = ctk.CTkFrame(engine_card, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(12, 4))
        chip = make_chip("bolt", GRAD_HOT, 34, CARD, 10)
        ctk.CTkLabel(row, image=ctk.CTkImage(chip, chip, (34, 34)), text="").pack(side="right")
        col = ctk.CTkFrame(row, fg_color="transparent")
        col.pack(side="right", padx=8)
        ctk.CTkLabel(col, text="موتور هوشمند", font=load_font(11, True),
                     text_color=TEXT).pack(anchor="e")
        ctk.CTkLabel(col, text="YuNet + SFace", font=load_font(10),
                     text_color=MUTED).pack(anchor="e")
        row2 = ctk.CTkFrame(engine_card, fg_color="transparent")
        row2.pack(fill="x", padx=14, pady=(0, 12))
        self._engine_dot = ctk.CTkFrame(row2, width=10, height=10, corner_radius=5, fg_color=GOOD)
        self._engine_dot.pack(side="right", padx=(0, 6), pady=4)
        ctk.CTkLabel(row2, text="آفلاین و آماده به کار", font=load_font(10),
                     text_color=GOOD).pack(side="right")

        ctk.CTkLabel(sidebar, text=f"نسخهٔ {self.VERSION}  ·  Aurora Neon",
                     font=load_font(9), text_color=DIM).grid(row=11, column=0, pady=(0, 14))

        # ---------- content ----------
        self.content = ctk.CTkFrame(self, fg_color=BG_PAGE, corner_radius=0)
        self.content.grid(row=0, column=1, sticky="nsew")

        # ---------- status bar ----------
        status = ctk.CTkFrame(self, height=34, fg_color=SIDEBAR, corner_radius=0)
        status.grid(row=1, column=1, sticky="ew")
        AccentLine(self, bg=SIDEBAR, height=2, fade=False).grid(row=1, column=1, sticky="new")
        self.status_label = ctk.CTkLabel(status, text="●  آماده", text_color=GOOD,
                                         font=load_font(11, True))
        self.status_label.pack(side="right", padx=18)
        self.clock_label = ctk.CTkLabel(status, text="", text_color=MUTED, font=load_font(10))
        self.clock_label.pack(side="left", padx=18)
        ctk.CTkLabel(status, text="پوهنځی کمپیوتر ساینس — پوهنتون غرجستان",
                     text_color=DIM, font=load_font(10)).pack(side="left", padx=(4, 0))

        # ---------- screens ----------
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_rowconfigure(0, weight=1)
        self._screens["dashboard"] = self._build_dashboard()
        self._screens["enroll"] = self._build_enroll()
        self._screens["people"] = self._build_people()
        self._screens["report"] = self._build_report()
        self._screens["settings"] = self._build_settings()

        self.bind("<Control-Key-1>", lambda e: self.show_screen("dashboard"))
        self.bind("<Control-Key-2>", lambda e: self.show_screen("enroll"))
        self.bind("<Control-Key-3>", lambda e: self.show_screen("people"))
        self.bind("<Control-Key-4>", lambda e: self.show_screen("report"))
        self.bind("<Control-Key-5>", lambda e: self.show_screen("settings"))
        self.bind("<Escape>", lambda e: self._on_escape())

    def show_screen(self, name):
        if name != "dashboard":
            self._stop_dash_camera()
        if name != "enroll":
            self._cancel_scan(silent=True)

        for key, frame in self._screens.items():
            if key == name:
                frame.grid(row=0, column=0, sticky="nsew", padx=18, pady=16)
            else:
                frame.grid_forget()
        for key, btn in self._nav_buttons:
            btn.set_active(key == name)
        if name == "people":
            self._refresh_people()
        elif name == "report":
            self._refresh_report()
        elif name == "dashboard":
            self._refresh_dashboard_stats()
            if not self._dash_running:
                self._cam_view.show_idle()

    def _tick_clock(self):
        try:
            self.clock_label.configure(text=datetime.now().strftime("%Y/%m/%d   %H:%M:%S"))
        except Exception:
            pass
        self.after(1000, self._tick_clock)

    def _pulse_dot(self):
        """Gentle heartbeat on the engine-ready indicator."""
        try:
            self._dot_on = not self._dot_on
            self._engine_dot.configure(fg_color=GOOD if self._dot_on else "#146b55")
        except Exception:
            return
        self.after(900, self._pulse_dot)

    # =====================================================
    # helpers
    # =====================================================

    def _get_engine(self):
        if self.engine is None:
            self.engine = FaceEngine()
        return self.engine

    def _refresh_mean_embeddings(self):
        try:
            self._mean_emb = self.db.get_mean_embeddings()
        except Exception:
            self._mean_emb = {}

    def _refresh_people_count(self):
        count = self.db.count_people()
        try:
            self._stat_people.value.configure(text=str(count))
        except AttributeError:
            pass
        return count

    def _toast(self, message, ok=True):
        toast = ctk.CTkToplevel(self)
        toast.overrideredirect(True)
        toast.attributes("-topmost", True)
        try:
            toast.attributes("-alpha", 0.97)
        except tk.TclError:
            pass
        accent = GOOD if ok else BAD
        x = self.winfo_rootx() + self.winfo_width() // 2 - 190
        y = self.winfo_rooty() + 24
        toast.geometry(f"380x56+{x}+{y}")
        toast.configure(fg_color=accent)
        inner = ctk.CTkFrame(toast, fg_color=CARD_2, corner_radius=0)
        inner.pack(fill="both", expand=True, padx=(0, 5))
        ctk.CTkLabel(inner, text=f"  {message}", text_color=TEXT, font=load_font(13, True),
                     image=icon_ctk("check" if ok else "info", 22, accent),
                     compound="right").pack(expand=True)
        toast.after(1900, toast.destroy)

    def _status(self, text, ok=True):
        self.status_label.configure(text=f"●  {text}", text_color=(GOOD if ok else BAD))

    def _get_photo_ctk(self, person, size=96, bg=CARD):
        key = (person["id"], person.get("photo_path"), size, bg)
        if key not in self._photo_cache:
            img = photo_or_avatar(person, size * 2, bg)          # 2× for HiDPI crispness
            self._photo_cache[key] = ctk.CTkImage(light_image=img, dark_image=img,
                                                  size=(size, size))
        return self._photo_cache[key]

    # ---- shared UI builders ----
    def _card(self, master, radius=20, **kw):
        return ctk.CTkFrame(master, fg_color=CARD, border_color=BORDER, border_width=1,
                            corner_radius=radius, **kw)

    def _page_title(self, master, text, icon, stops=GRAD_MAIN):
        """Right-aligned page title with a gradient icon chip."""
        row = ctk.CTkFrame(master, fg_color="transparent")
        chip = make_chip(icon, stops, 40, BG_PAGE)
        ctk.CTkLabel(row, image=ctk.CTkImage(chip, chip, (40, 40)), text="").pack(side="right")
        ctk.CTkLabel(row, text=text, font=load_font(21, True), text_color=TEXT).pack(
            side="right", padx=12)
        return row

    def _pill(self, master, text, fg, bg, size=10, bold=True):
        """Small tinted status chip."""
        f = ctk.CTkFrame(master, fg_color=bg, corner_radius=9)
        ctk.CTkLabel(f, text=text, font=load_font(size, bold), text_color=fg).pack(
            padx=10, pady=2)
        return f

    # =====================================================
    # SCREEN 1: DASHBOARD
    # =====================================================

    def _build_dashboard(self):
        scr = ctk.CTkFrame(self.content, fg_color="transparent")
        scr.grid_rowconfigure(1, weight=1)
        scr.grid_columnconfigure(0, weight=3)
        scr.grid_columnconfigure(1, weight=2)

        # ---- hero banner ----
        HeroBanner(scr, "سیستم هوشمند شناسایی چهره",
                   "کاملاً آفلاین   ·   YuNet + SFace   ·   پوهنتون غرجستان").grid(
            row=0, column=0, columnspan=2, sticky="ew", pady=(0, 14))

        # ---- camera card ----
        cam_card = self._card(scr, 22)
        cam_card.grid(row=1, column=0, sticky="nsew", padx=(0, 12))
        cam_card.grid_rowconfigure(2, weight=1)
        cam_card.grid_columnconfigure(0, weight=1)

        head = ctk.CTkFrame(cam_card, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", padx=20, pady=(16, 6))
        chip = make_chip("camera", GRAD_MAIN, 38, CARD, 11)
        ctk.CTkLabel(head, image=ctk.CTkImage(chip, chip, (38, 38)), text="").pack(side="right")
        title_box = ctk.CTkFrame(head, fg_color="transparent")
        title_box.pack(side="right", padx=10)
        ctk.CTkLabel(title_box, text="دوربین زنده", font=load_font(17, True),
                     text_color=TEXT).pack(anchor="e")
        self._fps_label = ctk.CTkLabel(title_box, text="شناسایی لحظه‌ای چهره‌ها",
                                       font=load_font(10), text_color=MUTED)
        self._fps_label.pack(anchor="e")
        self._live_chip = ctk.CTkFrame(head, fg_color=CARD_3, corner_radius=11)
        self._live_chip.pack(side="left")
        self._live_text = ctk.CTkLabel(self._live_chip, text="○  OFFLINE", font=load_font(10, True),
                                       text_color=MUTED)
        self._live_text.pack(padx=12, pady=4)
        AccentLine(cam_card, bg=CARD, height=2).grid(row=1, column=0, sticky="ew", padx=20)

        self._cam_view = CameraView(cam_card, "دوربین آماده است",
                                    "برای شروع شناسایی، دکمهٔ «شروع دوربین» را بزنید", bg=CARD)
        self._cam_view.grid(row=2, column=0, sticky="nsew", padx=18, pady=10)

        ctl = ctk.CTkFrame(cam_card, fg_color="transparent")
        ctl.grid(row=3, column=0, sticky="ew", padx=18, pady=(2, 18))
        self._cam_toggle_btn = GradientButton(
            ctl, text="شروع دوربین", icon="play", colors=GRAD_MAIN, text_color=ON_CYAN,
            width=210, height=50, bg=CARD, size=15, command=self._toggle_dash_camera)
        self._cam_toggle_btn.pack(side="right")
        faces_chip = ctk.CTkFrame(ctl, fg_color=CARD_2, corner_radius=12, border_color=BORDER,
                                  border_width=1)
        faces_chip.pack(side="left")
        self._faces_label = ctk.CTkLabel(faces_chip, text="چهره‌ها:  0", font=load_font(12, True),
                                         text_color=TEXT)
        self._faces_label.pack(padx=16, pady=10)

        # ---- right column ----
        right = ctk.CTkFrame(scr, fg_color="transparent")
        right.grid(row=1, column=1, sticky="nsew")
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)

        stats = ctk.CTkFrame(right, fg_color="transparent")
        stats.grid(row=0, column=0, sticky="ew")
        stats.grid_columnconfigure((0, 1, 2), weight=1, uniform="stat")
        self._stat_people = StatCard(stats, "people", "افراد ثبت‌شده", CYAN, GRAD_MAIN)
        self._stat_people.grid(row=0, column=2, sticky="nsew", padx=(8, 0))
        self._stat_today = StatCard(stats, "check", "حضور امروز", GOOD, GRAD_GOOD)
        self._stat_today.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        self._stat_model = StatCard(stats, "cpu", "وضعیت مدل‌ها", MAGENTA, GRAD_HOT)
        self._stat_model.grid(row=0, column=0, sticky="nsew")

        # last recognition card (border pulses on a new match)
        self._last_card = ctk.CTkFrame(right, fg_color=CARD, border_color=BORDER,
                                       border_width=2, corner_radius=22)
        self._last_card.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        hdr = ctk.CTkFrame(self._last_card, fg_color="transparent")
        hdr.pack(fill="x", padx=18, pady=(16, 0))
        ctk.CTkLabel(hdr, text="آخرین شناسایی", font=load_font(15, True),
                     text_color=TEXT).pack(side="right")
        ctk.CTkLabel(hdr, image=icon_ctk("scan", 22, CYAN), text="").pack(side="right", padx=8)
        AccentLine(self._last_card, bg=CARD, height=2).pack(fill="x", padx=18, pady=(8, 0))

        body = ctk.CTkFrame(self._last_card, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=18, pady=6)
        self._ph_avatar = ctk.CTkImage(make_avatar("؟", 208, True, CARD), make_avatar("؟", 208, True, CARD), (104, 104))
        self._last_avatar = ctk.CTkLabel(body, text="", image=self._ph_avatar)
        self._last_avatar.pack(pady=(10, 6))
        self._last_name = ctk.CTkLabel(body, text="—", font=load_font(20, True), text_color=TEXT)
        self._last_name.pack()
        self._last_meta = ctk.CTkLabel(body, text="هنوز شناسایی‌ای انجام نشده",
                                       font=load_font(11), text_color=MUTED)
        self._last_meta.pack(pady=(2, 8))
        self._gauge = RingGauge(body, 132, CARD, GRAD_MAIN)
        self._gauge.pack(pady=(0, 6))
        self._gauge.set(0, big="—", small="شباهت")
        self._match_pct = ctk.CTkLabel(body, text="", font=load_font(10), text_color=MUTED)
        self._match_pct.pack()
        return scr

    def _set_live(self, on):
        self._live_chip.configure(fg_color=(GOOD_BG if on else CARD_3))
        self._live_text.configure(text=("●  LIVE" if on else "○  OFFLINE"),
                                  text_color=(GOOD if on else MUTED))

    def _refresh_dashboard_stats(self):
        self._stat_people.value.configure(text=str(self.db.count_people()))
        self._stat_today.value.configure(text=str(self.db.attendance_today_count()))
        self._stat_model.value.configure(text="آماده" if models_present() else "خطا",
                                         text_color=(MAGENTA if models_present() else BAD))
        self._status("آماده" if models_present() else "مدل‌ها یافت نشدند", ok=models_present())

    def _toggle_dash_camera(self):
        if self._dash_running:
            self._stop_dash_camera()
        else:
            self._start_dash_camera()

    def _start_dash_camera(self):
        try:
            cam_index = int(self.db.get_setting("camera_index", "0"))
        except (ValueError, TypeError):
            cam_index = 0
        self._dash_cam = cv2.VideoCapture(cam_index)
        if not self._dash_cam.isOpened():
            self._status("دوربین باز نشد", ok=False)
            messagebox.showerror("خطا", "دوربین باز نشد. اتصال و اجازهٔ دسترسی را بررسی کنید.")
            return
        self._dash_running = True
        self._cam_toggle_btn.configure(text="توقف دوربین", icon="stop", fg_color=GRAD_BAD,
                                       text_color="#ffffff")
        self._set_live(True)
        self._refresh_mean_embeddings()
        self._status("دوربین روشن است")
        self._dash_loop()

    def _stop_dash_camera(self):
        self._dash_running = False
        if self._dash_cam is not None:
            self._dash_cam.release()
            self._dash_cam = None
        try:
            self._cam_toggle_btn.configure(text="شروع دوربین", icon="play", fg_color=GRAD_MAIN,
                                           text_color=ON_CYAN)
            self._set_live(False)
            self._cam_view.show_idle()
            self._faces_label.configure(text="چهره‌ها:  0")
            self._fps_label.configure(text="شناسایی لحظه‌ای چهره‌ها")
        except Exception:
            pass

    def _dash_loop(self):
        if not self._dash_running or self._dash_cam is None:
            return
        ok, frame = self._dash_cam.read()
        if not ok:
            self._status("خطا در خواندن فریم", ok=False)
            self._stop_dash_camera()
            return

        engine = self._get_engine()
        faces = engine.detect(frame)
        threshold = float(self.db.get_setting("match_threshold", COSINE_THRESHOLD))

        boxes, best_show = [], None
        for f in faces:
            feat = engine.embed(frame, f)
            best_id, best_sim = None, 0.0
            for pid, mean in self._mean_emb.items():
                sim = cosine_sim(feat, mean)
                if sim > best_sim:
                    best_id, best_sim = pid, sim
            known = best_id is not None and best_sim >= threshold
            x, y, w, h = [int(v) for v in f[:4]]
            if known:
                person = self.db.get_person(best_id)
                name = person["full_name"] if person else str(best_id)
                caption = f"{name}   {best_sim * 100:.0f}%"
                self._log_attendance_gui(best_id)
                best_show = (person, best_sim)
            else:
                caption = f"Unknown   {best_sim * 100:.0f}%"
            boxes.append((x, y, w, h, caption, GOOD if known else BAD))

        if best_show:
            self._show_last_recognition(best_show[0], best_show[1])

        now = time.time()
        self._fps_history.append(now)
        self._fps_history = [t for t in self._fps_history if now - t < 1.0]
        fps = len(self._fps_history)
        self._faces_label.configure(text=f"چهره‌ها:  {len(faces)}")
        self._fps_label.configure(text=f"{fps} FPS  ·  {len(faces)} چهره در قاب")
        self._cam_view.show_frame(frame, boxes,
                                  hud=[("LIVE", BAD), (f"{fps} FPS", CYAN)])
        self.after(33, self._dash_loop)

    def _log_attendance_gui(self, person_id):
        now = datetime.now()
        last = self._att_cache.get(person_id)
        if last and (now - last).total_seconds() < 60:
            return
        self.db.add_attendance(person_id)
        self._att_cache[person_id] = now
        self._stat_today.value.configure(text=str(self.db.attendance_today_count()))

    def _show_last_recognition(self, person, sim):
        if person is None:
            return
        img = self._get_photo_ctk(person, 104, CARD)
        self._last_avatar.configure(image=img)
        self._last_avatar._image = img
        self._last_name.configure(text=person["full_name"])
        meta = "  ·  ".join(p for p in [person.get("student_id"), person.get("department")] if p)
        self._last_meta.configure(text=meta or "—")
        good = sim >= 0.363
        self._gauge.set(max(0.0, min(1.0, sim)), stops=(GRAD_GOOD if good else GRAD_GOLD),
                        big=f"{sim * 100:.0f}", small="٪ شباهت")
        self._match_pct.configure(text=f"شباهت کسینوسی:  {sim:.3f}")
        try:                                   # pulse the card border
            self._last_card.configure(border_color=(GOOD if good else WARN))
            self.after(1400, lambda: self._last_card.configure(border_color=BORDER))
        except Exception:
            pass


    # ---- extra shared builders ----
    def _ghost_btn(self, master, text, command, icon=None, bg=CARD, width=130,
                   height=44, color=TEXT, size=12):
        """Quiet outlined button (secondary actions)."""
        return GradientButton(master, text=text, command=command, icon=icon,
                              colors=(CARD_3, CARD_3), text_color=color, width=width,
                              height=height, bg=bg, size=size, bold=True, border=BORDER_HI)

    def _card_header(self, card, title, subtitle=None, icon="info", stops=GRAD_MAIN):
        """Chip + title (+ subtitle) + glowing divider, right aligned."""
        bg = CARD
        head = ctk.CTkFrame(card, fg_color="transparent")
        head.pack(fill="x", padx=20, pady=(18, 6))
        chip = make_chip(icon, stops, 38, bg, 11)
        ctk.CTkLabel(head, image=ctk.CTkImage(chip, chip, (38, 38)), text="").pack(side="right")
        box = ctk.CTkFrame(head, fg_color="transparent")
        box.pack(side="right", padx=10)
        ctk.CTkLabel(box, text=title, font=load_font(16, True), text_color=TEXT).pack(anchor="e")
        if subtitle:
            ctk.CTkLabel(box, text=subtitle, font=load_font(10), text_color=MUTED).pack(anchor="e")
        AccentLine(card, bg=bg, height=2).pack(fill="x", padx=20, pady=(0, 6))
        return head

    @staticmethod
    def _focus_glow(entry):
        """Cyan border while an entry has focus."""
        entry.bind("<FocusIn>", lambda e: entry.configure(border_color=CYAN), add="+")
        entry.bind("<FocusOut>", lambda e: entry.configure(border_color=BORDER), add="+")

    # =====================================================
    # SCREEN 2: ENROLL — Face-ID style guided scan
    # =====================================================

    def _build_enroll(self):
        scr = ctk.CTkFrame(self.content, fg_color="transparent")
        scr.grid_columnconfigure(0, weight=2, uniform="enroll")
        scr.grid_columnconfigure(1, weight=3, uniform="enroll")
        scr.grid_rowconfigure(0, weight=1)

        # ---- form card ----
        form_card = self._card(scr, 22)
        form_card.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        self._card_header(form_card, "مشخصات فرد", "فیلدهای ستاره‌دار الزامی‌اند",
                          "user_plus", GRAD_HOT)
        form = ctk.CTkScrollableFrame(form_card, fg_color="transparent", corner_radius=0,
                                      scrollbar_button_color=CARD_3,
                                      scrollbar_button_hover_color=BORDER_HI)
        form.pack(fill="both", expand=True, padx=(8, 4), pady=(0, 6))

        def field(label, placeholder):
            ctk.CTkLabel(form, text=label, font=load_font(12, True), text_color=TEXT).pack(
                anchor="e", padx=12, pady=(8, 3))
            entry = ctk.CTkEntry(form, placeholder_text=placeholder, height=42,
                                 font=load_font(13), justify="right", corner_radius=12,
                                 fg_color=CARD_2, border_color=BORDER, border_width=1,
                                 text_color=TEXT, placeholder_text_color=DIM)
            entry.pack(fill="x", padx=12, pady=(0, 2))
            self._focus_glow(entry)
            return entry

        self._f_name = field("نام کامل  *", "مثلاً: سید احمد رضایی")
        self._f_sid = field("شمارهٔ کارت / پوهنځی", "مثلاً: CS-1402-015")
        self._f_dept = field("دانشکده / بخش", "مثلاً: کمپیوتر ساینس")
        self._f_phone = field("شمارهٔ تماس", "مثلاً: 07xx xxx xxx")
        self._f_email = field("ایمیل", "name@example.com")
        ctk.CTkLabel(form, text="یادداشت", font=load_font(12, True),
                     text_color=TEXT).pack(anchor="e", padx=12, pady=(10, 3))
        self._f_notes = ctk.CTkTextbox(form, height=64, font=load_font(12), corner_radius=12,
                                       fg_color=CARD_2, border_color=BORDER, border_width=1,
                                       text_color=TEXT)
        self._f_notes.pack(fill="x", padx=12, pady=(0, 8))
        self._focus_glow(self._f_notes)

        self._form_error = ctk.CTkLabel(form, text="", font=load_font(11, True), text_color=BAD)
        self._form_error.pack(anchor="e", padx=12, pady=(0, 2))

        btn_row = ctk.CTkFrame(form_card, fg_color="transparent")
        btn_row.pack(fill="x", padx=20, pady=(2, 20))
        self._scan_btn = GradientButton(btn_row, text="شروع اسکن چهره", icon="scan",
                                        colors=GRAD_HOT, text_color="#ffffff", width=220,
                                        height=50, bg=CARD, size=14, command=self._start_scan)
        self._scan_btn.pack(side="right", fill="x", expand=True, padx=(10, 0))
        self._ghost_btn(btn_row, "پاک کردن", self._clear_enroll_form, width=110, height=50).pack(
            side="left")

        # ---- scan card ----
        scan = self._card(scr, 22)
        scan.grid(row=0, column=1, sticky="nsew")
        scan.grid_columnconfigure(0, weight=1)
        scan.grid_rowconfigure(2, weight=1)

        head = ctk.CTkFrame(scan, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", padx=20, pady=(18, 6))
        chip = make_chip("scan", GRAD_MAIN, 38, CARD, 11)
        ctk.CTkLabel(head, image=ctk.CTkImage(chip, chip, (38, 38)), text="").pack(side="right")
        box = ctk.CTkFrame(head, fg_color="transparent")
        box.pack(side="right", padx=10)
        ctk.CTkLabel(box, text="اسکن چهره — چند زاویه‌ای", font=load_font(16, True),
                     text_color=TEXT).pack(anchor="e")
        ctk.CTkLabel(box, text="سر را آرام بچرخانید تا همهٔ زاویه‌ها ثبت شود", font=load_font(10),
                     text_color=MUTED).pack(anchor="e")
        AccentLine(scan, bg=CARD, height=2).grid(row=1, column=0, sticky="ew", padx=20)

        self._scan_view = CameraView(scan, "آماده برای اسکن چهره",
                                     "پس از پر کردن مشخصات، «شروع اسکن چهره» را بزنید", bg=CARD)
        self._scan_view.grid(row=2, column=0, sticky="nsew", padx=18, pady=10)

        self._scan_instruction = ctk.CTkLabel(scan, text="آماده برای اسکن", font=load_font(18, True),
                                              text_color=CYAN)
        self._scan_instruction.grid(row=3, column=0, pady=(0, 6))

        chips = ctk.CTkFrame(scan, fg_color="transparent")
        chips.grid(row=4, column=0, pady=(0, 6), padx=18)
        for c in range(4):
            chips.grid_columnconfigure(c, weight=1, uniform="zone")
        self._zone_chips = {}
        for i, zone in enumerate(ZONE_ORDER):
            chip_lbl = ctk.CTkLabel(chips, text=self._zone_text(zone, 0, False),
                                    font=load_font(10), fg_color=CARD_2, corner_radius=10,
                                    width=150, height=30, text_color=MUTED)
            chip_lbl.grid(row=i // 4, column=3 - (i % 4), padx=3, pady=3)
            self._zone_chips[zone] = chip_lbl

        prog = ctk.CTkFrame(scan, fg_color="transparent")
        prog.grid(row=5, column=0, sticky="ew", padx=22)
        self._scan_progress = GradientBar(prog, bg=CARD, stops=GRAD_FULL, height=12, track=CARD_3)
        self._scan_progress.pack(fill="x")
        self._scan_count = ctk.CTkLabel(scan, text=f"0 / {sum(ZONE_TARGETS.values())} نمونه",
                                        font=load_font(11, True), text_color=MUTED)
        self._scan_count.grid(row=6, column=0, pady=(6, 0))
        self._scan_reason = ctk.CTkLabel(scan, text=" ", font=load_font(10, True), text_color=BAD)
        self._scan_reason.grid(row=7, column=0)

        btn_row2 = ctk.CTkFrame(scan, fg_color="transparent")
        btn_row2.grid(row=8, column=0, sticky="ew", padx=20, pady=(6, 20))
        self._ghost_btn(btn_row2, "لغو اسکن", self._cancel_scan, icon="stop", width=140,
                        height=46).pack(side="right")
        self._finish_btn = GradientButton(btn_row2, text="پایان و ذخیره", icon="check",
                                          colors=GRAD_GOOD, text_color=ON_GOOD, width=190,
                                          height=46, bg=CARD, size=13, command=self._finish_scan)
        self._finish_btn.configure(state="disabled")
        self._finish_btn.pack(side="left")
        return scr

    @staticmethod
    def _zone_text(zone, count, done):
        label = ZONE_LABELS[zone].split("  ", 1)[1]
        target = ZONE_TARGETS[zone]
        if done:
            return f"✓  {label}"
        if count:
            return f"{count}/{target}  {label}"
        return f"○  {label}"

    def _clear_enroll_form(self):
        for entry in (self._f_name, self._f_sid, self._f_dept, self._f_phone, self._f_email):
            entry.delete(0, "end")
        self._f_notes.delete("1.0", "end")
        self._form_error.configure(text="")

    def _validated_form(self):
        name = self._f_name.get().strip()
        if len(name) < 2:
            self._form_error.configure(text="⚠ نام کامل الزامی است (حداقل ۲ حرف)")
            return None
        sid = self._f_sid.get().strip()
        if sid and self.db.student_id_exists(sid):
            self._form_error.configure(text="⚠ این شمارهٔ کارت قبلاً ثبت شده است")
            return None
        self._form_error.configure(text="")
        return {
            "full_name": name,
            "student_id": sid or None,
            "department": self._f_dept.get().strip() or None,
            "phone": self._f_phone.get().strip() or None,
            "email": self._f_email.get().strip() or None,
            "notes": self._f_notes.get("1.0", "end").strip() or None,
        }

    def _start_scan(self):
        data = self._validated_form()
        if data is None:
            return
        try:
            cam_index = int(self.db.get_setting("camera_index", "0"))
        except (ValueError, TypeError):
            cam_index = 0
        self._enroll_cam = cv2.VideoCapture(cam_index)
        if not self._enroll_cam.isOpened():
            self._release_enroll_camera()
            messagebox.showerror("خطا", "دوربین باز نشد.")
            return

        width = int(self._enroll_cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._enroll_cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        engine = self._get_engine()
        engine.set_input_size(width, height)

        self._enroll_state = {
            "person": data,
            "samples": [],
            "zones": dict(ZONE_TARGETS),
            "counts": {z: 0 for z in ZONE_TARGETS},
            "last_capture": 0.0,
            "started": time.time(),
        }
        self._scan_btn.configure(state="disabled")
        self._finish_btn.configure(state="disabled")
        self._status("در حال اسکن چهره")
        self._enroll_loop()

    def _missing_zones(self):
        st = self._enroll_state
        return [z for z in ZONE_ORDER if st["counts"][z] < st["zones"][z]]

    def _enroll_loop(self):
        st = self._enroll_state
        if st is None or self._enroll_cam is None:
            return
        ok, frame = self._enroll_cam.read()
        if not ok:
            self._status("خطا در خواندن فریم", ok=False)
            self._cancel_scan()
            return

        engine = self._get_engine()
        faces = engine.detect(frame)
        faces = [f for f in faces if f[2] >= 50 and f[3] >= 50]
        self._scan_angle = (self._scan_angle + 7) % 360          # rotating comet arc

        boxes = []
        now = time.time()
        target_total = sum(st["zones"].values())
        if len(faces) == 1:
            f = max(faces, key=lambda r: r[2] * r[3])
            x, y, w, h = [int(v) for v in f[:4]]

            quality_ok, reason, score = engine.sample_quality(frame, f)
            boxes.append((x, y, w, h, "", GOOD if quality_ok else WARN))
            self._scan_reason.configure(
                text=" " if quality_ok else QUALITY_REASON_TEXT.get(reason, reason),
                text_color=(GOOD if quality_ok else BAD))

            yaw_zone, pitch_zone, _yaw, _pitch = engine.head_pose(f)
            if yaw_zone != "center":
                zone = yaw_zone
            elif pitch_zone != "level":
                zone = pitch_zone
            else:
                zone = "center"

            if quality_ok:
                target_zone = zone if st["counts"].get(zone, 99) < st["zones"].get(zone, 0) else None
                if target_zone is None and st["counts"].get("center", 99) < st["zones"].get("center", 0):
                    target_zone = "center"
                if now - st["last_capture"] >= 0.45:
                    st["last_capture"] = now
                    emb = engine.embed(frame, f)
                    crop = FaceEngine.crop_face(frame, f)
                    capture_zone = target_zone if target_zone else "overflow"
                    st["samples"].append({"emb": emb, "zone": capture_zone,
                                          "quality": score, "crop": crop})
                    if target_zone:
                        st["counts"][target_zone] += 1
                        self._update_zone_chips()

            missing = self._missing_zones()
            total = len(st["samples"])
            self._scan_progress.set(min(1.0, total / target_total))
            self._scan_count.configure(text=f"{total} / {target_total} نمونه",
                                       text_color=(GOOD if not missing else TEXT))
            if total >= 12:
                self._finish_btn.configure(state="normal", fg_color=GRAD_GOOD,
                                           text_color=ON_GOOD)

            if missing:
                self._scan_instruction.configure(text=MISSING_PROMPT[missing[0]], text_color=CYAN)
            else:
                self._scan_instruction.configure(text="✓ همهٔ زاویه‌ها ثبت شد!", text_color=GOOD)

            if not missing or total >= target_total or (
                    now - st["started"] > 40 and total >= 12):
                self._finish_scan()
                return
        else:
            msg = "دوربین را تنظیم کنید — یک چهره در قاب باشد" if faces else \
                  "چهره‌ای دیده نشد — نزدیک‌تر شوید و نور را تنظیم کنید"
            self._scan_instruction.configure(text=msg, text_color=BAD)

        total = len(st["samples"])
        self._scan_view.show_frame(
            frame, boxes,
            hud=[("SCAN", MAGENTA), (f"{total}/{target_total}", CYAN)],
            scan=dict(progress=min(1.0, total / target_total), angle=self._scan_angle))
        self.after(30, self._enroll_loop)

    def _update_zone_chips(self):
        st = self._enroll_state
        for zone in ZONE_ORDER:
            count, need = st["counts"][zone], st["zones"][zone]
            done = count >= need
            partial = 0 < count < need
            self._zone_chips[zone].configure(
                text=self._zone_text(zone, count, done),
                text_color=(GOOD if done else (VIOLET_HI if partial else MUTED)),
                fg_color=(GOOD_BG if done else (VIOLET_BG if partial else CARD_2)))

    def _release_enroll_camera(self):
        if self._enroll_cam is not None:
            self._enroll_cam.release()
            self._enroll_cam = None

    def _cancel_scan(self, silent=False):
        was_active = self._enroll_state is not None
        self._enroll_state = None
        self._release_enroll_camera()
        self._scan_btn.configure(state="normal")
        self._finish_btn.configure(state="disabled")
        self._scan_view.show_idle()
        self._scan_instruction.configure(text="آماده برای اسکن", text_color=CYAN)
        self._scan_progress.set(0)
        self._scan_count.configure(text=f"0 / {sum(ZONE_TARGETS.values())} نمونه",
                                   text_color=MUTED)
        self._scan_reason.configure(text=" ")
        for zone in ZONE_ORDER:
            self._zone_chips[zone].configure(text=self._zone_text(zone, 0, False),
                                             text_color=MUTED, fg_color=CARD_2)
        if not silent and was_active:
            self._status("اسکن لغو شد", ok=False)
            self._toast("اسکن لغو شد", ok=False)

    def _finish_scan(self):
        st = self._enroll_state
        if st is None:
            return
        self._release_enroll_camera()
        if not st["samples"]:
            self._cancel_scan()
            return

        feats = [s["emb"] for s in st["samples"]]
        qualities = [s["quality"] for s in st["samples"]]
        mean = np.mean(np.stack(feats), axis=0)
        mean /= (np.linalg.norm(mean) + 1e-10)
        avg_quality = float(np.mean(qualities)) if qualities else 0.0

        person_id = self.db.add_person(**st["person"])
        for s in st["samples"]:
            self.db.add_face(person_id, s["emb"], s["zone"], s["quality"])

        best = max(st["samples"], key=lambda s: s["quality"])
        photo_path = PROFILES_DIR / f"person_{person_id}.jpg"
        cv2.imwrite(str(photo_path), best["crop"])
        self.db.update_person(person_id, photo_path=str(photo_path))

        name = st["person"]["full_name"]
        self._enroll_state = None
        self._scan_btn.configure(state="normal")
        self._finish_btn.configure(state="disabled")
        self._photo_cache.clear()
        self._refresh_mean_embeddings()
        self._refresh_people_count()
        self._status(f"«{name}» ثبت شد")
        self._show_enroll_summary(person_id, name, len(feats), avg_quality)
        self._clear_enroll_form()
        self._cancel_scan(silent=True)

    def _show_enroll_summary(self, person_id, name, sample_count, avg_quality):
        win = ctk.CTkToplevel(self)
        win.title("ثبت موفق")
        w, h = 470, 510
        x = self.winfo_rootx() + max(0, (self.winfo_width() - w) // 2)
        y = self.winfo_rooty() + max(0, (self.winfo_height() - h) // 2)
        win.geometry(f"{w}x{h}+{x}+{y}")
        win.resizable(False, False)
        win.transient(self)
        win.configure(fg_color=BG_PAGE)
        win.after(150, lambda: self._safe_grab(win))

        card = ctk.CTkFrame(win, fg_color=CARD, border_color=BORDER, border_width=1,
                            corner_radius=26)
        card.pack(fill="both", expand=True, padx=14, pady=14)

        badge = make_chip("check", GRAD_GOOD, 64, CARD, 32)
        ctk.CTkLabel(card, image=ctk.CTkImage(badge, badge, (64, 64)), text="").pack(pady=(24, 6))
        ctk.CTkLabel(card, text="ثبت‌نام موفق!", font=load_font(23, True),
                     text_color=GOOD).pack()

        person = self.db.get_person(person_id)
        if person:
            img = photo_or_avatar(person, 260, CARD)
            ctk.CTkLabel(card, image=ctk.CTkImage(img, img, (130, 130)), text="").pack(pady=(14, 6))
        ctk.CTkLabel(card, text=name, font=load_font(19, True), text_color=TEXT).pack()

        pills = ctk.CTkFrame(card, fg_color="transparent")
        pills.pack(pady=(10, 4))
        self._pill(pills, f"{sample_count} نمونه از زوایای مختلف", CYAN, CYAN_BG, 11).pack(
            side="right", padx=4)
        self._pill(pills, f"کیفیت {avg_quality:.0f} / 100", GOOD, GOOD_BG, 11).pack(
            side="right", padx=4)
        bar = GradientBar(card, bg=CARD, stops=GRAD_GOOD, height=10, track=CARD_3)
        bar.pack(fill="x", padx=60, pady=(8, 0))
        bar.set(max(0.0, min(1.0, avg_quality / 100.0)))

        def go(target):
            win.destroy()
            self.show_screen(target)

        row = ctk.CTkFrame(card, fg_color="transparent")
        row.pack(side="bottom", pady=(8, 22))
        GradientButton(row, text="ثبت فرد دیگر", icon="user_plus", colors=GRAD_MAIN,
                       text_color=ON_CYAN, width=170, height=46, bg=CARD, size=13,
                       command=lambda: go("enroll")).pack(side="right", padx=6)
        self._ghost_btn(row, "فهرست افراد", lambda: go("people"), icon="people", width=160,
                        height=46).pack(side="left", padx=6)

    @staticmethod
    def _safe_grab(win):
        try:
            win.grab_set()
            win.focus_force()
        except tk.TclError:
            pass

    def _on_escape(self):
        if self._enroll_state is not None:
            self._cancel_scan()

    # =====================================================
    # SCREEN 3: PEOPLE
    # =====================================================

    def _build_people(self):
        scr = ctk.CTkFrame(self.content, fg_color="transparent")
        scr.grid_columnconfigure(0, weight=1)
        scr.grid_rowconfigure(1, weight=1)

        top = ctk.CTkFrame(scr, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        self._page_title(top, "افراد ثبت‌شده", "people", GRAD_MAIN).pack(side="right")
        self._people_count = self._pill(top, "۰ نفر", CYAN, CYAN_BG, 11)
        self._people_count.pack(side="right", padx=12)
        self._people_count_lbl = self._people_count.winfo_children()[0]

        GradientButton(top, text="افزودن فرد", icon="user_plus", colors=GRAD_HOT,
                       text_color="#ffffff", width=170, height=44, bg=BG_PAGE, size=13,
                       command=lambda: self.show_screen("enroll")).pack(side="left")

        sbox = ctk.CTkFrame(top, fg_color=CARD_2, corner_radius=14, border_color=BORDER,
                            border_width=1, height=44)
        sbox.pack(side="left", padx=14)
        sbox.pack_propagate(False)
        ctk.CTkLabel(sbox, image=icon_ctk("search", 18, MUTED), text="").pack(
            side="right", padx=(8, 12))
        self._search_entry = ctk.CTkEntry(
            sbox, placeholder_text="جستجو: نام، شمارهٔ کارت، دانشکده…", width=300, height=34,
            justify="right", font=load_font(12), fg_color="transparent", border_width=0,
            text_color=TEXT, placeholder_text_color=DIM)
        self._search_entry.pack(side="right", padx=(10, 0))
        self._search_entry.bind("<KeyRelease>", lambda e: self._refresh_people())
        self._search_entry.bind("<FocusIn>", lambda e: sbox.configure(border_color=CYAN), add="+")
        self._search_entry.bind("<FocusOut>", lambda e: sbox.configure(border_color=BORDER), add="+")

        self._people_list = ctk.CTkScrollableFrame(
            scr, fg_color=CARD, border_color=BORDER, border_width=1, corner_radius=22,
            scrollbar_button_color=CARD_3, scrollbar_button_hover_color=BORDER_HI)
        self._people_list.grid(row=1, column=0, sticky="nsew")
        return scr

    @staticmethod
    def _hover_border(row, on_color=BORDER_HI, off_color=BORDER):
        """Highlight a row's border while the pointer is anywhere inside it."""
        def inside():
            try:
                px, py = row.winfo_pointerxy()
                return (row.winfo_rootx() <= px <= row.winfo_rootx() + row.winfo_width() and
                        row.winfo_rooty() <= py <= row.winfo_rooty() + row.winfo_height())
            except tk.TclError:
                return False

        def enter(_e=None):
            row.configure(border_color=on_color)

        def leave(_e=None):
            row.after(30, lambda: row.configure(border_color=on_color if inside() else off_color)
                      if row.winfo_exists() else None)

        row.bind("<Enter>", enter, add="+")
        row.bind("<Leave>", leave, add="+")
        for child in row.winfo_children():
            child.bind("<Enter>", enter, add="+")
            child.bind("<Leave>", leave, add="+")

    def _refresh_people(self):
        for widget in self._people_list.winfo_children():
            widget.destroy()
        search = self._search_entry.get().strip() if hasattr(self, "_search_entry") else ""
        people = self.db.list_people(search)
        self._people_count_lbl.configure(text=f"{len(people)} نفر")
        if not people:
            empty = ctk.CTkFrame(self._people_list, fg_color="transparent")
            empty.pack(pady=40)
            glyph = make_glow_glyph(120, alpha=80, bg=CARD)
            ctk.CTkLabel(empty, image=ctk.CTkImage(glyph, glyph, glyph.size), text="").pack()
            ctk.CTkLabel(empty, text="نتیجه‌ای پیدا نشد" if search else "هنوز کسی ثبت نشده",
                         font=load_font(16, True), text_color=TEXT).pack(pady=(8, 2))
            ctk.CTkLabel(empty, text="عبارت دیگری را جستجو کنید" if search else
                         "اولین نفر را با اسکن چند‌زاویه‌ای ثبت کنید",
                         font=load_font(11), text_color=MUTED).pack(pady=(0, 14))
            if not search:
                GradientButton(empty, text="افزودن فرد", icon="user_plus", colors=GRAD_MAIN,
                               text_color=ON_CYAN, width=170, height=46, bg=CARD, size=13,
                               command=lambda: self.show_screen("enroll")).pack()
            return

        for person in people:
            row = ctk.CTkFrame(self._people_list, fg_color=CARD_2, corner_radius=20, height=92,
                               border_color=BORDER, border_width=1)
            row.pack(fill="x", padx=10, pady=6)
            row.pack_propagate(False)
            ctk.CTkLabel(row, image=self._get_photo_ctk(person, 68, CARD_2), text="").pack(
                side="right", padx=(16, 10), pady=10)

            info = ctk.CTkFrame(row, fg_color="transparent")
            info.pack(side="right", fill="both", expand=True, pady=12)
            ctk.CTkLabel(info, text=person["full_name"], font=load_font(15, True),
                         text_color=TEXT, anchor="e").pack(fill="x")
            pills = ctk.CTkFrame(info, fg_color="transparent")
            pills.pack(anchor="e", pady=(6, 0))
            if person.get("student_id"):
                self._pill(pills, person["student_id"], CYAN, CYAN_BG).pack(side="right", padx=(0, 6))
            if person.get("department"):
                self._pill(pills, person["department"], VIOLET_HI, VIOLET_BG).pack(
                    side="right", padx=(0, 6))
            self._pill(pills, f"{person.get('face_count', 0)} نمونه", GOOD, GOOD_BG).pack(
                side="right", padx=(0, 6))

            ctk.CTkButton(row, text="حذف", width=96, height=38, font=load_font(11, True),
                          image=icon_ctk("trash", 16, BAD), compound="right",
                          fg_color="transparent", hover_color=BAD_BG, text_color=BAD,
                          border_width=1, border_color=BAD_BG, corner_radius=12,
                          command=lambda p=person: self._delete_person(p)).pack(
                side="left", padx=16)
            self._hover_border(row)

    def _delete_person(self, person):
        ok = messagebox.askyesno(
            "حذف فرد",
            f"«{person['full_name']}» و تمام {person.get('face_count', 0)} نمونهٔ چهره‌اش حذف می‌شود.\n"
            "این عمل برگشت‌ناپذیر است. ادامه می‌دهید؟")
        if not ok:
            return
        self.db.delete_person(person["id"])
        self._photo_cache.clear()
        self._refresh_mean_embeddings()
        self._refresh_people()
        self._refresh_people_count()
        self._status(f"«{person['full_name']}» حذف شد", ok=False)
        self._toast("فرد حذف شد", ok=False)

    # =====================================================
    # SCREEN 4: REPORT
    # =====================================================

    REPORT_COLS = ((4, "#", 44), (3, "نام", 0), (2, "شمارهٔ کارت", 150), (1, "دانشکده", 170),
                   (0, "زمان ثبت", 190))
    REPORT_MAX_ROWS = 400

    def _build_report(self):
        scr = ctk.CTkFrame(self.content, fg_color="transparent")
        scr.grid_columnconfigure(0, weight=1)
        scr.grid_rowconfigure(1, weight=1)

        top = ctk.CTkFrame(scr, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        self._page_title(top, "گزارش حضور", "report", GRAD_GOOD).pack(side="right")
        count_pill = self._pill(top, "۰ رکورد", CYAN, CYAN_BG, 11)
        count_pill.pack(side="right", padx=12)
        self._report_count = count_pill.winfo_children()[0]

        GradientButton(top, text="خروجی CSV", icon="download", colors=GRAD_HOT,
                       text_color="#ffffff", width=170, height=44, bg=BG_PAGE, size=13,
                       command=self._export_csv).pack(side="left")
        self._report_scope = ctk.CTkSegmentedButton(
            top, values=["امروز", "همه"], font=load_font(12, True), height=40, corner_radius=12,
            fg_color=CARD_2, selected_color=VIOLET, selected_hover_color=VIOLET_HI,
            unselected_color=CARD_2, unselected_hover_color=CARD_3, text_color=TEXT,
            command=lambda v: self._refresh_report())
        self._report_scope.set("امروز")
        self._report_scope.pack(side="left", padx=14)

        self._report_list = ctk.CTkScrollableFrame(
            scr, fg_color=CARD, border_color=BORDER, border_width=1, corner_radius=22,
            scrollbar_button_color=CARD_3, scrollbar_button_hover_color=BORDER_HI)
        self._report_list.grid(row=1, column=0, sticky="nsew")
        return scr

    def _report_row_frame(self, parent, fg, height):
        frame = ctk.CTkFrame(parent, fg_color=fg, corner_radius=12, height=height)
        frame.pack(fill="x", padx=12, pady=3)
        frame.pack_propagate(False)
        inner = ctk.CTkFrame(frame, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=10)
        for col, _t, width in self.REPORT_COLS:
            inner.grid_columnconfigure(col, weight=1 if width == 0 else 0, minsize=width)
        inner.grid_rowconfigure(0, weight=1)
        return inner

    def _small_avatar(self, name, bg):
        key = ("rep", name, bg)
        if key not in self._photo_cache:
            img = make_avatar(name, 60, True, bg)
            self._photo_cache[key] = ctk.CTkImage(img, img, (30, 30))
        return self._photo_cache[key]

    def _refresh_report(self):
        for widget in self._report_list.winfo_children():
            widget.destroy()
        today_only = self._report_scope.get() == "امروز"
        rows = self.db.attendance_today() if today_only else self.db.attendance_all()
        self._report_count.configure(text=f"{len(rows)} رکورد")

        header = self._report_row_frame(self._report_list, "transparent", 34)
        for col, text, _w in self.REPORT_COLS:
            ctk.CTkLabel(header, text=text, font=load_font(11, True), text_color=CYAN,
                         anchor="e").grid(row=0, column=col, sticky="e", padx=8)
        AccentLine(self._report_list, bg=CARD, height=2).pack(fill="x", padx=16, pady=(0, 4))

        if not rows:
            empty = ctk.CTkFrame(self._report_list, fg_color="transparent")
            empty.pack(pady=40)
            chip = make_chip("inbox", GRAD_MAIN, 64, CARD, 20)
            ctk.CTkLabel(empty, image=ctk.CTkImage(chip, chip, (64, 64)), text="").pack()
            ctk.CTkLabel(empty, text="رکوردی ثبت نشده است", font=load_font(15, True),
                         text_color=TEXT).pack(pady=(10, 2))
            ctk.CTkLabel(empty, text="با روشن کردن دوربین در داشبورد، حضور افراد ثبت می‌شود",
                         font=load_font(11), text_color=MUTED).pack()
            return

        shown = rows[:self.REPORT_MAX_ROWS]
        for i, r in enumerate(shown):
            bg = CARD_2 if i % 2 == 0 else CARD
            row = self._report_row_frame(self._report_list, bg, 46)
            ctk.CTkLabel(row, text=str(i + 1), font=load_font(10), text_color=DIM).grid(
                row=0, column=4, sticky="e", padx=8)
            who = ctk.CTkFrame(row, fg_color="transparent")
            who.grid(row=0, column=3, sticky="e", padx=8)
            ctk.CTkLabel(who, image=self._small_avatar(r["full_name"], bg), text="").pack(
                side="right")
            ctk.CTkLabel(who, text=r["full_name"], font=load_font(12, True), text_color=TEXT,
                         anchor="e").pack(side="right", padx=8)
            sid = r.get("student_id")
            if sid:
                self._pill(row, sid, CYAN, CYAN_BG).grid(row=0, column=2, sticky="e", padx=8)
            else:
                ctk.CTkLabel(row, text="—", font=load_font(11), text_color=DIM).grid(
                    row=0, column=2, sticky="e", padx=8)
            ctk.CTkLabel(row, text=r.get("department") or "—", font=load_font(11),
                         text_color=MUTED, anchor="e").grid(row=0, column=1, sticky="e", padx=8)
            ctk.CTkLabel(row, text=r["recorded_at"], font=load_font(11), text_color=TEXT,
                         anchor="e").grid(row=0, column=0, sticky="e", padx=8)
        if len(rows) > len(shown):
            ctk.CTkLabel(self._report_list,
                         text=f"فقط {len(shown)} رکورد اول نمایش داده شد — برای همهٔ رکوردها از «خروجی CSV» استفاده کنید",
                         font=load_font(10), text_color=WARN).pack(pady=(8, 10))

    def _export_csv(self):
        today_only = self._report_scope.get() == "امروز"
        OUTPUTS_DIR.mkdir(exist_ok=True)
        default = OUTPUTS_DIR / f"attendance_{datetime.now():%Y-%m-%d}.csv"
        path = filedialog.asksaveasfilename(
            title="ذخیرهٔ گزارش حضور", defaultextension=".csv",
            initialfile=default.name, initialdir=str(OUTPUTS_DIR),
            filetypes=[("CSV", "*.csv")])
        if not path:
            return
        count = self.db.export_attendance_csv(path, today_only=today_only)
        self._toast(f"{count} رکورد ذخیره شد")
        self._status("گزارش CSV ذخیره شد")

    # =====================================================
    # SCREEN 5: SETTINGS & HEALTH
    # =====================================================

    def _build_settings(self):
        scr = ctk.CTkScrollableFrame(self.content, fg_color="transparent",
                                     scrollbar_button_color=CARD_3,
                                     scrollbar_button_hover_color=BORDER_HI)
        scr.grid_columnconfigure(0, weight=1, uniform="set")
        scr.grid_columnconfigure(1, weight=1, uniform="set")

        self._page_title(scr, "تنظیمات و سلامت سیستم", "settings", GRAD_HOT).grid(
            row=0, column=0, columnspan=2, sticky="e", pady=(2, 12), padx=6)

        # --- camera & threshold (right column) ---
        card = self._card(scr, 22)
        card.grid(row=1, column=1, sticky="nsew", padx=(8, 6), pady=6)
        self._card_header(card, "دوربین و آستانهٔ تطبیق", "منبع تصویر و میزان سخت‌گیری شناسایی",
                          "camera", GRAD_MAIN)

        row1 = ctk.CTkFrame(card, fg_color="transparent")
        row1.pack(fill="x", padx=22, pady=(10, 4))
        ctk.CTkLabel(row1, text="شمارهٔ دوربین", font=load_font(12, True),
                     text_color=TEXT).pack(side="right")
        saved_cam = self.db.get_setting("camera_index", "0")
        cam_menu = ctk.CTkOptionMenu(
            row1, values=["0", "1", "2", "3"], width=96, height=36, corner_radius=12,
            fg_color=CARD_2, button_color=VIOLET, button_hover_color=VIOLET_HI,
            dropdown_fg_color=CARD_2, dropdown_hover_color=CARD_3, dropdown_text_color=TEXT,
            text_color=TEXT, font=load_font(12, True),
            command=lambda v: self.db.set_setting("camera_index", v))
        cam_menu.set(saved_cam)
        cam_menu.pack(side="left")

        row2 = ctk.CTkFrame(card, fg_color="transparent")
        row2.pack(fill="x", padx=22, pady=(16, 6))
        ctk.CTkLabel(row2, text="آستانهٔ تطبیق (سخت‌گیری)", font=load_font(12, True),
                     text_color=TEXT).pack(side="right")
        thr_pill = self._pill(row2, "0.36", CYAN, CYAN_BG, 12)
        thr_pill.pack(side="left")
        self._threshold_label = thr_pill.winfo_children()[0]
        saved_thr = float(self.db.get_setting("match_threshold", str(COSINE_THRESHOLD)))
        slider = ctk.CTkSlider(card, from_=0.20, to=0.60, number_of_steps=40, height=20,
                               progress_color=CYAN, button_color=CYAN_HI,
                               button_hover_color="#ffffff", fg_color=CARD_3,
                               command=self._on_threshold_change)
        slider.set(saved_thr)
        slider.pack(fill="x", padx=22, pady=(2, 2))
        self._threshold_label.configure(text=f"{saved_thr:.2f}")
        caps = ctk.CTkFrame(card, fg_color="transparent")
        caps.pack(fill="x", padx=22)
        ctk.CTkLabel(caps, text="سخت‌گیر ›", font=load_font(10), text_color=MUTED).pack(side="right")
        ctk.CTkLabel(caps, text="‹ آسان‌گیر", font=load_font(10), text_color=MUTED).pack(side="left")
        ctk.CTkLabel(card, text="مقدار پیشنهادی برای SFace: 0.36 — بالاتر = سخت‌گیرتر (کمتر اشتباه می‌گیرد)",
                     font=load_font(10), text_color=DIM, wraplength=420,
                     justify="right").pack(anchor="e", padx=22, pady=(8, 20))

        # --- backup (right column) ---
        card3 = self._card(scr, 22)
        card3.grid(row=2, column=1, sticky="nsew", padx=(8, 6), pady=6)
        self._card_header(card3, "پشتیبان‌گیری", "یک کپی از دیتابیس (افراد + چهره‌ها + حضور)",
                          "db", GRAD_GOLD)
        GradientButton(card3, text="ذخیرهٔ پشتیبان دیتابیس", icon="download",
                       colors=GRAD_GOLD, text_color="#2b1600", width=260, height=46, bg=CARD,
                       size=13, command=self._backup_db).pack(anchor="e", padx=22, pady=(10, 22))

        # --- health check (left column, tall) ---
        card2 = self._card(scr, 22)
        card2.grid(row=1, column=0, rowspan=2, sticky="nsew", padx=(6, 8), pady=6)
        self._card_header(card2, "بررسی سلامت پروژه", "مدل‌ها، دیتابیس و تشخیص چهره", "pulse",
                          GRAD_GOOD)
        self._health_box = ctk.CTkTextbox(card2, height=210, corner_radius=14,
                                          font=ctk.CTkFont(family="Consolas", size=12),
                                          fg_color=BG_PAGE, border_color=BORDER, border_width=1,
                                          text_color=TEXT)
        self._health_box.pack(fill="both", expand=True, padx=20, pady=(8, 4))
        for tag, col in (("pass", GOOD), ("fail", BAD), ("head", CYAN), ("note", MUTED)):
            self._health_box.tag_config(tag, foreground=col)
        self._health_box.insert("end", "برای اجرای بررسی «اجرای تست سلامت» را بزنید…", "note")
        GradientButton(card2, text="اجرای تست سلامت", icon="pulse", colors=GRAD_GOOD,
                       text_color=ON_GOOD, width=230, height=46, bg=CARD, size=13,
                       command=self._run_health_check).pack(anchor="e", padx=20, pady=(8, 20))

        # --- about (full width) ---
        card4 = self._card(scr, 22)
        card4.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=6, pady=(6, 22))
        wrap = ctk.CTkFrame(card4, fg_color="transparent")
        wrap.pack(fill="x", padx=22, pady=20)
        logo = make_app_logo(64, bg=CARD)
        ctk.CTkLabel(wrap, image=ctk.CTkImage(logo, logo, logo.size), text="").pack(side="right")
        col = ctk.CTkFrame(wrap, fg_color="transparent")
        col.pack(side="right", padx=14)
        ctk.CTkLabel(col, text="چهره‌یاب — سیستم شناسایی هوشمند چهره", font=load_font(16, True),
                     text_color=TEXT).pack(anchor="e")
        ctk.CTkLabel(col, text=f"نسخهٔ {self.VERSION}  ·  Aurora Neon  ·  پوهنځی کمپیوتر ساینس — "
                               "پوهنتون غرجستان  |  2026", font=load_font(11),
                     text_color=MUTED).pack(anchor="e", pady=(2, 0))
        ctk.CTkLabel(col, text="سازنده: Sayed Mohammad Naqi Sadat", font=load_font(11),
                     text_color=VIOLET_HI).pack(anchor="e", pady=(2, 0))
        chips = ctk.CTkFrame(wrap, fg_color="transparent")
        chips.pack(side="left")
        self._pill(chips, "SFace · 128-D Embedding", MAGENTA, PINK_BG, 10).pack(side="right", padx=4)
        self._pill(chips, "YuNet · تشخیص", CYAN, CYAN_BG, 10).pack(side="right", padx=4)
        self._pill(chips, "100٪ آفلاین", GOOD, GOOD_BG, 10).pack(side="right", padx=4)
        return scr

    def _on_threshold_change(self, value):
        self.db.set_setting("match_threshold", f"{float(value):.2f}")
        self._threshold_label.configure(text=f"{float(value):.2f}")

    def _run_health_check(self):
        box = self._health_box
        box.delete("1.0", "end")
        box.insert("end", "== Smart Face Recognition — Health Check ==\n\n", "head")

        def report(label, ok):
            box.insert("end", f"[{'PASS' if ok else 'FAIL'}] ", "pass" if ok else "fail")
            box.insert("end", f"{label}\n")

        report("YuNet model file present",
               (PROJECT_ROOT / "models/face_detection_yunet_2023mar.onnx").exists())
        report("SFace model file present",
               (PROJECT_ROOT / "models/face_recognition_sface_2021dec.onnx").exists())

        engine = None
        try:
            engine = self._get_engine()
            report("Models load into OpenCV DNN", True)
        except Exception as exc:
            report(f"Models load into OpenCV DNN ({exc})", False)

        if TEST_IMAGE.exists() and engine is not None:
            img = cv2.imread(str(TEST_IMAGE))
            faces = engine.detect(img) if img is not None else []
            report(f"Detection on test.jpg ({len(faces)} faces, expected >= 2)", len(faces) >= 2)

        for label, ok in self.db.integrity_check():
            report(label, ok)

        box.insert("end", f"\nافراد ثبت‌شده: {self.db.count_people()}   |   نمونه‌های چهره: "
                          f"{sum(self.db.get_face_count(p['id']) for p in self.db.list_people())}\n",
                   "note")
        box.insert("end", "== تمام ==\n", "head")

    def _backup_db(self):
        path = filedialog.asksaveasfilename(
            title="ذخیرهٔ پشتیبان دیتابیس", defaultextension=".db",
            initialfile=f"face_recognition_backup_{datetime.now():%Y%m%d}.db",
            filetypes=[("SQLite DB", "*.db")])
        if not path:
            return
        self.db.backup(path)
        self._toast("پشتیبان ذخیره شد")
        self._status("پشتیبان‌گیری انجام شد")

    # =====================================================
    # start / destroy
    # =====================================================

    def destroy(self):
        self._stop_dash_camera()
        self._release_enroll_camera()
        try:
            self.db.close()
        except Exception:
            pass
        super().destroy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoketest", action="store_true",
                        help="build all screens, cycle them, then exit")
    args = parser.parse_args()

    app = FaceRecognitionApp()
    if args.smoketest:
        for name in ("dashboard", "enroll", "people", "report", "settings", "dashboard"):
            app.show_screen(name)
            app.update()
        app.after(700, app.destroy)
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
