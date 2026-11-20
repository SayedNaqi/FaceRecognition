#!/usr/bin/env python3
# ============================================================
# ui_kit.py — Vector icons, gradient widgets and visual effects
# for the Smart Face Recognition GUI (Aurora v3)
# ============================================================

import math

import numpy as np
import customtkinter as ctk
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# palette
CYAN = "#22d3ee"
VIOLET = "#8b5cf6"
MAGENTA = "#e879f9"
GOOD = "#34d399"
BAD = "#fb7185"
CARD = "#1c1f42"
CARD_2 = "#242852"
BORDER = "#2d3159"
TEXT = "#efedff"
MUTED = "#8f92bb"
ON_DARK = "#052b34"
FONT = "Tahoma"


# ---------------- gradient raster helpers ----------------

def _lerp(a, b, t):
    return a + (b - a) * t


def gradient(size, c1, c2, diagonal=True):
    """RGB gradient image from two hex colors."""
    w, h = size
    a = _hex(c1)
    b = _hex(c2)
    if diagonal:
        x = np.linspace(0, 1, w, dtype="float32")
        y = np.linspace(0, 1, h, dtype="float32")
        t = (x[None, :] + y[:, None]) / 2.0
        arr = np.zeros((h, w, 3), dtype="float32")
        for i in range(3):
            arr[:, :, i] = _lerp(a[i], b[i], t)
        return Image.fromarray(arr.astype("uint8"), "RGB")
    img = Image.new("RGB", size, a)
    d = ImageDraw.Draw(img)
    for y in range(h):
        t = y / max(h - 1, 1)
        d.line([(0, y), (w, y)], fill=tuple(int(_lerp(a[i], b[i], t)) for i in range(3)))
    return img


def _hex(color):
    color = color.lstrip("#")
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


def rounded_alpha(size, radius):
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, size[0] - 1, size[1] - 1],
                                        radius=radius, fill=255)
    return m


def gradient_pill(width, height, c1=CYAN, c2=VIOLET, radius=None, shine=True):
    """Rounded gradient button face with a glossy top edge."""
    ss = 3
    w, h = width * ss, height * ss
    r = (radius or height // 2) * ss
    img = gradient((w, h), c1, c2).convert("RGBA")
    img.putalpha(rounded_alpha((w, h), r))
    if shine:
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([w * 0.04, h * 0.08, w * 0.96, h * 0.50],
                            radius=r * 0.8, outline=(255, 255, 255, 42), width=max(1, ss))
    return img.resize((width, height), Image.LANCZOS)


def ctk_img(pil, size=None):
    if size:
        pil = pil.resize(size, Image.LANCZOS)
    return ctk.CTkImage(light_image=pil, dark_image=pil, size=pil.size)


def face_glyph_local(size, color=(255, 255, 255, 255), width=None):
    """Face-ID style glyph drawn locally (no dependency on gui module)."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    w = width or max(3, size // 14)
    m, arm = int(size * 0.17), int(size * 0.16)
    for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        x = m if sx > 0 else size - m
        y = m if sy > 0 else size - m
        d.line([(x, y), (x - sx * arm, y)], fill=color, width=w)
        d.line([(x, y), (x, y - sy * arm)], fill=color, width=w)
    c = size / 2
    r = size * 0.16
    d.ellipse([c - r, c - r, c + r, c + r], outline=color, width=w)
    d.line([(c - size * 0.07, c - size * 0.03), (c - size * 0.07, c + size * 0.03)],
           fill=color, width=w)
    d.line([(c + size * 0.07, c - size * 0.03), (c + size * 0.07, c + size * 0.03)],
           fill=color, width=w)
    d.arc([c - size * 0.10, c + size * 0.02, c + size * 0.10, c + size * 0.20],
          20, 160, fill=color, width=w)
    return img


# ---------------- vector icon library ----------------

def icon(name, size=24, color="#ffffff", width=None):
    """Draw a crisp white-line vector icon. Names: see ICONS below."""
    ss = 4
    S = size * ss
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    col = _hex(color) + (255,)
    w = width or max(2, S // 22)
    k = S / 64.0

    def pts(*vals):
        if vals and isinstance(vals[0], (tuple, list)):
            return [(v[0] * k, v[1] * k) for v in vals]
        return [(vals[i] * k, vals[i + 1] * k)
                for i in range(0, len(vals) - 1, 2)]

    def line(*coords, fill=col, width=w):
        d.line(pts(*coords), fill=fill, width=width, joint="curve")

    def rect(x1, y1, x2, y2, r=0):
        box = [x1 * k, y1 * k, x2 * k, y2 * k]
        if r:
            d.rounded_rectangle(box, radius=r * k, outline=col, width=w)
        else:
            d.rectangle(box, outline=col, width=w)

    def circle(cx, cy, r, outline=col, width=w):
        d.ellipse([(cx - r) * k, (cy - r) * k, (cx + r) * k, (cy + r) * k],
                  outline=outline, width=width)

    if name == "home":
        line(8, 32, 32, 12, 56, 32)
        rect(14, 29, 50, 56, r=4)
        rect(27, 43, 37, 56, r=2)
    elif name == "add_user":
        circle(26, 23, 10)
        d.arc([10 * k, 36 * k, 42 * k, 68 * k], 180, 360, fill=col, width=w)
        line(50, 22, 50, 42)
        line(40, 32, 60, 32)
    elif name == "users":
        circle(23, 22, 9)
        d.arc([7 * k, 34 * k, 39 * k, 64 * k], 180, 360, fill=col, width=w)
        circle(46, 26, 7)
        d.arc([35 * k, 35 * k, 59 * k, 61 * k], 180, 360, fill=col, width=w)
    elif name == "report":
        rect(12, 8, 52, 56, r=5)
        line(20, 22, 44, 22)
        line(20, 32, 44, 32)
        line(20, 42, 36, 42)
    elif name == "settings":
        for y in (18, 32, 46):
            line(12, y, 52, y)
            cx = 22 + (12 if y == 32 else 0)
            d.ellipse([(cx - 6) * k, (y - 6) * k, (cx + 6) * k, (y + 6) * k],
                      fill=col)
    elif name == "camera":
        rect(7, 18, 57, 50, r=6)
        circle(32, 34, 9)
        line(20, 18, 25, 12, 39, 12, 44, 18)
    elif name == "play":
        line(22, 16, 22, 48)
        d.polygon(pts((24, 17), (46, 32), (24, 47)), fill=col)
    elif name == "stop":
        rect(18, 18, 46, 46, r=4)
    elif name == "search":
        circle(28, 28, 15)
        line(39, 39, 54, 54)
    elif name == "download":
        line(32, 10, 32, 42)
        line(21, 32, 32, 43, 43, 32)
        line(12, 52, 52, 52)
    elif name == "trash":
        line(14, 18, 50, 18)
        line(24, 18, 26, 12, 38, 12, 40, 18)
        rect(18, 18, 46, 54, r=4)
        line(26, 28, 26, 45)
        line(38, 28, 38, 45)
    elif name == "shield":
        d.line(pts((32, 7), (52, 15), (50, 38), (32, 56), (14, 38), (12, 15), (32, 7)),
               fill=col, width=w, joint="curve")
        line(23, 32, 30, 39, 43, 25)
    elif name == "pulse":
        line(7, 34, 20, 34, 26, 22, 34, 46, 40, 34, 57, 34)
    elif name == "check":
        line(14, 34, 27, 47, 51, 18)
    elif name == "sparkles":
        for cx, cy, r in ((24, 22, 11), (47, 41, 8)):
            d.polygon(pts((cx, cy - r), (cx + r * .28, cy - r * .28),
                          (cx + r, cy), (cx + r * .28, cy + r * .28),
                          (cx, cy + r), (cx - r * .28, cy + r * .28),
                          (cx - r, cy), (cx - r * .28, cy - r * .28)), fill=col)
    elif name == "clock":
        circle(32, 32, 22)
        line(32, 18, 32, 33, 43, 39)
    elif name == "database":
        d.ellipse([12 * k, 12 * k, 52 * k, 28 * k], outline=col, width=w)
        line(12, 20, 12, 46)
        line(52, 20, 52, 46)
        d.arc([12 * k, 36 * k, 52 * k, 52 * k], 0, 180, fill=col, width=w)
        d.arc([12 * k, 44 * k, 52 * k, 60 * k], 0, 180, fill=col, width=w)
    elif name == "arrow_left":
        line(54, 32, 12, 32)
        line(24, 20, 12, 32, 24, 44)
    elif name == "arrow_right":
        line(10, 32, 52, 32)
        line(40, 20, 52, 32, 40, 44)
    elif name == "arrow_up":
        line(32, 54, 32, 12)
        line(20, 24, 32, 12, 44, 24)
    elif name == "arrow_down":
        line(32, 10, 32, 52)
        line(20, 40, 32, 52, 44, 40)
    elif name == "face":
        m, a = 12 * k, 11 * k
        for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
            x = (32 + sx * 20) * k
            y = (32 + sy * 20) * k
            line(x, y, x - sx * a / k * k, y)
            line(x, y, x, y - sy * a / k * k)
        circle(32, 32, 15)
        d.line([(26 * k, 28 * k), (26 * k, 35 * k)], fill=col, width=w)
        d.line([(38 * k, 28 * k), (38 * k, 35 * k)], fill=col, width=w)
        d.arc([24 * k, 34 * k, 40 * k, 47 * k], 20, 160, fill=col, width=w)
    elif name == "faceid":
        return face_glyph_local(size, col, width or max(3, size // 14))
    else:
        circle(32, 32, 18)
    return img.resize((size, size), Image.LANCZOS)


# ---------------- gradient button ----------------

class GradientButton(ctk.CTkButton):
    """CTkButton with a gradient pill image, glossy shine and hover/disabled states."""

    def __init__(self, master, text="", c1=CYAN, c2=VIOLET, width=180, height=44,
                 text_color=TEXT, icon_name=None, font=None, command=None,
                 compound="left", **kwargs):
        self._c1, self._c2 = c1, c2
        self._w, self._h = width, height
        self._disabled = False
        self._base = ctk_img(gradient_pill(width, height, c1, c2))
        self._hover = ctk_img(gradient_pill(width, height, c1, c2, shine=True))
        self._off = ctk_img(gradient_pill(width, height, "#3a3e63", "#4a4f78",
                                           shine=False))
        self._icon = ctk_img(icon(icon_name, 20, text_color)) if icon_name else None
        super().__init__(
            master, text=text, image=self._base, compound=compound,
            fg_color=CARD, hover_color=CARD,
            text_color=text_color, width=width, height=height,
            corner_radius=height // 2, font=font, command=command,
            **kwargs)
        self.bind("<Enter>", lambda e: self._set_image(self._hover))
        self.bind("<Leave>", lambda e: self._set_image(
            self._off if self._disabled else self._base))

    def _set_image(self, img):
        if not self._disabled:
            self.configure(image=img)

    def set_gradient(self, c1, c2):
        """Swap the button face (used by play/stop toggles)."""
        self._c1, self._c2 = c1, c2
        self._base = ctk_img(gradient_pill(self._w, self._h, c1, c2))
        self._hover = ctk_img(gradient_pill(self._w, self._h, c1, c2))
        self.configure(image=self._base)

    def set_disabled(self, disabled):
        self._disabled = bool(disabled)
        self.configure(state="disabled" if disabled else "normal",
                       image=self._off if disabled else self._base,
                       text_color=(MUTED if disabled else self._text_color_normal))
        if disabled:
            self.unbind("<Enter>")
            self.unbind("<Leave>")
        else:
            self.bind("<Enter>", lambda e: self._set_image(self._hover))
            self.bind("<Leave>", lambda e: self._set_image(self._base))

    @property
    def _text_color_normal(self):
        return getattr(self, "_saved_text_color", TEXT)


# ---------------- toast with fade animation ----------------

class Toast(ctk.CTkToplevel):
    def __init__(self, master, message, ok=True, width=360, height=50):
        super().__init__(master)
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        self.configure(fg_color=(GOOD if ok else BAD))
        x = master.winfo_rootx() + master.winfo_width() // 2 - width // 2
        y = master.winfo_rooty() + 20
        self.geometry(f"{width}x{height}+{x}+{y}")
        ctk.CTkLabel(self, text=message, text_color="#0b1b16",
                     font=ctk.CTkFont(family=FONT, size=13, weight="bold")).pack(expand=True)
        self._alpha = 0.0
        self._fade_in()

    def _fade_in(self):
        self._alpha = min(0.96, self._alpha + 0.16)
        try:
            self.attributes("-alpha", self._alpha)
        except Exception:
            pass
        if self._alpha < 0.96:
            self.after(28, self._fade_in)
        else:
            self.after(1500, self._fade_out)

    def _fade_out(self):
        self._alpha -= 0.14
        try:
            self.attributes("-alpha", max(0.0, self._alpha))
        except Exception:
            pass
        if self._alpha > 0:
            self.after(28, self._fade_out)
        else:
            self.destroy()


# ---------------- circular match gauge ----------------

def match_gauge(value, size=112):
    """Semicircle gauge with a colored arc and a numeric percentage."""
    ss = 4
    S = size * ss
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pad = S * 0.10
    box = [pad, pad, S - pad, S - pad]
    d.arc(box, 180, 360, fill=(45, 49, 89, 255), width=int(S * 0.075))
    v = max(0.0, min(1.0, value))
    end = 180 + 180 * v
    steps = max(1, int(40 * v))
    for i in range(steps):
        a1 = 180 + 180 * v * i / steps
        a2 = 180 + 180 * v * (i + 1) / steps + 0.6
        t = i / max(steps - 1, 1)
        col = (int(_lerp(139, 34, t)), int(_lerp(92, 211, t)), int(_lerp(246, 238, t)), 255)
        d.arc(box, a1, a2, fill=col, width=int(S * 0.075))
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/tahomabd.ttf", int(S * 0.30))
        small = ImageFont.truetype("C:/Windows/Fonts/tahoma.ttf", int(S * 0.12))
    except OSError:
        font = small = ImageFont.load_default()
    txt = f"{v * 100:.0f}%"
    bbox = d.textbbox((0, 0), txt, font=font)
    d.text(((S - (bbox[2] - bbox[0])) / 2 - bbox[0], S * 0.50),
           txt, font=font, fill=(239, 237, 255, 255))
    bbox2 = d.textbbox((0, 0), "MATCH", font=small)
    d.text(((S - (bbox2[2] - bbox2[0])) / 2 - bbox2[0], S * 0.79),
           "MATCH", font=small, fill=(143, 146, 187, 255))
    return img.resize((size, size), Image.LANCZOS)


# ---------------- animated hero banner ----------------

def hero_frame(width, height, phase=0):
    """A glowing banner frame (Latin text only — Persian is overlaid by Tk)."""
    import numpy as np
    x = np.linspace(0, 1, width, dtype="float32")
    y = np.linspace(0, 1, height, dtype="float32")
    t = (x[None, :] + y[:, None]) / 2.0
    c1, c2 = _hex("#15173a"), _hex("#0e5b78")
    arr = np.zeros((height, width, 3), dtype="float32")
    for i in range(3):
        arr[:, :, i] = _lerp(c1[i], c2[i], t)
    img = Image.fromarray(arr.astype("uint8"), "RGB").convert("RGBA")
    img.putalpha(rounded_alpha((width, height), 22))

    glow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    off = math.sin(phase * 0.7) * 40
    gd.ellipse([width * 0.58 + off, -height * 0.65, width * 1.0 + off, height * 1.15],
               fill=(139, 92, 246, 125))
    gd.ellipse([-width * 0.08 - off, height * 0.20, width * 0.30 - off, height * 1.6],
               fill=(34, 211, 238, 105))
    gd.ellipse([width * 0.36, height * 0.9, width * 0.62, height * 1.5],
               fill=(232, 121, 249, 70))
    glow = glow.filter(ImageFilter.GaussianBlur(48))
    img.alpha_composite(glow)
    d = ImageDraw.Draw(img)

    # floating indicator dots
    for i in range(3):
        px = int(width * 0.86 + i * 16)
        py = int(height * (0.30 + 0.15 * i) + math.sin(phase + i) * 6)
        d.ellipse([px, py, px + 7, py + 7], fill=(34, 211, 238, 150 - i * 35))

    # watermark face glyph
    gsize = int(height * 0.72)
    g = face_glyph_local(gsize, (255, 255, 255, 42))
    img.alpha_composite(g, (width - gsize - 24, (height - gsize) // 2))

    try:
        small = ImageFont.truetype("C:/Windows/Fonts/tahoma.ttf", 13)
    except OSError:
        small = ImageFont.load_default()
    d.text((26, height - 26), "YUNET + SFACE   ·   OFFLINE   ·   GHORJISTAN UNIVERSITY",
           font=small, fill=(154, 201, 220, 255))
    d.line([(26, height - 34), (150, height - 34)], fill=(34, 211, 238, 255), width=3)
    return img.convert("RGB")


class AnimatedHero(ctk.CTkLabel):
    """Hero label that cycles pre-rendered glow frames and resizes with debounce."""

    def __init__(self, master, width=900, height=118):
        self._width, self._height = width, height
        self._frames = [ctk_img(hero_frame(width, height, i), (width, height))
                        for i in range(6)]
        self._phase = 0
        super().__init__(master, image=self._frames[0], text="")
        self._loop()
        self._resize_job = None
        self.bind("<Configure>", self._on_resize)

    def _loop(self):
        self._phase = (self._phase + 1) % len(self._frames)
        try:
            self.configure(image=self._frames[self._phase])
        except Exception:
            pass
        self.after(180, self._loop)

    def _on_resize(self, event):
        w = max(620, min(event.width, 1500))
        if abs(w - self._width) < 24:
            return
        self._width = w
        if self._resize_job:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(180, self._rebuild)

    def _rebuild(self):
        self._frames = [ctk_img(hero_frame(self._width, self._height, i),
                                (self._width, self._height)) for i in range(6)]
        self.configure(image=self._frames[self._phase])
