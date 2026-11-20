#!/usr/bin/env python3
# ============================================================
# چهره‌یاب (Chehrahyab) — Smart Face Recognition System
# Professional GUI · Faculty of Computer Science — Ghorjistan University
#
# Design: "Aurora" theme — deep midnight indigo, neon cyan → violet
#         → magenta accents, glassy cards, custom-drawn PIL artwork.
#
# Run:  python src/gui.py            (normal)
#       python src/gui.py --smoketest  (build all screens then exit)
# ============================================================

import argparse
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from db import Database, PROFILES_DIR
from face_engine import FaceEngine, COSINE_THRESHOLD, models_present, cosine_sim
from ui_kit import (
    icon, ctk_img, GradientButton, Toast, match_gauge,
    AnimatedHero, gradient_pill,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_IMAGE = PROJECT_ROOT / "data" / "images" / "test.jpg"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# ---------- AURORA palette ----------
BG_PAGE   = "#0e1027"   # app background  (deep midnight indigo)
SIDEBAR   = "#131431"   # sidebar background
SIDEBAR_2 = "#1a1b3c"   # sidebar hover
CARD      = "#1c1f42"   # card background
CARD_2    = "#242852"   # elevated card / inputs
BORDER    = "#2d3159"   # card border
TEXT      = "#efedff"   # primary text (near-white lavender)
MUTED     = "#8f92bb"   # secondary text
CYAN      = "#22d3ee"   # primary accent
VIOLET    = "#8b5cf6"   # secondary accent
MAGENTA   = "#e879f9"   # highlight accent
GOOD      = "#34d399"   # emerald
BAD       = "#fb7185"   # rose
WARN      = "#fbbf24"   # amber
ON_CYAN   = "#052b34"   # dark text on cyan
FONT      = "Tahoma"

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


def load_font(size, bold=False):
    return ctk.CTkFont(family=FONT, size=size, weight="bold" if bold else "normal")


# =====================================================
# PIL artwork helpers (AURORA)
# =====================================================

def _lerp(a, b, t):
    return a + (b - a) * t


def _diag_gradient(size, c1, c2):
    """Diagonal linear gradient c1 → c2 as numpy array (RGB)."""
    w, h = size
    x = np.linspace(0, 1, w, dtype=np.float32)
    y = np.linspace(0, 1, h, dtype=np.float32)
    t = (x[None, :] + y[:, None]) / 2.0
    img = np.zeros((h, w, 3), dtype=np.float32)
    for i in range(3):
        img[:, :, i] = _lerp(c1[i], c2[i], t)
    return img.astype(np.uint8)


def _rounded_alpha(size, radius):
    m = Image.new("L", size, 0)
    d = ImageDraw.Draw(m)
    d.rounded_rectangle([0, 0, size[0] - 1, size[1] - 1], radius=radius, fill=255)
    return m


def draw_face_id_glyph(size, color=(255, 255, 255, 255), width=None):
    """Original face-ID-style glyph (corner brackets + face) on transparency."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    w = width or max(4, size // 15)
    m = int(size * 0.16)          # bracket margin
    arm = int(size * 0.17)        # bracket arm length
    s = size

    def bracket(x, y, dx, dy):
        d.line([(x, y), (x + dx * arm, y)], fill=color, width=w)
        d.line([(x, y), (x, y + dy * arm)], fill=color, width=w)

    bracket(m, m, +1, +1)                     # top-left
    bracket(s - m, m, -1, +1)                 # top-right
    bracket(m, s - m, +1, -1)                 # bottom-left
    bracket(s - m, s - m, -1, -1)             # bottom-right

    c = s / 2
    r = s * 0.155
    d.ellipse([c - r, c - r, c + r, c + r], outline=color, width=w)

    eye_h = s * 0.042
    eye_dy = s * 0.038
    lx, rx = c - s * 0.075, c + s * 0.075
    for ex in (lx, rx):
        d.line([(ex, c - eye_dy - eye_h), (ex, c - eye_dy + eye_h)], fill=color, width=w)

    d.line([(c, c + s * 0.01), (c, c + s * 0.075)], fill=color, width=w)
    d.arc([c - s * 0.10, c + s * 0.02, c + s * 0.10, c + s * 0.22],
          start=25, end=155, fill=color, width=w)
    return img


def make_app_logo(size=96):
    """Supersampled gradient tile with the glyph — crisp at any size."""
    ss = 4
    S = size * ss
    grad = _diag_gradient((S, S), (34, 211, 238), (232, 121, 249))
    tile = Image.fromarray(grad, "RGB").convert("RGBA")
    tile.putalpha(_rounded_alpha((S, S), int(S * 0.24)))
    glyph = draw_face_id_glyph(int(S * 0.78), (255, 255, 255, 255), int(S * 0.018))
    pos = ((S - glyph.width) // 2, (S - glyph.height) // 2)
    tile.alpha_composite(glyph, pos)
    return tile.resize((size, size), Image.LANCZOS)


def make_glow_glyph(size=170, alpha=90):
    return draw_face_id_glyph(size, (34, 211, 238, alpha), max(4, size // 13))


def make_avatar(name, size=96, ring=True):
    """Gradient-ring initials avatar."""
    ss = 4
    S = size * ss
    parts = [p for p in name.split() if p]
    initials = (parts[0][0] + (parts[-1][0] if len(parts) > 1 else "")).upper() if parts else "؟"
    if ring:
        img = Image.fromarray(_diag_gradient((S, S), (34, 211, 238), (232, 121, 249))).convert("RGBA")
        inner = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        d = ImageDraw.Draw(inner)
        d.ellipse([S * 0.09, S * 0.09, S * 0.91, S * 0.91], fill=(26, 27, 60, 255))
        img.alpha_composite(inner)
    else:
        img = Image.new("RGBA", (S, S), (26, 27, 60, 255))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/tahomabd.ttf", int(S * 0.40))
    except OSError:
        font = ImageFont.load_default()
    bbox = d.textbbox((0, 0), initials, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text(((S - tw) / 2 - bbox[0], (S - th) / 2 - bbox[1]), initials,
           fill=(239, 237, 255, 255), font=font)
    return img.resize((size, size), Image.LANCZOS).convert("RGB")


def photo_or_avatar(person, size=96):
    photo = person.get("photo_path")
    if photo and Path(photo).exists():
        try:
            return Image.open(photo).convert("RGB").resize((size, size))
        except OSError:
            pass
    return make_avatar(person.get("full_name", "؟"), size)


def make_hero_banner(width=940, height=128):
    """Latin-only hero banner (PIL cannot shape Persian script)."""
    grad = _diag_gradient((width, height), (23, 24, 58), (13, 116, 144))
    img = Image.fromarray(grad, "RGB").convert("RGBA")
    img.putalpha(_rounded_alpha((width, height), 26))
    d = ImageDraw.Draw(img)
    # glow orbs
    glow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.ellipse([width * 0.62, -height * 0.7, width * 1.02, height * 1.2],
               fill=(139, 92, 246, 110))
    gd.ellipse([-width * 0.10, height * 0.25, width * 0.28, height * 1.7],
               fill=(34, 211, 238, 90))
    glow = glow.filter(ImageFilter.GaussianBlur(60))
    img.alpha_composite(glow)
    d = ImageDraw.Draw(img)
    # glyph watermark
    glyph = draw_face_id_glyph(int(height * 0.82), (255, 255, 255, 46))
    img.alpha_composite(glyph, (width - int(height * 0.98), int(height * 0.09)))
    # Latin wordmarks (PIL-safe)
    try:
        big = ImageFont.truetype("C:/Windows/Fonts/tahomabd.ttf", 34)
        small = ImageFont.truetype("C:/Windows/Fonts/tahoma.ttf", 15)
    except OSError:
        big = small = ImageFont.load_default()
    d.text((30, 30), "FACE RECOGNITION", font=big, fill=(239, 237, 255, 255))
    d.text((30, 76), "YuNet + SFace  ·  Offline  ·  Ghorjistan University",
           font=small, fill=(154, 201, 220, 255))
    d.line([(30, 66), (168, 66)], fill=(34, 211, 238, 255), width=3)
    return img.convert("RGB")


def make_ctk_image(pil_img, size=None):
    if size:
        pil_img = pil_img.resize(size, Image.LANCZOS)
    return ctk.CTkImage(light_image=pil_img, dark_image=pil_img,
                        size=pil_img.size)


# =====================================================
# COLLAPSIBLE HELPER WIDGETS
# =====================================================

class NavButton(ctk.CTkFrame):
    """Sidebar item: animated accent bar + vector icon + label."""

    def __init__(self, master, icon_name, label, command):
        super().__init__(master, fg_color="transparent", height=48)
        self.grid_columnconfigure(1, weight=1)
        self.grid_propagate(False)
        self.bar = ctk.CTkFrame(self, width=4, height=30, corner_radius=2,
                                fg_color="transparent")
        self.bar.grid(row=0, column=0, sticky="ns", padx=(0, 8), pady=9)
        self.bar.grid_propagate(False)
        self._ic = ctk_img(icon(icon_name, 22, TEXT))
        self.button = ctk.CTkButton(self, image=self._ic, text=f"   {label}",
                                    compound="left", font=load_font(14), height=46,
                                    anchor="w", fg_color="transparent",
                                    hover_color=SIDEBAR_2, text_color=TEXT,
                                    corner_radius=12, command=command)
        self.button.grid(row=0, column=1, sticky="nsew")

    def set_active(self, active):
        self.bar.configure(fg_color=(CYAN if active else "transparent"))
        self.button.configure(fg_color=(SIDEBAR_2 if active else "transparent"),
                              text_color=(CYAN if active else TEXT))
        if active:  # tiny pulse animation on the accent bar
            self._pulse(6)

    def _pulse(self, width):
        try:
            self.bar.configure(width=width)
            if width > 4:
                self.after(70, lambda: self._pulse(width - 1))
        except Exception:
            pass


class StatCard(ctk.CTkFrame):
    def __init__(self, master, icon_name, title, accent):
        super().__init__(master, fg_color=CARD, border_color=BORDER,
                         border_width=1, corner_radius=18)
        self.grid_columnconfigure(0, weight=1)
        chip = ctk.CTkFrame(self, fg_color=CARD_2, corner_radius=12,
                            width=44, height=44)
        chip.grid(row=0, column=0, sticky="e", padx=12, pady=(12, 2))
        chip.grid_propagate(False)
        ctk.CTkLabel(chip, image=ctk_img(icon(icon_name, 22, accent)),
                     text="").place(relx=0.5, rely=0.5, anchor="center")
        self.value = ctk.CTkLabel(self, text="0", font=load_font(26, True),
                                  text_color=accent)
        self.value.grid(row=1, column=0, sticky="e", padx=14)
        ctk.CTkLabel(self, text=title, font=load_font(10),
                     text_color=MUTED).grid(row=2, column=0, sticky="e",
                                            padx=14, pady=(0, 12))


# =====================================================
# MAIN APP
# =====================================================

class FaceRecognitionApp(ctk.CTk):
    VERSION = "2.1.0"

    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.title("چهره‌یاب | Face Recognition — پوهنتون غرجستان")
        self.geometry("1240x760")
        self.minsize(1080, 680)
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

        self._build_layout()
        self._refresh_mean_embeddings()
        self.show_screen("dashboard")
        self._tick_clock()

    # =====================================================
    # LAYOUT — sidebar + content + statusbar
    # =====================================================

    def _build_layout(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ---------- sidebar ----------
        sidebar = ctk.CTkFrame(self, width=250, fg_color=SIDEBAR, corner_radius=0)
        sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew")
        sidebar.grid_propagate(False)
        sidebar.grid_rowconfigure(9, weight=1)
        sidebar.grid_columnconfigure(0, weight=1)

        logo_row = ctk.CTkFrame(sidebar, fg_color="transparent")
        logo_row.grid(row=0, column=0, sticky="ew", padx=20, pady=(26, 2))
        logo_img = make_ctk_image(make_app_logo(84))
        ctk.CTkLabel(logo_row, image=logo_img, text="").pack(side="right")
        word = ctk.CTkFrame(logo_row, fg_color="transparent")
        word.pack(side="right", padx=(0, 12), pady=6)
        ctk.CTkLabel(word, text="چهره‌یاب", font=load_font(25, True),
                     text_color=TEXT).pack(anchor="e")
        ctk.CTkLabel(word, text="Face Recognition", font=load_font(10),
                     text_color=CYAN).pack(anchor="e")
        ctk.CTkFrame(sidebar, height=2, fg_color=BORDER).grid(
            row=1, column=0, sticky="ew", padx=22, pady=(12, 16))

        nav = [
            ("dashboard", "home",      "داشبورد زنده"),
            ("enroll",    "add_user",  "ثبت فرد جدید"),
            ("people",    "users",     "افراد ثبت‌شده"),
            ("report",    "report",    "گزارش حضور"),
            ("settings",  "settings",  "تنظیمات و سلامت"),
        ]
        self._nav_buttons = []
        for i, (key, icon_name, label) in enumerate(nav):
            btn = NavButton(sidebar, icon_name, label,
                            command=lambda k=key: self.show_screen(k))
            btn.grid(row=2 + i, column=0, sticky="ew", padx=12, pady=3)
            self._nav_buttons.append((key, btn))

        # engine card (bottom)
        engine_card = ctk.CTkFrame(sidebar, fg_color=CARD, corner_radius=16,
                                   border_color=BORDER, border_width=1)
        engine_card.grid(row=10, column=0, sticky="ew", padx=16, pady=(6, 12))
        row = ctk.CTkFrame(engine_card, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(10, 2))
        ctk.CTkLabel(row, text="⚡", font=ctk.CTkFont(size=17)).pack(side="right")
        ctk.CTkLabel(row, text="موتور: YuNet + SFace", font=load_font(11, True),
                     text_color=TEXT).pack(side="right", padx=6)
        row2 = ctk.CTkFrame(engine_card, fg_color="transparent")
        row2.pack(fill="x", padx=12, pady=(0, 10))
        dot = ctk.CTkFrame(row2, width=10, height=10, corner_radius=5,
                           fg_color=GOOD)
        dot.pack(side="right", padx=(0, 6), pady=4)
        dot.pack_propagate(False)
        ctk.CTkLabel(row2, text="آفلاین و آماده به کار", font=load_font(10),
                     text_color=MUTED).pack(side="right")

        ctk.CTkLabel(sidebar, text=f"نسخهٔ {self.VERSION} — Aurora",
                     font=load_font(9), text_color=MUTED).grid(row=11, column=0,
                                                               pady=(0, 14))

        # ---------- content ----------
        self.content = ctk.CTkFrame(self, fg_color=BG_PAGE, corner_radius=0)
        self.content.grid(row=0, column=1, sticky="nsew")

        # ---------- status bar ----------
        status = ctk.CTkFrame(self, height=32, fg_color=SIDEBAR, corner_radius=0)
        status.grid(row=1, column=1, sticky="ew")
        ctk.CTkFrame(self, height=1, fg_color=MAGENTA).grid(row=1, column=1,
                                                              sticky="new")
        self.status_label = ctk.CTkLabel(status, text="●  آماده", text_color=GOOD,
                                         font=load_font(11, True))
        self.status_label.pack(side="right", padx=16)
        self.clock_label = ctk.CTkLabel(status, text="", text_color=MUTED,
                                        font=load_font(10))
        self.clock_label.pack(side="left", padx=16)
        ctk.CTkLabel(status, text="پوهنځی کمپیوتر ساینس — پوهنتون غرجستان",
                     text_color=MUTED, font=load_font(10)).pack(side="left",
                                                                padx=(4, 0))

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
                frame.grid(row=0, column=0, sticky="nsew", padx=16, pady=16)
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

    def _tick_clock(self):
        try:
            self.clock_label.configure(
                text=datetime.now().strftime("%Y/%m/%d   %H:%M:%S"))
        except Exception:
            pass
        self.after(1000, self._tick_clock)

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
        # Floating toast with fade-in / fade-out animation
        Toast(self, message, ok)

    def _status(self, text, ok=True):
        self.status_label.configure(text=f"●  {text}",
                                    text_color=(GOOD if ok else BAD))

    @staticmethod
    def _frame_to_ctk(frame, width, height):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((width, height))
        return ctk.CTkImage(light_image=img, dark_image=img, size=(width, height))

    def _get_photo_ctk(self, person, size=96):
        key = (person["id"], person.get("photo_path"), size)
        if key not in self._photo_cache:
            img = photo_or_avatar(person, size)
            self._photo_cache[key] = ctk.CTkImage(light_image=img, dark_image=img,
                                                  size=(size, size))
        return self._photo_cache[key]

    # =====================================================
    # SCREEN 1: DASHBOARD
    # =====================================================

    def _build_dashboard(self):
        scr = ctk.CTkFrame(self.content, fg_color="transparent")
        scr.grid_rowconfigure(1, weight=1)
        scr.grid_columnconfigure(0, weight=3)
        scr.grid_columnconfigure(1, weight=2)

        # ---- animated hero banner (dynamic glow + Persian overlay) ----
        hero_box = ctk.CTkFrame(scr, fg_color="transparent", height=118)
        hero_box.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        hero_box.grid_propagate(False)
        self._hero = AnimatedHero(hero_box, 940, 118)
        self._hero.pack(fill="x")
        ctk.CTkLabel(hero_box, text="به چهره‌یاب خوش آمدید",
                     font=load_font(19, True), text_color=TEXT).place(
            relx=0.955, rely=0.30, anchor="e")
        ctk.CTkLabel(hero_box,
                     text="ثبت چهره   ·   شناسایی زنده   ·   حضور خودکار",
                     font=load_font(11), text_color="#bae6fd").place(
            relx=0.955, rely=0.56, anchor="e")

        # ---- camera card ----
        cam_card = ctk.CTkFrame(scr, fg_color=CARD, border_color=BORDER,
                                border_width=1, corner_radius=20)
        cam_card.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        cam_card.grid_rowconfigure(1, weight=1)
        cam_card.grid_columnconfigure(0, weight=1)

        head = ctk.CTkFrame(cam_card, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", padx=18, pady=(14, 4))
        title_box = ctk.CTkFrame(head, fg_color="transparent")
        title_box.pack(side="right")
        ctk.CTkLabel(title_box, text="دوربین زنده", font=load_font(17, True),
                     text_color=TEXT).pack(anchor="e")
        self._fps_label = ctk.CTkLabel(title_box, text="", font=load_font(10),
                                       text_color=MUTED)
        self._fps_label.pack(anchor="e")
        live_chip = ctk.CTkFrame(head, fg_color="#134e4a", corner_radius=10)
        live_chip.pack(side="left")
        ctk.CTkLabel(live_chip, text="LIVE", font=load_font(10, True),
                     text_color=GOOD).pack(padx=10, pady=3)

        self._cam_label = ctk.CTkLabel(cam_card, text="", image=None)
        self._cam_label.grid(row=1, column=0, sticky="nsew", padx=18, pady=6)
        self._idle_hint = self._make_camera_idle_state(cam_card)
        self._idle_hint.grid(row=1, column=0)

        ctl = ctk.CTkFrame(cam_card, fg_color="transparent")
        ctl.grid(row=2, column=0, sticky="ew", padx=18, pady=(4, 18))
        self._cam_toggle_btn = GradientButton(
            ctl, text="شروع دوربین", icon_name="camera",
            c1=CYAN, c2="#06b6d4", text_color=ON_CYAN,
            width=190, height=48, font=load_font(15, True),
            command=self._toggle_dash_camera)
        self._cam_toggle_btn.pack(side="right")
        faces_chip = ctk.CTkFrame(ctl, fg_color=CARD_2, corner_radius=10)
        faces_chip.pack(side="left")
        self._faces_label = ctk.CTkLabel(faces_chip, text="چهره‌ها:  0",
                                         font=load_font(12, True),
                                         text_color=TEXT)
        self._faces_label.pack(padx=14, pady=8)

        # ---- right column ----
        right = ctk.CTkFrame(scr, fg_color="transparent")
        right.grid(row=1, column=1, sticky="nsew")
        right.grid_rowconfigure(2, weight=1)

        stats = ctk.CTkFrame(right, fg_color="transparent")
        stats.pack(fill="x")
        self._stat_people = StatCard(stats, "users", "افراد ثبت‌شده", CYAN)
        self._stat_people.pack(side="right", fill="x", expand=True, padx=(0, 8))
        self._stat_today = StatCard(stats, "check", "حضور امروز", GOOD)
        self._stat_today.pack(side="right", fill="x", expand=True, padx=(0, 8))
        self._stat_model = StatCard(stats, "shield", "وضعیت مدل‌ها", MAGENTA)
        self._stat_model.pack(side="right", fill="x", expand=True)

        # last recognition card with pulsing border
        self._last_card = ctk.CTkFrame(right, fg_color=CARD, border_color=BORDER,
                                       border_width=2, corner_radius=20)
        self._last_card.pack(fill="both", expand=True, pady=(12, 0))
        ctk.CTkLabel(self._last_card, text="آخرین شناسایی",
                     font=load_font(15, True), text_color=TEXT).pack(
            anchor="e", padx=18, pady=(16, 0))

        body = ctk.CTkFrame(self._last_card, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=18, pady=10)
        self._last_avatar = ctk.CTkLabel(body, text="؟",
                                         font=ctk.CTkFont(size=52),
                                         width=110, height=110,
                                         text_color=MUTED)
        self._last_avatar.pack(pady=(8, 10))
        self._last_name = ctk.CTkLabel(body, text="—", font=load_font(21, True),
                                       text_color=TEXT)
        self._last_name.pack()
        self._last_meta = ctk.CTkLabel(body, text="هنوز شناسایی‌ای انجام نشده",
                                       font=load_font(11), text_color=MUTED)
        self._last_meta.pack(pady=(4, 12))
        self._gauge_img = ctk_img(match_gauge(0.0, 116))
        self._match_gauge = ctk.CTkLabel(body, image=self._gauge_img, text="")
        self._match_gauge.pack()
        self._match_pct = ctk.CTkLabel(body, text="", font=load_font(11),
                                       text_color=MUTED)
        self._match_pct.pack(pady=(6, 0))

        return scr

    def _make_camera_idle_state(self, parent):
        idle = ctk.CTkFrame(parent, fg_color="transparent")
        glyph = make_ctk_image(make_glow_glyph(150, alpha=80))
        ctk.CTkLabel(idle, image=glyph, text="").pack(pady=(18, 10))
        ctk.CTkLabel(idle, text="دوربین آماده است", font=load_font(16, True),
                     text_color=TEXT).pack()
        ctk.CTkLabel(idle, text="برای شروع شناسایی، دکمهٔ «شروع دوربین» را بزنید",
                     font=load_font(11), text_color=MUTED).pack(pady=(4, 18))
        return idle

    def _refresh_dashboard_stats(self):
        self._stat_people.value.configure(text=str(self.db.count_people()))
        self._stat_today.value.configure(text=str(self.db.attendance_today_count()))
        self._stat_model.value.configure(text="آماده" if models_present() else "خطا")
        self._status("آماده" if models_present() else "مدل‌ها یافت نشدند",
                    ok=models_present())

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
        self._cam_toggle_btn.configure(text="توقف دوربین", text_color="#2b0d12")
        self._cam_toggle_btn.set_gradient(BAD, "#fb923c")
        self._idle_hint.grid_forget()
        self._refresh_mean_embeddings()
        self._status("دوربین روشن است")
        self._dash_loop()

    def _stop_dash_camera(self):
        self._dash_running = False
        if self._dash_cam is not None:
            self._dash_cam.release()
            self._dash_cam = None
        try:
            self._cam_toggle_btn.configure(text="شروع دوربین", text_color=ON_CYAN)
            self._cam_toggle_btn.set_gradient(CYAN, "#06b6d4")
            self._cam_label.configure(image=None)
            if not self._idle_hint.grid_info():
                self._idle_hint.grid()
            self._faces_label.configure(text="چهره‌ها:  0")
            self._fps_label.configure(text="")
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

        try:
            engine = self._get_engine()
            faces = engine.detect(frame)
        except Exception as exc:
            self._status(f"خطای مدل: {exc}", ok=False)
            self._stop_dash_camera()
            return
        display = frame.copy()
        threshold = float(self.db.get_setting("match_threshold", COSINE_THRESHOLD))

        # ---- camera HUD: radar rings, brackets, LIVE badge ----
        fh, fw = display.shape[:2]
        t = time.time()
        cx, cy = fw // 2, fh // 2
        pulse = int(18 * (1 + math.sin(t * 2.2)) / 2)
        for rr in (105 + pulse, 165 + pulse):
            for a0 in range(0, 360, 45):
                cv2.ellipse(display, (cx, cy), (rr, int(rr * 0.72)), 0,
                            a0 + 8, a0 + 34, (60, 70, 120), 1, cv2.LINE_AA)
        bm, bl = 12, 34
        for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
            px = bm if sx > 0 else fw - bm
            py = bm if sy > 0 else fh - bm
            cv2.line(display, (px, py), (px - sx * bl, py), (34, 211, 238), 2)
            cv2.line(display, (px, py), (px, py - sy * bl), (34, 211, 238), 2)
        cv2.putText(display, "LIVE", (22, fh - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (34, 211, 238), 2, cv2.LINE_AA)
        cv2.putText(display, datetime.now().strftime("%H:%M:%S"), (fw - 110, fh - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 170, 200), 1, cv2.LINE_AA)

        best_show = None
        for f in faces:
            feat = engine.embed(frame, f)
            best_id, best_sim = None, 0.0
            for pid, mean in self._mean_emb.items():
                sim = cosine_sim(feat, mean)
                if sim > best_sim:
                    best_id, best_sim = pid, sim
            known = best_id is not None and best_sim >= threshold
            x, y, w, h = [int(v) for v in f[:4]]
            color = (52, 211, 153) if known else (251, 113, 133)

            # corner-bracket face box + confidence rail
            arm = max(10, min(w, h) // 4)
            for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
                px = x if sx > 0 else x + w
                py = y if sy > 0 else y + h
                cv2.line(display, (px, py), (px - sx * arm, py), color, 3)
                cv2.line(display, (px, py), (px, py - sy * arm), color, 3)
            cv2.rectangle(display, (x, y + h + 5), (x + w, y + h + 9), (45, 49, 89), -1)
            cv2.rectangle(display, (x, y + h + 5),
                          (x + int(w * max(0.0, min(1.0, best_sim))), y + h + 9),
                          color, -1)
            if known:
                person = self.db.get_person(best_id)
                caption = f"MATCH  {best_sim * 100:.0f}%"
                self._log_attendance_gui(best_id)
                best_show = (person, best_sim)
            else:
                caption = f"UNKNOWN  {best_sim * 100:.0f}%"
            cv2.putText(display, caption, (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)

        if best_show:
            self._show_last_recognition(best_show[0], best_show[1])

        self._faces_label.configure(text=f"چهره‌ها:  {len(faces)}")
        now = time.time()
        self._fps_history.append(now)
        self._fps_history = [t for t in self._fps_history if now - t < 1.0]
        self._fps_label.configure(text=f"{len(self._fps_history)} FPS")

        img = self._frame_to_ctk(display, 640, 400)
        self._cam_label.configure(image=img)
        self._cam_label._image = img
        self.after(33, self._dash_loop)

    def _log_attendance_gui(self, person_id):
        now = datetime.now()
        last = self._att_cache.get(person_id)
        if last and (now - last).total_seconds() < 60:
            return
        self.db.add_attendance(person_id)
        self._att_cache[person_id] = now
        self._stat_today.value.configure(
            text=str(self.db.attendance_today_count()))

    def _show_last_recognition(self, person, sim):
        if person is None:
            return
        self._last_avatar.configure(image=self._get_photo_ctk(person, 110))
        self._last_avatar._image = self._get_photo_ctk(person, 110)
        self._last_name.configure(text=person["full_name"])
        meta = "  ·  ".join(p for p in [person.get("student_id"),
                                        person.get("department")] if p)
        self._last_meta.configure(text=meta or "—")
        self._gauge_img = ctk_img(match_gauge(sim, 116))
        self._match_gauge.configure(image=self._gauge_img)
        self._match_gauge._image = self._gauge_img
        self._match_pct.configure(text=f"شباهت:  {sim * 100:.1f}٪")
        # pulse the card border (delight!)
        try:
            self._last_card.configure(border_color=CYAN)
            self.after(1400, lambda: self._last_card.configure(border_color=BORDER))
        except Exception:
            pass

    # =====================================================
    # SCREEN 2: ENROLL — Face-ID style guided scan
    # =====================================================

    def _build_enroll(self):
        scr = ctk.CTkFrame(self.content, fg_color="transparent")
        scr.grid_columnconfigure(0, weight=2)
        scr.grid_columnconfigure(1, weight=3)
        scr.grid_rowconfigure(0, weight=1)

        # ---- form card ----
        form = ctk.CTkScrollableFrame(scr, fg_color=CARD, border_color=BORDER,
                                      border_width=1, corner_radius=20)
        form.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        ctk.CTkLabel(form, text="مشخصات فرد", font=load_font(18, True),
                     text_color=TEXT).pack(anchor="e", pady=(12, 0), padx=10)
        ctk.CTkLabel(form, text="فیلدهای ستاره‌دار الزامی‌اند", font=load_font(10),
                     text_color=MUTED).pack(anchor="e", padx=10, pady=(0, 12))

        def field(label, placeholder, required=False):
            ctk.CTkLabel(form, text=label + ("  *" if required else ""),
                         font=load_font(12, True), text_color=TEXT).pack(
                anchor="e", padx=10, pady=(4, 2))
            entry = ctk.CTkEntry(form, placeholder_text=placeholder, height=40,
                                 font=load_font(13), justify="right",
                                 fg_color=CARD_2, border_color=BORDER,
                                 border_width=1, text_color=TEXT,
                                 placeholder_text_color=MUTED)
            entry.pack(fill="x", padx=10, pady=(0, 2))
            return entry

        self._f_name = field("نام کامل", "مثلاً: سید احمد رضایی", required=True)
        self._f_sid = field("شمارهٔ کارت / پوهنځی", "مثلاً: CS-1402-015")
        self._f_dept = field("دانشکده / بخش", "مثلاً: کمپیوتر ساینس")
        self._f_phone = field("شمارهٔ تماس", "مثلاً: 07xx xxx xxx")
        self._f_email = field("ایمیل", "name@example.com")
        ctk.CTkLabel(form, text="یادداشت", font=load_font(12, True),
                     text_color=TEXT).pack(anchor="e", padx=10, pady=(6, 2))
        self._f_notes = ctk.CTkTextbox(form, height=56, font=load_font(12),
                                       fg_color=CARD_2, border_color=BORDER,
                                       border_width=1, text_color=TEXT)
        self._f_notes.pack(fill="x", padx=10, pady=(0, 10))

        self._form_error = ctk.CTkLabel(form, text="", font=load_font(11),
                                        text_color=BAD)
        self._form_error.pack(anchor="e", padx=10)

        btn_row = ctk.CTkFrame(form, fg_color="transparent")
        btn_row.pack(fill="x", padx=10, pady=(4, 18))
        self._scan_btn = GradientButton(
            btn_row, text="شروع اسکن چهره", icon_name="sparkles",
            c1=VIOLET, c2=MAGENTA, text_color="#ffffff",
            width=230, height=46, font=load_font(14, True),
            command=self._start_scan)
        self._scan_btn.pack(side="right", padx=(0, 8))
        ctk.CTkButton(btn_row, text="پاک کردن", font=load_font(12), height=46,
                      fg_color=CARD_2, hover_color=BORDER, text_color=TEXT,
                      corner_radius=14,
                      command=self._clear_enroll_form).pack(side="left")

        # ---- scan card ----
        scan = ctk.CTkFrame(scr, fg_color=CARD, border_color=BORDER,
                            border_width=1, corner_radius=20)
        scan.grid(row=0, column=1, sticky="nsew")
        scan.grid_columnconfigure(0, weight=1)
        scan.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(scan, text="اسکن چهره — چند زاویه‌ای",
                     font=load_font(17, True), text_color=TEXT).grid(
            row=0, column=0, sticky="e", padx=20, pady=(16, 2))

        self._scan_preview = ctk.CTkLabel(scan,
                                          text="پس از پر کردن مشخصات،\n«شروع اسکن چهره» را بزنید",
                                          font=load_font(14), text_color=MUTED)
        self._scan_preview.grid(row=1, column=0, sticky="nsew", padx=20, pady=4)

        self._scan_instruction = ctk.CTkLabel(scan, text="آماده برای اسکن",
                                              font=load_font(18, True),
                                              text_color=CYAN)
        self._scan_instruction.grid(row=2, column=0, pady=(2, 0))

        chips = ctk.CTkFrame(scan, fg_color="transparent")
        chips.grid(row=3, column=0, pady=8)
        self._zone_chips = {}
        for zone in ZONE_ORDER:
            chip = ctk.CTkLabel(chips, text="⬜  " + ZONE_LABELS[zone].split("  ", 1)[1],
                                font=load_font(10), fg_color=CARD_2,
                                corner_radius=8, width=158, height=28,
                                text_color=MUTED)
            chip.pack(side="right", padx=3, pady=2)
            self._zone_chips[zone] = chip

        progress_row = ctk.CTkFrame(scan, fg_color="transparent")
        progress_row.grid(row=4, column=0, sticky="ew", padx=20)
        self._scan_progress = ctk.CTkProgressBar(progress_row, progress_color=CYAN,
                                                 fg_color=CARD_2, height=14,
                                                 corner_radius=7)
        self._scan_progress.set(0)
        self._scan_progress.pack(fill="x")
        self._scan_count = ctk.CTkLabel(scan, text=f"0 / {sum(ZONE_TARGETS.values())} نمونه",
                                        font=load_font(11), text_color=MUTED)
        self._scan_count.grid(row=5, column=0, pady=(4, 2))
        self._scan_reason = ctk.CTkLabel(scan, text="", font=load_font(10),
                                         text_color=BAD)
        self._scan_reason.grid(row=6, column=0)

        btn_row2 = ctk.CTkFrame(scan, fg_color="transparent")
        btn_row2.grid(row=7, column=0, sticky="ew", padx=20, pady=(6, 18))
        ctk.CTkButton(btn_row2, text="لغو اسکن", font=load_font(12),
                      fg_color=CARD_2, hover_color=BAD, text_color=TEXT,
                      corner_radius=12, height=40,
                      command=self._cancel_scan).pack(side="right")
        self._finish_btn = GradientButton(
            btn_row2, text="پایان و ذخیره", icon_name="check",
            c1=GOOD, c2="#0ea5e9", text_color="#08120d",
            width=170, height=40, font=load_font(12, True),
            command=self._finish_scan)
        self._finish_btn.set_disabled(True)
        self._finish_btn.pack(side="left")

        return scr

    def _clear_enroll_form(self):
        for entry in (self._f_name, self._f_sid, self._f_dept, self._f_phone,
                      self._f_email):
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
        self._scan_btn.set_disabled(True)
        self._finish_btn.set_disabled(True)
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

        try:
            engine = self._get_engine()
            faces = engine.detect(frame)
        except Exception as exc:
            self._status(f"خطای مدل: {exc}", ok=False)
            self._cancel_scan()
            return
        display = frame.copy()
        faces = [f for f in faces if f[2] >= 50 and f[3] >= 50]

        # animated scanning arc around the face guide
        hgt, wid = display.shape[:2]
        self._scan_angle = (self._scan_angle + 7) % 360
        center = (wid // 2, hgt // 2)
        axes = (wid // 4, hgt // 3)
        cv2.ellipse(display, center, axes, 0, 0, 360, (45, 49, 89), 2)
        cv2.ellipse(display, center, axes, 0, int(self._scan_angle),
                    int(self._scan_angle + 80), (34, 211, 238), 4)

        # 7 Face-ID-style checkpoint dots + direction cue + live counter
        for i, zone in enumerate(ZONE_ORDER):
            ang = math.radians(-90 + i * (360 / len(ZONE_ORDER)))
            dx = int(center[0] + axes[0] * math.cos(ang))
            dy = int(center[1] + axes[1] * math.sin(ang))
            done = st["counts"][zone] >= st["zones"][zone]
            cv2.circle(display, (dx, dy), 7 if done else 5,
                       (52, 211, 153) if done else (95, 100, 150), -1, cv2.LINE_AA)
        missing_now = self._missing_zones()
        if missing_now:
            mz = missing_now[0]
            if mz in ("slight_left", "far_left"):
                cv2.arrowedLine(display, (center[0] + 26, 40), (center[0] - 16, 40),
                                (232, 121, 249), 3, tipLength=0.35)
            elif mz in ("slight_right", "far_right"):
                cv2.arrowedLine(display, (center[0] - 16, 40), (center[0] + 26, 40),
                                (232, 121, 249), 3, tipLength=0.35)
            elif mz == "up":
                cv2.arrowedLine(display, (center[0], 62), (center[0], 22),
                                (232, 121, 249), 3, tipLength=0.35)
            elif mz == "down":
                cv2.arrowedLine(display, (center[0], 22), (center[0], 62),
                                (232, 121, 249), 3, tipLength=0.35)
        cv2.putText(display, f"SCAN {len(st['samples'])}/{sum(st['zones'].values())}",
                    (16, hgt - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (34, 211, 238), 2, cv2.LINE_AA)

        now = time.time()
        if len(faces) == 1:
            f = max(faces, key=lambda r: r[2] * r[3])
            x, y, w, h = [int(v) for v in f[:4]]
            cv2.rectangle(display, (x, y), (x + w, y + h), (139, 92, 246), 2)

            quality_ok, reason, score = engine.sample_quality(frame, f)
            self._scan_reason.configure(
                text="" if quality_ok else QUALITY_REASON_TEXT.get(reason, reason),
                text_color=(GOOD if quality_ok else BAD))

            yaw_zone, pitch_zone, _yaw, _pitch = engine.head_pose(f)
            missing = self._missing_zones()

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

            total = len(st["samples"])
            target_total = sum(st["zones"].values())
            self._scan_progress.set(min(1.0, total / target_total))
            self._scan_count.configure(text=f"{total} / {target_total} نمونه")
            if total >= 12:
                self._finish_btn.set_disabled(False)

            if missing:
                self._scan_instruction.configure(text=MISSING_PROMPT[missing[0]],
                                                 text_color=CYAN)
            else:
                self._scan_instruction.configure(text="✓ همهٔ زاویه‌ها ثبت شد!",
                                                 text_color=GOOD)

            if not missing or total >= target_total or (
                    now - st["started"] > 40 and total >= 12):
                self._finish_scan()
                return
        else:
            msg = "دوربین را تنظیم کنید — یک چهره در قاب باشد" if faces else \
                  "چهره‌ای دیده نشد — نزدیک‌تر شوید و نور را تنظیم کنید"
            self._scan_instruction.configure(text=msg, text_color=BAD)

        img = self._frame_to_ctk(display, 460, 330)
        self._scan_preview.configure(image=img, text="")
        self._scan_preview._image = img
        self.after(30, self._enroll_loop)

    def _update_zone_chips(self):
        st = self._enroll_state
        for zone in ZONE_ORDER:
            done = st["counts"][zone] >= st["zones"][zone]
            self._zone_chips[zone].configure(
                text=("✅ " if done else f"{st['counts'][zone]}/{st['zones'][zone]} ") +
                     ZONE_LABELS[zone].split("  ", 1)[1],
                text_color=(GOOD if done else MUTED),
                fg_color=("#134e4a" if done else CARD_2))

    def _release_enroll_camera(self):
        if self._enroll_cam is not None:
            self._enroll_cam.release()
            self._enroll_cam = None

    def _cancel_scan(self, silent=False):
        was_active = self._enroll_state is not None
        self._enroll_state = None
        self._release_enroll_camera()
        self._scan_btn.set_disabled(False)
        self._finish_btn.set_disabled(True)
        self._scan_preview.configure(image=None,
                                     text="پس از پر کردن مشخصات،\n«شروع اسکن چهره» را بزنید")
        self._scan_instruction.configure(text="آماده برای اسکن", text_color=CYAN)
        self._scan_progress.set(0)
        self._scan_count.configure(text=f"0 / {sum(ZONE_TARGETS.values())} نمونه")
        self._scan_reason.configure(text="")
        for zone in ZONE_ORDER:
            self._zone_chips[zone].configure(
                text="⬜  " + ZONE_LABELS[zone].split("  ", 1)[1],
                text_color=MUTED, fg_color=CARD_2)
        if not silent and was_active:
            self._status("اسکن لغو شد", ok=False)
            self._toast("اسکن لغو شد", ok=False)

    def _finish_scan(self):
        st = self._enroll_state
        if st is None:
            return
        self._release_enroll_camera()

        feats = [s["emb"] for s in st["samples"]]
        qualities = [s["quality"] for s in st["samples"]]
        mean = np.mean(np.stack(feats), axis=0)
        mean /= (np.linalg.norm(mean) + 1e-10)
        avg_quality = float(np.mean(qualities)) if qualities else 0.0

        person_id = self.db.add_person(**st["person"])
        for s in st["samples"]:
            self.db.add_face(person_id, s["emb"], s["zone"], s["quality"])

        if st["samples"]:
            best = max(st["samples"], key=lambda s: s["quality"])
            photo_path = PROFILES_DIR / f"person_{person_id}.jpg"
            cv2.imwrite(str(photo_path), best["crop"])
            self.db.update_person(person_id, photo_path=str(photo_path))

        name = st["person"]["full_name"]
        self._enroll_state = None
        self._scan_btn.set_disabled(False)
        self._finish_btn.set_disabled(True)
        self._refresh_mean_embeddings()
        self._refresh_people_count()
        self._status(f"«{name}» ثبت شد")
        self._show_enroll_summary(person_id, name, len(feats), avg_quality)
        self._clear_enroll_form()
        self._cancel_scan(silent=True)

    def _show_enroll_summary(self, person_id, name, sample_count, avg_quality):
        win = ctk.CTkToplevel(self)
        win.title("ثبت موفق")
        win.geometry("430x420")
        win.transient(self)
        win.grab_set()
        win.configure(fg_color=CARD)

        badge = ctk.CTkLabel(win, text="✅", font=ctk.CTkFont(size=44))
        badge.pack(pady=(24, 2))
        ctk.CTkLabel(win, text="ثبت‌نام موفق!", font=load_font(23, True),
                     text_color=GOOD).pack()

        person = self.db.get_person(person_id)
        if person:
            img = photo_or_avatar(person, 128)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(128, 128))
            ctk.CTkLabel(win, image=ctk_img, text="").pack(pady=10)

        ctk.CTkLabel(win, text=name, font=load_font(18, True),
                     text_color=TEXT).pack()
        ctk.CTkLabel(win,
                     text=f"{sample_count} نمونهٔ چهره از زوایای مختلف ذخیره شد\n"
                          f"امتیاز کیفیت:  {avg_quality:.0f} از 100",
                     font=load_font(12), text_color=MUTED, justify="center").pack(pady=8)

        def go(target):
            win.destroy()
            self.show_screen(target)

        row = ctk.CTkFrame(win, fg_color="transparent")
        row.pack(pady=12)
        GradientButton(row, text="مشاهدهٔ فهرست افراد", icon_name="users",
                       c1="#3a3e63", c2="#4a4f78", text_color=TEXT,
                       width=190, height=40, font=load_font(12, True),
                       command=lambda: go("people")).pack(side="right", padx=6)
        GradientButton(row, text="ثبت فرد دیگر", icon_name="add_user",
                       c1=CYAN, c2="#06b6d4", text_color=ON_CYAN,
                       width=160, height=40, font=load_font(12, True),
                       command=lambda: go("enroll")).pack(side="left", padx=6)

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
        top.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        ctk.CTkLabel(top, text="افراد ثبت‌شده", font=load_font(21, True),
                     text_color=TEXT).pack(side="right")
        # gradient-bordered search field
        search_wrap = ctk.CTkFrame(top, fg_color="transparent", width=320, height=42)
        search_wrap.pack(side="right", padx=(0, 14))
        search_wrap.pack_propagate(False)
        search_bg = ctk.CTkLabel(
            search_wrap,
            image=ctk_img(gradient_pill(320, 42, CYAN, VIOLET, radius=14)),
            text="")
        search_bg.place(relx=0.5, rely=0.5, anchor="center")
        self._search_entry = ctk.CTkEntry(
            search_wrap, placeholder_text="جستجو: نام، شمارهٔ کارت، دانشکده…",
            justify="right", font=load_font(12), fg_color=CARD,
            border_width=0, text_color=TEXT, placeholder_text_color=MUTED,
            height=36)
        self._search_entry.place(relx=0.5, rely=0.5, anchor="center",
                                 relwidth=0.985, relheight=0.86)
        self._search_entry._search_bg = search_bg
        self._search_entry.bind("<KeyRelease>", lambda e: self._refresh_people())
        GradientButton(top, text="افزودن فرد", icon_name="add_user",
                       c1=CYAN, c2="#06b6d4", text_color=ON_CYAN,
                       width=165, height=40, font=load_font(12, True),
                       command=lambda: self.show_screen("enroll")).pack(side="left")

        self._people_list = ctk.CTkScrollableFrame(scr, fg_color=CARD,
                                                   border_color=BORDER,
                                                   border_width=1,
                                                   corner_radius=20)
        self._people_list.grid(row=1, column=0, sticky="nsew")
        return scr

    def _refresh_people(self):
        for widget in self._people_list.winfo_children():
            widget.destroy()
        search = self._search_entry.get().strip() if hasattr(self, "_search_entry") else ""
        people = self.db.list_people(search)
        if not people:
            empty = ctk.CTkFrame(self._people_list, fg_color="transparent")
            empty.pack(pady=44)
            glyph = make_ctk_image(make_glow_glyph(110, alpha=70))
            ctk.CTkLabel(empty, image=glyph, text="").pack()
            ctk.CTkLabel(empty,
                         text="هنوز کسی ثبت نشده",
                         font=load_font(15, True), text_color=TEXT).pack(pady=(8, 2))
            ctk.CTkLabel(empty, text="از دکمهٔ «افزودن فرد» شروع کنید",
                         font=load_font(11), text_color=MUTED).pack()
            return
        for person in people:
            row = ctk.CTkFrame(self._people_list, fg_color=CARD_2,
                               corner_radius=16, height=82,
                               border_color=BORDER, border_width=1)
            row.pack(fill="x", padx=12, pady=6)
            row.pack_propagate(False)
            avatar = photo_or_avatar(person, 64)
            ctk_img = ctk.CTkImage(light_image=avatar, dark_image=avatar,
                                   size=(64, 64))
            ctk.CTkLabel(row, image=ctk_img, text="").pack(side="right",
                                                           padx=14, pady=9)

            info = ctk.CTkFrame(row, fg_color="transparent")
            info.pack(side="right", fill="both", expand=True, pady=10)
            ctk.CTkLabel(info, text=person["full_name"], font=load_font(14, True),
                         text_color=TEXT, anchor="e").pack(fill="x")
            meta = "  ·  ".join(p for p in [person.get("student_id"),
                                            person.get("department"),
                                            f"{person.get('face_count', 0)} نمونه"]
                                if p)
            ctk.CTkLabel(info, text=meta, font=load_font(10), text_color=MUTED,
                         anchor="e").pack(fill="x")

            ctk.CTkButton(row, text="🗑  حذف", width=86, height=36,
                          fg_color="transparent", hover_color=BAD,
                          text_color=TEXT, corner_radius=10,
                          command=lambda p=person: self._delete_person(p)).pack(
                side="left", padx=14)

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

    def _build_report(self):
        scr = ctk.CTkFrame(self.content, fg_color="transparent")
        scr.grid_columnconfigure(0, weight=1)
        scr.grid_rowconfigure(1, weight=1)

        top = ctk.CTkFrame(scr, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        ctk.CTkLabel(top, text="گزارش حضور", font=load_font(21, True),
                     text_color=TEXT).pack(side="right")
        self._report_scope = ctk.CTkSegmentedButton(
            top, values=["امروز", "همه"], font=load_font(12),
            selected_color=CYAN, selected_hover_color="#67e8f9",
            unselected_color=CARD_2, unselected_hover_color=BORDER,
            command=lambda v: self._refresh_report())
        self._report_scope.set("امروز")
        self._report_scope.pack(side="left", padx=(12, 0))
        GradientButton(top, text="خروجی CSV", icon_name="download",
                       c1=VIOLET, c2=MAGENTA, text_color="#ffffff",
                       width=160, height=40, font=load_font(12, True),
                       command=self._export_csv).pack(side="left")

        count_chip = ctk.CTkFrame(top, fg_color=CARD_2, corner_radius=10)
        count_chip.pack(side="left", padx=12)
        self._report_count = ctk.CTkLabel(count_chip, text="۰ رکورد",
                                          font=load_font(12, True),
                                          text_color=CYAN)
        self._report_count.pack(padx=14, pady=8)

        self._report_list = ctk.CTkScrollableFrame(scr, fg_color=CARD,
                                                   border_color=BORDER,
                                                   border_width=1,
                                                   corner_radius=20)
        self._report_list.grid(row=1, column=0, sticky="nsew")
        return scr

    def _refresh_report(self):
        for widget in self._report_list.winfo_children():
            widget.destroy()
        today_only = self._report_scope.get() == "امروز"
        rows = self.db.attendance_today() if today_only else self.db.attendance_all()
        self._report_count.configure(text=f"{len(rows)} رکورد")

        header = ctk.CTkFrame(self._report_list, fg_color="transparent")
        header.pack(fill="x", padx=14, pady=(10, 4))
        for text, width in [("زمان", 150), ("دانشکده", 160), ("شمارهٔ کارت", 170), ("نام", 0)]:
            ctk.CTkLabel(header, text=text, font=load_font(11, True),
                         text_color=CYAN, width=width, anchor="e").pack(
                side="right", padx=10)

        if not rows:
            empty = ctk.CTkFrame(self._report_list, fg_color="transparent")
            empty.pack(pady=30)
            ctk.CTkLabel(empty, text="📭", font=ctk.CTkFont(size=40)).pack()
            ctk.CTkLabel(empty, text="رکوردی ثبت نشده است",
                         font=load_font(13), text_color=MUTED).pack(pady=(6, 0))
            return
        for i, r in enumerate(rows):
            row = ctk.CTkFrame(self._report_list,
                               fg_color=(CARD_2 if i % 2 == 0 else CARD),
                               corner_radius=10, height=42)
            row.pack(fill="x", padx=14, pady=3)
            row.pack_propagate(False)
            ctk.CTkLabel(row, text=r["recorded_at"], font=load_font(11),
                         text_color=TEXT, width=150, anchor="e").pack(
                side="right", padx=10)
            ctk.CTkLabel(row, text=r.get("department") or "—", font=load_font(11),
                         text_color=MUTED, width=160, anchor="e").pack(
                side="right", padx=10)
            ctk.CTkLabel(row, text=r.get("student_id") or "—", font=load_font(11),
                         text_color=MUTED, width=170, anchor="e").pack(
                side="right", padx=10)
            ctk.CTkLabel(row, text=r["full_name"], font=load_font(11, True),
                         text_color=TEXT, anchor="e").pack(side="right", padx=10)

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
        scr = ctk.CTkScrollableFrame(self.content, fg_color="transparent")

        ctk.CTkLabel(scr, text="تنظیمات و سلامت سیستم", font=load_font(21, True),
                     text_color=TEXT).pack(anchor="e", pady=(4, 12), padx=6)

        # --- camera & threshold ---
        card = ctk.CTkFrame(scr, fg_color=CARD, border_color=BORDER,
                            border_width=1, corner_radius=20)
        card.pack(fill="x", padx=6, pady=6)
        ctk.CTkLabel(card, text="دوربین و آستانهٔ تطبیق", font=load_font(15, True),
                     text_color=TEXT).pack(anchor="e", padx=18, pady=(14, 8))

        row1 = ctk.CTkFrame(card, fg_color="transparent")
        row1.pack(fill="x", padx=18, pady=4)
        ctk.CTkLabel(row1, text="شمارهٔ دوربین:", font=load_font(12),
                     text_color=TEXT).pack(side="right", padx=(0, 10))
        saved_cam = self.db.get_setting("camera_index", "0")
        cam_menu = ctk.CTkOptionMenu(row1, values=["0", "1", "2", "3"], width=90,
                                     fg_color=CARD_2, button_color=VIOLET,
                                     button_hover_color="#a78bfa",
                                     text_color=TEXT,
                                     command=lambda v: self.db.set_setting("camera_index", v))
        cam_menu.set(saved_cam)
        cam_menu.pack(side="right")

        row2 = ctk.CTkFrame(card, fg_color="transparent")
        row2.pack(fill="x", padx=18, pady=(12, 6))
        ctk.CTkLabel(row2, text="آستانهٔ تطبیق (سخت‌گیری):", font=load_font(12),
                     text_color=TEXT).pack(side="right", padx=(0, 10))
        self._threshold_label = ctk.CTkLabel(row2, text="", font=load_font(12, True),
                                             text_color=CYAN)
        self._threshold_label.pack(side="left")
        saved_thr = float(self.db.get_setting("match_threshold",
                                              str(COSINE_THRESHOLD)))
        slider = ctk.CTkSlider(card, from_=0.20, to=0.60, number_of_steps=40,
                               progress_color=CYAN, button_color=CYAN,
                               button_hover_color="#67e8f9", fg_color=CARD_2,
                               command=self._on_threshold_change)
        slider.set(saved_thr)
        slider.pack(fill="x", padx=18, pady=(0, 6))
        self._threshold_label.configure(text=f"{saved_thr:.2f}")
        ctk.CTkLabel(card, text="مقدار پیشنهادی برای SFace: 0.36 — بالاتر = سخت‌گیرتر (کمتر اشتباه می‌گیرد)",
                     font=load_font(10), text_color=MUTED).pack(anchor="e",
                                                                padx=18, pady=(0, 16))

        # --- health check ---
        card2 = ctk.CTkFrame(scr, fg_color=CARD, border_color=BORDER,
                             border_width=1, corner_radius=20)
        card2.pack(fill="x", padx=6, pady=6)
        ctk.CTkLabel(card2, text="🩺  بررسی سلامت پروژه", font=load_font(15, True),
                     text_color=TEXT).pack(anchor="e", padx=18, pady=(14, 8))
        self._health_box = ctk.CTkTextbox(card2, height=150,
                                          font=ctk.CTkFont(family="Consolas",
                                                           size=12),
                                          fg_color=CARD_2, text_color=TEXT)
        self._health_box.pack(fill="x", padx=18, pady=4)
        self._health_box.insert("1.0", "برای اجرای بررسی «اجرای تست» را بزنید…")
        GradientButton(card2, text="اجرای تست سلامت", icon_name="pulse",
                       c1=CYAN, c2="#06b6d4", text_color=ON_CYAN,
                       width=190, height=42, font=load_font(12, True),
                       command=self._run_health_check).pack(anchor="e",
                                                            padx=18, pady=(6, 16))

        # --- backup ---
        card3 = ctk.CTkFrame(scr, fg_color=CARD, border_color=BORDER,
                             border_width=1, corner_radius=20)
        card3.pack(fill="x", padx=6, pady=6)
        ctk.CTkLabel(card3, text="💾  پشتیبان‌گیری", font=load_font(15, True),
                     text_color=TEXT).pack(anchor="e", padx=18, pady=(14, 2))
        ctk.CTkLabel(card3, text="یک کپی از دیتابیس (افراد + چهره‌ها + حضور) ذخیره می‌شود.",
                     font=load_font(11), text_color=MUTED).pack(anchor="e", padx=18)
        GradientButton(card3, text="ذخیرهٔ پشتیبان دیتابیس", icon_name="database",
                       c1=VIOLET, c2="#2563eb", text_color="#ffffff",
                       width=225, height=42, font=load_font(12, True),
                       command=self._backup_db).pack(anchor="e", padx=18,
                                                     pady=(8, 16))

        # --- about ---
        card4 = ctk.CTkFrame(scr, fg_color=CARD, border_color=BORDER,
                             border_width=1, corner_radius=20)
        card4.pack(fill="x", padx=6, pady=(6, 22))
        ctk.CTkLabel(card4, text="دربارهٔ پروژه", font=load_font(15, True),
                     text_color=TEXT).pack(anchor="e", padx=18, pady=(14, 4))
        about = ("چهره‌یاب — سیستم شناسایی هوشمند چهره · نسخهٔ 2.1 (Aurora)\n"
                 "تشخیص: YuNet (OpenCV Zoo)   شناسایی: SFace — ۱۲۸-D Embedding\n"
                 "پوهنځی کمپیوتر ساینس — پوهنتون غرجستان | 2026\n"
                 "سازنده: Sayed Mohammad Naqi Sadat")
        ctk.CTkLabel(card4, text=about, font=load_font(11), text_color=MUTED,
                     justify="right").pack(anchor="e", padx=18, pady=(0, 18))
        return scr

    def _on_threshold_change(self, value):
        self.db.set_setting("match_threshold", f"{float(value):.2f}")
        self._threshold_label.configure(text=f"{float(value):.2f}")

    def _run_health_check(self):
        box = self._health_box
        box.delete("1.0", "end")
        lines = ["== Smart Face Recognition — Health Check ==\n"]

        def report(label, ok):
            lines.append(f"[{'PASS' if ok else 'FAIL'}] {label}\n")

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
            report(f"Detection on test.jpg ({len(faces)} faces, expected >= 2)",
                   len(faces) >= 2)

        for label, ok in self.db.integrity_check():
            report(label, ok)

        lines.append(f"\nافراد ثبت‌شده: {self.db.count_people()}   |   نمونه‌های چهره: "
                     f"{sum(self.db.get_face_count(p['id']) for p in self.db.list_people())}\n")
        lines.append("== تمام ==\n")
        box.insert("1.0", "".join(lines))

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
