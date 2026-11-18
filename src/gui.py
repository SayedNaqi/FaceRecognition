#!/usr/bin/env python3
# ============================================================
# Smart Face Recognition — Professional GUI
# Faculty of Computer Science — Ghorjistan University
#
# Features
#   • Dashboard with live camera (start/stop) + last-recognition card
#   • Guided Face-ID-style enrollment (head-turn zones) + profile form
#   • People registry (search, view, delete)
#   • Attendance report (export CSV)
#   • Settings (camera, threshold, health check, DB backup)
#
# Run:  python src/gui.py            (normal)
#       python src/gui.py --smoketest  (build all screens then exit)
# ============================================================

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
from PIL import Image, ImageDraw, ImageFont

from db import Database, PROFILES_DIR
from face_engine import FaceEngine, COSINE_THRESHOLD, models_present

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_IMAGE = PROJECT_ROOT / "data" / "images" / "test.jpg"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# ---------- palette (matches the exhibition poster) ----------
NAVY = "#0c2b33"
PANEL = "#123744"
PANEL_2 = "#173f4e"
CARD = "#1a4553"
CREAM = "#f5f1e6"
MUTED = "#9fb4b8"
GOLD = "#d9a441"
TEAL = "#2f8f9d"
GOOD = "#3ecf6f"
BAD = "#ff6b5e"
FONT = "Tahoma"

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
POSE_OK_REASON = "ok"
QUALITY_REASON_TEXT = {
    "too_small": "چهره خیلی کوچک است — نزدیک‌تر شوید",
    "too_dark": "نور کم است — محیط را روشن‌تر کنید",
    "too_bright": "نور خیلی زیاد است",
    "low_contrast": "کنتراست چهره کم است",
    "blurry": "تصویر تار است — ثابت بمانید",
}


def load_font(size, bold=False):
    return ctk.CTkFont(family=FONT, size=size, weight="bold" if bold else "normal")


def initials_image(name, size=96):
    """Draw a navy avatar with the person's initials."""
    parts = [p for p in name.split() if p]
    initials = (parts[0][0] + (parts[-1][0] if len(parts) > 1 else "")).upper() if parts else "؟"
    img = Image.new("RGB", (size, size), NAVY)
    draw = ImageDraw.Draw(img)
    draw.ellipse([2, 2, size - 2, size - 2], outline=GOLD, width=4)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/tahomabd.ttf", size // 2)
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), initials, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((size - tw) / 2 - bbox[0], (size - th) / 2 - bbox[1]),
              initials, fill=GOLD, font=font)
    return img


def photo_or_avatar(person, size=96):
    photo = person.get("photo_path")
    if photo and Path(photo).exists():
        try:
            return Image.open(photo).convert("RGB").resize((size, size))
        except OSError:
            pass
    return initials_image(person.get("full_name", "؟"), size)


class FaceRecognitionApp(ctk.CTk):
    VERSION = "2.0.0"

    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.title("سیستم شناسایی هوشمند چهره | پوهنتون غرجستان")
        self.geometry("1180x720")
        self.minsize(1024, 640)
        self.configure(fg_color=NAVY)

        self.db = Database()
        self.engine = None  # lazy — created on first camera use
        self._mean_emb = {}
        self._photo_cache = {}
        self._att_cache = {}          # person_id -> datetime of last attendance log
        self._screens = {}
        self._dash_cam = None
        self._dash_running = False
        self._enroll_cam = None
        self._enroll_state = None
        self._fps_history = []

        self._build_layout()
        self._refresh_mean_embeddings()
        self.show_screen("dashboard")

    # =====================================================
    # LAYOUT
    # =====================================================

    def _build_layout(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ---------- sidebar ----------
        sidebar = ctk.CTkFrame(self, width=232, fg_color=NAVY, corner_radius=0)
        sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew")
        sidebar.grid_propagate(False)
        sidebar.grid_rowconfigure(8, weight=1)

        logo = ctk.CTkLabel(sidebar, text="👤", font=ctk.CTkFont(size=44))
        logo.grid(row=0, column=0, pady=(26, 0))
        ctk.CTkLabel(sidebar, text="سیستم شناسایی", font=load_font(21, True),
                     text_color=CREAM).grid(row=1, column=0)
        ctk.CTkLabel(sidebar, text="هوشمند چهره", font=load_font(21, True),
                     text_color=GOLD).grid(row=2, column=0, pady=(0, 2))
        ctk.CTkLabel(sidebar, text="پوهنتون غرجستان", font=load_font(11),
                     text_color=MUTED).grid(row=3, column=0, pady=(0, 18))

        nav = [
            ("dashboard", "🏠  داشبورد زنده"),
            ("enroll",    "➕  ثبت فرد جدید"),
            ("people",    "👥  افراد ثبت‌شده"),
            ("report",    "📋  گزارش حضور"),
            ("settings",  "⚙️  تنظیمات و سلامت"),
        ]
        self._nav_buttons = []
        for i, (key, label) in enumerate(nav):
            btn = ctk.CTkButton(
                sidebar, text=label, font=load_font(14), height=44,
                anchor="w", fg_color="transparent", hover_color=PANEL_2,
                text_color=CREAM, corner_radius=10,
                command=lambda k=key: self.show_screen(k))
            btn.grid(row=4 + i, column=0, sticky="ew", padx=14, pady=3)
            self._nav_buttons.append((key, btn))

        ctk.CTkLabel(sidebar, text="مدل‌ها: YuNet + SFace\nآفلاین — بدون اینترنت",
                     font=load_font(10), text_color=MUTED,
                     justify="center").grid(row=9, column=0, pady=(0, 6))
        ctk.CTkLabel(sidebar, text=f"نسخهٔ {self.VERSION}",
                     font=load_font(10), text_color=GOLD).grid(row=10, column=0, pady=(0, 14))

        # ---------- content ----------
        self.content = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=0)
        self.content.grid(row=0, column=1, sticky="nsew")

        # ---------- status bar ----------
        status = ctk.CTkFrame(self, height=30, fg_color=NAVY, corner_radius=0)
        status.grid(row=1, column=1, sticky="ew")
        self.status_label = ctk.CTkLabel(status, text="● آماده", text_color=GOOD,
                                         font=load_font(11))
        self.status_label.pack(side="right", padx=14)
        ctk.CTkLabel(status, text="پوهنځی کمپیوتر ساینس", text_color=MUTED,
                     font=load_font(10)).pack(side="left", padx=14)

        # ---------- screens ----------
        self._screens["dashboard"] = self._build_dashboard()
        self._screens["enroll"] = self._build_enroll()
        self._screens["people"] = self._build_people()
        self._screens["report"] = self._build_report()
        self._screens["settings"] = self._build_settings()

        # keyboard shortcuts (HCI: accelerators)
        self.bind("<Control-Key-1>", lambda e: self.show_screen("dashboard"))
        self.bind("<Control-Key-2>", lambda e: self.show_screen("enroll"))
        self.bind("<Control-Key-3>", lambda e: self.show_screen("people"))
        self.bind("<Control-Key-4>", lambda e: self.show_screen("report"))
        self.bind("<Control-Key-5>", lambda e: self.show_screen("settings"))
        self.bind("<Escape>", lambda e: self._on_escape())

    def show_screen(self, name):
        # stop cameras when leaving their screens
        if name != "dashboard":
            self._stop_dash_camera()
        if name != "enroll":
            self._cancel_scan(silent=True)

        for key, frame in self._screens.items():
            if key == name:
                frame.grid(row=0, column=0, sticky="nsew", padx=14, pady=14)
            else:
                frame.grid_forget()
        for key, btn in self._nav_buttons:
            btn.configure(fg_color=TEAL if key == name else "transparent")
        if name == "people":
            self._refresh_people()
        elif name == "report":
            self._refresh_report()
        elif name == "dashboard":
            self._refresh_dashboard_stats()

    # =====================================================
    # HELPERS
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
            self._stat_people.configure(text=str(count))
        except AttributeError:
            pass
        return count

    def _toast(self, message, ok=True):
        toast = ctk.CTkToplevel(self)
        toast.overrideredirect(True)
        toast.attributes("-topmost", True)
        x = self.winfo_rootx() + self.winfo_width() // 2 - 170
        y = self.winfo_rooty() + 24
        toast.geometry(f"340x46+{x}+{y}")
        toast.configure(fg_color=(GOOD if ok else BAD))
        ctk.CTkLabel(toast, text=message, text_color="#0c2b33",
                     font=load_font(13, True)).pack(expand=True)
        toast.after(1800, toast.destroy)

    def _status(self, text, ok=True):
        self.status_label.configure(text=f"● {text}", text_color=(GOOD if ok else BAD))

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
        scr.grid_columnconfigure(0, weight=3)
        scr.grid_columnconfigure(1, weight=2)
        scr.grid_rowconfigure(0, weight=1)

        # ---- camera card ----
        cam_card = ctk.CTkFrame(scr, fg_color=PANEL_2, corner_radius=16)
        cam_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        cam_card.grid_rowconfigure(1, weight=1)
        cam_card.grid_columnconfigure(0, weight=1)

        head = ctk.CTkFrame(cam_card, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 6))
        ctk.CTkLabel(head, text="دوربین زنده", font=load_font(17, True),
                     text_color=CREAM).pack(side="right")
        self._fps_label = ctk.CTkLabel(head, text="", font=load_font(11),
                                       text_color=MUTED)
        self._fps_label.pack(side="left")

        self._cam_label = ctk.CTkLabel(cam_card, text="دوربین خاموش است\n\nدکمهٔ «شروع دوربین» را بزنید",
                                       font=load_font(16), text_color=MUTED)
        self._cam_label.grid(row=1, column=0, sticky="nsew", padx=16, pady=6)

        ctl = ctk.CTkFrame(cam_card, fg_color="transparent")
        ctl.grid(row=2, column=0, sticky="ew", padx=16, pady=(6, 16))
        self._cam_toggle_btn = ctk.CTkButton(
            ctl, text="▶  شروع دوربین", font=load_font(15, True), height=46,
            fg_color=GOLD, hover_color="#c2913a", text_color=NAVY, corner_radius=12,
            command=self._toggle_dash_camera)
        self._cam_toggle_btn.pack(side="right")
        self._faces_label = ctk.CTkLabel(ctl, text="چهره‌ها: 0", font=load_font(13),
                                         text_color=CREAM)
        self._faces_label.pack(side="left")

        # ---- right column ----
        right = ctk.CTkFrame(scr, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_rowconfigure(3, weight=1)

        stats = ctk.CTkFrame(right, fg_color="transparent")
        stats.pack(fill="x")
        self._stat_people = self._stat_card(stats, "افراد ثبت‌شده", "0", "👥")
        self._stat_today = self._stat_card(stats, "حضور امروز", "0", "✅")
        self._stat_model = self._stat_card(stats, "وضعیت مدل‌ها", "آماده", "🧠")

        last = ctk.CTkFrame(right, fg_color=PANEL_2, corner_radius=16)
        last.pack(fill="both", expand=True, pady=(10, 0))
        ctk.CTkLabel(last, text="آخرین شناسایی", font=load_font(16, True),
                     text_color=CREAM).pack(anchor="e", padx=18, pady=(16, 6))

        body = ctk.CTkFrame(last, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=18, pady=8)
        self._last_avatar = ctk.CTkLabel(body, text="؟", font=ctk.CTkFont(size=52),
                                         width=110, height=110)
        self._last_avatar.pack(pady=(6, 8))
        self._last_name = ctk.CTkLabel(body, text="—", font=load_font(20, True),
                                       text_color=CREAM)
        self._last_name.pack()
        self._last_meta = ctk.CTkLabel(body, text="هنوز شناسایی‌ای انجام نشده",
                                       font=load_font(12), text_color=MUTED)
        self._last_meta.pack(pady=(2, 10))
        self._match_bar = ctk.CTkProgressBar(body, width=220, progress_color=GOOD,
                                             fg_color=CARD, height=14)
        self._match_bar.set(0)
        self._match_bar.pack()
        self._match_pct = ctk.CTkLabel(body, text="", font=load_font(11),
                                       text_color=MUTED)
        self._match_pct.pack(pady=(4, 0))

        return scr

    def _stat_card(self, parent, title, value, icon):
        card = ctk.CTkFrame(parent, fg_color=PANEL_2, corner_radius=16, width=150, height=96)
        card.pack(side="right", fill="x", expand=True, padx=(0, 8))
        card.pack_propagate(False)
        ctk.CTkLabel(card, text=icon, font=ctk.CTkFont(size=22)).pack(
            anchor="e", padx=10, pady=(8, 0))
        lbl = ctk.CTkLabel(card, text=value, font=load_font(24, True),
                           text_color=GOLD)
        lbl.pack(anchor="e", padx=12)
        ctk.CTkLabel(card, text=title, font=load_font(10),
                     text_color=MUTED).pack(anchor="e", padx=12, pady=(0, 8))
        return lbl

    def _refresh_dashboard_stats(self):
        self._stat_people.configure(text=str(self.db.count_people()))
        self._stat_today.configure(text=str(self.db.attendance_today_count()))
        self._stat_model.configure(text="آماده" if models_present() else "خطا")
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
        self._cam_toggle_btn.configure(text="■  توقف دوربین", fg_color=BAD,
                                       hover_color="#d84e41", text_color="#1a0b08")
        self._cam_label.configure(text="")
        self._refresh_mean_embeddings()
        self._status("دوربین روشن است")
        self._dash_loop()

    def _stop_dash_camera(self):
        self._dash_running = False
        if self._dash_cam is not None:
            self._dash_cam.release()
            self._dash_cam = None
        try:
            self._cam_toggle_btn.configure(text="▶  شروع دوربین", fg_color=GOLD,
                                           hover_color="#c2913a", text_color=NAVY)
            self._cam_label.configure(text="دوربین خاموش است\n\nدکمهٔ «شروع دوربین» را بزنید",
                                      image=None)
            self._faces_label.configure(text="چهره‌ها: 0")
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

        engine = self._get_engine()
        display = frame.copy()
        faces = engine.detect(frame)
        threshold = float(self.db.get_setting("match_threshold", COSINE_THRESHOLD))

        from face_engine import cosine_sim
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
            color = (60, 220, 60) if known else (60, 60, 230)
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
            if known:
                person = self.db.get_person(best_id)
                name = person["full_name"] if person else str(best_id)
                caption = f"{name}  {best_sim * 100:.0f}%"
                self._log_attendance_gui(best_id)
                best_show = (person, best_sim)
            else:
                caption = f"Unknown  {best_sim * 100:.0f}%"
            cv2.putText(display, caption, (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

        if best_show:
            self._show_last_recognition(best_show[0], best_show[1])

        self._faces_label.configure(text=f"چهره‌ها: {len(faces)}")
        now = time.time()
        self._fps_history.append(now)
        self._fps_history = [t for t in self._fps_history if now - t < 1.0]
        self._fps_label.configure(text=f"FPS: {len(self._fps_history)}")

        img = self._frame_to_ctk(display, 640, 400)
        self._cam_label.configure(image=img, text="")
        self._cam_label._image = img
        self.after(33, self._dash_loop)

    def _log_attendance_gui(self, person_id):
        now = datetime.now()
        last = self._att_cache.get(person_id)
        if last and (now - last).total_seconds() < 60:
            return
        self.db.add_attendance(person_id)
        self._att_cache[person_id] = now
        self._stat_today.configure(text=str(self.db.attendance_today_count()))

    def _show_last_recognition(self, person, sim):
        if person is None:
            return
        self._last_avatar.configure(image=self._get_photo_ctk(person, 110))
        self._last_name.configure(text=person["full_name"])
        meta = " | ".join(p for p in [person.get("student_id"),
                                      person.get("department")] if p)
        self._last_meta.configure(text=meta or "—")
        self._match_bar.set(max(0.0, min(1.0, sim)))
        self._match_bar.configure(progress_color=(GOOD if sim >= 0.363 else GOLD))
        self._match_pct.configure(text=f"شباهت: {sim * 100:.1f}٪")

    # =====================================================
    # SCREEN 2: ENROLL (Face-ID style guided scan)
    # =====================================================

    def _build_enroll(self):
        scr = ctk.CTkFrame(self.content, fg_color="transparent")
        scr.grid_columnconfigure(0, weight=2)
        scr.grid_columnconfigure(1, weight=3)
        scr.grid_rowconfigure(0, weight=1)

        # ---- form card ----
        form = ctk.CTkScrollableFrame(scr, fg_color=PANEL_2, corner_radius=16)
        form.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        ctk.CTkLabel(form, text="مشخصات فرد", font=load_font(18, True),
                     text_color=CREAM).pack(anchor="e", pady=(10, 2), padx=8)
        ctk.CTkLabel(form, text="فیلدهای ستاره‌دار الزامی‌اند", font=load_font(10),
                     text_color=MUTED).pack(anchor="e", padx=8, pady=(0, 10))

        def field(label, placeholder, required=False):
            ctk.CTkLabel(form, text=label + (" *" if required else ""),
                         font=load_font(12, True), text_color=CREAM).pack(
                anchor="e", padx=8, pady=(6, 2))
            entry = ctk.CTkEntry(form, placeholder_text=placeholder, height=38,
                                 font=load_font(13), justify="right",
                                 fg_color=CARD, border_color=TEAL)
            entry.pack(fill="x", padx=8, pady=(0, 2))
            return entry

        self._f_name = field("نام کامل", "مثلاً: سید احمد رضایی", required=True)
        self._f_sid = field("شمارهٔ کارت / پوهنځی", "مثلاً: CS-1402-015")
        self._f_dept = field("دانشکده / بخش", "مثلاً: کمپیوتر ساینس")
        self._f_phone = field("شمارهٔ تماس", "مثلاً: 07xx xxx xxx")
        self._f_email = field("ایمیل", "name@example.com")
        ctk.CTkLabel(form, text="یادداشت", font=load_font(12, True),
                     text_color=CREAM).pack(anchor="e", padx=8, pady=(6, 2))
        self._f_notes = ctk.CTkTextbox(form, height=58, font=load_font(12),
                                       fg_color=CARD, border_color=TEAL, border_width=2)
        self._f_notes.pack(fill="x", padx=8, pady=(0, 8))

        self._form_error = ctk.CTkLabel(form, text="", font=load_font(11),
                                        text_color=BAD)
        self._form_error.pack(anchor="e", padx=8, pady=(0, 6))

        btn_row = ctk.CTkFrame(form, fg_color="transparent")
        btn_row.pack(fill="x", padx=8, pady=(4, 16))
        self._scan_btn = ctk.CTkButton(btn_row, text="🎥  شروع اسکن چهره",
                                       font=load_font(14, True), height=44,
                                       fg_color=GOLD, hover_color="#c2913a",
                                       text_color=NAVY, corner_radius=12,
                                       command=self._start_scan)
        self._scan_btn.pack(side="right", fill="x", expand=True, padx=(0, 6))
        ctk.CTkButton(btn_row, text="پاک کردن", font=load_font(12), height=44,
                      fg_color=CARD, hover_color=TEAL, corner_radius=12,
                      command=self._clear_enroll_form).pack(side="left")

        # ---- scan card ----
        scan = ctk.CTkFrame(scr, fg_color=PANEL_2, corner_radius=16)
        scan.grid(row=0, column=1, sticky="nsew")
        scan.grid_columnconfigure(0, weight=1)
        scan.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(scan, text="اسکن چهره (چند زاویه‌ای)", font=load_font(17, True),
                     text_color=CREAM).grid(row=0, column=0, sticky="e",
                                            padx=18, pady=(14, 2))

        self._scan_preview = ctk.CTkLabel(scan,
                                          text="پس از پر کردن مشخصات،\n«شروع اسکن چهره» را بزنید",
                                          font=load_font(15), text_color=MUTED)
        self._scan_preview.grid(row=1, column=0, sticky="nsew", padx=18, pady=6)

        self._scan_instruction = ctk.CTkLabel(scan, text="آماده برای اسکن",
                                              font=load_font(18, True), text_color=GOLD)
        self._scan_instruction.grid(row=2, column=0, pady=(4, 0))

        chips = ctk.CTkFrame(scan, fg_color="transparent")
        chips.grid(row=3, column=0, pady=6)
        self._zone_chips = {}
        for zone in ZONE_ORDER:
            chip = ctk.CTkLabel(chips, text="⬜", font=load_font(10),
                                fg_color=CARD, corner_radius=8,
                                width=150, height=26, text_color=CREAM)
            chip.pack(side="right", padx=3, pady=2)
            self._zone_chips[zone] = chip

        progress_row = ctk.CTkFrame(scan, fg_color="transparent")
        progress_row.grid(row=4, column=0, sticky="ew", padx=18)
        self._scan_progress = ctk.CTkProgressBar(progress_row, progress_color=GOLD,
                                                 fg_color=CARD, height=14)
        self._scan_progress.set(0)
        self._scan_progress.pack(fill="x")
        self._scan_count = ctk.CTkLabel(scan, text="0 / 24 نمونه",
                                        font=load_font(11), text_color=MUTED)
        self._scan_count.grid(row=5, column=0, pady=(2, 2))
        self._scan_reason = ctk.CTkLabel(scan, text="", font=load_font(10),
                                         text_color=BAD)
        self._scan_reason.grid(row=6, column=0)

        btn_row2 = ctk.CTkFrame(scan, fg_color="transparent")
        btn_row2.grid(row=7, column=0, sticky="ew", padx=18, pady=(4, 16))
        ctk.CTkButton(btn_row2, text="لغو اسکن", font=load_font(12),
                      fg_color=CARD, hover_color=BAD, corner_radius=10,
                      command=self._cancel_scan).pack(side="right")
        self._finish_btn = ctk.CTkButton(btn_row2, text="پایان و ذخیره ✓",
                                         font=load_font(12, True), height=38,
                                         fg_color=TEAL, hover_color="#25707c",
                                         state="disabled", corner_radius=10,
                                         command=self._finish_scan)
        self._finish_btn.pack(side="left")

        return scr

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
            messagebox.showerror("خطا", "دوربین باز نشد.")
            return

        width = int(self._enroll_cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._enroll_cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        engine = self._get_engine()
        engine.set_input_size(width, height)

        self._enroll_state = {
            "person": data,
            "samples": [],          # list of dicts: emb, zone, quality, crop(bgr)
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
        display = frame.copy()
        faces = engine.detect(frame)
        faces = [f for f in faces if f[2] >= 50 and f[3] >= 50]

        # draw face guide (ellipse) + largest face box
        hgt, wid = display.shape[:2]
        cv2.ellipse(display, (wid // 2, hgt // 2), (wid // 4, hgt // 3),
                    0, 0, 360, (217, 164, 65), 2)

        now = time.time()
        if len(faces) == 1:
            f = max(faces, key=lambda r: r[2] * r[3])
            x, y, w, h = [int(v) for v in f[:4]]
            cv2.rectangle(display, (x, y), (x + w, y + h), (217, 164, 65), 2)

            quality_ok, reason, score = engine.sample_quality(frame, f)
            self._scan_reason.configure(
                text="" if quality_ok else QUALITY_REASON_TEXT.get(reason, reason),
                text_color=(GOOD if quality_ok else BAD))

            yaw_zone, pitch_zone, _yaw, _pitch = engine.head_pose(f)
            missing = self._missing_zones()

            # decide which zone this frame fills
            if yaw_zone != "center":
                zone = yaw_zone
            elif pitch_zone != "level":
                zone = pitch_zone
            else:
                zone = "center"

            if quality_ok:
                # if the natural zone is full, capture anyway as "overflow"
                # (counts toward total but not toward the zone quota)
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
            self._finish_btn.configure(state=("normal" if total >= 12 else "disabled"))
            if total >= 12:
                self._finish_btn.configure(fg_color=GOOD, text_color=NAVY)

            if missing:
                self._scan_instruction.configure(
                    text=MISSING_PROMPT[missing[0]],
                    text_color=GOLD)
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
                text_color=(GOOD if done else CREAM))

    def _release_enroll_camera(self):
        if self._enroll_cam is not None:
            self._enroll_cam.release()
            self._enroll_cam = None

    def _cancel_scan(self, silent=False):
        was_active = self._enroll_state is not None
        self._enroll_state = None
        self._release_enroll_camera()
        # always restore usable UI state (also when leaving the screen)
        self._scan_btn.configure(state="normal")
        self._finish_btn.configure(state="disabled", fg_color=TEAL)
        self._scan_preview.configure(image=None,
                                     text="پس از پر کردن مشخصات،\n«شروع اسکن چهره» را بزنید")
        self._scan_instruction.configure(text="آماده برای اسکن", text_color=GOLD)
        self._scan_progress.set(0)
        self._scan_count.configure(text=f"0 / {sum(ZONE_TARGETS.values())} نمونه")
        self._scan_reason.configure(text="")
        for zone in ZONE_ORDER:
            self._zone_chips[zone].configure(
                text="⬜ " + ZONE_LABELS[zone].split("  ", 1)[1],
                text_color=CREAM)
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

        # save person + faces
        person_id = self.db.add_person(**st["person"])
        for s in st["samples"]:
            self.db.add_face(person_id, s["emb"], s["zone"], s["quality"])

        # profile photo: best-quality crop
        if st["samples"]:
            best = max(st["samples"], key=lambda s: s["quality"])
            photo_path = PROFILES_DIR / f"person_{person_id}.jpg"
            cv2.imwrite(str(photo_path), best["crop"])
            self.db.update_person(person_id, photo_path=str(photo_path))

        name = st["person"]["full_name"]
        self._enroll_state = None
        self._scan_btn.configure(state="normal")
        self._finish_btn.configure(state="disabled", fg_color=TEAL)
        self._refresh_mean_embeddings()
        self._refresh_people_count()
        self._status(f"«{name}» ثبت شد")
        self._show_enroll_summary(person_id, name, len(feats), avg_quality)
        # reset form for the next person
        self._clear_enroll_form()
        self._cancel_scan(silent=True)

    def _show_enroll_summary(self, person_id, name, sample_count, avg_quality):
        win = ctk.CTkToplevel(self)
        win.title("ثبت موفق")
        win.geometry("420x400")
        win.transient(self)
        win.grab_set()
        ctk.CTkLabel(win, text="✅  ثبت‌نام موفق!", font=load_font(22, True),
                     text_color=GOOD).pack(pady=(22, 6))
        person = self.db.get_person(person_id)
        if person:
            img = photo_or_avatar(person, 130)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(130, 130))
            ctk.CTkLabel(win, image=ctk_img, text="").pack(pady=6)
        ctk.CTkLabel(win, text=name, font=load_font(18, True),
                     text_color=CREAM).pack()
        ctk.CTkLabel(win,
                     text=f"{sample_count} نمونهٔ چهره از زوایای مختلف ذخیره شد\n"
                          f"امتیاز کیفیت: {avg_quality:.0f} از 100",
                     font=load_font(12), text_color=MUTED, justify="center").pack(pady=8)

        def go(target):
            win.destroy()
            self.show_screen(target)

        row = ctk.CTkFrame(win, fg_color="transparent")
        row.pack(pady=10)
        ctk.CTkButton(row, text="مشاهدهٔ فهرست افراد", font=load_font(12, True),
                      fg_color=TEAL, hover_color="#25707c", corner_radius=10,
                      command=lambda: go("people")).pack(side="right", padx=6)
        ctk.CTkButton(row, text="ثبت فرد دیگر", font=load_font(12, True),
                      fg_color=GOLD, hover_color="#c2913a", text_color=NAVY,
                      corner_radius=10,
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
        top.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ctk.CTkLabel(top, text="افراد ثبت‌شده", font=load_font(20, True),
                     text_color=CREAM).pack(side="right")
        self._search_entry = ctk.CTkEntry(top, placeholder_text="جستجو: نام، شمارهٔ کارت، دانشکده…",
                                          width=300, height=38, justify="right",
                                          fg_color=CARD, border_color=TEAL,
                                          font=load_font(12))
        self._search_entry.pack(side="right", padx=(0, 12))
        self._search_entry.bind("<KeyRelease>", lambda e: self._refresh_people())
        ctk.CTkButton(top, text="➕  افزودن فرد", font=load_font(12, True),
                      fg_color=GOLD, hover_color="#c2913a", text_color=NAVY,
                      corner_radius=10, height=38,
                      command=lambda: self.show_screen("enroll")).pack(side="left")

        self._people_list = ctk.CTkScrollableFrame(scr, fg_color=PANEL_2,
                                                   corner_radius=16)
        self._people_list.grid(row=1, column=0, sticky="nsew")
        return scr

    def _refresh_people(self):
        for widget in self._people_list.winfo_children():
            widget.destroy()
        search = self._search_entry.get().strip() if hasattr(self, "_search_entry") else ""
        people = self.db.list_people(search)
        if not people:
            ctk.CTkLabel(self._people_list,
                         text="هنوز کسی ثبت نشده — از «ثبت فرد جدید» شروع کنید",
                         font=load_font(13), text_color=MUTED).pack(pady=30)
            return
        for person in people:
            row = ctk.CTkFrame(self._people_list, fg_color=CARD, corner_radius=12,
                               height=78)
            row.pack(fill="x", padx=10, pady=5)
            row.pack_propagate(False)
            avatar = photo_or_avatar(person, 62)
            ctk_img = ctk.CTkImage(light_image=avatar, dark_image=avatar, size=(62, 62))
            ctk.CTkLabel(row, image=ctk_img, text="").pack(side="right", padx=12, pady=8)

            info = ctk.CTkFrame(row, fg_color="transparent")
            info.pack(side="right", fill="both", expand=True, pady=9)
            ctk.CTkLabel(info, text=person["full_name"], font=load_font(14, True),
                         text_color=CREAM, anchor="e").pack(fill="x")
            meta = " | ".join(p for p in [person.get("student_id"),
                                          person.get("department"),
                                          f"{person.get('face_count', 0)} نمونه"]
                              if p)
            ctk.CTkLabel(info, text=meta, font=load_font(10), text_color=MUTED,
                         anchor="e").pack(fill="x")

            ctk.CTkButton(row, text="🗑 حذف", width=84, height=34,
                          fg_color="transparent", hover_color=BAD, text_color=CREAM,
                          corner_radius=8,
                          command=lambda p=person: self._delete_person(p)).pack(
                side="left", padx=12)

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
        top.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ctk.CTkLabel(top, text="گزارش حضور", font=load_font(20, True),
                     text_color=CREAM).pack(side="right")
        self._report_scope = ctk.CTkSegmentedButton(
            top, values=["امروز", "همه"], font=load_font(12),
            command=lambda v: self._refresh_report())
        self._report_scope.set("امروز")
        self._report_scope.pack(side="left", padx=(10, 0))
        ctk.CTkButton(top, text="⬇ خروجی CSV", font=load_font(12, True),
                      fg_color=TEAL, hover_color="#25707c", corner_radius=10,
                      height=38, command=self._export_csv).pack(side="left")

        self._report_count = ctk.CTkLabel(top, text="۰ رکورد", font=load_font(12),
                                          text_color=MUTED)
        self._report_count.pack(side="left", padx=12)

        self._report_list = ctk.CTkScrollableFrame(scr, fg_color=PANEL_2,
                                                   corner_radius=16)
        self._report_list.grid(row=1, column=0, sticky="nsew")
        return scr

    def _refresh_report(self):
        for widget in self._report_list.winfo_children():
            widget.destroy()
        today_only = self._report_scope.get() == "امروز"
        rows = self.db.attendance_today() if today_only else self.db.attendance_all()
        self._report_count.configure(text=f"{len(rows)} رکورد")

        header = ctk.CTkFrame(self._report_list, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=(8, 2))
        for text, width in [("زمان", 150), ("دانشکده", 160), ("شمارهٔ کارت", 170), ("نام", 0)]:
            ctk.CTkLabel(header, text=text, font=load_font(11, True),
                         text_color=GOLD, width=width, anchor="e").pack(
                side="right", padx=8)

        if not rows:
            ctk.CTkLabel(self._report_list, text="رکوردی ثبت نشده است",
                         font=load_font(13), text_color=MUTED).pack(pady=26)
            return
        for i, r in enumerate(rows):
            row = ctk.CTkFrame(self._report_list,
                               fg_color=(CARD if i % 2 == 0 else PANEL_2),
                               corner_radius=8, height=40)
            row.pack(fill="x", padx=10, pady=2)
            row.pack_propagate(False)
            ctk.CTkLabel(row, text=r["recorded_at"], font=load_font(11),
                         text_color=CREAM, width=150, anchor="e").pack(side="right", padx=8)
            ctk.CTkLabel(row, text=r.get("department") or "—", font=load_font(11),
                         text_color=MUTED, width=160, anchor="e").pack(side="right", padx=8)
            ctk.CTkLabel(row, text=r.get("student_id") or "—", font=load_font(11),
                         text_color=MUTED, width=170, anchor="e").pack(side="right", padx=8)
            ctk.CTkLabel(row, text=r["full_name"], font=load_font(11, True),
                         text_color=CREAM, anchor="e").pack(side="right", padx=8)

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

        ctk.CTkLabel(scr, text="تنظیمات و سلامت سیستم", font=load_font(20, True),
                     text_color=CREAM).pack(anchor="e", pady=(4, 10), padx=6)

        # --- camera & threshold card ---
        card = ctk.CTkFrame(scr, fg_color=PANEL_2, corner_radius=16)
        card.pack(fill="x", padx=6, pady=6)
        ctk.CTkLabel(card, text="دوربین و آستانهٔ تطبیق", font=load_font(15, True),
                     text_color=CREAM).pack(anchor="e", padx=16, pady=(12, 6))

        row1 = ctk.CTkFrame(card, fg_color="transparent")
        row1.pack(fill="x", padx=16, pady=4)
        ctk.CTkLabel(row1, text="شمارهٔ دوربین:", font=load_font(12),
                     text_color=CREAM).pack(side="right", padx=(0, 8))
        saved_cam = self.db.get_setting("camera_index", "0")
        cam_menu = ctk.CTkOptionMenu(row1, values=["0", "1", "2", "3"], width=80,
                                     fg_color=CARD, button_color=TEAL,
                                     command=lambda v: self.db.set_setting("camera_index", v))
        cam_menu.set(saved_cam)
        cam_menu.pack(side="right")

        row2 = ctk.CTkFrame(card, fg_color="transparent")
        row2.pack(fill="x", padx=16, pady=(10, 4))
        ctk.CTkLabel(row2, text="آستانهٔ تطبیق (سخت‌گیری):", font=load_font(12),
                     text_color=CREAM).pack(side="right", padx=(0, 8))
        self._threshold_label = ctk.CTkLabel(row2, text="", font=load_font(12, True),
                                             text_color=GOLD)
        self._threshold_label.pack(side="left")
        saved_thr = float(self.db.get_setting("match_threshold", str(COSINE_THRESHOLD)))
        slider = ctk.CTkSlider(card, from_=0.20, to=0.60, number_of_steps=40,
                               command=self._on_threshold_change)
        slider.set(saved_thr)
        slider.pack(fill="x", padx=16, pady=(0, 4))
        self._threshold_label.configure(text=f"{saved_thr:.2f}")
        ctk.CTkLabel(card, text="مقدار پیشنهادی برای SFace: 0.36 — بالاتر = سخت‌گیرتر (کمتر اشتباه می‌گیرد)",
                     font=load_font(10), text_color=MUTED).pack(anchor="e",
                                                                padx=16, pady=(0, 14))

        # --- health check card ---
        card2 = ctk.CTkFrame(scr, fg_color=PANEL_2, corner_radius=16)
        card2.pack(fill="x", padx=6, pady=6)
        ctk.CTkLabel(card2, text="بررسی سلامت پروژه", font=load_font(15, True),
                     text_color=CREAM).pack(anchor="e", padx=16, pady=(12, 6))
        self._health_box = ctk.CTkTextbox(card2, height=150, font=ctk.CTkFont(
            family="Consolas", size=12), fg_color=CARD, text_color=CREAM)
        self._health_box.pack(fill="x", padx=16, pady=4)
        self._health_box.insert("1.0", "برای اجرای بررسی «اجرای تست» را بزنید…")
        ctk.CTkButton(card2, text="🩺  اجرای تست سلامت", font=load_font(12, True),
                      fg_color=TEAL, hover_color="#25707c", corner_radius=10,
                      height=40, command=self._run_health_check).pack(anchor="e",
                                                                      padx=16, pady=(4, 14))

        # --- backup card ---
        card3 = ctk.CTkFrame(scr, fg_color=PANEL_2, corner_radius=16)
        card3.pack(fill="x", padx=6, pady=6)
        ctk.CTkLabel(card3, text="پشتیبان‌گیری", font=load_font(15, True),
                     text_color=CREAM).pack(anchor="e", padx=16, pady=(12, 4))
        ctk.CTkLabel(card3, text="یک کپی از فایل دیتابیس (افراد + چهره‌ها + حضور) در مسیر دلخواه ذخیره می‌شود.",
                     font=load_font(11), text_color=MUTED).pack(anchor="e", padx=16)
        ctk.CTkButton(card3, text="💾  ذخیرهٔ پشتیبان دیتابیس", font=load_font(12, True),
                      fg_color=CARD, hover_color=TEAL, corner_radius=10, height=40,
                      command=self._backup_db).pack(anchor="e", padx=16, pady=(6, 14))

        # --- about card ---
        card4 = ctk.CTkFrame(scr, fg_color=PANEL_2, corner_radius=16)
        card4.pack(fill="x", padx=6, pady=(6, 20))
        ctk.CTkLabel(card4, text="دربارهٔ پروژه", font=load_font(15, True),
                     text_color=CREAM).pack(anchor="e", padx=16, pady=(12, 4))
        ctk.CTkLabel(
            card4,
            text=("سیستم شناسایی هوشمند چهره — نسخهٔ 2.0\n"
                  "تشخیص: YuNet (OpenCV Zoo)   شناسایی: SFace (128-D Embedding)\n"
                  "پوهنځی کمپیوتر ساینس — پوهنتون غرجستان | 2026\n"
                  "سازنده: Sayed Mohammad Naqi Sadat"),
            font=load_font(11), text_color=MUTED, justify="right").pack(
            anchor="e", padx=16, pady=(0, 16))
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

        report("YuNet model file present", (PROJECT_ROOT / "models/face_detection_yunet_2023mar.onnx").exists())
        report("SFace model file present", (PROJECT_ROOT / "models/face_recognition_sface_2021dec.onnx").exists())

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
