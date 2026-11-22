#!/usr/bin/env python3
# ============================================================
# Smart Face Recognition — Local Web Server (Flask)
# Frontend: HTML/CSS/JS (src/web) · Engine: YuNet + SFace
#
# Run:  python src/web_app.py
# Then open:  http://127.0.0.1:5000
# ============================================================

import base64
import os
import sqlite3
import sys
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_file, send_from_directory

from db import Database, PROFILES_DIR
from face_engine import (
    COSINE_THRESHOLD,
    FaceEngine,
    build_match_matrix,
    cosine_sim,
    match_matrix,
    models_present,
)

WEB_DIR = Path(__file__).resolve().parent / "web"
app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path="/static")

_engine = None
_engine_lock = threading.Lock()
_att_cache = {}
_cache_lock = threading.Lock()

REASON_TEXT = {
    "too_small": "چهره خیلی کوچک است — نزدیک‌تر شوید",
    "too_dark": "نور کم است — محیط را روشن‌تر کنید",
    "too_bright": "نور خیلی زیاد است",
    "low_contrast": "کنتراست چهره کم است",
    "blurry": "تصویر تار است — ثابت بمانید",
}


@contextmanager
def db_session():
    """One short-lived DB connection per request (thread-safe)."""
    db = Database()
    try:
        yield db
    finally:
        db.close()


def get_engine():
    global _engine
    if _engine is None:
        _engine = FaceEngine()
    return _engine


def decode_image(data) -> np.ndarray:
    """Accept a raw upload, multipart file, or base64 string."""
    if hasattr(data, "read"):
        buf = np.frombuffer(data.read(), dtype=np.uint8)
    elif isinstance(data, str) and "," in data:
        raw = base64.b64decode(data.split(",", 1)[1])
        buf = np.frombuffer(raw, dtype=np.uint8)
    elif isinstance(data, str):
        buf = np.frombuffer(base64.b64decode(data), dtype=np.uint8)
    else:
        raise ValueError("unsupported image payload")
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("could not decode image")
    return frame


# =====================================================
# pages / static
# =====================================================

@app.route("/")
def index():
    return send_from_directory(str(WEB_DIR), "index.html")


# =====================================================
# health & settings
# =====================================================

@app.get("/api/health")
def api_health():
    with db_session() as db:
        people = db.count_people()
        today = db.attendance_today_count()
    return jsonify({
        "models": models_present(),
        "people": people,
        "attendance_today": today,
        "version": "3.0-web",
    })


@app.get("/api/settings")
def api_get_settings():
    with db_session() as db:
        return jsonify({
            "match_threshold": float(db.get_setting("match_threshold",
                                                    str(COSINE_THRESHOLD))),
            "camera_index": db.get_setting("camera_index", "0"),
        })


@app.post("/api/settings")
def api_set_settings():
    data = request.get_json(force=True, silent=True) or {}
    with db_session() as db:
        for key, value in data.items():
            if key in ("match_threshold", "camera_index"):
                db.set_setting(key, value)
    return jsonify({"ok": True})


# =====================================================
# people
# =====================================================

@app.get("/api/people")
def api_people():
    search = request.args.get("search", "").strip()
    with db_session() as db:
        people = db.list_people(search)
    return jsonify(people)


@app.post("/api/people")
def api_create_person():
    data = request.get_json(force=True, silent=True) or {}
    name = (data.get("full_name") or "").strip()
    if len(name) < 2:
        return jsonify({"error": "نام کامل الزامی است"}), 400
    with db_session() as db:
        sid = (data.get("student_id") or "").strip()
        if sid and db.student_id_exists(sid):
            return jsonify({"error": "این شمارهٔ کارت قبلاً ثبت شده است"}), 409
        person_id = db.add_person(
            full_name=name,
            student_id=sid or None,
            department=data.get("department") or None,
            phone=data.get("phone") or None,
            email=data.get("email") or None,
            notes=data.get("notes") or None,
        )
    return jsonify({"id": person_id, "ok": True})


@app.delete("/api/people/<int:person_id>")
def api_delete_person(person_id):
    with db_session() as db:
        person = db.get_person(person_id)
        if not person:
            return jsonify({"error": "یافت نشد"}), 404
        db.delete_person(person_id)
    return jsonify({"ok": True})


# =====================================================
# recognition
# =====================================================

@app.post("/api/recognize")
def api_recognize():
    frame = decode_image(request.files.get("image") or request.get_data())
    with _engine_lock:
        engine = get_engine()
        with db_session() as db:
            mean_emb = db.get_mean_embeddings()
            people = {p["id"]: p for p in db.list_people()}
            threshold = float(db.get_setting("match_threshold",
                                             str(COSINE_THRESHOLD)))
        ids, matrix = build_match_matrix(mean_emb)
        faces = engine.detect(frame)

    matches = []
    last_known = None
    for f in faces:
        feat = engine.embed(frame, f)
        best_id, best_sim = match_matrix(feat, ids, matrix, threshold)
        known = best_id is not None and best_sim >= threshold
        person = people.get(best_id) if known else None
        matches.append({
            "x": int(f[0]), "y": int(f[1]), "w": int(f[2]), "h": int(f[3]),
            "sim": round(best_sim, 4),
            "known": known,
            "person_id": best_id,
            "name": person["full_name"] if person else None,
            "sid": person.get("student_id") if person else None,
            "dept": person.get("department") if person else None,
        })
        if known:
            # attendance only after at least 2 confirmations (anti false positive)
            now = datetime.now()
            with _cache_lock:
                entry = _att_cache.get(best_id)
                if entry is None or (now - entry["last"]).total_seconds() > 3:
                    entry = {"count": 0, "last": now, "logged": datetime.min}
                entry["count"] += 1
                entry["last"] = now
                _att_cache[best_id] = entry
                if (entry["count"] >= 2 and
                        (now - entry["logged"]).total_seconds() >= 60):
                    with db_session() as db:
                        db.add_attendance(best_id)
                    entry["logged"] = now
            last_known = {
                "person_id": best_id,
                "full_name": person["full_name"],
                "student_id": person.get("student_id"),
                "department": person.get("department"),
                "sim": round(best_sim, 4),
            }
    return jsonify({"matches": matches, "last_known": last_known})


# =====================================================
# enrollment
# =====================================================

@app.post("/api/enroll/analyze")
def api_enroll_analyze():
    frame = decode_image(request.files.get("image") or request.get_data())
    with _engine_lock:
        engine = get_engine()
        faces = engine.detect(frame)
        faces = [f for f in faces if f[2] >= 50 and f[3] >= 50]
        if len(faces) != 1:
            return jsonify({"ok": False,
                            "reason_text": "یک چهره در قاب باشد"
                            if faces else "چهره‌ای دیده نشد"})
        f = max(faces, key=lambda r: r[2] * r[3])
        ok, reason, score = engine.sample_quality(frame, f)
        if not ok:
            return jsonify({"ok": False, "reason": reason,
                            "reason_text": REASON_TEXT.get(reason, reason)})
        yaw_zone, pitch_zone, yaw, pitch = engine.head_pose(f)
    return jsonify({"ok": True, "score": score, "yaw_zone": yaw_zone,
                    "pitch_zone": pitch_zone, "yaw": yaw, "pitch": pitch})


@app.post("/api/enroll")
def api_enroll():
    data = request.get_json(force=True, silent=True) or {}
    person = data.get("person") or {}
    samples = data.get("samples") or []
    name = (person.get("full_name") or "").strip()
    if len(name) < 2:
        return jsonify({"error": "نام کامل الزامی است"}), 400
    if not samples:
        return jsonify({"error": "هیچ نمونه‌ای ثبت نشده"}), 400

    with _engine_lock:
        engine = get_engine()
        with db_session() as db:
            sid = (person.get("student_id") or "").strip()
            if sid and db.student_id_exists(sid):
                return jsonify({"error": "این شمارهٔ کارت قبلاً ثبت شده است"}), 409
            person_id = db.add_person(
                full_name=name,
                student_id=sid or None,
                department=person.get("department") or None,
                phone=person.get("phone") or None,
                email=person.get("email") or None,
                notes=person.get("notes") or None,
            )
            feats, qualities, crops = [], [], []
            for s in samples:
                try:
                    frame = decode_image(s.get("data", ""))
                except ValueError:
                    continue
                faces = engine.detect(frame)
                faces = [f for f in faces if f[2] >= 50 and f[3] >= 50]
                if not faces:
                    continue
                f = max(faces, key=lambda r: r[2] * r[3])
                _ok, _reason, score = engine.sample_quality(frame, f)
                if not _ok:
                    continue
                emb = engine.embed(frame, f)
                feats.append(emb)
                qualities.append(score)
                crops.append(FaceEngine.crop_face(frame, f))
                db.add_face(person_id, emb, s.get("pose"), score)

            if not feats:
                db.delete_person(person_id)
                return jsonify({"error": "نمونه‌های ارسالی معتبر نبودند"}), 400

            best_i = int(np.argmax(qualities))
            photo_path = PROFILES_DIR / f"person_{person_id}.jpg"
            cv2.imwrite(str(photo_path), crops[best_i])
            db.update_person(person_id, photo_path=str(photo_path))

    quality = float(np.mean(qualities)) if qualities else 0.0
    return jsonify({"ok": True, "person_id": person_id,
                    "samples": len(feats), "quality": quality})


# =====================================================
# attendance / export / backup
# =====================================================

@app.get("/api/attendance")
def api_attendance():
    scope = request.args.get("scope", "today")
    with db_session() as db:
        rows = db.attendance_today() if scope == "today" else db.attendance_all()
    return jsonify(rows)


@app.get("/api/export")
def api_export():
    today_only = request.args.get("today", "1") == "1"
    with db_session() as db:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        count = db.export_attendance_csv(tmp_path, today_only=today_only)
    fname = f"attendance_{datetime.now():%Y-%m-%d}.csv"
    return send_file(tmp_path, as_attachment=True, download_name=fname,
                     mimetype="text/csv")


@app.get("/api/backup")
def api_backup():
    """Safe backup: SQLite backup API handles WAL state correctly."""
    with db_session() as db:
        src = db._conn
        fd, path = tempfile.mkstemp(suffix=".db", prefix="facerec_backup_")
        os.close(fd)
        try:
            dst = sqlite3.connect(path)
            src.backup(dst)
            dst.close()
        except Exception:
            Path(path).unlink(missing_ok=True)
            raise
    fname = f"face_recognition_backup_{datetime.now():%Y%m%d_%H%M%S}.db"
    return send_file(path, as_attachment=True, download_name=fname,
                     mimetype="application/x-sqlite3")


# =====================================================
# health check (runs CLI-style selftest and returns lines)
# =====================================================

@app.post("/api/selftest")
def api_selftest():
    import subprocess
    exe = sys.executable
    try:
        out = subprocess.run(
            [exe, str(Path(__file__).parent / "face_app.py"), "selftest"],
            capture_output=True, text=True, timeout=120,
            cwd=str(Path(__file__).parent.parent))
        lines = [l for l in (out.stdout or "").splitlines() if l.strip()]
        return jsonify(lines[-14:])
    except Exception as exc:
        return jsonify([f"[FAIL] selftest could not run: {exc}"])


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    if not models_present():
        print("WARNING: model files missing. Run: python src/download_models.py")
    print("=" * 55)
    print("  چهره‌یاب — Smart Face Recognition (web)")
    print("  Open:  http://127.0.0.1:5000")
    print("=" * 55)
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)


if __name__ == "__main__":
    main()
