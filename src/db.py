#!/usr/bin/env python3
# ============================================================
# SQLite Database Layer for Smart Face Recognition
# Faculty of Computer Science — Ghorjistan University
#
# Tables:
#   people     — person profile (name, student_id, department, ...)
#   faces      — one row per enrolled sample (128-d SFace embedding, pose, quality)
#   attendance — recognition log (who, when)
#   settings   — key/value app settings
# ============================================================

import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "face_recognition.db"
PROFILES_DIR = PROJECT_ROOT / "data" / "profiles"

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS people (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name   TEXT NOT NULL,
    student_id  TEXT UNIQUE,
    department  TEXT,
    phone       TEXT,
    email       TEXT,
    notes       TEXT,
    photo_path  TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now','localtime'))
);

CREATE TABLE IF NOT EXISTS faces (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id   INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    embedding   BLOB NOT NULL,
    pose        TEXT,
    quality     REAL,
    captured_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
);

CREATE TABLE IF NOT EXISTS attendance (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id   INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    recorded_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def _to_blob(embedding):
    return np.ascontiguousarray(embedding, dtype=np.float32).tobytes()


def _to_array(blob):
    return np.frombuffer(blob, dtype=np.float32)


class Database:
    def __init__(self, path=None):
        self.path = Path(path) if path else DB_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        PROFILES_DIR.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)

    def close(self):
        self._conn.close()

    # ---------------- people ----------------

    def add_person(self, full_name, student_id=None, department=None,
                   phone=None, email=None, notes=None, photo_path=None):
        cur = self._conn.execute(
            """INSERT INTO people (full_name, student_id, department, phone, email, notes, photo_path)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (full_name.strip(),
             (student_id or "").strip() or None,
             (department or "").strip() or None,
             (phone or "").strip() or None,
             (email or "").strip() or None,
             (notes or "").strip() or None,
             photo_path),
        )
        self._conn.commit()
        return cur.lastrowid

    def update_person(self, person_id, **fields):
        allowed = {"full_name", "student_id", "department", "phone", "email", "notes", "photo_path"}
        sets, values = [], []
        for key, value in fields.items():
            if key in allowed:
                sets.append(f"{key} = ?")
                values.append((value or "").strip() if isinstance(value, str) else value)
        if not sets:
            return
        values.append(person_id)
        self._conn.execute(f"UPDATE people SET {', '.join(sets)} WHERE id = ?", values)
        self._conn.commit()

    def get_person(self, person_id):
        row = self._conn.execute(
            "SELECT * FROM people WHERE id = ?", (person_id,)).fetchone()
        return dict(row) if row else None

    def student_id_exists(self, student_id, exclude_id=None):
        student_id = (student_id or "").strip()
        if not student_id:
            return False
        if exclude_id is None:
            row = self._conn.execute(
                "SELECT 1 FROM people WHERE student_id = ?", (student_id,)).fetchone()
        else:
            row = self._conn.execute(
                "SELECT 1 FROM people WHERE student_id = ? AND id != ?",
                (student_id, exclude_id)).fetchone()
        return row is not None

    def list_people(self, search=""):
        search = (search or "").strip()
        query = """
            SELECT p.*, COUNT(f.id) AS face_count
            FROM people p LEFT JOIN faces f ON f.person_id = p.id
        """
        params = []
        if search:
            query += " WHERE p.full_name LIKE ? OR p.student_id LIKE ? OR p.department LIKE ?"
            like = f"%{search}%"
            params = [like, like, like]
        query += " GROUP BY p.id ORDER BY p.created_at DESC"
        return [dict(r) for r in self._conn.execute(query, params).fetchall()]

    def delete_person(self, person_id):
        person = self.get_person(person_id)
        self._conn.execute("DELETE FROM people WHERE id = ?", (person_id,))
        self._conn.commit()
        if person and person.get("photo_path"):
            try:
                Path(person["photo_path"]).unlink(missing_ok=True)
            except OSError:
                pass

    def count_people(self):
        return self._conn.execute("SELECT COUNT(*) FROM people").fetchone()[0]

    # ---------------- faces / embeddings ----------------

    def add_face(self, person_id, embedding, pose=None, quality=None):
        self._conn.execute(
            "INSERT INTO faces (person_id, embedding, pose, quality) VALUES (?, ?, ?, ?)",
            (person_id, _to_blob(embedding), pose, quality),
        )
        self._conn.commit()

    def get_mean_embeddings(self):
        """Return {person_id: mean L2-normalized embedding}."""
        rows = self._conn.execute(
            "SELECT person_id, embedding FROM faces").fetchall()
        by_person = {}
        for row in rows:
            by_person.setdefault(row["person_id"], []).append(_to_array(row["embedding"]))
        means = {}
        for person_id, feats in by_person.items():
            mean = np.mean(np.stack(feats), axis=0)
            norm = np.linalg.norm(mean)
            if norm > 0:
                mean = mean / norm
            means[person_id] = mean.astype(np.float32)
        return means

    def get_face_count(self, person_id):
        return self._conn.execute(
            "SELECT COUNT(*) FROM faces WHERE person_id = ?", (person_id,)).fetchone()[0]

    def get_best_face_quality(self, person_id):
        row = self._conn.execute(
            "SELECT AVG(quality) FROM faces WHERE person_id = ?", (person_id,)).fetchone()
        return float(row[0]) if row and row[0] is not None else 0.0

    # ---------------- attendance ----------------

    def add_attendance(self, person_id):
        self._conn.execute("INSERT INTO attendance (person_id) VALUES (?)", (person_id,))
        self._conn.commit()

    def attendance_today(self):
        rows = self._conn.execute(
            """SELECT a.recorded_at, p.full_name, p.student_id, p.department
               FROM attendance a JOIN people p ON p.id = a.person_id
               WHERE date(a.recorded_at) = date('now','localtime')
               ORDER BY a.recorded_at DESC""").fetchall()
        return [dict(r) for r in rows]

    def attendance_today_count(self):
        return self._conn.execute(
            "SELECT COUNT(DISTINCT person_id) FROM attendance"
            " WHERE date(recorded_at) = date('now','localtime')").fetchone()[0]

    def attendance_all(self, limit=None):
        sql = ("""SELECT a.recorded_at, p.full_name, p.student_id, p.department
                  FROM attendance a JOIN people p ON p.id = a.person_id
                  ORDER BY a.recorded_at DESC""")
        if limit:
            sql += f" LIMIT {int(limit)}"
        rows = self._conn.execute(sql).fetchall()
        return [dict(r) for r in rows]

    def export_attendance_csv(self, csv_path, today_only=True):
        import csv
        rows = self.attendance_today() if today_only else self.attendance_all()
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Student ID", "Department", "Date", "Time"])
            for r in rows:
                dt = datetime.strptime(r["recorded_at"], "%Y-%m-%d %H:%M:%S")
                writer.writerow([r["full_name"], r["student_id"] or "", r["department"] or "",
                                 f"{dt:%Y-%m-%d}", f"{dt:%H:%M:%S}"])
        return len(rows)

    # ---------------- settings ----------------

    def get_setting(self, key, default=None):
        row = self._conn.execute(
            "SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def set_setting(self, key, value):
        self._conn.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, str(value)))
        self._conn.commit()

    # ---------------- maintenance ----------------

    def integrity_check(self):
        """Return list of (label, ok) checks."""
        checks = []
        row = self._conn.execute("PRAGMA integrity_check").fetchone()
        ok = bool(row) and str(row[0]).strip().lower() == "ok"
        checks.append(("SQLite database readable & valid", ok))
        checks.append(("People table accessible",
                       self._conn.execute("SELECT COUNT(*) FROM people").fetchone() is not None))
        checks.append(("Faces table accessible",
                       self._conn.execute("SELECT COUNT(*) FROM faces").fetchone() is not None))
        try:
            test = PROFILES_DIR / ".write_test"
            test.write_text("ok", encoding="utf-8")
            test.unlink()
            checks.append(("Profiles directory writable", True))
        except OSError:
            checks.append(("Profiles directory writable", False))
        return checks

    def backup(self, dest_path):
        shutil.copy2(self.path, dest_path)
        return dest_path
