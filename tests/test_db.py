"""Unit tests for the SQLite layer (src/db.py)."""
import numpy as np

from db import Database


def test_person_crud(tmp_path):
    db = Database(tmp_path / "t.db")
    assert db.count_people() == 0
    pid = db.add_person("احمد رضایی", student_id="CS-1", department="CS")
    assert db.count_people() == 1
    assert db.student_id_exists("CS-1")
    assert db.get_person(pid)["full_name"] == "احمد رضایی"
    assert len(db.list_people()) == 1
    assert len(db.list_people("احمد")) == 1
    assert len(db.list_people("NOTFOUND")) == 0
    db.update_person(pid, phone="0712")
    assert db.get_person(pid)["phone"] == "0712"
    db.delete_person(pid)
    assert db.count_people() == 0
    db.close()


def test_faces_and_mean_embeddings(tmp_path):
    db = Database(tmp_path / "t.db")
    pid = db.add_person("A", "1")
    e1 = np.ones(128, dtype=np.float32) / np.sqrt(128)
    e2 = np.zeros(128, dtype=np.float32)
    e2[0] = 1.0
    db.add_face(pid, e1, "center", 90.0)
    db.add_face(pid, e2, "up", 80.0)
    assert db.get_face_count(pid) == 2
    means = db.get_mean_embeddings()
    assert set(means) == {pid}
    assert abs(float(np.linalg.norm(means[pid])) - 1.0) < 1e-5
    assert db.get_best_face_quality(pid) == 85.0
    db.close()


def test_attendance_export_and_integrity(tmp_path):
    db = Database(tmp_path / "t.db")
    pid = db.add_person("A", "1", "CS")
    db.add_attendance(pid)
    db.add_attendance(pid)
    assert db.attendance_today_count() == 1       # distinct people
    assert len(db.attendance_today()) == 2
    assert len(db.attendance_all()) == 2
    csv_path = tmp_path / "attendance.csv"
    db.export_attendance_csv(csv_path)
    text = csv_path.read_text(encoding="utf-8-sig")
    assert "Name" in text and "A" in text
    assert all(dict(db.integrity_check()).values())
    db.delete_person(pid)                          # cascade
    assert db.attendance_all() == []
    db.close()
