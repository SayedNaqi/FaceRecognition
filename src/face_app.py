#!/usr/bin/env python3
# ============================================================
# Smart Face Recognition System - CLI
# Faculty of Computer Science - Ghorjistan University
#
# Detection : YuNet  |  Recognition: SFace  |  Storage: SQLite
#
# Commands:
#   enroll    Register a person from webcam samples
#   recognize Live webcam recognition (demo mode)
#   image     Recognize faces in an image file
#   list      List enrolled people
#   remove    Delete an enrolled person
#   export    Export attendance to CSV
#   selftest  Headless end-to-end check (no webcam needed)
# ============================================================

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from db import Database, PROFILES_DIR
from face_engine import (
    COSINE_THRESHOLD,
    FaceEngine,
    build_match_matrix,
    cosine_sim,
    match_matrix,
    models_present,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

MIN_SAMPLES = 5
DEFAULT_SAMPLES = 30
CAPTURE_INTERVAL = 0.40
ATTENDANCE_COOLDOWN = 60

GREEN = (60, 220, 60)
RED = (60, 60, 230)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


# -------------------------
# shared helpers
# -------------------------

def open_camera(index):
    camera = cv2.VideoCapture(index)
    if not camera.isOpened():
        print("ERROR: Cannot access webcam. Check connection/permissions.")
        sys.exit(1)
    return camera


def draw_label(frame, x1, y1, text, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
    ty = max(y1, th + 14)
    cv2.rectangle(frame, (x1, ty - th - 10), (x1 + tw + 10, ty + 2), color, -1)
    cv2.putText(frame, text, (x1 + 5, ty - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, BLACK, 2, cv2.LINE_AA)


def draw_face_box(frame, face_row, color, label=None):
    x, y, w, h = [int(v) for v in face_row[:4]]
    x, y = max(x, 0), max(y, 0)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    if label:
        draw_label(frame, x, y, label, color)


def draw_footer(frame, text):
    h = frame.shape[0]
    cv2.putText(frame, text, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)


def match_against_db(feat, mean_embeddings, people_by_id, threshold):
    ids, matrix = build_match_matrix(mean_embeddings)
    best_id, best_sim = match_matrix(feat, ids, matrix, threshold)
    if best_id is not None:
        person = people_by_id.get(best_id)
        name = person["full_name"] if person else str(best_id)
        return name, best_sim
    return "Unknown", best_sim


def load_people_map(db):
    return {p["id"]: p for p in db.list_people()}


# -------------------------
# COMMAND: enroll
# -------------------------

def cmd_enroll(args):
    if not models_present():
        print("Model file(s) missing. Run: python src/download_models.py")
        return 1

    name = args.name.strip()
    if not name:
        print("ERROR: Please provide a name:  --name \"Ahmad\"")
        return 1

    db = Database()
    if args.student_id and db.student_id_exists(args.student_id):
        print(f"ERROR: student id '{args.student_id}' is already registered.")
        db.close()
        return 1

    camera = open_camera(args.camera)
    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    engine = FaceEngine((width, height), args.score_threshold)

    target = args.samples
    captures = []  # list of (embedding, quality, crop)
    last_capture = 0.0

    print("=" * 55)
    print(f"  ENROLLMENT MODE - {name}")
    print("=" * 55)
    print(f"Look at the camera. Capturing {target} samples automatically.")
    print("Press Q to cancel.\n")

    while len(captures) < target:
        ok, frame = camera.read()
        if not ok:
            print("ERROR: Cannot read webcam frame.")
            break

        faces = [f for f in engine.detect(frame) if f[2] >= 60 and f[3] >= 60]
        display = frame.copy()

        if faces:
            f0 = max(faces, key=lambda f: f[2] * f[3])
            quality_ok, reason, score = engine.sample_quality(frame, f0)
            draw_face_box(display, f0, GREEN if quality_ok else RED)
            if quality_ok:
                now = time.time()
                if now - last_capture >= CAPTURE_INTERVAL:
                    last_capture = now
                    captures.append({
                        "emb": engine.embed(frame, f0),
                        "quality": score,
                        "crop": FaceEngine.crop_face(frame, f0),
                    })
                    print(f"  Sample {len(captures)}/{target} captured (quality {score:.0f})")
            else:
                cv2.putText(display, f"Fix: {reason}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2, cv2.LINE_AA)
        else:
            cv2.putText(display, "No face detected - move closer", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2, cv2.LINE_AA)

        bar_w = int((len(captures) / target) * (width - 20))
        cv2.rectangle(display, (10, height - 55), (width - 10, height - 40), WHITE, 1)
        cv2.rectangle(display, (10, height - 55), (10 + bar_w, height - 40), GREEN, -1)
        draw_footer(display, f"Samples: {len(captures)}/{target}   |   Q: cancel")

        cv2.imshow("Enroll - Face Recognition", display)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            print("\nEnrollment cancelled by user.")
            camera.release()
            cv2.destroyAllWindows()
            db.close()
            return 1

    camera.release()
    cv2.destroyAllWindows()

    if len(captures) < MIN_SAMPLES:
        print(f"\nOnly {len(captures)} samples captured (min {MIN_SAMPLES}). Not saved.")
        db.close()
        return 1

    feats = np.stack([c["emb"] for c in captures])
    mean = feats.mean(axis=0)
    mean /= (np.linalg.norm(mean) + 1e-10)
    quality = float(np.mean([cosine_sim(c["emb"], mean) for c in captures]))
    avg_quality = float(np.mean([c["quality"] for c in captures]))

    person_id = db.add_person(
        full_name=name,
        student_id=args.student_id,
        department=args.department,
        phone=args.phone,
        email=args.email,
    )
    for c in captures:
        db.add_face(person_id, c["emb"], pose="frontal", quality=c["quality"])
    best = max(captures, key=lambda c: c["quality"])
    photo_path = PROFILES_DIR / f"person_{person_id}.jpg"
    cv2.imwrite(str(photo_path), best["crop"])
    db.update_person(person_id, photo_path=str(photo_path))
    db.close()

    print("\n" + "=" * 55)
    print(f"  Enrollment complete: {name}")
    print(f"  Samples saved : {len(captures)}")
    print(f"  Consistency   : {quality * 100:.1f}%  (higher is better)")
    print(f"  Avg. quality  : {avg_quality:.0f}/100")
    print("=" * 55)
    return 0


# -------------------------
# COMMAND: recognize (live demo)
# -------------------------

def cmd_recognize(args):
    if not models_present():
        print("Model file(s) missing. Run: python src/download_models.py")
        return 1

    db = Database()
    mean_emb = db.get_mean_embeddings()
    if not mean_emb:
        print("No enrolled faces found.")
        print("Enroll someone first:  python src/face_app.py enroll --name \"Your Name\"")
        print("Or use the GUI:        python src/gui.py")
        db.close()
        return 1

    people = load_people_map(db)
    threshold = float(db.get_setting("match_threshold", str(COSINE_THRESHOLD)))
    print(f"Loaded {len(mean_emb)} enrolled identities (threshold={threshold:.2f}).")

    camera = open_camera(args.camera)
    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    engine = FaceEngine((width, height), args.score_threshold)

    print("\nLIVE RECOGNITION - Press Q to quit, S to save a screenshot.\n")
    frame_count, start = 0, time.time()
    att_cache = {}

    while True:
        ok, frame = camera.read()
        if not ok:
            print("ERROR: Cannot read webcam frame.")
            break

        faces = engine.detect(frame)
        for f in faces:
            feat = engine.embed(frame, f)
            name, sim = match_against_db(feat, mean_emb, people, threshold)
            known = name != "Unknown"
            color = GREEN if known else RED
            draw_face_box(frame, f, color, f"{name}  {sim * 100:.0f}%")
            if known and args.log:
                pid = next((pid for pid, p in people.items()
                            if p["full_name"] == name), None)
                now = datetime.now()
                if pid is not None:
                    last = att_cache.get(pid)
                    if not last or (now - last).total_seconds() >= ATTENDANCE_COOLDOWN:
                        db.add_attendance(pid)
                        att_cache[pid] = now

        frame_count += 1
        fps = frame_count / max(time.time() - start, 1e-6)
        cv2.putText(frame, f"Faces: {len(faces)} | FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        draw_footer(frame, "Q: quit   |   S: screenshot")

        cv2.imshow("Smart Face Recognition - LIVE DEMO", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            OUTPUTS_DIR.mkdir(exist_ok=True)
            shot = OUTPUTS_DIR / f"screenshot_{datetime.now():%Y%m%d_%H%M%S}.png"
            cv2.imwrite(str(shot), frame)
            print(f"Screenshot saved: {shot}")

    camera.release()
    cv2.destroyAllWindows()
    db.close()
    print("Demo stopped.")
    return 0


# -------------------------
# COMMAND: image
# -------------------------

def cmd_image(args):
    if not models_present():
        print("Model file(s) missing. Run: python src/download_models.py")
        return 1

    image_path = Path(args.path)
    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        return 1

    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"ERROR: Could not read image: {image_path}")
        return 1

    height, width = frame.shape[:2]
    engine = FaceEngine((width, height), args.score_threshold)

    db = Database()
    mean_emb = db.get_mean_embeddings()
    people = load_people_map(db)
    threshold = float(db.get_setting("match_threshold", str(COSINE_THRESHOLD)))
    if not mean_emb:
        print("NOTE: No enrolled identities yet - faces will be labelled 'Unknown'.")
        print("      Enroll first:  python src/face_app.py enroll --name \"Your Name\"\n")

    faces = engine.detect(frame)
    print(f"Image: {image_path}  ({width}x{height})")
    print(f"Faces detected: {len(faces)}\n")

    for i, f in enumerate(faces, 1):
        feat = engine.embed(frame, f)
        name, sim = match_against_db(feat, mean_emb, people, threshold)
        color = GREEN if name != "Unknown" else RED
        draw_face_box(frame, f, color, f"{name}  {sim * 100:.0f}%")
        print(f"  Face {i}: {name:<20} similarity={sim * 100:.1f}%")

    OUTPUTS_DIR.mkdir(exist_ok=True)
    out_path = OUTPUTS_DIR / f"result_{image_path.stem}_{datetime.now():%Y%m%d_%H%M%S}.png"
    cv2.imwrite(str(out_path), frame)
    db.close()
    print(f"\nAnnotated image saved: {out_path}")
    return 0


# -------------------------
# COMMAND: list / remove / export
# -------------------------

def cmd_list(_args):
    db = Database()
    people = db.list_people()
    if not people:
        print("No enrolled identities yet.")
        db.close()
        return 0
    print(f"{'#':<4}{'Name':<28}{'Student ID':<18}{'Samples':<10}")
    print("-" * 60)
    for i, p in enumerate(people, 1):
        face_count = db.get_face_count(p["id"])
        print(f"{i:<4}{p['full_name']:<28}{(p.get('student_id') or '-'):<18}{face_count:<10}")
    print(f"\nTotal: {len(people)} identities")
    db.close()
    return 0


def cmd_remove(args):
    db = Database()
    name = args.name.strip().lower()
    matches = [p for p in db.list_people() if p["full_name"].lower() == name]
    if not matches:
        # fall back to substring match
        matches = [p for p in db.list_people() if name in p["full_name"].lower()]
    if not matches:
        print(f"No enrolled person matching '{args.name}'.")
        db.close()
        return 1
    for person in matches:
        db.delete_person(person["id"])
        print(f"Removed '{person['full_name']}' (id={person['id']}).")
    db.close()
    return 0


def cmd_export(args):
    db = Database()
    OUTPUTS_DIR.mkdir(exist_ok=True)
    target = Path(args.path) if args.path else \
        OUTPUTS_DIR / f"attendance_{datetime.now():%Y-%m-%d}.csv"
    count = db.export_attendance_csv(target, today_only=args.today)
    db.close()
    print(f"Exported {count} attendance record(s) to {target}")
    return 0


# -------------------------
# COMMAND: selftest (headless, no webcam)
# -------------------------

def cmd_selftest(args):
    import tempfile

    print("=" * 55)
    print("  SELF-TEST (no webcam / no display required)")
    print("=" * 55)
    failures = []

    def check(label, condition):
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {label}")
        if not condition:
            failures.append(label)

    check("Model files present", models_present())
    test_image = PROJECT_ROOT / "data" / "images" / "test.jpg"
    frame = cv2.imread(str(test_image))
    check("test image readable", frame is not None)

    height, width = frame.shape[:2]
    engine = FaceEngine((width, height))
    check("YuNet detector loads", engine.detector is not None)
    check("SFace recognizer loads", engine.recognizer is not None)

    faces = engine.detect(frame)
    check(f"YuNet detects faces in test.jpg (found {len(faces)}, expected >= 2)",
          len(faces) >= 2)

    feats = [engine.embed(frame, f) for f in faces]
    check("SFace embeddings are 128-dim", all(f.shape == (128,) for f in feats))
    check("Embeddings are L2-normalized",
          all(abs(np.linalg.norm(f) - 1.0) < 1e-4 for f in feats))

    # database round-trip in a temp location (never touches real data)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_db = Database(Path(tmp) / "test.db")
        person_id = tmp_db.add_person(full_name="Test Person",
                                      student_id="TEST-001",
                                      department="CS")
        for f in feats:
            tmp_db.add_face(person_id, f, pose="test", quality=99.0)
        mean = tmp_db.get_mean_embeddings()
        fresh = engine.embed(frame, faces[0])
        sim = cosine_sim(fresh, mean[person_id]) if person_id in mean else -1.0
        check(f"DB enroll + mean-embedding round-trip (sim={sim:.3f})", sim > 0.5)
        check("DB stores person with face count", tmp_db.get_face_count(person_id) == len(feats))
        tmp_db.add_attendance(person_id)
        check("DB attendance logging works", tmp_db.attendance_today_count() >= 1)
        check("DB integrity check passes",
              all(ok for _, ok in tmp_db.integrity_check()))
        tmp_db.close()

    check(f"Threshold {COSINE_THRESHOLD} within valid range (0, 1)", 0.0 < COSINE_THRESHOLD < 1.0)

    print("\n" + "=" * 55)
    if failures:
        print(f"  SELF-TEST FAILED ({len(failures)} issue(s))")
        print("=" * 55)
        return 1
    print("  SELF-TEST PASSED - project is technically ready.")
    print("=" * 55)
    return 0


# -------------------------
# ENTRY POINT
# -------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Smart Face Recognition System (YuNet + SFace, offline, SQLite)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python src/face_app.py enroll --name "Ahmad" --student-id "CS-01"
  python src/face_app.py recognize
  python src/face_app.py image --path data/images/test.jpg
  python src/face_app.py list
  python src/face_app.py export
  python src/face_app.py selftest

GUI: python src/gui.py
""")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--camera", type=int, default=0, help="webcam index (default: 0)")
        sp.add_argument("--score-threshold", type=float, default=0.9,
                        help="YuNet detection confidence (default: 0.9)")

    sp = sub.add_parser("enroll", help="register a person from webcam samples")
    sp.add_argument("--name", required=True, help="person's full name")
    sp.add_argument("--student-id", help="student / card number (unique)")
    sp.add_argument("--department", help="faculty / department")
    sp.add_argument("--phone", help="phone number")
    sp.add_argument("--email", help="email address")
    sp.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                    help=f"number of samples to capture (default: {DEFAULT_SAMPLES})")
    add_common(sp)
    sp.set_defaults(func=cmd_enroll)

    sp = sub.add_parser("recognize", help="live webcam recognition demo")
    sp.add_argument("--no-log", dest="log", action="store_false",
                    help="disable attendance logging")
    add_common(sp)
    sp.set_defaults(func=cmd_recognize, log=True)

    sp = sub.add_parser("image", help="recognize faces in an image file")
    sp.add_argument("--path", required=True, help="path to the image")
    add_common(sp)
    sp.set_defaults(func=cmd_image)

    sp = sub.add_parser("list", help="list enrolled identities")
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("remove", help="remove an enrolled identity")
    sp.add_argument("--name", required=True, help="name to remove (exact or substring)")
    sp.set_defaults(func=cmd_remove)

    sp = sub.add_parser("export", help="export attendance report to CSV")
    sp.add_argument("--path", help="output CSV path (default: outputs/attendance_<date>.csv)")
    sp.add_argument("--all", dest="today", action="store_false",
                    help="export all records instead of today only")
    sp.set_defaults(func=cmd_export, today=True)

    sp = sub.add_parser("selftest", help="headless end-to-end check")
    sp.set_defaults(func=cmd_selftest)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
