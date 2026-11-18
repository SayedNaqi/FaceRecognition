#!/usr/bin/env python3
# ============================================================
# Smart Face Recognition System
# Faculty of Computer Science - Ghorjistan University
#
# Detection : YuNet  (face_detection_yunet_2023mar.onnx)
# Recognition: SFace (face_recognition_sface_2021dec.onnx)
# Both models run 100% offline via OpenCV DNN.
#
# Commands:
#   enroll    Register a person from webcam samples
#   recognize  Live webcam recognition (demo mode)
#   image     Recognize faces in an image file
#   list      List enrolled people
#   remove    Delete an enrolled person
#   selftest  Headless end-to-end check (no webcam needed)
# ============================================================

import argparse
import csv
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# -------------------------
# PATHS & SETTINGS
# -------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
REGISTERED_DIR = DATA_DIR / "registered_faces"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

YUNET_MODEL = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL = MODELS_DIR / "face_recognition_sface_2021dec.onnx"
EMBEDDINGS_PATH = MODELS_DIR / "face_embeddings.npz"

# SFace reference threshold (OpenCV Zoo). Higher = stricter matching.
COSINE_THRESHOLD = 0.363

MIN_SAMPLES = 5          # refuse to enroll fewer samples than this
DEFAULT_SAMPLES = 30     # target samples per person
CAPTURE_INTERVAL = 0.40  # seconds between automatic captures
ATTENDANCE_COOLDOWN = 60 # seconds between logs of the same person


# -------------------------
# MODEL HELPERS
# -------------------------

def check_models_exist():
    missing = [p for p in (YUNET_MODEL, SFACE_MODEL) if not p.exists()]
    if missing:
        print("Model file(s) missing:")
        for p in missing:
            print("  -", p)
        print("\nRun:  python src/download_models.py")
        sys.exit(1)


def load_detector(frame_width, frame_height, score_threshold=0.9):
    detector = cv2.FaceDetectorYN.create(
        str(YUNET_MODEL), "", (frame_width, frame_height),
        score_threshold=score_threshold, nms_threshold=0.3, top_k=5000,
    )
    return detector


def load_recognizer():
    return cv2.FaceRecognizerSF.create(str(SFACE_MODEL), "")


def detect_faces(detector, frame):
    _, faces = detector.detect(frame)
    if faces is None:
        return []
    return list(faces)


def get_embedding(recognizer, frame, face_row):
    """Aligned face -> L2-normalized 128-dim SFace embedding."""
    aligned = recognizer.alignCrop(frame, face_row)
    feat = recognizer.feature(aligned).flatten().astype(np.float32)
    norm = np.linalg.norm(feat)
    if norm > 0:
        feat = feat / norm
    return feat


def cosine_sim(a, b):
    # embeddings are L2-normalized, so dot product == cosine similarity
    return float(np.dot(a, b))


def safe_dirname(name):
    return re.sub(r'[\\/:*?"<>|]', "_", name).strip()


# -------------------------
# EMBEDDING DATABASE (models/face_embeddings.npz)
# -------------------------

def load_database():
    if not EMBEDDINGS_PATH.exists():
        return []
    z = np.load(EMBEDDINGS_PATH, allow_pickle=False)
    names = z["names"]
    embs = z["embeddings"]
    counts = z["counts"] if "counts" in z else np.ones(len(names), dtype=np.int32)
    db = []
    for i in range(len(names)):
        db.append({
            "name": str(names[i]),
            "embedding": embs[i].astype(np.float32),
            "count": int(counts[i]),
        })
    return db


def save_database(db):
    MODELS_DIR.mkdir(exist_ok=True)
    if not db:
        if EMBEDDINGS_PATH.exists():
            EMBEDDINGS_PATH.unlink()
        return
    names = np.array([d["name"] for d in db])
    embs = np.vstack([d["embedding"] for d in db]).astype(np.float32)
    counts = np.array([d["count"] for d in db], dtype=np.int32)
    np.savez(EMBEDDINGS_PATH, names=names, embeddings=embs, counts=counts)


def match_embedding(db, feat):
    """Return (name, similarity) of best match, or ('Unknown', best_sim)."""
    if not db:
        return "Unknown", 0.0
    sims = [cosine_sim(feat, d["embedding"]) for d in db]
    idx = int(np.argmax(sims))
    best = sims[idx]
    if best >= COSINE_THRESHOLD:
        return db[idx]["name"], best
    return "Unknown", best


# -------------------------
# OVERLAY DRAWING
# -------------------------

GREEN = (60, 220, 60)
RED = (60, 60, 230)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


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
    return x, y, w, h


def draw_footer(frame, text):
    h = frame.shape[0]
    cv2.putText(frame, text, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)


# -------------------------
# ATTENDANCE LOG (outputs/attendance_YYYY-MM-DD.csv)
# -------------------------

def log_attendance(name, _cache={}):
    now = datetime.now()
    last = _cache.get(name)
    if last and (now - last).total_seconds() < ATTENDANCE_COOLDOWN:
        return False
    OUTPUTS_DIR.mkdir(exist_ok=True)
    path = OUTPUTS_DIR / f"attendance_{now:%Y-%m-%d}.csv"
    new_file = not path.exists()
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["Name", "Date", "Time"])
        writer.writerow([name, f"{now:%Y-%m-%d}", f"{now:%H:%M:%S}"])
    _cache[name] = now
    return True


# -------------------------
# COMMAND: enroll
# -------------------------

def cmd_enroll(args):
    check_models_exist()
    name = args.name.strip()
    if not name:
        print("ERROR: Please provide a name:  --name \"Ahmad\"")
        return 1

    safe = safe_dirname(name)
    person_dir = REGISTERED_DIR / safe
    person_dir.mkdir(parents=True, exist_ok=True)

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        print("ERROR: Cannot access webcam. Check connection/permissions.")
        return 1

    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    detector = load_detector(width, height, args.score_threshold)
    recognizer = load_recognizer()

    target = args.samples
    feats, paths = [], []
    last_capture = 0.0

    print("=" * 55)
    print(f"  ENROLLMENT MODE - {name}")
    print("=" * 55)
    print(f"Look at the camera. Capturing {target} samples automatically.")
    print("Press Q to cancel.\n")

    while len(feats) < target:
        ok, frame = camera.read()
        if not ok:
            print("ERROR: Cannot read webcam frame.")
            break

        faces = detect_faces(detector, frame)
        faces = [f for f in faces if f[2] >= 60 and f[3] >= 60]

        if faces:
            f0 = max(faces, key=lambda f: f[2] * f[3])  # largest face
            draw_face_box(frame, f0, GREEN)
            now = time.time()
            if now - last_capture >= CAPTURE_INTERVAL:
                last_capture = now
                idx = len(feats)
                img_path = person_dir / f"img_{idx:03d}.jpg"
                cv2.imwrite(str(img_path), frame)
                feats.append(get_embedding(recognizer, frame, f0))
                paths.append(img_path)
                print(f"  Sample {len(feats)}/{target} captured")
        else:
            cv2.putText(frame, "No face detected - move closer", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2, cv2.LINE_AA)

        # progress bar + counter
        bar_w = int((len(feats) / target) * (width - 20))
        cv2.rectangle(frame, (10, height - 55), (width - 10, height - 40), WHITE, 1)
        cv2.rectangle(frame, (10, height - 55), (10 + bar_w, height - 40), GREEN, -1)
        draw_footer(frame, f"Samples: {len(feats)}/{target}   |   Q: cancel")

        cv2.imshow("Enroll - Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            print("\nEnrollment cancelled by user.")
            camera.release()
            cv2.destroyAllWindows()
            return 1

    camera.release()
    cv2.destroyAllWindows()

    if len(feats) < MIN_SAMPLES:
        print(f"\nOnly {len(feats)} samples captured (min {MIN_SAMPLES}). Not saved.")
        return 1

    feats_arr = np.vstack(feats)
    mean_feat = feats_arr.mean(axis=0)
    mean_feat /= (np.linalg.norm(mean_feat) + 1e-10)
    quality = float(np.mean([cosine_sim(f, mean_feat) for f in feats]))

    db = load_database()
    db = [d for d in db if d["name"].lower() != name.lower()]  # re-enrollment replaces
    db.append({"name": name, "embedding": mean_feat.astype(np.float32), "count": len(feats)})
    save_database(db)

    print("\n" + "=" * 55)
    print(f"  Enrollment complete: {name}")
    print(f"  Samples saved : {len(feats)}  ->  data/registered_faces/{safe}/")
    print(f"  Quality score : {quality * 100:.1f}%  (higher is better)")
    print("=" * 55)
    return 0


# -------------------------
# COMMAND: recognize (live demo)
# -------------------------

def cmd_recognize(args):
    check_models_exist()
    db = load_database()
    if not db:
        print("No enrolled faces found.")
        print("Enroll someone first:  python src/face_app.py enroll --name \"Your Name\"")
        return 1
    print(f"Loaded {len(db)} enrolled identities: {', '.join(d['name'] for d in db)}")

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        print("ERROR: Cannot access webcam. Check connection/permissions.")
        return 1

    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    detector = load_detector(width, height, args.score_threshold)
    recognizer = load_recognizer()

    print("\nLIVE RECOGNITION - Press Q to quit, S to save a screenshot.\n")
    frame_count, start = 0, time.time()

    while True:
        ok, frame = camera.read()
        if not ok:
            print("ERROR: Cannot read webcam frame.")
            break

        faces = detect_faces(detector, frame)
        for f in faces:
            feat = get_embedding(recognizer, frame, f)
            name, sim = match_embedding(db, feat)
            known = name != "Unknown"
            color = GREEN if known else RED
            draw_face_box(frame, f, color, f"{name}  {sim * 100:.0f}%")
            if known and args.log:
                log_attendance(name)

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
    print("Demo stopped.")
    return 0


# -------------------------
# COMMAND: image
# -------------------------

def cmd_image(args):
    check_models_exist()
    image_path = Path(args.path)
    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        return 1

    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"ERROR: Could not read image: {image_path}")
        return 1

    height, width = frame.shape[:2]
    detector = load_detector(width, height, args.score_threshold)
    recognizer = load_recognizer()
    db = load_database()
    if not db:
        print("NOTE: No enrolled identities yet - faces will be labelled 'Unknown'.")
        print("      Enroll first:  python src/face_app.py enroll --name \"Your Name\"\n")

    faces = detect_faces(detector, frame)
    print(f"Image: {image_path}  ({width}x{height})")
    print(f"Faces detected: {len(faces)}\n")

    results = []
    for i, f in enumerate(faces, 1):
        feat = get_embedding(recognizer, frame, f)
        name, sim = match_embedding(db, feat)
        color = GREEN if name != "Unknown" else RED
        draw_face_box(frame, f, color, f"{name}  {sim * 100:.0f}%")
        results.append((i, name, sim))
        print(f"  Face {i}: {name:<20} similarity={sim * 100:.1f}%")

    OUTPUTS_DIR.mkdir(exist_ok=True)
    out_path = OUTPUTS_DIR / f"result_{image_path.stem}_{datetime.now():%Y%m%d_%H%M%S}.png"
    cv2.imwrite(str(out_path), frame)
    print(f"\nAnnotated image saved: {out_path}")
    return 0


# -------------------------
# COMMAND: list / remove
# -------------------------

def cmd_list(_args):
    db = load_database()
    if not db:
        print("No enrolled identities yet.")
        return 0
    print(f"{'#':<4}{'Name':<25}{'Samples':<10}")
    print("-" * 39)
    for i, d in enumerate(db, 1):
        print(f"{i:<4}{d['name']:<25}{d['count']:<10}")
    print(f"\nTotal: {len(db)} identities")
    return 0


def cmd_remove(args):
    name = args.name.strip()
    db = load_database()
    matched = [d for d in db if d["name"].lower() == name.lower()]
    if not matched:
        print(f"No enrolled person named '{name}'.")
        return 1
    db = [d for d in db if d["name"].lower() != name.lower()]
    save_database(db)
    person_dir = REGISTERED_DIR / safe_dirname(name)
    if person_dir.exists():
        shutil.rmtree(person_dir)
    print(f"Removed '{name}' (embedding + {person_dir} samples).")
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

    # 1. models present and loadable
    check_models_exist()
    test_image = PROJECT_ROOT / "data" / "images" / "test.jpg"
    frame = cv2.imread(str(test_image))
    check("test image readable", frame is not None)
    height, width = frame.shape[:2]
    detector = load_detector(width, height)
    recognizer = load_recognizer()
    check("YuNet + SFace models loaded", detector is not None and recognizer is not None)

    # 2. detection works
    faces = detect_faces(detector, frame)
    check(f"YuNet detects faces in test.jpg (found {len(faces)}, expected >= 2)", len(faces) >= 2)

    # 3. embedding pipeline works
    feats = [get_embedding(recognizer, frame, f) for f in faces]
    check("SFace embeddings are 128-dim", all(f.shape == (128,) for f in feats))
    check("Embeddings are L2-normalized",
          all(abs(np.linalg.norm(f) - 1.0) < 1e-4 for f in feats))

    # 4. database round-trip (temp file, does not touch real data)
    global EMBEDDINGS_PATH
    original_path = EMBEDDINGS_PATH
    with tempfile.TemporaryDirectory() as tmp:
        EMBEDDINGS_PATH = Path(tmp) / "face_embeddings.npz"
        db = [{"name": f"Person{i + 1}", "embedding": f, "count": 1}
              for i, f in enumerate(feats)]
        save_database(db)
        loaded = load_database()
        check("Embedding database save/load round-trip",
              len(loaded) == len(db) and
              all(cosine_sim(a["embedding"], b["embedding"]) > 0.99
                  for a, b in zip(db, loaded)))
        best_name, best_sim = match_embedding(loaded, feats[0])
        check(f"Matching returns enrolled identity (got '{best_name}', sim={best_sim:.3f})",
              best_name == "Person1" and best_sim > 0.99)
    EMBEDDINGS_PATH = original_path

    # 5. cosine threshold sanity
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
        description="Smart Face Recognition System (YuNet + SFace, offline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python src/face_app.py enroll --name "Ahmad"
  python src/face_app.py recognize
  python src/face_app.py image --path data/images/test.jpg
  python src/face_app.py list
  python src/face_app.py selftest
""")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--camera", type=int, default=0, help="webcam index (default: 0)")
        sp.add_argument("--score-threshold", type=float, default=0.9,
                        help="YuNet detection confidence (default: 0.9)")

    sp = sub.add_parser("enroll", help="register a person from webcam samples")
    sp.add_argument("--name", required=True, help="person's name")
    sp.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                    help=f"number of samples to capture (default: {DEFAULT_SAMPLES})")
    add_common(sp)
    sp.set_defaults(func=cmd_enroll)

    sp = sub.add_parser("recognize", help="live webcam recognition demo")
    sp.add_argument("--no-log", dest="log", action="store_false",
                    help="disable attendance CSV logging")
    add_common(sp)
    sp.set_defaults(func=cmd_recognize, log=True)

    sp = sub.add_parser("image", help="recognize faces in an image file")
    sp.add_argument("--path", required=True, help="path to the image")
    add_common(sp)
    sp.set_defaults(func=cmd_image)

    sp = sub.add_parser("list", help="list enrolled identities")
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("remove", help="remove an enrolled identity")
    sp.add_argument("--name", required=True, help="exact name to remove")
    sp.set_defaults(func=cmd_remove)

    sp = sub.add_parser("selftest", help="headless end-to-end check")
    sp.set_defaults(func=cmd_selftest)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
