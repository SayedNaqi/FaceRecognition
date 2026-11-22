"""Unit tests for the YuNet + SFace engine (src/face_engine.py)."""
from pathlib import Path

import cv2
import numpy as np

from face_engine import (
    COSINE_THRESHOLD,
    FaceEngine,
    build_match_matrix,
    cosine_sim,
    match_matrix,
    models_present,
)

IMAGE = Path(__file__).resolve().parent.parent / "data" / "images" / "test.jpg"


def test_models_present():
    assert models_present()


def test_detection_and_embeddings():
    img = cv2.imread(str(IMAGE))
    assert img is not None
    h, w = img.shape[:2]
    engine = FaceEngine((w, h))
    faces = engine.detect(img)
    assert len(faces) >= 2
    feats = [engine.embed(img, f) for f in faces]
    assert all(f.shape == (128,) for f in feats)
    assert all(abs(float(np.linalg.norm(f)) - 1.0) < 1e-4 for f in feats)
    assert cosine_sim(feats[0], feats[0]) > 0.99
    # the three people in test.jpg are different identities
    assert cosine_sim(feats[0], feats[1]) < COSINE_THRESHOLD


def test_head_pose_and_quality():
    img = cv2.imread(str(IMAGE))
    h, w = img.shape[:2]
    engine = FaceEngine((w, h))
    faces = engine.detect(img)
    zones = {"center", "slight_left", "slight_right", "far_left", "far_right"}
    pitches = {"up", "level", "down"}
    qualities = []
    for f in faces:
        yaw, pitch, _y, _p = engine.head_pose(f)
        assert yaw in zones and pitch in pitches
        ok, _reason, score = engine.sample_quality(img, f)
        if ok:
            qualities.append(score)
    assert qualities


def test_vectorized_match():
    img = cv2.imread(str(IMAGE))
    h, w = img.shape[:2]
    engine = FaceEngine((w, h))
    faces = engine.detect(img)
    f1 = engine.embed(img, faces[0])
    f2 = engine.embed(img, faces[1])
    ids, matrix = build_match_matrix({"a": f1, "b": f2})
    assert matrix.shape == (2, 128)
    assert match_matrix(f1, ids, matrix, COSINE_THRESHOLD)[0] == "a"
    assert match_matrix(f2, ids, matrix, COSINE_THRESHOLD)[0] == "b"
    unseen = np.zeros(128, dtype=np.float32)
    unseen[1] = 1.0
    assert match_matrix(unseen, ids, matrix, COSINE_THRESHOLD)[0] is None
    assert match_matrix(f1, [], np.zeros((0, 128), np.float32)) == (None, 0.0)
