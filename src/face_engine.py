#!/usr/bin/env python3
# ============================================================
# Face Engine — shared YuNet + SFace wrapper
# Detection : YuNet  (face_detection_yunet_2023mar.onnx)
# Recognition: SFace (face_recognition_sface_2021dec.onnx)
# ============================================================

import cv2
import numpy as np

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
YUNET_MODEL = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL = MODELS_DIR / "face_recognition_sface_2021dec.onnx"

# SFace reference cosine threshold (OpenCV Zoo)
COSINE_THRESHOLD = 0.363


def models_present():
    return YUNET_MODEL.exists() and SFACE_MODEL.exists()


def cosine_sim(a, b):
    # embeddings are L2-normalized, so dot == cosine
    return float(np.dot(a, b))


class FaceEngine:
    def __init__(self, input_size=(640, 480), score_threshold=0.9):
        if not models_present():
            raise FileNotFoundError(
                f"Face models missing in {MODELS_DIR}. Run: python src/download_models.py")
        self.input_size = tuple(input_size)
        self.detector = cv2.FaceDetectorYN.create(
            str(YUNET_MODEL), "", self.input_size,
            score_threshold=score_threshold, nms_threshold=0.3, top_k=5000)
        self.recognizer = cv2.FaceRecognizerSF.create(str(SFACE_MODEL), "")

    def set_input_size(self, width, height):
        size = (int(width), int(height))
        if size != self.input_size:
            self.input_size = size
            self.detector.setInputSize(size)

    def detect(self, frame):
        """Return list of Nx15 face rows (float32)."""
        self.set_input_size(frame.shape[1], frame.shape[0])
        _, faces = self.detector.detect(frame)
        if faces is None:
            return []
        return list(faces)

    def embed(self, frame, face_row):
        aligned = self.recognizer.alignCrop(frame, face_row)
        feat = self.recognizer.feature(aligned).flatten().astype(np.float32)
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
        return feat

    def match(self, frame, face_row, mean_embeddings):
        """Return (person_id, similarity). person_id is None when below threshold."""
        feat = self.embed(frame, face_row)
        if not mean_embeddings:
            return None, 0.0, feat
        best_id, best_sim = None, -1.0
        for person_id, mean in mean_embeddings.items():
            sim = cosine_sim(feat, mean)
            if sim > best_sim:
                best_id, best_sim = person_id, sim
        if best_sim >= COSINE_THRESHOLD:
            return best_id, best_sim, feat
        return None, best_sim, feat

    # ---------------- head pose (rough, from YuNet landmarks) ----------------

    @staticmethod
    def head_pose(face_row):
        """Return (yaw_zone, pitch_zone, yaw_offset, pitch_ratio).

        yaw_offset       : nose offset from eye-midpoint / eye distance
                           (>0 means the face is turned towards the person's left)
        pitch_ratio      : nose position between eye and mouth lines
                           (<0.48 head tilted up, >0.72 head tilted down)
        """
        x_re, y_re, x_le, y_le, x_ne, y_ne, _x_rcm, _y_rcm, _x_lcm, _y_lcm = face_row[4:14]
        mid_eye_x = (x_re + x_le) / 2.0
        mid_eye_y = (y_re + y_le) / 2.0
        eye_dist = abs(x_le - x_re) + 1e-6
        yaw = float((x_ne - mid_eye_x) / eye_dist)

        mid_mouth_x = (_x_rcm + _x_lcm) / 2.0
        mid_mouth_y = (_y_rcm + _y_lcm) / 2.0
        pitch = float((y_ne - mid_eye_y) / (mid_mouth_y - mid_eye_y + 1e-6))

        if yaw >= 0.32:
            yaw_zone = "far_left"
        elif yaw >= 0.13:
            yaw_zone = "slight_left"
        elif yaw <= -0.32:
            yaw_zone = "far_right"
        elif yaw <= -0.13:
            yaw_zone = "slight_right"
        else:
            yaw_zone = "center"

        if pitch <= 0.48:
            pitch_zone = "up"
        elif pitch >= 0.72:
            pitch_zone = "down"
        else:
            pitch_zone = "level"

        return yaw_zone, pitch_zone, yaw, pitch

    # ---------------- sample quality gate ----------------

    @staticmethod
    def sample_quality(frame, face_row):
        """Return (ok, reason, score 0..100)."""
        x, y, w, h = [int(v) for v in face_row[:4]]
        if w < 60 or h < 60:
            return False, "too_small", 0.0
        height, width = frame.shape[:2]
        x1, y1 = max(x, 0), max(y, 0)
        x2, y2 = min(x + w, width), min(y + h, height)
        if x2 - x1 < 40 or y2 - y1 < 40:
            return False, "too_small", 0.0
        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        brightness = float(gray.mean())
        contrast = float(gray.std())
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        if brightness < 45:
            return False, "too_dark", 10.0
        if brightness > 225:
            return False, "too_bright", 10.0
        if contrast < 18:
            return False, "low_contrast", 15.0
        if blur < 60:
            return False, "blurry", 20.0

        # score: mix of size, sharpness, contrast
        size_term = min(w / 160.0, 1.0) * 40
        sharp_term = min(blur / 400.0, 1.0) * 40
        contrast_term = min(contrast / 64.0, 1.0) * 20
        return True, "ok", round(size_term + sharp_term + contrast_term, 1)

    @staticmethod
    def crop_face(frame, face_row, pad_ratio=0.35):
        x, y, w, h = [int(v) for v in face_row[:4]]
        height, width = frame.shape[:2]
        pad_x, pad_y = int(w * pad_ratio), int(h * pad_ratio)
        x1, y1 = max(x - pad_x, 0), max(y - pad_y, 0)
        x2, y2 = min(x + w + pad_x, width), min(y + h + pad_y, height)
        return frame[y1:y2, x1:x2].copy()
