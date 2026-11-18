#!/usr/bin/env python3
# ============================================================
# Downloads the YuNet + SFace ONNX models from OpenCV Zoo.
# Usage:  python src/download_models.py
# ============================================================

import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

MODELS = {
    "face_detection_yunet_2023mar.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_recognition_sface/face_recognition_sface_2021dec.onnx",
}


def download(url, dest):
    print(f"Downloading {dest.name} ...")
    with urllib.request.urlopen(url, timeout=300) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        downloaded = 0
        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 // total
                    print(f"\r  {downloaded / 1e6:.1f} MB / {total / 1e6:.1f} MB ({pct}%)",
                          end="", flush=True)
    print(f"\n  Saved: {dest}")


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    ok = True
    for filename, url in MODELS.items():
        dest = MODELS_DIR / filename
        if dest.exists() and dest.stat().st_size > 0:
            print(f"Already exists, skipping: {dest.name}")
            continue
        try:
            download(url, dest)
        except Exception as exc:  # noqa: BLE001 - report and continue
            ok = False
            print(f"\n  FAILED: {filename}: {exc}")
    if not ok:
        print("\nSome models failed to download. Check your internet connection.")
        return 1
    print("\nAll models ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
