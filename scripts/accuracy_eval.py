#!/usr/bin/env python3
"""
scripts/accuracy_eval.py — Accuracy evaluation for the web app
Measures:
  - TP rate (correct acceptance): for each enrolled person, recognize the face
    samples we have (profile crops) and check that the server reports the
    correct person_id.
  - False Accept Rate (FAR): for each enrolled person P, send samples of
    every OTHER enrolled person Q != P; count the number of times Q is
    (incorrectly) accepted as P (or as any other person).

Prerequisite: python src/web_app.py (running, http://127.0.0.1:5000)
Run:  python scripts/accuracy_eval.py
"""

import base64
import sys
from pathlib import Path

import cv2
import requests

BASE = "http://127.0.0.1:5000"
PROFILES_DIR = Path(__file__).resolve().parent.parent / "data" / "profiles"


def to_jpeg_bytes(path):
    img = cv2.imread(str(path))
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes(), img


def recognize(path):
    data, _ = to_jpeg_bytes(path)
    r = requests.post(f"{BASE}/api/recognize",
                      files={"image": ("f.jpg", data, "image/jpeg")})
    r.raise_for_status()
    return r.json()["matches"]


def main():
    people = requests.get(f"{BASE}/api/people").json()
    if len(people) < 2:
        print("Need at least 2 enrolled people for a meaningful evaluation.")
        return 2

    # profile crops of every enrolled person
    profiles = {}
    for p in people:
        prof = Path(p.get("photo_path", "") or "")
        if prof.exists():
            profiles[p["id"]] = prof

    tp = {p["id"]: 0 for p in people}
    total = {p["id"]: 0 for p in people}
    false_accept = {p["id"]: 0 for p in people}  # accepted as someone else
    attempts = {p["id"]: 0 for p in people}

    for pid, prof in profiles.items():
        m = recognize(prof)
        if not m:
            continue
        matches = [x for x in m if x["known"]]
        if matches and matches[0]["person_id"] == pid:
            tp[pid] += 1
        total[pid] += 1

    # false acceptance: person B's crop recognized as person A (A != B)
    for pid, prof in profiles.items():
        for qid, qprof in profiles.items():
            if qid == pid:
                continue
            m = recognize(qprof)
            attempts[pid] += 1
            for match in m:
                if match["known"] and match["person_id"] != qid and match["person_id"] == pid:
                    false_accept[pid] += 1

    print(f"{'Name':<30} {'TP %':>7} {'FAR %':>7}")
    all_tp, all_fa, n = 0, 0, 0
    for p in people:
        pid = p["id"]
        if pid in profiles:
            tp_rate = (tp[pid] / total[pid] * 100) if total[pid] else 0
            fa_rate = (false_accept[pid] / attempts[pid] * 100) if attempts[pid] else 0
        else:
            tp_rate = fa_rate = 0.0
        name = p["full_name"]
        print(f"{name[:30]:<30} {tp_rate:>6.1f}% {fa_rate:>6.1f}%")
        all_tp += tp[pid]; all_fa += false_accept[pid]
        n += 1
    print("-" * 45)
    tot = sum(total.values()) or 1
    print(f"Overall True Positive Rate: {sum(tp.values()) / tot * 100:.1f}%")
    att = sum(attempts.values()) or 1
    print(f"Overall False Accept Rate : {sum(false_accept.values()) / att * 100:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
