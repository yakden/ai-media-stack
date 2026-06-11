#!/usr/bin/env python3
"""Enroll a person into VMS face recognition from GOOD reference photos.

The auto-captured face exemplars from camera views are often low-quality (small,
angled) — which is why live face matches are weak. Enrolling a few clean, frontal
photos per person fixes that: their ArcFace embeddings become strong gallery anchors.

Usage (run on the box; localhost needs no auth):
    python3 scripts/enroll_person.py --name "Иван Петров" photo1.jpg photo2.jpg ...
    python3 scripts/enroll_person.py --name "Иван Петров" --dir /path/to/photos

Tips: 3–5 photos per person, frontal + a couple of mild angles, face clearly visible,
≥112 px. The server runs SCRFD+ArcFace on each and stores the embedding.
"""
import argparse
import json
import mimetypes
import os
import sys
import urllib.request

BASE = os.environ.get("VMS_BASE", "http://127.0.0.1:8120")


def _post_json(path, obj):
    req = urllib.request.Request(BASE + path, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _post_file(path, filepath):
    boundary = "----vmsenroll7e3f"
    fn = os.path.basename(filepath)
    ctype = mimetypes.guess_type(fn)[0] or "application/octet-stream"
    with open(filepath, "rb") as f:
        data = f.read()
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{fn}\"\r\n"
            f"Content-Type: {ctype}\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(BASE + path, data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status, json.loads(r.read() or b"{}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="person display name")
    ap.add_argument("--dir", help="directory of photos (jpg/png)")
    ap.add_argument("photos", nargs="*", help="photo files")
    a = ap.parse_args()

    photos = list(a.photos)
    if a.dir:
        photos += [os.path.join(a.dir, f) for f in sorted(os.listdir(a.dir))
                   if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
    photos = [p for p in photos if os.path.isfile(p)]
    if not photos:
        print("нет фото для загрузки"); sys.exit(1)

    person = _post_json("/api/people", {"name": a.name})
    pid = person.get("id")
    print(f"создана персона #{pid} «{a.name}»")
    ok = 0
    for p in photos:
        try:
            st, resp = _post_file(f"/api/people/{pid}/faces", p)
            print(f"  ✓ {os.path.basename(p)} -> лицо #{resp.get('id','?')}")
            ok += 1
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:200]
            print(f"  ✗ {os.path.basename(p)} -> {e.code}: {detail}  (нет лица / несколько лиц / мелко?)")
        except Exception as e:
            print(f"  ✗ {os.path.basename(p)} -> {e}")
    print(f"\nготово: {ok}/{len(photos)} фото записано для «{a.name}». "
          f"Лицо станет узнаваться на камерах сразу (галерея перестроится).")


if __name__ == "__main__":
    main()
