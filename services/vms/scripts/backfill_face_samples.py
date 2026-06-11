#!/usr/bin/env python3
"""Backfill FaceSample rows from existing per-identity FaceExemplars.

The dedicated face-grouping layer (FaceSample) only began capturing crops after
it was added; this one-shot backfill seeds it from the ArcFace vectors already
collected by the body-Re-ID layer (FaceExemplar), and recovers a real face crop
by re-detecting the face inside each sighting's stored body thumbnail. Vectors
that yield no face crop are still inserted (they cluster; just no thumbnail).

Run in the container:  python3 scripts/backfill_face_samples.py
"""
from __future__ import annotations

import os

import cv2

from app.config import get_settings
from app.db.database import SessionLocal
from app.db.models import AppearanceExemplar, FaceExemplar, FaceSample, Sighting


def _resolve(settings, stored):
    if not stored:
        return None
    data_root = os.path.abspath(str(settings.data_dir))
    cand = (os.path.abspath(stored) if os.path.isabs(stored)
            else os.path.abspath(os.path.join(os.path.dirname(data_root), stored)))
    if not os.path.exists(cand):
        alt = os.path.abspath(os.path.join(data_root, stored))
        if os.path.exists(alt):
            cand = alt
    return cand if os.path.exists(cand) else None


def main() -> int:
    settings = get_settings()
    samples_dir = settings.face_samples_dir
    samples_dir.mkdir(parents=True, exist_ok=True)

    rec = None
    try:
        from app.faces.recognizer import FaceRecognizer
        rec = FaceRecognizer(models_dir=str(settings.insightface_root),
                             pack_name=settings.insightface_pack, device="cpu",
                             det_size=(settings.face_det_size, settings.face_det_size))
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] recognizer unavailable ({exc}); inserting vectors without face crops")

    s = SessionLocal()
    existing = {sid for (sid,) in s.query(FaceSample.sighting_id).filter(FaceSample.sighting_id.isnot(None)).all()}
    made = thumbs = 0
    try:
        for ex in s.query(FaceExemplar).all():
            if ex.sighting_id in existing:
                continue
            sighting = s.get(Sighting, ex.sighting_id) if ex.sighting_id else None
            app = (s.query(AppearanceExemplar)
                   .filter(AppearanceExemplar.sighting_id == ex.sighting_id).first()
                   if ex.sighting_id else None)
            fs = FaceSample(
                camera_id=ex.camera_id,
                ts=(sighting.ts if sighting else ex.created_at),
                vector=ex.vector,
                app_vector=app.vector if app else None,
                quality=float(ex.det_score or 0.0),
                identity_id=ex.identity_id,
                sighting_id=ex.sighting_id,
            )
            s.add(fs); s.commit()
            made += 1
            # Recover a face crop from the sighting's body thumbnail.
            if rec is not None and sighting is not None:
                p = _resolve(settings, sighting.thumb_path)
                if p:
                    img = cv2.imread(p)
                    if img is not None:
                        # Body thumbs are small; upscale so SCRFD can find the
                        # (tiny) face before cropping it.
                        if max(img.shape[:2]) < 400:
                            img = cv2.resize(img, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
                        try:
                            faces = rec.detect(img)
                        except Exception:
                            faces = []
                        if faces:
                            f = max(faces, key=lambda x: x.area)
                            x1, y1, x2, y2 = (int(v) for v in f.bbox)
                            h, w = img.shape[:2]
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(w, x2), min(h, y2)
                            if x2 > x1 and y2 > y1:
                                out = samples_dir / f"{fs.id}.jpg"
                                if cv2.imwrite(str(out), img[y1:y2, x1:x2]):
                                    fs.thumb_path = f"data/face_samples/{fs.id}.jpg"
                                    s.commit(); thumbs += 1
        print(f"[ok] created {made} FaceSamples ({thumbs} with face crops)")
    finally:
        s.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
