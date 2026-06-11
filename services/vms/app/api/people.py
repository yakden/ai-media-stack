"""People / face-DB API.

Person CRUD and face enrollment endpoints. Enrollment accepts an uploaded
photo, runs the insightface SCRFD+ArcFace recognizer to produce a 512-d
L2-normalized embedding, persists it as a ``FaceEmbedding`` BLOB and adds it to
the in-memory FAISS index. Deletions remove embeddings (and source images) and
keep the FAISS index in sync (rebuilt from the DB, the single source of truth).

Routes (mounted under ``/api/people``):

    GET    /api/people                  -> [Person{...,num_faces}]
    POST   /api/people                  -> Person (201)
    GET    /api/people/{id}             -> Person
    PUT    /api/people/{id}             -> Person
    DELETE /api/people/{id}             -> 204  (cascade faces, rebuild FAISS,
                                                 events.person_id -> NULL)
    POST   /api/people/{id}/faces       -> {embedding_id, faces_detected, image_path}
    GET    /api/people/{id}/faces       -> [FaceEmbedding{id,image_url,created_at}]
    DELETE /api/people/{id}/faces/{fid} -> 204
"""

from __future__ import annotations

import os
import uuid
from typing import List

import numpy as np
from fastapi import (
    APIRouter,
    Body,
    Depends,
    File,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from ..auth import require_auth
from ..config import Settings, get_settings
from ..db.database import get_db
from ..db.models import Event, FaceEmbedding, Person
from .. import schemas

router = APIRouter(prefix="/api/people", tags=["people"], dependencies=[Depends(require_auth)])

# Cap upload size defensively even though nginx also enforces client_max_body_size.
MAX_IMAGE_BYTES = 32 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
    "application/octet-stream",  # some clients omit a proper type
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _get_person_or_404(db: Session, person_id: int) -> Person:
    person = db.get(Person, person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="Person not found")
    return person


def _num_faces(db: Session, person_id: int) -> int:
    return int(
        db.scalar(
            select(func.count(FaceEmbedding.id)).where(
                FaceEmbedding.person_id == person_id
            )
        )
        or 0
    )


def _person_out(db: Session, person: Person) -> schemas.Person:
    """Serialize a Person, attaching the derived ``num_faces`` count."""
    return schemas.Person.model_validate(
        {
            "id": person.id,
            "name": person.name,
            "notes": person.notes,
            "created_at": person.created_at,
            "num_faces": _num_faces(db, person.id),
        }
    )


def _face_out(face: FaceEmbedding) -> schemas.FaceEmbedding:
    return schemas.FaceEmbedding.model_validate(
        {
            "id": face.id,
            "image_url": f"/api/people/{face.person_id}/faces/{face.id}/image",
            "created_at": face.created_at,
        }
    )


def _faces_dir(settings: Settings, person_id: int) -> str:
    base = getattr(settings, "faces_dir", None) or os.path.join(
        getattr(settings, "data_dir", "data"), "faces"
    )
    path = os.path.join(str(base), str(person_id))
    os.makedirs(path, exist_ok=True)
    return path


def _decode_image(raw: bytes) -> np.ndarray:
    """Decode raw image bytes into a BGR uint8 ndarray (OpenCV layout)."""
    import cv2  # imported lazily to keep import-time light / testable

    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    return img


def _vector_to_blob(vec: np.ndarray) -> bytes:
    """Normalize to unit length and serialize as little-endian float32 bytes."""
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    if v.size != 512:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected embedding dimension {v.size} (expected 512)",
        )
    norm = float(np.linalg.norm(v))
    if norm > 0:
        v = v / norm
    return np.ascontiguousarray(v, dtype="<f4").tobytes()


def _rebuild_face_index(request: Request, db: Session) -> None:
    """Rebuild the FAISS index from the DB (single source of truth)."""
    index = getattr(request.app.state, "face_index", None)
    if index is None:
        return
    if hasattr(index, "rebuild_from_db"):
        index.rebuild_from_db(db)
        return

    # Generic fallback: stream all embeddings back in.
    rows = db.execute(
        select(FaceEmbedding.id, FaceEmbedding.person_id, FaceEmbedding.vector)
    ).all()
    if hasattr(index, "reset"):
        index.reset()
    elif hasattr(index, "clear"):
        index.clear()
    for emb_id, person_id, blob in rows:
        vec = np.frombuffer(blob, dtype="<f4").astype(np.float32)
        faiss_pos = index.add(vec, person_id=person_id, embedding_id=emb_id)
        if faiss_pos is not None:
            db.execute(
                update(FaceEmbedding)
                .where(FaceEmbedding.id == emb_id)
                .values(faiss_id=int(faiss_pos))
            )
    db.commit()


# --------------------------------------------------------------------------- #
# Person CRUD
# --------------------------------------------------------------------------- #
@router.get("", response_model=List[schemas.Person])
def list_people(db: Session = Depends(get_db)) -> List[schemas.Person]:
    """List all enrolled people with their face counts."""
    people = db.scalars(select(Person).order_by(Person.name)).all()
    return [_person_out(db, p) for p in people]


@router.post("", response_model=schemas.Person, status_code=status.HTTP_201_CREATED)
def create_person(
    body: schemas.PersonCreate, db: Session = Depends(get_db)
) -> schemas.Person:
    """Create a new (face-less) person record. Enroll photos separately."""
    person = Person(name=body.name, notes=body.notes)
    db.add(person)
    db.commit()
    db.refresh(person)
    return _person_out(db, person)


@router.get("/{person_id}", response_model=schemas.Person)
def get_person(person_id: int, db: Session = Depends(get_db)) -> schemas.Person:
    person = _get_person_or_404(db, person_id)
    return _person_out(db, person)


@router.put("/{person_id}", response_model=schemas.Person)
def update_person(
    person_id: int,
    body: schemas.PersonUpdate,
    db: Session = Depends(get_db),
) -> schemas.Person:
    person = _get_person_or_404(db, person_id)
    data = body.model_dump(exclude_unset=True)
    if "name" in data and data["name"] is not None:
        person.name = data["name"]
    if "notes" in data:
        person.notes = data["notes"]
    db.commit()
    db.refresh(person)
    return _person_out(db, person)


@router.delete("/{person_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_person(
    person_id: int,
    request: Request,
    db: Session = Depends(get_db),
) -> None:
    """Delete a person: cascades face embeddings, removes source images,
    detaches the identity from past events, and rebuilds the FAISS index."""
    person = _get_person_or_404(db, person_id)

    # Detach from events (FK is ON DELETE SET NULL, but be explicit so the
    # denormalized person_name snapshot is preserved on the events themselves).
    db.execute(
        update(Event).where(Event.person_id == person_id).values(person_id=None)
    )

    # Remove source enrollment images from disk.
    faces = db.scalars(
        select(FaceEmbedding).where(FaceEmbedding.person_id == person_id)
    ).all()
    settings = get_settings()
    for face in faces:
        _safe_unlink(_resolve_image_path(settings, face.image_path))
    _safe_rmdir(_faces_dir(settings, person_id))

    db.delete(person)  # cascade deletes face_embeddings rows
    db.commit()

    _rebuild_face_index(request, db)
    return None


def _purge_person_files(settings, db, person_id: int) -> None:
    faces = db.scalars(
        select(FaceEmbedding).where(FaceEmbedding.person_id == person_id)
    ).all()
    for face in faces:
        _safe_unlink(_resolve_image_path(settings, face.image_path))
    _safe_rmdir(_faces_dir(settings, person_id))


@router.post("/bulk-delete")
def bulk_delete_people(
    ids: list[int] = Body(..., embed=True),
    request: Request = None,
    db: Session = Depends(get_db),
) -> dict:
    """Delete the given people (+ enrolment images); rebuild the index once."""
    if not ids:
        return {"deleted": 0}
    ids = [int(i) for i in ids]
    db.execute(update(Event).where(Event.person_id.in_(ids)).values(person_id=None))
    settings = get_settings()
    n = 0
    for p in db.query(Person).filter(Person.id.in_(ids)).all():
        _purge_person_files(settings, db, p.id)
        db.delete(p)
        n += 1
    db.commit()
    _rebuild_face_index(request, db)
    return {"deleted": n}


@router.post("/clear-all")
def clear_all_people(
    confirm: bool = Body(False, embed=True),
    request: Request = None,
    db: Session = Depends(get_db),
) -> dict:
    """Delete ALL enrolled people (+ images). Requires confirm. Index rebuilt once."""
    if not confirm:
        raise HTTPException(status_code=400, detail="confirm=true required")
    settings = get_settings()
    n = 0
    while True:
        batch = db.query(Person).limit(200).all()
        if not batch:
            break
        ids = [int(p.id) for p in batch]
        db.execute(update(Event).where(Event.person_id.in_(ids)).values(person_id=None))
        for p in batch:
            _purge_person_files(settings, db, p.id)
            db.delete(p)
            n += 1
        db.commit()
        if len(batch) < 200:
            break
    _rebuild_face_index(request, db)
    return {"deleted": n}


# --------------------------------------------------------------------------- #
# Face enrollment / management
# --------------------------------------------------------------------------- #
@router.post("/{person_id}/faces", status_code=status.HTTP_201_CREATED)
async def enroll_face(
    person_id: int,
    request: Request,
    file: UploadFile = File(..., description="Photo containing exactly one clear face."),
    db: Session = Depends(get_db),
) -> dict:
    """Detect a face in the uploaded photo, embed it (ArcFace 512-d), persist
    the embedding + source image, and add it to the FAISS index."""
    person = _get_person_or_404(db, person_id)

    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported image type: {file.content_type}",
        )

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload")
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image too large")

    img = _decode_image(raw)

    recognizer = getattr(request.app.state, "recognizer", None)
    if recognizer is None:
        raise HTTPException(status_code=503, detail="Face recognizer not available")

    # recognizer returns detected faces, each carrying a 512-d embedding.
    faces = recognizer.detect_and_embed(img)
    faces_detected = len(faces)
    if faces_detected == 0:
        raise HTTPException(status_code=422, detail="No face detected in image")

    # Pick the largest / highest-confidence face for enrollment.
    best = _select_best_face(faces)
    embedding = _extract_embedding(best)
    blob = _vector_to_blob(embedding)

    # Persist the source image to data/faces/<person_id>/<uuid>.<ext>.
    settings = get_settings()
    ext = _safe_ext(file.filename)
    fname = f"{uuid.uuid4().hex}{ext}"
    dest_dir = _faces_dir(settings, person_id)
    abs_path = os.path.join(dest_dir, fname)
    with open(abs_path, "wb") as fh:
        fh.write(raw)
    rel_path = _relative_image_path(settings, abs_path, person_id, fname)

    face_row = FaceEmbedding(
        person_id=person.id,
        vector=blob,
        image_path=rel_path,
    )
    db.add(face_row)
    db.commit()
    db.refresh(face_row)

    # Add to FAISS and record its position for remove/rebuild mapping.
    index = getattr(request.app.state, "face_index", None)
    if index is not None:
        try:
            faiss_pos = index.add(
                np.frombuffer(blob, dtype="<f4"),
                person_id=person.id,
                embedding_id=face_row.id,
            )
            if faiss_pos is not None:
                face_row.faiss_id = int(faiss_pos)
                db.commit()
        except Exception:  # noqa: BLE001 - index stays consistent via DB rebuild
            db.rollback()
            _rebuild_face_index(request, db)

    return {
        "embedding_id": face_row.id,
        "faces_detected": faces_detected,
        "image_path": rel_path,
    }


@router.get("/{person_id}/faces", response_model=List[schemas.FaceEmbedding])
def list_faces(
    person_id: int, db: Session = Depends(get_db)
) -> List[schemas.FaceEmbedding]:
    _get_person_or_404(db, person_id)
    faces = db.scalars(
        select(FaceEmbedding)
        .where(FaceEmbedding.person_id == person_id)
        .order_by(FaceEmbedding.created_at)
    ).all()
    return [_face_out(f) for f in faces]


@router.get("/{person_id}/faces/{face_id}/image")
def get_face_image(
    person_id: int, face_id: int, db: Session = Depends(get_db)
):
    """Serve the source enrollment photo for a face embedding."""
    from fastapi.responses import FileResponse

    face = db.get(FaceEmbedding, face_id)
    if face is None or face.person_id != person_id:
        raise HTTPException(status_code=404, detail="Face not found")
    if not face.image_path:
        raise HTTPException(status_code=404, detail="No source image stored")
    abs_path = _resolve_image_path(get_settings(), face.image_path)
    if not abs_path or not os.path.isfile(abs_path):
        raise HTTPException(status_code=404, detail="Image file missing")
    return FileResponse(abs_path, media_type="image/jpeg")


@router.delete(
    "/{person_id}/faces/{face_id}", status_code=status.HTTP_204_NO_CONTENT
)
def delete_face(
    person_id: int,
    face_id: int,
    request: Request,
    db: Session = Depends(get_db),
) -> None:
    """Remove a single embedding, delete its source image, and rebuild FAISS."""
    face = db.get(FaceEmbedding, face_id)
    if face is None or face.person_id != person_id:
        raise HTTPException(status_code=404, detail="Face not found")

    _safe_unlink(_resolve_image_path(get_settings(), face.image_path))
    db.delete(face)
    db.commit()

    _rebuild_face_index(request, db)
    return None


# --------------------------------------------------------------------------- #
# Embedding / face-object extraction helpers (tolerant of recognizer shape)
# --------------------------------------------------------------------------- #
def _extract_embedding(face) -> np.ndarray:
    """Pull a 512-d embedding out of a recognizer face object/dict.

    insightface Face objects expose ``normed_embedding`` / ``embedding``; we
    also accept plain dicts or raw ndarrays for testability.
    """
    if isinstance(face, np.ndarray):
        return face
    for attr in ("normed_embedding", "embedding"):
        val = getattr(face, attr, None)
        if val is not None:
            return np.asarray(val, dtype=np.float32)
    if isinstance(face, dict):
        for key in ("normed_embedding", "embedding", "vector"):
            if key in face and face[key] is not None:
                return np.asarray(face[key], dtype=np.float32)
    raise HTTPException(
        status_code=500, detail="Recognizer returned a face without an embedding"
    )


def _face_area(face) -> float:
    """Best-effort bounding-box area, used to pick the dominant face."""
    bbox = None
    if isinstance(face, dict):
        bbox = face.get("bbox")
    else:
        bbox = getattr(face, "bbox", None)
    if bbox is None:
        return 0.0
    b = np.asarray(bbox, dtype=np.float32).reshape(-1)
    if b.size >= 4:
        return float(max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]))
    return 0.0


def _select_best_face(faces):
    """Return the largest detected face (the intended subject for enrollment)."""
    return max(faces, key=_face_area)


# --------------------------------------------------------------------------- #
# Path helpers
# --------------------------------------------------------------------------- #
def _safe_ext(filename: str | None) -> str:
    if not filename:
        return ".jpg"
    ext = os.path.splitext(filename)[1].lower()
    if ext in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        return ext
    return ".jpg"


def _data_dir(settings: Settings) -> str:
    return str(getattr(settings, "data_dir", "data"))


def _relative_image_path(
    settings: Settings, abs_path: str, person_id: int, fname: str
) -> str:
    """Store path relative to the data dir, matching ``data/faces/<id>/<file>``."""
    data_dir = os.path.abspath(_data_dir(settings))
    abs_path = os.path.abspath(abs_path)
    try:
        rel = os.path.relpath(abs_path, data_dir)
        # Guard against escaping the data dir.
        if not rel.startswith(".."):
            return os.path.join("data", rel)
    except ValueError:
        pass
    return os.path.join("data", "faces", str(person_id), fname)


def _resolve_image_path(settings: Settings, stored: str | None) -> str | None:
    """Resolve a stored (relative) image path to an absolute on-disk path."""
    if not stored:
        return None
    if os.path.isabs(stored):
        return stored
    data_dir = os.path.abspath(_data_dir(settings))
    # Stored paths are like 'data/faces/<id>/<file>'; strip the leading 'data/'.
    norm = stored.replace("\\", "/")
    if norm.startswith("data/"):
        norm = norm[len("data/"):]
    return os.path.join(data_dir, norm)


def _safe_unlink(path: str | None) -> None:
    if not path:
        return
    try:
        if os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass


def _safe_rmdir(path: str | None) -> None:
    if not path:
        return
    try:
        if os.path.isdir(path) and not os.listdir(path):
            os.rmdir(path)
    except OSError:
        pass
