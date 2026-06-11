"""Recording-artifact lifecycle: keep on-disk files in sync with the DB.

The VMS writes three kinds of per-event/per-camera artifacts:

  * ``data/thumbnails/<event_id>.jpg``
  * ``data/recordings/<camera_id>/<event_id>.mp4``
  * ``data/segments/<camera_id>/seg_*.mp4`` (warm rolling buffer)
  * ``/dev/shm/vms_frames/cam_<camera_id>.jpg`` (live frame slot)

Integer ids are reused by SQLite after deletion, so an artifact left behind by
a deleted camera/event can be silently served for a *new* row that happens to
reuse the id — which is exactly how "old footage from a camera that no longer
exists" shows up. These helpers delete a camera's artifacts on delete/repoint,
and an orphan sweep at startup removes any file with no owning DB row.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Iterable

logger = logging.getLogger("vms.recording.cleanup")


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:  # pragma: no cover - best effort
        logger.warning("could not delete %s: %s", path, exc)


def _rmtree(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass
    except OSError as exc:  # pragma: no cover
        logger.warning("could not remove dir %s: %s", path, exc)


def delete_camera_artifacts(settings, camera_id: int, event_ids: Iterable[int]) -> None:
    """Delete every on-disk artifact owned by one camera.

    Call this BEFORE the camera row (and its cascade-deleted events) are
    removed, passing the event ids so their thumbnails are cleaned too.
    """
    cam_id = int(camera_id)
    for eid in event_ids:
        _unlink(Path(settings.thumbnails_dir) / f"{int(eid)}.jpg")
    _rmtree(Path(settings.recordings_dir) / str(cam_id))
    _rmtree(Path(settings.segments_dir) / str(cam_id))
    delete_hls_artifacts(settings, cam_id)
    # Live frame slot (kept in shared memory).
    frames_dir = Path("/dev/shm/vms_frames")
    _unlink(frames_dir / f"cam_{cam_id}.jpg")
    logger.info("Cleaned artifacts for camera %s", cam_id)


def delete_hls_artifacts(settings, camera_id: int) -> None:
    """Remove a camera's on-demand HLS segment dir."""
    hls = getattr(settings, "hls_dir", None)
    if hls is not None:
        _rmtree(Path(hls) / str(int(camera_id)))


def delete_event_artifacts(settings, camera_id: int, event_id: int) -> None:
    """Delete one event's clip + thumbnail."""
    _unlink(Path(settings.thumbnails_dir) / f"{int(event_id)}.jpg")
    _unlink(Path(settings.recordings_dir) / str(int(camera_id)) / f"{int(event_id)}.mp4")


def purge_orphans(settings) -> dict:
    """Remove artifacts with no owning DB row. Returns a small stats dict.

    Safe to run at startup: a file is an orphan only when its camera/event id
    is absent from the DB, so concurrent writers (workers) never lose live data
    (their cameras/events exist).
    """
    from sqlalchemy import select

    from ..db.database import SessionLocal
    from ..db.models import Camera, Event

    stats = {"thumbnails": 0, "recordings": 0, "recording_dirs": 0,
             "segment_dirs": 0, "frame_slots": 0, "hls_dirs": 0}
    session = SessionLocal()
    try:
        event_ids = {int(e) for e in session.scalars(select(Event.id)).all()}
        camera_ids = {int(c) for c in session.scalars(select(Camera.id)).all()}
    finally:
        session.close()

    # Thumbnails: <event_id>.jpg
    thumbs = Path(settings.thumbnails_dir)
    if thumbs.is_dir():
        for f in thumbs.glob("*.jpg"):
            try:
                eid = int(f.stem)
            except ValueError:
                continue
            if eid not in event_ids:
                _unlink(f); stats["thumbnails"] += 1

    # Recordings: <camera_id>/<event_id>.mp4
    recs = Path(settings.recordings_dir)
    if recs.is_dir():
        for cam_dir in recs.iterdir():
            if not cam_dir.is_dir():
                continue
            try:
                cid = int(cam_dir.name)
            except ValueError:
                continue
            if cid not in camera_ids:
                _rmtree(cam_dir); stats["recording_dirs"] += 1
                continue
            for f in cam_dir.glob("*.mp4"):
                try:
                    eid = int(f.stem)
                except ValueError:
                    continue
                if eid not in event_ids:
                    _unlink(f); stats["recordings"] += 1

    # Segment dirs for cameras that no longer exist.
    segs = Path(settings.segments_dir)
    if segs.is_dir():
        for cam_dir in segs.iterdir():
            if not cam_dir.is_dir():
                continue
            try:
                cid = int(cam_dir.name)
            except ValueError:
                continue
            if cid not in camera_ids:
                _rmtree(cam_dir); stats["segment_dirs"] += 1

    # HLS dirs for cameras that no longer exist (live-with-sound leftovers).
    hls = getattr(settings, "hls_dir", None)
    if hls is not None and Path(hls).is_dir():
        for cam_dir in Path(hls).iterdir():
            if not cam_dir.is_dir():
                continue
            try:
                cid = int(cam_dir.name)
            except ValueError:
                continue
            if cid not in camera_ids:
                _rmtree(cam_dir); stats["hls_dirs"] += 1

    # Frame slots for cameras that no longer exist.
    frames_dir = Path("/dev/shm/vms_frames")
    if frames_dir.is_dir():
        for f in frames_dir.glob("cam_*.jpg"):
            try:
                cid = int(f.stem.split("_", 1)[1])
            except (ValueError, IndexError):
                continue
            if cid not in camera_ids:
                _unlink(f); stats["frame_slots"] += 1

    if any(stats.values()):
        logger.info("Orphan purge removed: %s", stats)
    return stats
