"""Clip assembly + thumbnail extraction.

On a person-trigger the camera worker calls :func:`clip_event`. Given the
trigger timestamp and the camera's warm segment buffer, it:

1. Waits (up to ``post_seconds`` + a small margin) for the post-roll segments
   covering ``trigger + post`` to be flushed to disk by the segmenter.
2. Selects every segment whose time span overlaps ``[trigger - pre, trigger + post]``.
3. Concatenates them with ffmpeg's concat demuxer using ``-c copy`` — no
   re-encode, so this is fast and GPU-free — into
   ``data/recordings/<camera_id>/<event_id>.mp4``.
4. Extracts a single thumbnail JPEG near the trigger instant into
   ``data/thumbnails/<event_id>.jpg``.

The function is deliberately synchronous and self-contained: it knows nothing
about the DB or detection. The worker passes in the paths/ids and stores the
returned :class:`ClipResult` into the ``events`` row (``clip_path``,
``thumb_path``, ``end_ts``). Returned paths are absolute; the caller is
responsible for storing them relative to the data root per the data model.
"""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .segmenter import SegmentInfo, list_segments

logger = logging.getLogger(__name__)


@dataclass
class ClipResult:
    """Outcome of an event clip assembly."""

    clip_path: Path | None  # absolute path to the assembled .mp4, or None on failure
    thumb_path: Path | None  # absolute path to the thumbnail .jpg, or None
    start_ts: datetime  # clip window start (UTC)
    end_ts: datetime  # clip window end (UTC)
    segments_used: int  # number of source segments concatenated

    @property
    def ok(self) -> bool:
        return self.clip_path is not None and self.clip_path.exists()


def _utc(dt: datetime) -> datetime:
    """Normalise a datetime to tz-aware UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _select_segments(
    segments: list[SegmentInfo],
    window_start: datetime,
    window_end: datetime,
    segment_seconds: int,
) -> list[SegmentInfo]:
    """Return segments whose [start, start+len) span overlaps the window.

    Segment length is taken as ``segment_seconds``; the final live segment may
    be shorter but we only use its start for overlap, which is conservative
    (we'd rather include a touching segment than drop footage).
    """
    seg_len = timedelta(seconds=segment_seconds)
    selected: list[SegmentInfo] = []
    for seg in segments:  # already sorted by start
        seg_end = seg.start + seg_len
        # overlap test: seg_start < window_end AND seg_end > window_start
        if seg.start < window_end and seg_end > window_start:
            selected.append(seg)
    return selected


def _wait_for_post_roll(
    segments_dir: Path,
    window_end: datetime,
    segment_seconds: int,
    *,
    poll_interval: float = 0.5,
    extra_margin: float = 2.0,
    timeout: float | None = None,
) -> list[SegmentInfo]:
    """Block until a segment starting at/after ``window_end`` exists, or timeout.

    The segmenter writes a segment to its final name only after it finishes, so
    the segment covering the tail of our window is fully on disk once a *newer*
    segment has appeared. We wait for that successor (or the deadline).
    """
    if timeout is None:
        # Enough time for the post segment to fill + finalise + a safety margin.
        timeout = float(segment_seconds) * 2 + extra_margin
    deadline = time.monotonic() + timeout
    while True:
        segs = list_segments(segments_dir)
        # A segment that starts at or after window_end means the post-roll
        # segment(s) before it have been finalised.
        if any(s.start >= window_end for s in segs):
            return segs
        if time.monotonic() >= deadline:
            logger.debug(
                "post-roll wait timed out after %.1fs for %s (window_end=%s); "
                "proceeding with available segments",
                timeout, segments_dir, window_end.isoformat(),
            )
            return segs
        time.sleep(poll_interval)


def _is_finalized(path: Path, *, ffprobe_bin: str = "ffprobe", timeout: float = 5.0) -> bool:
    """Return True if ``path`` is a complete, readable mp4 (has a moov atom).

    A segment ffmpeg is still writing has no moov atom yet, so concatenating it
    fails the whole clip. We cheaply probe each candidate and skip the ones that
    aren't finalized — making clip assembly robust to a stalled/flaky camera.
    """
    try:
        if not path.is_file() or path.stat().st_size < 1024:
            return False
        proc = subprocess.run(
            [ffprobe_bin, "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=timeout,
        )
        return proc.returncode == 0 and proc.stdout.strip() not in (b"", b"N/A")
    except Exception:
        return False


def _concat_segments(
    segments: list[SegmentInfo],
    out_path: Path,
    *,
    ffmpeg_bin: str = "ffmpeg",
    timeout: float = 120.0,
) -> bool:
    """Concatenate segments into ``out_path`` via the concat demuxer (stream copy)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Build a concat list file. Paths must be quoted/escaped per ffmpeg rules.
    list_path = out_path.with_suffix(".concat.txt")
    try:
        with list_path.open("w", encoding="utf-8") as fh:
            for seg in segments:
                # ffmpeg concat: escape single quotes, wrap in single quotes.
                p = str(seg.path.resolve()).replace("'", r"'\''")
                fh.write(f"file '{p}'\n")

        cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel", "error",
            "-nostdin",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_path),
            "-c", "copy",
            "-movflags", "+faststart",
            "-y",
            str(out_path),
        ]
        logger.debug("concat cmd: %s", " ".join(cmd))
        proc = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        if proc.returncode != 0:
            logger.error(
                "ffmpeg concat failed rc=%s for %s: %s",
                proc.returncode, out_path.name,
                proc.stderr.decode("utf-8", "replace")[-500:].strip(),
            )
            return False
        return out_path.exists() and out_path.stat().st_size > 0
    except subprocess.TimeoutExpired:
        logger.error("ffmpeg concat timed out for %s", out_path.name)
        return False
    except Exception:  # pragma: no cover
        logger.exception("ffmpeg concat error for %s", out_path.name)
        return False
    finally:
        try:
            list_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass


def extract_thumbnail(
    source: Path,
    thumb_path: Path,
    *,
    offset_seconds: float = 0.0,
    width: int = 320,
    ffmpeg_bin: str = "ffmpeg",
    timeout: float = 30.0,
) -> bool:
    """Grab a single JPEG frame from ``source`` at ``offset_seconds``.

    Scales to ``width`` px (preserving aspect, even height) for a compact thumb.
    Falls back to the first frame if seeking past the clip's end yields nothing.
    """
    thumb_path.parent.mkdir(parents=True, exist_ok=True)

    def _run(seek: float) -> bool:
        cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel", "error",
            "-nostdin",
        ]
        if seek > 0:
            # Input-side seek is fast; accuracy within a segment is fine for a thumb.
            cmd += ["-ss", f"{seek:.3f}"]
        cmd += [
            "-i", str(source),
            "-frames:v", "1",
            "-vf", f"scale={width}:-2:flags=fast_bilinear",
            "-q:v", "3",
            "-y",
            str(thumb_path),
        ]
        try:
            proc = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            logger.error("thumbnail extraction timed out for %s", source.name)
            return False
        if proc.returncode != 0:
            logger.debug(
                "thumbnail extract rc=%s seek=%.2f %s: %s",
                proc.returncode, seek, source.name,
                proc.stderr.decode("utf-8", "replace")[-300:].strip(),
            )
            return False
        return thumb_path.exists() and thumb_path.stat().st_size > 0

    if _run(max(0.0, offset_seconds)):
        return True
    # Retry at the very start of the clip.
    if offset_seconds > 0 and _run(0.0):
        return True
    logger.warning("could not extract thumbnail from %s", source)
    return False


@dataclass
class BuiltClip:
    """Result of :func:`build_clip` with DB-relative path strings.

    Paths are stored relative to the app working dir (e.g.
    ``data/recordings/<cam>/<id>.mp4``) to match the convention the events API
    expects; ``None`` when the artifact could not be produced.
    """

    clip_path: str | None
    thumb_path: str | None
    end_ts: datetime | None
    segments_used: int


def build_clip(
    *,
    segmenter: object,
    camera_id: int,
    event_id: int,
    trigger_time: datetime,
    data_dir: Path,
    fallback_frame: object | None = None,
    pre_seconds: int | None = None,
    post_seconds: int | None = None,
) -> BuiltClip:
    """Worker-facing adapter around :func:`clip_event`.

    The camera worker owns a :class:`~app.recording.segmenter.Segmenter` and a
    ``data_dir``; this assembles the event clip + thumbnail from the warm
    segment buffer and returns paths relative to the app working dir so they can
    be stored directly on the ``events`` row.

    ``fallback_frame`` is accepted for API compatibility — the worker writes its
    own thumbnail from the trigger frame when this returns no thumbnail.
    """
    data_dir = Path(data_dir)
    segments_dir = Path(getattr(segmenter, "segments_dir", data_dir / "segments" / str(camera_id)))
    segment_seconds = int(getattr(segmenter, "segment_seconds", 4))
    if pre_seconds is None:
        pre_seconds = int(getattr(segmenter, "pre_seconds", 5) or 5)
    if post_seconds is None:
        post_seconds = int(getattr(segmenter, "post_seconds", 10) or 10)

    result = clip_event(
        event_id=event_id,
        camera_id=camera_id,
        trigger_ts=trigger_time,
        segments_dir=segments_dir,
        recordings_root=data_dir / "recordings",
        thumbnails_root=data_dir / "thumbnails",
        pre_seconds=pre_seconds,
        post_seconds=post_seconds,
        segment_seconds=segment_seconds,
    )

    def _rel(p: Path | None) -> str | None:
        if p is None:
            return None
        p = Path(p)
        try:
            # Store as "data/<...>" to match the worker's thumbnail convention.
            return str(Path(data_dir.name) / p.relative_to(data_dir))
        except ValueError:
            return str(p)

    return BuiltClip(
        clip_path=_rel(result.clip_path) if result.ok else None,
        thumb_path=_rel(result.thumb_path),
        end_ts=result.end_ts,
        segments_used=result.segments_used,
    )


def build_clip_from_track(
    *,
    segmenter: object,
    camera_id: int,
    event_id: int,
    enter_ts: float,
    last_ts: float,
    data_dir: Path,
    pre_seconds: int | None = None,
    post_seconds: int | None = None,
    fallback_frame: object | None = None,
) -> BuiltClip:
    """Assemble a clip spanning a track's whole presence: ``[enter-pre, last+post]``.

    ``enter_ts``/``last_ts`` are epoch seconds (UTC). The window is bounded by the
    segmenter's retention; a presence longer than retention loses its pre-roll
    head (logged). Used by track-driven recording.
    """
    data_dir = Path(data_dir)
    segments_dir = Path(getattr(segmenter, "segments_dir", data_dir / "segments" / str(camera_id)))
    segment_seconds = int(getattr(segmenter, "segment_seconds", 4))
    retention = float(getattr(segmenter, "retention_seconds", 120))
    if pre_seconds is None:
        pre_seconds = int(getattr(segmenter, "pre_seconds", 5) or 5)
    if post_seconds is None:
        post_seconds = int(getattr(segmenter, "post_seconds", 5) or 5)

    # Naive UTC, matching the segment-filename UTC convention (clip_event._utc
    # treats naive as UTC).
    window_start = datetime.utcfromtimestamp(float(enter_ts) - pre_seconds)
    window_end = datetime.utcfromtimestamp(float(last_ts) + post_seconds)
    if (window_end - window_start).total_seconds() > retention:
        logger.warning(
            "camera %s event %s: presence %.0fs exceeds retention %.0fs; pre-roll head will be truncated",
            camera_id, event_id, (window_end - window_start).total_seconds(), retention,
        )
    # Anchor the thumbnail/fallback inside the presence (not at window_start,
    # which may predate the retained buffer).
    trigger_ts = datetime.utcfromtimestamp(float(enter_ts))

    result = clip_event(
        event_id=event_id,
        camera_id=camera_id,
        trigger_ts=trigger_ts,
        segments_dir=segments_dir,
        recordings_root=data_dir / "recordings",
        thumbnails_root=data_dir / "thumbnails",
        pre_seconds=pre_seconds,
        post_seconds=post_seconds,
        segment_seconds=segment_seconds,
        window_start=window_start,
        window_end=window_end,
    )

    def _rel(p: Path | None) -> str | None:
        if p is None:
            return None
        p = Path(p)
        try:
            return str(Path(data_dir.name) / p.relative_to(data_dir))
        except ValueError:
            return str(p)

    return BuiltClip(
        clip_path=_rel(result.clip_path) if result.ok else None,
        thumb_path=_rel(result.thumb_path),
        end_ts=result.end_ts,
        segments_used=result.segments_used,
    )


def clip_event(
    *,
    event_id: int,
    camera_id: int,
    trigger_ts: datetime,
    segments_dir: Path,
    recordings_root: Path,
    thumbnails_root: Path,
    pre_seconds: int = 5,
    post_seconds: int = 10,
    segment_seconds: int = 4,
    ffmpeg_bin: str = "ffmpeg",
    wait_for_post: bool = True,
    thumb_width: int = 320,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
) -> ClipResult:
    """Assemble the clip + thumbnail for a person-trigger event.

    If ``window_start`` and ``window_end`` are both given they define the clip
    span directly (track-driven recording); otherwise the window is computed
    from ``trigger_ts +/- pre/post``. ``trigger_ts`` is still used for the
    thumbnail offset and the nearest-segment fallback.

    Parameters
    ----------
    event_id, camera_id:
        Identify the output files: clip ->
        ``recordings_root/<camera_id>/<event_id>.mp4``, thumb ->
        ``thumbnails_root/<event_id>.jpg``.
    trigger_ts:
        When the person was detected (UTC; naive treated as UTC).
    segments_dir:
        The camera's warm segment directory (``Segmenter.segments_dir``).
    pre_seconds / post_seconds:
        Clip window around the trigger.
    segment_seconds:
        The segmenter's segment length (used for overlap math + post-roll wait).
    wait_for_post:
        If True, block until post-roll segments are finalised before concat.

    Returns
    -------
    ClipResult
        ``ok`` is True iff a non-empty clip was produced. Thumbnail failure does
        not fail the clip (thumb_path will simply be None).
    """
    trigger_ts = _utc(trigger_ts)
    if window_start is not None and window_end is not None:
        window_start = _utc(window_start)
        window_end = _utc(window_end)
    else:
        window_start = trigger_ts - timedelta(seconds=pre_seconds)
        window_end = trigger_ts + timedelta(seconds=post_seconds)

    segments_dir = Path(segments_dir)
    clip_path = Path(recordings_root) / str(camera_id) / f"{event_id}.mp4"
    thumb_path = Path(thumbnails_root) / f"{event_id}.jpg"

    result = ClipResult(
        clip_path=None,
        thumb_path=None,
        start_ts=window_start,
        end_ts=window_end,
        segments_used=0,
    )

    # 1. Ensure post-roll is on disk.
    if wait_for_post:
        segs = _wait_for_post_roll(segments_dir, window_end, segment_seconds)
    else:
        segs = list_segments(segments_dir)

    if not segs:
        logger.warning(
            "event %s (cam %s): no segments available in %s; cannot build clip",
            event_id, camera_id, segments_dir,
        )
        return result

    # The newest file on disk is very likely the segment ffmpeg is *currently*
    # writing — it has no `moov` atom yet, so concatenating it fails the whole
    # clip ("moov atom not found"). Treat it as live and never feed it to
    # concat. In the normal case the post-roll wait has already produced a
    # finalized successor that supersedes it, so we lose no footage.
    live_tail = segs[-1] if len(segs) > 1 else None

    # 2. Select overlapping segments (excluding the still-growing tail).
    selected = _select_segments(segs, window_start, window_end, segment_seconds)
    if live_tail is not None:
        selected = [s for s in selected if s.path != live_tail.path]
    if not selected:
        # Window may predate the buffer (e.g. just-started camera). Fall back to
        # the nearest *finalized* segment so we still capture something.
        finalized = [s for s in segs if live_tail is None or s.path != live_tail.path]
        if not finalized:
            logger.warning(
                "event %s (cam %s): only a live segment available; cannot build clip yet",
                event_id, camera_id,
            )
            return result
        nearest = min(finalized, key=lambda s: abs((s.start - trigger_ts).total_seconds()))
        selected = [nearest]
        logger.info(
            "event %s (cam %s): window had no overlapping segments; "
            "falling back to nearest segment %s",
            event_id, camera_id, nearest.name,
        )

    # Drop any segment that isn't a finalized, readable mp4 (still being
    # written, truncated by a camera drop, etc.) so one bad file can't fail the
    # whole concat.
    finalized_sel = [s for s in selected if _is_finalized(s.path)]
    if not finalized_sel:
        logger.warning(
            "event %s (cam %s): no finalized segments in window "
            "(camera stalled/just-started?); cannot build clip",
            event_id, camera_id,
        )
        return result
    selected = finalized_sel
    result.segments_used = len(selected)

    # 3. Concatenate (stream copy, no re-encode).
    if not _concat_segments(selected, clip_path, ffmpeg_bin=ffmpeg_bin):
        logger.error("event %s (cam %s): clip concat failed", event_id, camera_id)
        return result
    result.clip_path = clip_path

    # 4. Thumbnail at the trigger instant relative to the clip start.
    clip_window_start = selected[0].start
    thumb_offset = max(0.0, (trigger_ts - clip_window_start).total_seconds())
    if extract_thumbnail(
        clip_path,
        thumb_path,
        offset_seconds=thumb_offset,
        width=thumb_width,
        ffmpeg_bin=ffmpeg_bin,
    ):
        result.thumb_path = thumb_path

    logger.info(
        "event %s (cam %s): clip ready (%d segs) -> %s%s",
        event_id, camera_id, result.segments_used, clip_path.name,
        "" if result.thumb_path else " (no thumbnail)",
    )
    return result
