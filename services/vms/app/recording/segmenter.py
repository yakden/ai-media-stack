"""Warm rolling ffmpeg segment buffer (per camera).

The :class:`Segmenter` owns a single long-lived ``ffmpeg`` process that reads
the camera's RTSP stream and continuously writes fixed-length ``.mp4`` segments
to ``data/segments/<camera_id>/``. Because the process is always running, a
configurable amount of *pre-roll* footage is already on disk the moment a
person is detected, so clips can include the seconds leading up to the event
without re-encoding or buffering frames in memory.

Design notes
------------
* **Stream copy, no decode.** ffmpeg is invoked with ``-c copy`` so it never
  re-encodes — negligible CPU and zero GPU. The camera worker decodes frames
  separately (via OpenCV) for detection; this process is purely an I/O recorder.
* **RTSP over TCP.** ``-rtsp_transport tcp`` for reliability over lossy links,
  matching the worker's capture settings.
* **Timestamped names.** Segments are named with ffmpeg's ``strftime`` so the
  clipper can select segments by wall-clock window. Each segment's start time is
  recoverable from its filename; this is the source of truth for clip selection.
  ffmpeg ``strftime`` uses the process's local timezone, so the segmenter pins
  the ffmpeg subprocess to ``TZ=UTC`` (see :meth:`Segmenter._spawn`) to keep
  filenames in UTC, matching the UTC timestamps the worker reads from the DB.
* **Bounded retention.** Old segments are pruned to ``retention_seconds`` worth
  of buffer so the segments dir never grows unbounded. Retention must comfortably
  exceed the largest ``pre_seconds`` any event will request.
* **Self-healing.** ffmpeg can die (camera reboot, network blip). A background
  watchdog thread restarts it with backoff while the segmenter is "started".

The segmenter does not know about events or triggers — it just keeps the buffer
warm. The clipper reads the segments it produces.
"""

from __future__ import annotations

from typing import Any
import logging
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def _set_pdeathsig() -> None:  # pragma: no cover - exercised only in child proc
    """Ask the kernel to SIGKILL this child when its parent thread dies (Linux).

    Runs in the forked child between fork() and exec(). On non-Linux or if
    prctl is unavailable this silently no-ops; the graceful ``stop()`` path
    still handles normal teardown.
    """
    try:
        import ctypes
        import signal as _signal

        PR_SET_PDEATHSIG = 1
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(PR_SET_PDEATHSIG, _signal.SIGKILL, 0, 0, 0)
    except Exception:
        pass

# Segment filename format. The strftime pattern is embedded directly so ffmpeg's
# `-strftime 1` fills it in at write time. Resolution is 1s, which is plenty —
# segments are several seconds long. Example: ``seg_20260606T143012.mp4``.
_SEG_PREFIX = "seg_"
_SEG_TIME_FMT = "%Y%m%dT%H%M%S"
_SEG_GLOB = f"{_SEG_PREFIX}*.mp4"
_SEG_RE = re.compile(rf"^{re.escape(_SEG_PREFIX)}(\d{{8}}T\d{{6}})\.mp4$")
# ffmpeg consumes this strftime pattern at segment-write time:
_SEG_FFMPEG_PATTERN = f"{_SEG_PREFIX}%Y%m%dT%H%M%S.mp4"


@dataclass(frozen=True)
class SegmentInfo:
    """A single on-disk segment file and its recovered start time (UTC)."""

    path: Path
    start: datetime  # wall-clock start of the segment (UTC, tz-aware)

    @property
    def name(self) -> str:
        return self.path.name


def parse_segment_name(name: str) -> datetime | None:
    """Recover a segment's UTC start time from its filename, or ``None``.

    Filenames are produced by ffmpeg with the subprocess pinned to ``TZ=UTC``
    (see :meth:`Segmenter._spawn`), so the embedded wall-clock time is UTC.
    """
    m = _SEG_RE.match(name)
    if not m:
        return None
    try:
        dt = datetime.strptime(m.group(1), _SEG_TIME_FMT)
    except ValueError:
        return None
    return dt.replace(tzinfo=timezone.utc)


def list_segments(segments_dir: Path) -> list[SegmentInfo]:
    """Return all parseable segments in ``segments_dir`` sorted by start time.

    In stream-copy ``-f segment`` mode ffmpeg writes each segment to its final
    name directly, so the most recent file may still be growing. The clipper
    accounts for this by waiting for post-roll segments to finalize.
    """
    out: list[SegmentInfo] = []
    if not segments_dir.is_dir():
        return out
    for child in segments_dir.iterdir():
        if not child.is_file():
            continue
        start = parse_segment_name(child.name)
        if start is None:
            continue
        out.append(SegmentInfo(path=child, start=start))
    out.sort(key=lambda s: s.start)
    return out


class Segmenter:
    """Manage one warm ffmpeg segment-recording process for a single camera.

    Use as::

        seg = Segmenter(camera_id, rtsp_url, segments_root)
        seg.start()
        ...                       # buffer fills in the background
        seg.stop()

    or as a context manager. All time bookkeeping uses UTC.
    """

    def __init__(
        self,
        camera_id: int,
        rtsp_url: str,
        segments_root: Path | None = None,
        *,
        settings: object | None = None,
        segment_seconds: int = 4,
        retention_seconds: int = 120,
        ffmpeg_bin: str = "ffmpeg",
        rtsp_transport: str = "tcp",
        prune_interval: float = 15.0,
        restart_backoff: float = 3.0,
        restart_backoff_max: float = 30.0,
        **_ignore: Any,
    ) -> None:
        # The worker constructs with settings=...; derive paths/durations from it.
        # **_ignore swallows pre_seconds/post_seconds (a clipper concern, not ours).
        if settings is not None:
            if segments_root is None:
                segments_root = getattr(settings, "segments_dir", None)
            segment_seconds = int(getattr(settings, "segment_seconds", segment_seconds))
            retention_seconds = int(getattr(settings, "segment_retention_seconds", retention_seconds))
        if segments_root is None:
            segments_root = Path("/app/data/segments")
        if retention_seconds < segment_seconds:
            retention_seconds = segment_seconds * 4
        if segment_seconds < 1:
            raise ValueError("segment_seconds must be >= 1")
        if retention_seconds < segment_seconds:
            raise ValueError("retention_seconds must be >= segment_seconds")

        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.segments_dir = Path(segments_root) / str(camera_id)
        self.segment_seconds = int(segment_seconds)
        self.retention_seconds = int(retention_seconds)
        self.ffmpeg_bin = ffmpeg_bin
        self.rtsp_transport = rtsp_transport
        self.prune_interval = prune_interval
        self.restart_backoff = restart_backoff
        self.restart_backoff_max = restart_backoff_max

        self._proc: subprocess.Popen[bytes] | None = None
        self._watchdog: threading.Thread | None = None
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._last_prune = 0.0

    # ------------------------------------------------------------------ API

    def start(self) -> None:
        """Start the warm segment buffer (idempotent)."""
        with self._lock:
            if self._watchdog and self._watchdog.is_alive():
                return
            self.segments_dir.mkdir(parents=True, exist_ok=True)
            # Purge any segments left by a previous session. A restart usually
            # means the RTSP source changed (URL edit, re-enable) — keeping old
            # segments in the same per-camera dir would let a clip splice
            # footage from two different streams together ("channels confused").
            self._purge_all_segments()
            self._stop_evt.clear()
            self._watchdog = threading.Thread(
                target=self._run,
                name=f"segmenter-cam{self.camera_id}",
                daemon=True,
            )
            self._watchdog.start()
            logger.info("Segmenter started for camera %s -> %s", self.camera_id, self.segments_dir)

    def stop(self, timeout: float = 10.0) -> None:
        """Stop the buffer and terminate ffmpeg (idempotent)."""
        self._stop_evt.set()
        with self._lock:
            self._terminate_proc(timeout=timeout)
            wd = self._watchdog
        if wd and wd.is_alive():
            wd.join(timeout=timeout)
        with self._lock:
            self._watchdog = None
        logger.info("Segmenter stopped for camera %s", self.camera_id)

    def close(self, timeout: float = 10.0) -> None:
        """Alias for :meth:`stop` so generic ``close()``-based teardown stops
        the ffmpeg process (the worker's cleanup loop calls ``close()``)."""
        self.stop(timeout=timeout)

    def _purge_all_segments(self) -> None:
        """Delete every existing segment file in this camera's dir."""
        try:
            for child in self.segments_dir.glob(_SEG_GLOB):
                try:
                    child.unlink()
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    logger.warning("camera %s: could not purge %s: %s", self.camera_id, child.name, exc)
        except Exception:  # pragma: no cover - best-effort
            logger.debug("camera %s: purge_all_segments failed", self.camera_id, exc_info=True)

    def is_running(self) -> bool:
        with self._lock:
            return (
                self._proc is not None
                and self._proc.poll() is None
                and not self._stop_evt.is_set()
            )

    def segments(self) -> list[SegmentInfo]:
        """Snapshot of the current on-disk segments (sorted by start time)."""
        return list_segments(self.segments_dir)

    def __enter__(self) -> "Segmenter":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # ------------------------------------------------------------- internals

    def _ffmpeg_cmd(self) -> list[str]:
        out_pattern = str(self.segments_dir / _SEG_FFMPEG_PATTERN)
        return [
            self.ffmpeg_bin,
            "-hide_banner",
            "-loglevel", "warning",
            "-nostdin",
            # Input: RTSP, force transport, modest probe so it starts quickly.
            "-rtsp_transport", self.rtsp_transport,
            "-fflags", "+genpts",
            "-use_wallclock_as_timestamps", "1",
            "-i", self.rtsp_url,
            # Keep BOTH video and audio. Video is stream-copied (no re-encode →
            # CPU/GPU near zero). Audio is transcoded to AAC because IP cameras
            # commonly emit PCM A-law/mu-law (G.711) or PCMU which the mp4/segment
            # muxer cannot stream-copy ("Error initializing output stream 0:1").
            # AAC is universally mp4-compatible and dirt cheap to encode.
            # 0:a? makes the audio map optional so cameras without audio still run.
            "-map", "0:v:0",
            "-c:v", "copy",
            # NO audio. Transcoding camera audio (often G.711 with broken timestamps) to
            # AAC backed up the encoder queue ("Non-monotonous DTS in stream 0:1") and
            # periodically HUNG the segment muxer — clips/thumbnails silently stopped.
            # Surveillance event clips are video-only; this makes the buffer rock-solid.
            "-an",
            # Rolling segments. reset_timestamps keeps each segment independently
            # playable; strftime drives the wall-clock filenames.
            "-f", "segment",
            "-segment_time", str(self.segment_seconds),
            "-segment_format", "mp4",
            # NB: no per-segment +faststart. It triggers a re-mux pass on every
            # rotation and only matters for progressive streaming of a single
            # file; the concatenated event clip gets +faststart in the clipper.
            "-reset_timestamps", "1",
            "-strftime", "1",
            "-y",
            out_pattern,
        ]

    def _spawn(self) -> subprocess.Popen[bytes]:
        cmd = self._ffmpeg_cmd()
        logger.debug("camera %s ffmpeg segment cmd: %s", self.camera_id, " ".join(cmd))
        # CRITICAL: ffmpeg's `-strftime` uses the *local* timezone. We pin the
        # process to UTC so segment filenames carry UTC wall-clock time, matching
        # the UTC timestamps the worker reads from the DB and our UTC parsing in
        # `parse_segment_name`. Without this, clip selection and pruning skew by
        # the box's UTC offset.
        env = dict(os.environ)
        env["TZ"] = "UTC"
        return subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=env,
            # Ensure ffmpeg is killed if the worker process dies for ANY reason
            # (including SIGKILL, where Python cleanup never runs) so the RTSP
            # session is always torn down with its owning camera worker.
            preexec_fn=_set_pdeathsig,
        )

    def _terminate_proc(self, timeout: float = 10.0) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        if proc.poll() is not None:
            return
        try:
            proc.terminate()
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                logger.warning("camera %s ffmpeg did not exit; killing", self.camera_id)
                proc.kill()
                proc.wait(timeout=timeout)
        except Exception:  # pragma: no cover - best-effort cleanup
            logger.exception("camera %s error terminating ffmpeg", self.camera_id)

    def _newest_segment_mtime(self) -> float:
        """Newest segment file mtime — advances while ffmpeg writes; frozen if it stalls.
        Robust to pruning (pruning removes OLD files, never the newest)."""
        try:
            return max((p.stat().st_mtime for p in self.segments_dir.glob("seg_*.mp4")),
                       default=0.0)
        except Exception:
            return 0.0

    def _run(self) -> None:
        """Watchdog loop: keep ffmpeg alive, prune old segments, until stopped."""
        backoff = self.restart_backoff
        while not self._stop_evt.is_set():
            try:
                proc = self._spawn()
            except FileNotFoundError:
                logger.error(
                    "camera %s: ffmpeg binary %r not found; segmenter disabled",
                    self.camera_id, self.ffmpeg_bin,
                )
                return
            except Exception:
                logger.exception("camera %s: failed to spawn ffmpeg", self.camera_id)
                if self._stop_evt.wait(backoff):
                    return
                backoff = min(backoff * 2, self.restart_backoff_max)
                continue

            with self._lock:
                self._proc = proc
            backoff = self.restart_backoff  # reset on a successful spawn

            # Supervise: poll for exit, prune periodically, AND detect a HUNG ffmpeg
            # — one that is alive (poll() is None) but produces no segments (RTSP
            # stalled mid-handshake / mangled timestamps). The old watchdog only
            # restarted on process *death*, so a hung ffmpeg could sit for hours
            # writing nothing → no clips, no thumbnails. We now also restart on stall.
            stall_timeout = max(20.0, self.segment_seconds * 6)
            last_progress = time.monotonic()
            last_mtime = self._newest_segment_mtime()
            while not self._stop_evt.is_set():
                if proc.poll() is not None:
                    break
                self._maybe_prune()
                mtime = self._newest_segment_mtime()
                if mtime > last_mtime:
                    last_mtime = mtime
                    last_progress = time.monotonic()
                elif time.monotonic() - last_progress > stall_timeout:
                    logger.warning(
                        "camera %s: ffmpeg segmenter produced no new segment in %.0fs "
                        "(stalled); killing it to restart", self.camera_id, stall_timeout)
                    self._terminate_proc()
                    break
                if self._stop_evt.wait(1.0):
                    break

            if self._stop_evt.is_set():
                self._terminate_proc()
                return

            # ffmpeg exited on its own — log stderr tail and restart with backoff.
            rc = proc.returncode
            stderr_tail = b""
            try:
                if proc.stderr is not None:
                    stderr_tail = proc.stderr.read() or b""
            except Exception:
                pass
            logger.warning(
                "camera %s: ffmpeg segmenter exited rc=%s; restarting in %.1fs. stderr: %s",
                self.camera_id, rc, backoff,
                stderr_tail.decode("utf-8", "replace")[-500:].strip(),
            )
            with self._lock:
                self._proc = None
            if self._stop_evt.wait(backoff):
                return
            backoff = min(backoff * 2, self.restart_backoff_max)

    def _maybe_prune(self) -> None:
        now = time.monotonic()
        if now - self._last_prune < self.prune_interval:
            return
        self._last_prune = now
        try:
            self._prune()
        except Exception:  # pragma: no cover
            logger.exception("camera %s: segment prune failed", self.camera_id)

    def _prune(self) -> None:
        """Delete segments older than the retention window.

        Selection is by filename start time. We keep the newest file unconditionally
        (it may be the one currently being written) and anything within the window.
        """
        segs = list_segments(self.segments_dir)
        if len(segs) <= 1:
            return
        cutoff = datetime.now(timezone.utc).timestamp() - self.retention_seconds
        # Never delete the most recent segment (it may be live / still growing).
        for info in segs[:-1]:
            if info.start.timestamp() < cutoff:
                try:
                    info.path.unlink()
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    logger.warning("camera %s: could not prune %s: %s", self.camera_id, info.name, exc)
