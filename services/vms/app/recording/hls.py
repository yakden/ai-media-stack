"""On-demand RTSP -> HLS sessions for live-view WITH sound.

The live grid streams MJPEG, which carries NO audio. When the operator expands
a camera we start a short-lived ffmpeg that pulls the camera's RTSP, stream-
copies the H.264 video (zero transcode) and transcodes only the audio to AAC,
writing an HLS playlist + .ts segments that hls.js plays in the browser with
sound. Sessions are reference/idle-managed and hard-capped so they never pile
ffmpeg processes or RTSP connections on the shared T4.

Lifecycle / orphan safety (mirrors the segmenter's hard-won lessons):
  * ffmpeg gets PR_SET_PDEATHSIG=SIGKILL so it dies if this process dies.
  * an idle reaper stops sessions not accessed within ``hls_idle_timeout`` and
    removes their segment dir.
  * the API stops all sessions on shutdown; ``cleanup.purge_orphans`` sweeps
    leftover hls/<id> dirs at startup.
This manager runs in the API process (it serves the playlist/segments); the
ffmpeg it spawns opens its own RTSP session regardless of the camera worker.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

from .segmenter import _set_pdeathsig  # reuse the PR_SET_PDEATHSIG helper

logger = logging.getLogger("vms.recording.hls")

SEG_RE = re.compile(r"^seg\d{5}\.ts$")


class _Session:
    __slots__ = ("proc", "dir", "last_access")

    def __init__(self, proc, directory: Path) -> None:
        self.proc = proc
        self.dir = directory
        self.last_access = time.time()


class HlsManager:
    """Manage on-demand RTSP->HLS ffmpeg sessions (one per expanded camera)."""

    def __init__(self, settings) -> None:
        self.root = Path(settings.hls_dir)
        self.segment_seconds = int(getattr(settings, "hls_segment_seconds", 2))
        self.list_size = int(getattr(settings, "hls_list_size", 6))
        self.idle_timeout = float(getattr(settings, "hls_idle_timeout", 60))
        self.max_sessions = int(getattr(settings, "hls_max_sessions", 2))
        self.rtsp_transport = str(getattr(settings, "rtsp_transport", "tcp"))
        self.ffmpeg_bin = "ffmpeg"
        self._sessions: dict[int, _Session] = {}
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self.root.mkdir(parents=True, exist_ok=True)
        self._reaper = threading.Thread(target=self._reap_loop, name="hls-reaper", daemon=True)
        self._reaper.start()

    # ------------------------------------------------------------------ API
    def start(self, camera_id: int, rtsp_url: str) -> bool:
        """Start (or touch) an HLS session. Returns False if capacity-capped."""
        cam = int(camera_id)
        with self._lock:
            s = self._sessions.get(cam)
            if s is not None and s.proc.poll() is None:
                s.last_access = time.time()
                return True
            # Drop any dead entry.
            if s is not None:
                self._sessions.pop(cam, None)
            self._reap_idle_locked()
            live = sum(1 for x in self._sessions.values() if x.proc.poll() is None)
            if live >= self.max_sessions:
                logger.warning("HLS session cap (%d) reached; refusing camera %s", self.max_sessions, cam)
                return False
            d = self.root / str(cam)
            self._rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
            try:
                proc = self._spawn(rtsp_url, d)
            except Exception:
                logger.exception("Failed to start HLS ffmpeg for camera %s", cam)
                return False
            self._sessions[cam] = _Session(proc, d)
            logger.info("HLS session started for camera %s", cam)
            return True

    def touch(self, camera_id: int) -> None:
        with self._lock:
            s = self._sessions.get(int(camera_id))
            if s is not None:
                s.last_access = time.time()

    def stop(self, camera_id: int) -> None:
        with self._lock:
            self._stop_locked(int(camera_id))

    def playlist_path(self, camera_id: int) -> Path:
        return self.root / str(int(camera_id)) / "index.m3u8"

    def segment_path(self, camera_id: int, name: str) -> Optional[Path]:
        if not SEG_RE.match(name):
            return None
        p = self.root / str(int(camera_id)) / name
        with self._lock:
            s = self._sessions.get(int(camera_id))
            if s is not None:
                s.last_access = time.time()
        return p

    def shutdown(self) -> None:
        self._stop_evt.set()
        with self._lock:
            for cam in list(self._sessions.keys()):
                self._stop_locked(cam)

    # ------------------------------------------------------------- internals
    def _spawn(self, rtsp_url: str, out_dir: Path) -> "subprocess.Popen[bytes]":
        out = str(out_dir / "index.m3u8")
        seg = str(out_dir / "seg%05d.ts")
        cmd = [
            self.ffmpeg_bin, "-hide_banner", "-loglevel", "warning", "-nostdin",
            "-rtsp_transport", self.rtsp_transport,
            "-fflags", "+genpts", "-use_wallclock_as_timestamps", "1",
            "-i", rtsp_url,
            "-map", "0:v:0", "-map", "0:a?",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "1",
            "-f", "hls",
            "-hls_time", str(self.segment_seconds),
            "-hls_list_size", str(self.list_size),
            "-hls_flags", "delete_segments+append_list+omit_endlist+independent_segments",
            "-hls_segment_type", "mpegts",
            "-hls_allow_cache", "0",
            "-hls_segment_filename", seg,
            "-y", out,
        ]
        # rtsp_url may embed credentials -> only log the command at DEBUG.
        logger.debug("HLS ffmpeg cmd: %s", " ".join(cmd))
        env = dict(os.environ)
        env["TZ"] = "UTC"
        return subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, env=env, preexec_fn=_set_pdeathsig,
        )

    def _stop_locked(self, cam: int) -> None:
        s = self._sessions.pop(cam, None)
        if s is None:
            return
        proc = s.proc
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=3)
        except Exception:  # pragma: no cover
            logger.exception("Error stopping HLS ffmpeg for camera %s", cam)
        self._rmtree(s.dir)
        logger.info("HLS session stopped for camera %s", cam)

    def _reap_idle_locked(self) -> None:
        now = time.time()
        for cam in list(self._sessions.keys()):
            s = self._sessions[cam]
            if s.proc.poll() is not None or (now - s.last_access) > self.idle_timeout:
                self._stop_locked(cam)

    def _reap_loop(self) -> None:
        while not self._stop_evt.wait(10.0):
            with self._lock:
                self._reap_idle_locked()

    @staticmethod
    def _rmtree(path: Path) -> None:
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            pass
        except OSError:  # pragma: no cover
            logger.debug("could not remove %s", path, exc_info=True)
