# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project
adheres to [Semantic Versioning](https://semver.org/).

## [1.5.2] — 2026-06-12

### Security
Third VMS hardening batch — **fail-closed authentication** (closes the HIGH
findings). The app previously failed *open*: `auth_required` defaulted to false
and the gateway didn't inject the identity header, so any process that could
reach the loopback port (SSH tunnel, SSRF, co-located container) had full
unauthenticated read/write/delete, and the SSO identity was spoofable.
- **`auth_required` now defaults to `true`** (fail-closed). Behind the SSO
  gateway the header is always present, so only direct/loopback callers without a
  key are rejected (`app/config.py`).
- **Gateway now injects the identity from the auth subrequest and overrides any
  client-supplied header** (`auth_request_set $sso_email $upstream_http_x_auth_email;
  proxy_set_header X-Email $sso_email;`) — the trusted email can no longer be
  spoofed by a client. (nginx vhost; documented in `.env.example`.)
- **Strong API key** for CLI/tunnel access, compared in constant time (added in
  1.5.0); `.env.example` now documents generating one with `openssl rand -hex 32`.

Verified live: loopback without credentials → 401, with the injected SSO header
→ 200, with a valid bearer key → 200, bad key → 401, browser without a cookie →
302 to login (no lockout), production VM unaffected.

## [1.5.1] — 2026-06-12

### Security
Second VMS hardening batch — credential exposure & SSRF (server-side only, no
change to the auth/SSO flow):
- **RTSP credentials are no longer returned by the API.** Every camera response
  now masks the URL userinfo (`rtsp://***@host:port/path`); the raw URL with
  credentials stays server-side (workers read it straight from the DB, so capture
  is unaffected). Previously `GET /api/cameras` returned `admin:password` in clear
  text to anyone who could reach the API (`app/api/cameras.py`).
- **Camera edit can't leak or clobber the secret.** A blank or masked `rtsp_url`
  on update is treated as "keep current", so the UI safely echoes the redacted
  value without overwriting the stored credentials.
- **SSRF / local-file-read hardening on ffmpeg.** Both the segmenter and the HLS
  spawner now pass `-protocol_whitelist rtsp,rtsps,rtp,rtcp,udp,tcp,tls,crypto`,
  and `rtsp_url` is validated to the `rtsp://`/`rtsps://` schemes — so an operator
  can't point the recorder at `file:///…`, `http://169.254.169.254`, etc.
- String length caps added to the camera `name`/`rtsp_url` fields.

## [1.5.0] — 2026-06-12

### Security
First batch of a defensive security-hardening pass over the VMS service (low-risk
hygiene; no change to the auth/SSO flow):
- **Timing-safe API-key comparison** — the optional bearer key is now checked with
  `hmac.compare_digest` instead of `==`, so the key can't be recovered via response
  timing (`app/auth.py`).
- **Decompression-bomb guard on uploads** — enrollment images are rejected if they
  decode to more than 50 MP, and the byte cap is lowered 32 MiB → 8 MiB, so a small
  crafted file can no longer OOM the worker (`app/api/people.py`).
- **Path-traversal containment** — the stored-image resolver now verifies the
  resolved path stays inside the data dir (`os.path.commonpath`) and no longer
  trusts absolute stored paths, matching the guard already used elsewhere.
- **Bounded request bodies (DoS)** — bulk delete/merge/split/enroll/label id-lists
  are capped (`max_length` 1000–5000) and free-text fields (name/notes/label) get
  `max_length`, so a single request can't blow up memory or build a giant `IN()`.
- **Face-group clustering cap** lowered (`max_samples` 20000 → 2000) to bound the
  O(n²) similarity matrix.
- **Build hardening** — `setuptools`/`wheel` upgraded in the image (pins out
  CVE-2022-40897 / CVE-2024-6345, build-time only).

## [1.4.2] — 2026-06-12

### Performance
- **Drop-to-latest decode thread (VMS).** Each camera worker now reads its RTSP
  stream on a dedicated producer thread that publishes only the *newest* frame;
  the detection loop consumes that latest frame instead of calling `cap.read()`
  inline. Under load — or when an analysis pass is slow — stale frames are dropped
  rather than queued, so the pipeline stays at real time instead of building
  latency (the correct behaviour for surveillance). A stuck or reconnecting stream
  no longer blocks the loop: the consumer waits with a timeout and stays
  responsive while the producer handles reconnects. Verified on the live box —
  `last_seen` advances in real time on a healthy camera while a flaky second
  stream reconnects independently; teardown stays clean (bounded thread join).

## [1.4.1] — 2026-06-12

### Performance
- **Thumbnail & face-sample writes moved off the detection loop (VMS).** Sighting
  body-crop thumbnails and captured face samples are now produced on a dedicated
  per-camera persist drain thread (mirroring the existing clip-assembly thread)
  instead of inline in the hot path. The detect loop only copies the crop and
  enqueues; the thread does the JPEG-encode + DB write. Jobs are **batched into a
  single transaction** per drain cycle, collapsing the previous 2–3 separate
  fsync'd commits per track into one and removing the per-frame disk-write stall —
  the main cause of the "hang under many objects". A bounded queue falls back to an
  inline write under saturation (no lost thumbnails), and pending jobs are flushed
  on shutdown. Behaviour and on-disk layout are unchanged; only the timing moves.

## [1.4.0] — 2026-06-12

### Added
- **Adaptive detection cadence (VMS).** Each camera now detects at full rate only
  while objects are present, then throttles to `detect_interval_idle` once the
  scene has been empty for `active_grace_seconds`. The live-preview frame slot
  follows the same active/idle state (`active_preview_fps` / `idle_preview_fps`)
  instead of re-encoding a JPEG for every decoded frame — so a quiet camera stops
  burning GPU/CPU/disk on an empty scene, and re-arms instantly when something enters.
- **Per-frame re-identification cap (VMS).** `max_reid_per_frame` bounds how many
  tracks are embedded per detection frame so a crowd can no longer stall the loop;
  fresh (unassigned) tracks get priority and the fast cadence, already-identified
  tracks refresh on the slower `reid_confident_sample_seconds` (identity is sticky).
  This is the direct fix for the "lags when many objects appear" symptom.

### Performance
- Measured on the live box: the busy all-classes camera worker dropped from ~73% of
  a core to ~27% with no loss of detection responsiveness; host load average fell
  accordingly. All changes are RAM-neutral (no extra model copies) and reduce disk
  writes — strictly safer for the co-hosted production VM.

## [1.3.5] — 2026-06-12

### Fixed
- **Static assets are now cache-busted** (`?v=` query on every CSS/JS link). Browsers were serving
  stale cached stylesheets, so the responsive redesign from 1.3.4 did not appear on already-visited
  clients without a hard refresh. New loads now always pick up the latest UI.

### Changed
- **Larger desktop preview thumbnails** — events `320→380`, identities `240→280`, People `190→220`,
  clip grid `200→240` px minimum tile width, with roomier gaps, for easier at-a-glance reviewing.

## [1.3.4] — 2026-06-12

### Changed
- **Mobile-first responsive VMS UI.** The interface adapts to phones: the tab bar becomes a
  swipeable strip (all tabs reachable), card grids and KPIs reflow for small screens, modals go
  full-width with stacked detail rows, controls are touch-sized, and wide tables scroll. On desktop
  the **preview thumbnails are larger** (events / identities / People / clip grids), for easier
  reviewing.

## [1.3.3] — 2026-06-12

### Changed
- **Camera-only face-quality floor.** Since identities are built purely from camera detections
  (no photo enrollment), only DECENT faces (det_score × frontalness ≥ threshold) now register a
  person and become exemplars — blurry / tiny / extreme-profile faces are dropped instead of
  polluting the gallery with garbage embeddings. New `reid_face_exemplar_min_quality` (default 0.35).

## [1.3.2] — 2026-06-11

### Changed
- **Multi-view (pose-diverse) face gallery** for angle-invariant recognition. The 5 facial landmarks
  now yield a signed yaw; face exemplars are bucketed frontal / left / right and kept pose-diverse
  (cap 8→12, pose-aware eviction). The old acceptance band rejected low-cosine faces — i.e. profiles —
  so the gallery stayed frontal-only and a turned head never matched. Now a profile view of a tracked
  person is KEPT as an exemplar, so the same person is matched from any angle (a profile query hits the
  stored profile exemplar). New `face_exemplars.pose` column (idempotent migration).

## [1.3.1] — 2026-06-11

### Changed
- **Re-ID is now face-anchored** (the durable fix for duplicate people). A faceless person crop —
  a back/side view — no longer spawns a NEW identity; it only attaches to an existing person by
  appearance *within a session*, else it is dropped. One person seen from behind no longer explodes
  into dozens of duplicates. Identity across clothing changes / angles / days rests on the **face**
  (the only stable cross-day signal); clothing-appearance is a within-session helper only. New
  setting `reid_require_face_for_new_person` (default on).

## [1.3.0] — 2026-06-11

### Added
- **Analytical People dashboard** — the auto cross-camera unique-people layer (person `Identity`)
  is now a face-first analytics view: KPI cards (unique people / new today / seen today & 7d /
  sightings), sightings-by-hour + per-camera charts, a face-first gallery (search · filter ·
  sort), and a detail modal with a face gallery, cross-camera **sightings timeline**, dwell and
  inline rename. Manual photo-enrollment folded in. New `GET /api/identities/analytics`
  aggregates + `face_thumb_url` on identity list items.
- Re-ID ops tooling: `scripts/consolidate_identities.py` (collapse over-split duplicate identities
  of one person), `scripts/calibrate_thresholds.py`, `scripts/enroll_person.py`.

### Fixed
- **Re-ID over-splitting** — a gallery polluted with duplicates made every new sighting
  "ambiguous" → a new identity (one person split into ~90). Consolidation + lower appearance
  match thresholds/margin restore merging. Faceless back/low-light crops stay inherently hard —
  good face enrollment is the durable fix.

## [1.2.1] — 2026-06-11

### Changed
- **VMS Re-ID upgraded to OSNet-AIN x1.0** (real MSMT17-trained weights from the OSNet author's
  mirror) — domain-generalizable, much better cross-camera/cross-angle appearance matching for
  people AND objects (class-scoped: cars match cars, etc.). Exported to ONNX FP16, runs on the GPU.
- Added `scripts/calibrate_thresholds.py` — derive cosine thresholds from the live gallery
  (same vs cross-identity distributions) instead of academic defaults.
- Documented model-weight sourcing in the VMS README.

## [1.2.0] — 2026-06-11

### Changed
- **VMS inference moved to the GPU.** Detection (YOLOv8n), face recognition (SCRFD + ArcFace) and
  ReID now run on the T4 via the **CUDAExecutionProvider** instead of the CPU. Root cause was a
  packaging conflict — `insightface` pulls the CPU `onnxruntime`, which installs into the same module
  and **shadows `onnxruntime-gpu`**; the Dockerfile now drops the CPU build and reinstalls the GPU one.
  Result: per-camera CPU fell from ~7 cores to ~0.5 core (~14 cores freed at 2 cameras), host load
  average ~20 → ~4 — the box can now run many more cameras (GPU VRAM / RAM become the new ceiling).

## [1.1.1] — 2026-06-11

### Fixed
- **VMS clips & thumbnails** now record reliably. The per-camera ffmpeg segment buffer could **hang**
  (alive but writing nothing) — so events were logged but had no clip/thumbnail. Two fixes: the
  segmenter watchdog now restarts a **stalled** ffmpeg (not only a dead one), and the warm buffer is
  **video-only** (`-an`) — transcoding camera audio backed up the muxer and was the stall's root cause.

## [1.1.0] — 2026-06-11

### Added
- **Video surveillance (VMS)** published as [`services/vms`](services/vms) — RTSP camera management,
  person-triggered clip recording (pre/post-roll), a live MJPEG monitoring grid, events history with
  playback, and face recognition (SCRFD + ArcFace + FAISS). Ships **code only** — no camera data, face
  embeddings, recordings or database (those stay on the operator's box).
- README now documents the **full platform**: VMS, creative media tools (IOPaint object removal,
  Wan/ComfyUI video generation), and the SD + ControlNet render service.

### Changed
- NOTICE expanded with the VMS third-party stack — including the **Ultralytics YOLOv8 AGPL-3.0** caveat,
  InsightFace, FAISS, ONNX Runtime, ffmpeg, IOPaint and Wan/ComfyUI.

## [1.0.0] — 2026-06-11

First stable public release. The platform grew from a set of GPU services into a coherent,
multi-tenant product with a translation API, billing, and a full admin control panel.

### Added
- **Translation API** behind the gateway:
  - `POST /v1/translate` — single text, optional source **auto-detect** (`detect`) and `skip_same`.
  - `POST /v1/translate/batch` — up to 200 strings in one request (data migration).
  - `POST /v1/translate/multi` — one or many texts → many target languages at once.
  - `POST /v1/detect` — language detection.
  - **Model selection** per request (`model`): `eurollm:9b`, `translategemma:12b`, `qwen2.5vl:7b`, `llama3.2:3b`.
- **Translation-tuned models**: EuroLLM-9B-Instruct and Google TranslateGemma-12B (Q6_K), with a
  side-by-side speed/quality benchmark across Russian, Polish, Ukrainian and Norwegian.
- **Concurrent serving** — 8-way parallel translation (~3 translations/sec on one T4) and a
  connection-reusing reference client.
- **Admin control panel** (`/admin/ui`) — rebuilt as a tabbed, mobile-first SPA:
  - **Обзор** — live GPU load + sparkline, "running now", queue, disk/RAM.
  - **Запуск** — launchpad to open every tool, quick-connect API card, and **model management**
    (Ollama pull + allow-list toggle, custom service links).
  - **Сервисы** — start / stop / restart every docker + systemd service from the UI.
  - **Ключи** — issue keys, set monthly quota & rate limit, see per-key billing.
  - **Активность** — recent operations feed (tail-read, paginated).
- **GPU orchestration** — translation and voice co-reside; heavy 3D/render jobs swap the translation
  model out on demand and the broker restores it on idle. Heavy models can also run with the whole GPU
  via the broker's `/api/llm` route.

### Changed
- Default translation model is **EuroLLM-9B** (fast, doesn't disturb voice); TranslateGemma stays opt-in.
- Admin monitoring works **with or without** the broker (reads `nvidia-smi` + Ollama directly).
- The model allow-list is now dynamic — models added through the panel become selectable immediately.

### Fixed
- Heavy-model broker route no longer piles up under concurrent bursts — it **fails fast (503 + retry)**
  instead of exhausting the thread pool.
- Admin activity feed prunes stale in-flight entries (no more phantom "hung" sessions).
- Resolved a GPU out-of-memory condition where a pinned translation model blocked every other service.

## [0.1.0] — 2026-06-08

Initial public release: GPU job-broker with graceful model-swap, the floor-plan → 3D pipeline,
live voice translation, dubbing, talking avatars, and the first multi-tenant API gateway
(keys, metering, quotas, rate limits, billing).

[1.0.0]: https://github.com/yakden/ai-media-stack/releases/tag/v1.0.0
[0.1.0]: https://github.com/yakden/ai-media-stack/releases/tag/v0.1.0
