# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project
adheres to [Semantic Versioning](https://semver.org/).

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
