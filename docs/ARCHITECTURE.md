# Architecture notes

## The core idea: GPU as a scheduled resource
A single NVIDIA T4 (16 GB) hosts many heavy models that don't fit together. The **gpu-broker** keeps exactly
**one** heavy model resident and swaps gracefully between jobs:

- `whisper-xtts` (STT + voice cloning) — the default "duty" model (also used by voice-stream / dub)
- `vlm` (Qwen2.5-VL) — floor-plan understanding
- `render` (SD + ControlNet) — image generation

Swaps use `docker stop`/`start` + `ollama stop` — **graceful, never `kill -9`**, and **never touch other VMs**
(production stays safe). One FIFO queue, rolling-average ETA, per-job position. When idle, the duty model is restored.

## The platform layer
The **api-gateway** wraps every service in one multi-tenant REST surface:
- **auth**: one API key (`X-API-Key`) for all services
- **metering**: requests + weighted *units* + LLM tokens, per key, per service
- **limits**: monthly unit **quota** (→ 402) and **rate limit** (→ 429), applied on the very next request
- **billing**: append-only `events.jsonl` audit log + price-per-unit + self/admin billing
- **queue visibility**: every job response includes position / ahead / ETA / total-in-queue

## Floor-plan → 3D pipeline (hybrid)
1. **Understand** (CPU + one VLM pass): OCR + Qwen read *all* labels/dimensions; a multi-source **scale solver**
   (dimension chains with outlier rejection + printed areas) yields a reliable metres-per-pixel + a quality flag.
2. **Geometry**: CubiCasa neural parse (walls/rooms/doors/windows) + a **medial-axis** vectorizer for walls of
   any angle/curve/thickness (load-bearing vs partition), + colour-based **apartment** segmentation on multi-unit floors.
3. **Render** (3D): a Three.js scene assembled from the above — saved, linkable, with a render library.
Renders are **off by default** (toggle) to save GPU; only the 3D plan is built.
