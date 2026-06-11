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

## Translation & on-box LLMs
The gateway exposes machine translation (`/v1/translate` · `/batch` · `/multi` · `/detect`) over on-box LLMs
(Ollama). Light models — **EuroLLM-9B** (the default, translation-tuned for 35 European languages) — run
**directly** and **concurrently** (`OLLAMA_NUM_PARALLEL`); a heavy model — **TranslateGemma-12B** — is
**routed through the broker** (`/api/llm`) so it gets the whole GPU (fully on-GPU, no CPU offload). Under
contention the broker route **fails fast with 503 + client backoff**, so concurrent bursts queue cleanly
instead of exhausting the thread pool. Model choice, source auto-detect and skip-if-already-in-target are
per-request; usage is metered like every other service.

## Cross-camera person & object identification (VMS)
Each RTSP camera runs an independent worker: **YOLOv8n** (person + COCO objects, on the GPU via
`onnxruntime-gpu`) → a greedy-IoU tracker for short-term single-camera continuity → periodic crop embedding →
an **IdentityManager** that assigns each track to a cross-camera **Identity** (or mints a new one).

Identity is **anchored on the face** — the only cue stable across clothing changes, viewpoint, lighting and
days:

- **Face** = SCRFD-10G detect + 5 landmarks → affine align → ArcFace 512-d. It is the primary, cross-day
  linker. A person is registered **only when a *decent* face is seen** (a `det_score × frontalness` quality
  gate); a faceless back/side view **never spawns a new identity** — it can only attach to an existing person
  by appearance *within a session*. This is what stops one person seen from behind from exploding into dozens
  of duplicate identities.
- **Multi-view face gallery** — exemplars are bucketed by a **signed yaw** (derived from the landmarks) into
  *frontal / left / right* and kept pose-diverse (pose-aware eviction). A profile query then matches the stored
  profile exemplar: angle-invariance **at the gallery level**, not just for whatever pose was captured first.
- **Appearance** (OSNet-AIN x1.0, MSMT17) is a **within-session** helper only — time-windowed + exponentially
  decayed — because clothing is not an identity across days.
- Matching is **class-scoped** (`Identity.object_class`: a car only matches cars, a dog only dogs), with a
  *best-minus-second-best margin* to reject ambiguous matches and quality/rate gates to resist explosion. A
  maintenance pass decays, prunes and merges.

Events record person-triggered clips (pre/post-roll, warm rolling ffmpeg segment buffer with a stall-watchdog);
the **People** tab is a face-first analytics dashboard over these auto, cross-camera unique people (KPIs,
per-camera and hourly charts, a per-person cross-camera sightings timeline).

**GPU sharing** — all VMS inference runs on the *same* T4 as translation and voice: `onnxruntime-gpu` for
detection / face / ReID (≈1 CPU core per camera vs ~7 on CPU), coexisting inside the 16 GB budget.

**Honest limit** — a camera that only ever sees the back of a head carries no biometric any method can use;
faces must be captured. The system is designed to be **correct under that constraint** (it does not fabricate
identities from non-identifiable views) rather than to pretend otherwise.
