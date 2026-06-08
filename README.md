<div align="center">

# 🧠 ai-media-stack

**A self-hosted, multi-tenant AI media platform — runs a whole fleet of generative services on a single GPU.**

Floor-plan → 3D · live voice-to-voice translation in your own voice · video dubbing with lip-sync ·
talking avatars · on-box LLMs — all behind one **GPU job-broker** with automatic model-swapping,
and one **API gateway** with keys, per-key metering, quotas and billing.

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-3776AB)
![FastAPI](https://img.shields.io/badge/FastAPI-009688)
![Docker](https://img.shields.io/badge/Docker-2496ED)
![GPU](https://img.shields.io/badge/GPU-single%20T4%2016GB-76B900)

</div>

---

## Why this exists

Running many heavy AI models usually means many GPUs. This stack squeezes **a full product suite onto one
16 GB GPU** by treating the GPU as a scheduled resource: a **broker** keeps exactly one heavy model resident,
swaps models gracefully between jobs (never `kill -9`, never touches other tenants), and exposes live
queue/ETA. On top sits a **REST API gateway** that turns every service into a metered, multi-tenant product —
API keys, monthly quotas, rate limits, usage accounting and billing — with a clean admin dashboard.

It’s a complete, opinionated reference for **how to ship multiple GPU AI services as one coherent platform.**

## ✨ Features

| Domain | What it does |
|---|---|
| 🏗️ **Floor-plan → 3D** | Upload a plan (or photos) → typed rooms, walls (any angle/curve via medial-axis), doors/windows, apartments on multi-unit floors, true scale from printed dimensions, a saved/linkable Three.js 3D scene + render library. |
| 🗣️ **Live voice translation** | Stream from the mic → STT → translate → speak back **in your own cloned voice** in another language, near-real-time. Voice library (save/select) + 58 built-in voices with flags & gender. |
| 🎥 **Video dubbing** | Short clip → translate → your voice → **lip-sync** (MuseTalk). |
| 🖼️ **Talking avatars** | Photo + text → talking-head video. |
| 🤖 **On-box LLMs** | Qwen / Llama via the gateway (`/v1/llm/chat`, `/v1/translate`), token-metered. |
| 🧮 **GPU broker** | One resident model on the T4, graceful auto-swap, single FIFO queue, live position/ETA. |
| 🔐 **API gateway** | API-key auth, per-key usage + tokens, **monthly quotas (402)**, **rate limits (429)**, billing, audit log, admin dashboard. |

## 🏛️ Architecture

```
                         ┌────────────────────────── clients (web / mobile / server / MCP) ──────────────────────────┐
                         │                                   X-API-Key                                                │
                         ▼                                                                                            
                 ┌───────────────┐   keys · quotas · rate-limit · usage · billing · queue info                       
                 │  API GATEWAY  │  /v1/3d/* /v1/voice/* /v1/avatar /v1/dub /v1/llm/* /v1/translate  + /admin/*       
                 └──────┬────────┘                                                                                    
        ┌───────────────┼───────────────────────────────┬───────────────────────────────┐                          
        ▼               ▼                                ▼                                ▼                          
 ┌─────────────┐ ┌───────────────┐               ┌───────────────┐                ┌───────────────┐                 
 │  GPU BROKER │ │ voice-stream  │               │ dub / animate │                │  ollama (LLM) │                 
 │ queue+swap  │ │ STT·MT·XTTS   │               │ MuseTalk      │                │ qwen / llama  │                 
 └──────┬──────┘ └───────────────┘               └───────────────┘                └───────────────┘                 
        ▼  (CPU parsers)                                                                                             
 ┌─────────────┐ ┌───────────────┐   single NVIDIA T4 (16 GB): exactly ONE heavy model resident at a time,           
 │ floorplan3d │ │ cubicasa-svc  │   broker swaps {whisper-xtts ⇄ vlm ⇄ render} gracefully between jobs.             
 └─────────────┘ └───────────────┘                                                                                  
```

## 📦 Services

| Service | Port | Role |
|---|---|---|
| [`api-gateway`](services/api-gateway) | 8190 | Multi-tenant REST: auth, metering, quotas, billing, admin UI |
| [`gpu-broker`](services/gpu-broker) | 8092 | GPU job queue + graceful model-swap + the 3D pipeline orchestrator |
| [`voice-stream`](services/voice-stream) | 8202 | Live mic → translated speech in a cloned/preset voice |
| [`dub-web`](services/dub-web) | 8200 | Webcam/clip → dubbed, lip-synced video |
| [`animate-web`](services/animate-web) | 8201 | Photo + text → talking avatar |
| [`interior3d-web`](services/interior3d-web) | 8203 | Floor-plan → 3D test console |
| [`control-plane`](services/control-plane) | 8090 | Ops dashboard (services, GPU, system) |
| [`floorplan3d`](services/floorplan3d) | 8204 | CPU wrapper: OCR + Mask-R-CNN + medial-axis wall vectorization |
| [`cubicasa-service`](services/cubicasa-service) | 8205 | CPU wrapper: neural plan parsing (walls/rooms/doors/windows) + colour-based apartments |

## 🚀 Quickstart (one service)

```bash
git clone https://github.com/yakden/ai-media-stack
cd ai-media-stack/services/api-gateway
cp .env.example .env          # fill in your secrets
pip install fastapi uvicorn httpx
uvicorn app:app --host 127.0.0.1 --port 8190
```

Each service is a single-file FastAPI app (`app.py` / `serve*.py`) — read it top-to-bottom, the docstrings
explain the design. The two ML wrappers (`floorplan3d`, `cubicasa-service`) have Dockerfiles and their own
READMEs (they reference upstream model repos — weights are **not** bundled, see [NOTICE](NOTICE.md)).

## 🔌 API example (one key, many services)

```bash
# admin issues a key
curl -X POST https://host/gw/admin/keys -H "X-Admin-Key: $ADMIN" -F "owner=Acme" -F "quota_units=5000"

# client uses ONE key for everything, gets queue position + ETA back
curl -X POST https://host/gw/v1/3d/project -H "X-API-Key: $KEY" -F "files=@plan.png"
#  → {"job_id":"…","queue":{"position":1,"ahead":0,"eta_seconds":42,"total_in_queue":1}}

curl -X POST https://host/gw/v1/translate  -H "X-API-Key: $KEY" -d '{"text":"Привет","to":"German"}'
curl     https://host/gw/v1/billing        -H "X-API-Key: $KEY"   # your usage + cost
```

## 🧩 Tech

FastAPI · Docker · NVIDIA T4 · PyTorch · TensorFlow · OpenCV · scikit-image · Three.js ·
Whisper · Coqui XTTS · MuseTalk · Mask R-CNN · CubiCasa5k · Ollama (Qwen / Llama).

## 📜 License

This project’s **own code** is [MIT](LICENSE). Third-party models and datasets it integrates keep their own
licenses (some non-commercial) and are **referenced, not redistributed** — see [NOTICE](NOTICE.md).

> Built and maintained by [@yakden](https://github.com/yakden).
