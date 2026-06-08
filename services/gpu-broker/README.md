# gpu-broker
GPU job queue + **graceful single-resident-model swap** on one T4, and the floor-plan→3D pipeline orchestrator.
Keeps exactly one heavy model loaded (whisper-xtts ⇄ VLM ⇄ SD render), swaps via `docker stop/start` + `ollama stop`
(never `kill -9`, never touches other VMs), restores the duty model when idle. Exposes a live dashboard with
queue position/ETA, the 3D scene viewer (Three.js) and the project library. Single file: `app.py` (FastAPI, :8092).
