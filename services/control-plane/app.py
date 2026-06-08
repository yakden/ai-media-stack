"""Control plane for the on-box AI services ("single pane of glass").

- Web dashboard at /  (served static; talks to the API with a Bearer key)
- JSON API under /api/*  (Bearer-key protected)
- Manages docker-compose services: status, start/stop/restart
- Reports GPU (nvidia-smi) and host system stats

Runs as a systemd service on 127.0.0.1 — reachable only via SSH tunnel /
behind the host firewall. The API key gates all /api/* calls.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

API_KEY = os.environ.get("CONTROL_PLANE_KEY", "")
HERE = os.path.dirname(os.path.abspath(__file__))

# --- Service registry -------------------------------------------------------
# Each managed service: a docker-compose project we can inspect and control.
SERVICES = [
    {
        "name": "tts",
        "title": "TTS / STT — Whisper + Piper/XTTS",
        "container": "whisper-xtts-server",
        "compose": "/home/deploy/whisper-xtts-server/docker-compose.yml",
        "health": "http://127.0.0.1:8000/health",
        "port": 8000,
        "docs": "/docs",
    },
    {
        "name": "avatar",
        "title": "Talking Avatar — MuseTalk",
        "container": "avatar-muse",
        "compose": "/home/deploy/avatar-muse/docker-compose.avatar.yml",
        "health": "http://127.0.0.1:8100/health",
        "port": 8100,
        "docs": "/health",
    },
    {
        "name": "floorplan3d",
        "title": "Интерьер 3D — FloorPlanTo3D (Mask R-CNN, CPU)",
        "container": "floorplan3d",
        "compose": "/home/deploy/FloorPlanTo3D-API/docker-compose.yml",
        "health": "http://127.0.0.1:8204/health",
        "port": 8204,
        "docs": None,
        "url": "https://ai.1c-rus.ru/interior3d/",
    },
    {
        "name": "webui",
        "title": "AI Web UI — Open WebUI (chat, model select, MCP)",
        "container": "open-webui",
        "compose": "/home/deploy/webui/docker-compose.yml",
        "health": "http://127.0.0.1:8088/health",
        "port": 8088,
        "docs": "/",
        "url": "https://webui.1c-rus.ru",
    },
    {
        "name": "ollama",
        "title": "Ollama — model backend (qwen2.5vl, llama3.2)",
        "container": "ollama",
        "compose": "/home/deploy/webui/docker-compose.yml",
        "health": "http://127.0.0.1:11434/api/version",
        "port": 11434,
        "docs": None,
    },
    {
        "name": "mcpo",
        "title": "MCP gateway — mcpo (MCP→OpenAPI tools)",
        "container": "mcpo",
        "compose": "/home/deploy/webui/docker-compose.yml",
        "health": "http://127.0.0.1:8089/time/openapi.json",
        "port": 8089,
        "docs": "/docs",
    },
    {
        "name": "iopaint",
        "title": "IOPaint — object removal (LaMa + SAM2)",
        "container": "iopaint",
        "compose": "/home/deploy/iopaint/docker-compose.yml",
        "health": "http://127.0.0.1:8080/",
        "port": 8080,
        "docs": "/",
        "url": "https://paint.1c-rus.ru",
    },
    {
        "name": "portainer",
        "title": "Portainer — Docker admin (logs, console, stats)",
        "container": "portainer",
        "compose": "/home/deploy/portainer/docker-compose.yml",
        "health": "http://127.0.0.1:9000/",
        "port": 9000,
        "docs": "/",
        "url": "https://portainer.1c-rus.ru",
    },
    {
        "name": "vms",
        "title": "VMS — cameras, recording, face recognition",
        "container": "vms",
        "compose": "/home/deploy/vms-platform/vms/docker-compose.yml",
        "health": "http://127.0.0.1:8120/health",
        "port": 8120,
        "docs": "/",
        "url": "https://vms.1c-rus.ru",
    },
]
SVC_BY_NAME = {s["name"]: s for s in SERVICES}

app = FastAPI(title="AI Control Plane", version="0.1.0")


def require_key(request: Request):
    if not API_KEY:
        return  # key not configured -> open (dev only)
    hdr = request.headers.get("authorization", "")
    token = hdr[7:] if hdr.lower().startswith("bearer ") else None
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


def _run(cmd: list[str], timeout: int = 120) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr).strip()
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def _container_state(container: str) -> str:
    rc, out = _run(["docker", "inspect", "-f", "{{.State.Status}}", container], timeout=15)
    return out if rc == 0 else "absent"


def _health(url: str) -> bool:
    try:
        r = httpx.get(url, timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# --- API --------------------------------------------------------------------
@app.get("/api/services", dependencies=[Depends(require_key)])
def list_services():
    out = []
    for s in SERVICES:
        state = _container_state(s["container"])
        out.append({
            "name": s["name"],
            "title": s["title"],
            "port": s["port"],
            "docs": s.get("docs"),
            "url": s.get("url"),
            "state": state,
            "healthy": _health(s["health"]) if state == "running" else False,
        })
    return {"services": out}


@app.post("/api/services/{name}/{action}", dependencies=[Depends(require_key)])
def service_action(name: str, action: str):
    s = SVC_BY_NAME.get(name)
    if not s:
        raise HTTPException(404, "unknown service")
    if action == "start":
        cmd = ["docker", "compose", "-f", s["compose"], "up", "-d"]
    elif action == "stop":
        cmd = ["docker", "compose", "-f", s["compose"], "stop"]
    elif action == "restart":
        cmd = ["docker", "compose", "-f", s["compose"], "restart"]
    else:
        raise HTTPException(400, "action must be start|stop|restart")
    rc, out = _run(cmd, timeout=180)
    return {"ok": rc == 0, "action": action, "name": name, "log": out[-2000:]}


@app.get("/api/gpu", dependencies=[Depends(require_key)])
def gpu():
    if not shutil.which("nvidia-smi"):
        return {"available": False}
    rc, out = _run([
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ], timeout=15)
    if rc != 0:
        return {"available": False}
    name, total, used, free, util, temp = [x.strip() for x in out.split(",")]
    rc2, apps = _run([
        "nvidia-smi", "--query-compute-apps=pid,used_memory,process_name",
        "--format=csv,noheader,nounits",
    ], timeout=15)
    procs = []
    if rc2 == 0 and apps:
        for line in apps.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3:
                procs.append({"pid": parts[0], "mem_mib": parts[1], "name": parts[2]})
    return {
        "available": True, "name": name,
        "mem_total": int(total), "mem_used": int(used), "mem_free": int(free),
        "util": int(util), "temp": int(temp), "procs": procs,
    }


@app.get("/api/system", dependencies=[Depends(require_key)])
def system():
    du = shutil.disk_usage("/")
    mem = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":")
                mem[k.strip()] = int(v.strip().split()[0])  # kB
    except Exception:
        pass
    total_kb = mem.get("MemTotal", 0)
    avail_kb = mem.get("MemAvailable", 0)
    return {
        "disk_total_gb": round(du.total / 1e9, 1),
        "disk_free_gb": round(du.free / 1e9, 1),
        "ram_total_gb": round(total_kb / 1e6, 1),
        "ram_avail_gb": round(avail_kb / 1e6, 1),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


@app.get("/api/ping")
def ping():
    return {"ok": True, "auth_required": bool(API_KEY)}


# --- Static dashboard -------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")
