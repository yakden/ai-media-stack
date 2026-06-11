# VMS — Video Management System (MVP)

A self-contained, single-box **Video Management System** for the T4 AI platform:
RTSP camera management, **person-triggered clip recording** with pre/post-roll,
a live monitoring grid, an events history with playback, and **face recognition**
(SCRFD + ArcFace + FAISS) to attach a recognized identity to each event.

Runs as one Docker Compose service bound to `127.0.0.1:8120`, reached publicly
through the existing **nginx + Let's Encrypt + cookie-SSO** gateway at
`https://vms.1c-rus.ru`, and managed from the **control plane**.

```
RTSP camera ──► per-camera worker ──► YOLOv8n (person?) ──► trigger
                     │                                         │
                     ├─ warm ffmpeg segment buffer ────────────┤ clip (pre/post)
                     ├─ latest annotated JPEG (MJPEG live grid) │ + thumbnail
                     └─ insightface SCRFD+ArcFace ─► FAISS match ┘
                                                              │
                                              SQLite (cameras/events/persons/faces)
                                                              │
                                            FastAPI + vanilla-JS SPA  (127.0.0.1:8120)
```

---

## Stack

| Concern        | Choice |
| -------------- | ------ |
| API            | Python 3.10 + FastAPI + Uvicorn (same lineage as `whisper-xtts-server`) |
| Metadata       | SQLite (WAL) via SQLAlchemy 2.x |
| Detection      | YOLOv8n exported to ONNX, run under `onnxruntime-gpu` (fp16); CPU fallback |
| Faces          | insightface `buffalo_l` (SCRFD + ArcFace r50, 512-d) + `faiss-cpu` IndexFlatIP |
| Recording      | ffmpeg segment buffer + concat (no re-encode) |
| Live view      | MJPEG (`multipart/x-mixed-replace`) + JPEG snapshots |
| Frontend       | vanilla-JS SPA (no build step), served by FastAPI StaticFiles |
| GPU budget     | ~2 GB total — coexists with the generative stack; evictable to CPU |

DeepStream 7.1 is wired in as an **optional** high-throughput backend
(`DETECTOR_BACKEND=deepstream`) but is **not required** to build or run the MVP.

---

## Quick start

```bash
cd vms

# 1. Configure
cp .env.example .env
#   edit .env — at minimum confirm SSO_HEADER and (optionally) set API_KEY

# 2. Download models into ./models (bind-mounted). Idempotent.
#    Easiest: run inside the built image so the deps are present.
docker compose build
docker compose run --rm --entrypoint python vms scripts/download_models.py
#    (or on the host: pip install requests insightface onnxruntime &&
#     python scripts/download_models.py)

# 3. Run
docker compose up -d

# 4. Verify
curl -s http://127.0.0.1:8120/health | python3 -m json.tool
```

Then add cameras via the UI (`https://vms.1c-rus.ru` once nginx is wired, or an
SSH tunnel — see below) or the API:

```bash
curl -s -X POST http://127.0.0.1:8120/api/cameras \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <API_KEY>' \
  -d '{"name":"Front door","rtsp_url":"rtsp://user:pass@10.0.0.50:554/stream1","enabled":true}'
```

### Layout

```
vms/
├── docker-compose.yml             # vms service, 127.0.0.1:8120, GPU reservation
├── docker-compose.deepstream.yml  # OPTIONAL DeepStream backend override
├── Dockerfile                     # CUDA 12.4 + python + ffmpeg + onnxruntime-gpu + faiss + insightface
├── pyproject.toml                 # deps
├── .env.example                   # all config knobs
├── scripts/download_models.py     # fetch yolov8n.onnx + insightface buffalo_l
├── models/   (gitignored)         # bind-mounted model files
├── data/     (gitignored)         # bind-mounted: vms.db, recordings/, thumbnails/, faces/, segments/
└── app/                           # FastAPI backend + workers + SPA
```

---

## Configuration

All knobs live in `.env` (see `.env.example` for the annotated list). Highlights:

| Var | Default | Purpose |
| --- | --- | --- |
| `PORT` | `8120` | HTTP port (container-internal; published on `127.0.0.1:8120`) |
| `SSO_HEADER` | `X-Email` | Trusted header injected by the SSO proxy = authenticated user |
| `API_KEY` | _(unset)_ | Optional bearer token for CLI / tunneled access |
| `DETECTOR_BACKEND` | `onnx` | `onnx` \| `cpu` \| `deepstream` |
| `DETECTOR_DEVICE` | `cuda` | `cuda` \| `cpu` — set `cpu` under GPU pressure |
| `DETECT_CONF` | `0.4` | Default person-confidence threshold (per-camera override) |
| `DETECT_FPS` | `5` | Detection rate per camera (keeps GPU light) |
| `PRE_SECONDS` / `POST_SECONDS` | `5` / `10` | Clip pre/post-roll (per-camera override) |
| `FACE_MATCH_THRESHOLD` | `0.45` | Cosine similarity for a positive identity match |

### GPU budget & eviction

YOLOv8n fp16 (~150 MB) + SCRFD/ArcFace fp16 (~600 MB) + the onnxruntime CUDA
context (~700 MB) ≈ **under 2 GB** for a handful of cameras, so VMS coexists with
the generative stack. When the control plane needs the whole T4 for a heavy job
(e.g. Wan 2.2), either stop the service or set `DETECTOR_DEVICE=cpu` /
`DETECTOR_BACKEND=cpu` to keep detection running on CPU during the squeeze.

---

## nginx + Let's Encrypt + cookie-SSO

Add a vhost mirroring the other services. Service ports stay on `127.0.0.1`;
the gateway terminates TLS, enforces SSO, and injects the user header.

`/etc/nginx/sites-available/vms.1c-rus.ru`:

```nginx
server {
    listen 80;
    server_name vms.1c-rus.ru;

    # ACME http-01 challenge
    location /.well-known/acme-challenge/ { root /var/www/certbot; }
    location / { return 301 https://$host$request_uri; }
}

server {
    listen 443 ssl http2;
    server_name vms.1c-rus.ru;

    ssl_certificate     /etc/letsencrypt/live/vms.1c-rus.ru/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/vms.1c-rus.ru/privkey.pem;

    # Face-photo uploads
    client_max_body_size 64m;

    # --- cookie-SSO gateway (same shape as webui.1c-rus.ru) ---
    location = /sso/auth {
        internal;
        proxy_pass              http://127.0.0.1:8095/auth;
        proxy_pass_request_body off;
        proxy_set_header        Content-Length "";
        proxy_set_header        X-Original-URI $request_uri;
    }
    location /sso/ {
        proxy_pass http://127.0.0.1:8095/;
    }
    location @login {
        return 302 https://ai.1c-rus.ru/sso/login?rd=https://$host$request_uri;
    }

    # --- application ---
    location / {
        auth_request     /sso/auth;
        error_page 401 = @login;

        # Pull the authenticated identity from the SSO subrequest and forward it
        # as the trusted header app/auth.py reads (SSO_HEADER).
        auth_request_set $sso_email $upstream_http_x_auth_email;
        auth_request_set $sso_user  $upstream_http_x_auth_user;
        proxy_set_header X-Email    $sso_email;
        proxy_set_header X-User     $sso_user;

        proxy_pass http://127.0.0.1:8120;
        proxy_set_header Host              $host;
        proxy_set_header X-Real-IP         $remote_addr;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # MJPEG live streams + long clip downloads
        proxy_read_timeout 1200s;
        proxy_buffering    off;   # REQUIRED so multipart MJPEG flushes per frame
    }
}
```

Enable + reload:

```bash
ln -s /etc/nginx/sites-available/vms.1c-rus.ru /etc/nginx/sites-enabled/
certbot --nginx -d vms.1c-rus.ru        # or certonly + the snippet above
nginx -t && systemctl reload nginx
```

`app/auth.py` treats the presence of the `SSO_HEADER` (default `X-Email`) as
authenticated. The optional `API_KEY` bearer path is kept for SSH-tunnel / CLI
use that bypasses the proxy.

### SSH-tunnel access (no nginx)

```bash
ssh -N -L 8120:127.0.0.1:8120 deploy@<box>
# then open http://127.0.0.1:8120 and send Authorization: Bearer <API_KEY>
```

---

## Control-plane registration

Append one dict to `SERVICES` in `/opt/control-plane/app.py`, then restart it.
This surfaces VMS start/stop/restart + health + GPU usage on the dashboard.

```python
{
    "name": "vms",
    "title": "Video Management — RTSP person detection + face recognition",
    "container": "vms",
    "compose": "/home/deploy/whisper-xtts-server/vms/docker-compose.yml",
    "health": "http://127.0.0.1:8120/health",
    "port": 8120,
    "docs": "/",
}
```

```bash
systemctl restart control-plane
```

VMS keeps ~2 GB VRAM and is eviction-friendly: when paused for a heavy
generative job its workers stop cleanly and resume on restart. To keep it
running on CPU during GPU pressure, set `DETECTOR_BACKEND=cpu` (or
`DETECTOR_DEVICE=cpu`) and restart.

---

## Optional: DeepStream backend

For many high-FPS streams (the DeepStream person-detect + tracker graph
benchmarks ~700 FPS on the T4):

```bash
# .env: DETECTOR_BACKEND=deepstream
docker compose -f docker-compose.yml -f docker-compose.deepstream.yml up -d
```

The `vms` backend then consumes person-detection metadata from the `deepstream`
service over the internal compose network (`DEEPSTREAM_URL=http://deepstream:8121`)
instead of running ONNX itself. The DeepStream pipeline config + metadata bridge
live under `./deepstream/` (provided when this backend is adopted; gitignored).
The default MVP build ignores this override entirely.

---

## API summary

Base: `https://vms.1c-rus.ru` (internally `127.0.0.1:8120`). All routes behind
SSO; `/health` is unauthenticated.

| Group | Endpoints |
| --- | --- |
| System | `GET /health`, `GET /api/system` |
| Cameras | `GET/POST /api/cameras`, `GET/PUT/DELETE /api/cameras/{id}`, `GET /api/cameras/{id}/status` |
| Live | `GET /api/live/{id}/stream` (MJPEG), `GET /api/live/{id}/snapshot` (JPEG) |
| Events | `GET /api/events`, `GET/DELETE /api/events/{id}`, `GET /api/events/{id}/clip` (Range mp4), `GET /api/events/{id}/thumbnail` |
| People | `GET/POST /api/people`, `GET/PUT/DELETE /api/people/{id}`, `POST/GET /api/people/{id}/faces`, `DELETE /api/people/{id}/faces/{fid}` |
| SPA | `GET /` and static assets |

See the architecture doc / OpenAPI at `/docs` for full request/response shapes.

---

## Development

```bash
pip install -e '.[dev]'
python scripts/download_models.py            # models into ./models
DATA_DIR=./data MODELS_DIR=./models \
  uvicorn app.main:app --reload --port 8120
pytest                                        # CRUD + FaceIndex tests (mocked detection/RTSP)
```

## Notes

- **Data is the source of truth.** `data/vms.db` holds all metadata; the FAISS
  index is derived state, rebuilt from `face_embeddings` on startup. Back up
  `data/` (db + recordings + thumbnails + faces) to back up everything.
- **No re-encode.** Clips are concatenated from the warm ffmpeg segment buffer,
  so recording stays CPU/GPU-light.
- **Loopback only.** The container port is published on `127.0.0.1`; reach it
  through nginx or an SSH tunnel, never the public IP.
