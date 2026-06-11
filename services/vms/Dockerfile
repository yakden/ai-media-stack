# CUDA 12.4 runtime + cuDNN — provides the CUDAExecutionProvider for
# onnxruntime-gpu (YOLOv8n) and insightface (SCRFD + ArcFace). Same base
# lineage as the whisper-xtts-server image on this box.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    # insightface caches its model packs here; we point it at the bind-mounted
    # models dir so buffalo_l is downloaded once and reused across rebuilds.
    INSIGHTFACE_HOME=/app/models/insightface

# Runtime deps:
#   ffmpeg          -> rolling segment recording + clip concat + thumbnails
#   libgl1/libglib  -> opencv-python-headless runtime shared libs
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        ffmpeg \
        libgl1 libglib2.0-0 \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better layer caching). onnxruntime-gpu pulls the
# matching CUDA EP; cuDNN/CUDA libs come from the base image.
COPY pyproject.toml ./
# NB: insightface hard-depends on `onnxruntime` (CPU build), which installs into the
# SAME `onnxruntime` package dir as `onnxruntime-gpu` and SHADOWS it — leaving only the
# CPUExecutionProvider at runtime (detection/face/ReID fall back to CPU ≈ 7 cores/camera).
# Drop the CPU build and reinstall the GPU build so the CUDAExecutionProvider is present —
# this is what offloads the heavy inference to the T4.
RUN pip3 install --upgrade pip \
    && pip3 install . \
    && pip3 uninstall -y onnxruntime \
    && pip3 install --force-reinstall --no-deps onnxruntime-gpu

# Application code (owned by other components; copied as a tree).
COPY app ./app
COPY scripts ./scripts

# data/ and models/ are bind-mounted at runtime (see docker-compose.yml).
# Create them so the image runs even without mounts (e.g. in CI smoke tests).
RUN mkdir -p \
        /app/models \
        /app/data/recordings \
        /app/data/thumbnails \
        /app/data/faces \
        /app/data/segments \
        /app/data/identities

EXPOSE 8120

# Bound to all interfaces inside the container; the published port is pinned to
# 127.0.0.1 by compose and reached only through the nginx SSO gateway.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8120"]
