#!/usr/bin/env python3
"""Fetch the models the VMS needs into ./models (bind-mounted into the container).

Downloads:
  1. yolov8n.onnx              — person detector for onnxruntime (fp16 capable).
  2. insightface buffalo_l     — SCRFD detector + ArcFace r50 (512-d embeddings).
  3. osnet_x0_25_msmt17.onnx   — OSNet appearance Re-ID model for automatic
                                 cross-camera identification (512-d, fp16-capable).

The DB is the single source of truth for face/identity embeddings; these are
just the inference models. Run once on first deploy (or after wiping ./models):

    # on the host, before/after building the image
    pip install requests insightface onnxruntime
    python scripts/download_models.py

The OSNet ONNX is fetched from a release asset when available; if it is not yet
published it can be produced offline (on a box with torch + torchreid) via
``scripts/export_reid_onnx.py`` — this script will invoke that exporter as a
fallback when the asset download fails and torch is importable.

    # or inside the built container
    docker compose run --rm --entrypoint python vms scripts/download_models.py

Re-runs are idempotent: existing files are skipped unless --force is given.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests

# Resolve the models dir: env override (matches config.MODELS_DIR) or ./models
# relative to the repo root (this script lives in vms/scripts/).
DEFAULT_MODELS_DIR = os.environ.get(
    "MODELS_DIR",
    str(Path(__file__).resolve().parent.parent / "models"),
)

# YOLOv8n exported to ONNX. Ultralytics publishes a ready-made ONNX export of the
# nano detector on their release assets; this avoids pulling the heavy torch +
# ultralytics stack just to export it. (COCO classes; class 0 == person.)
YOLO_ONNX_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.onnx"
)
YOLO_FILENAME = "yolov8n.onnx"

# insightface model pack name. Downloaded via the insightface helper so the
# directory layout (models/<pack>/*.onnx) matches what FaceAnalysis expects.
INSIGHTFACE_PACK = os.environ.get("FACE_MODEL_PACK", "buffalo_l")

# OSNet appearance Re-ID model exported to ONNX. Default points at the
# osnet_x0_25 model trained on MSMT17 (lightest OSNet, ~0.2M params, good
# cross-camera generalization). REID_ONNX_URL can override the release asset;
# when unset/unreachable we fall back to the offline torchreid exporter.
REID_FILENAME = os.environ.get("REID_MODEL", "osnet_x0_25_msmt17.onnx")
REID_ONNX_URL = os.environ.get("REID_ONNX_URL", "").strip()


def _download(url: str, dest: Path, force: bool) -> None:
    if dest.exists() and not force:
        print(f"[skip] {dest} already exists ({dest.stat().st_size:,} bytes)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[get ] {url}\n   ->  {dest}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        written = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                f.write(chunk)
                written += len(chunk)
                if total:
                    pct = 100 * written / total
                    print(f"\r   ... {written:,}/{total:,} bytes ({pct:5.1f}%)", end="")
        print()
    tmp.rename(dest)
    print(f"[ok  ] {dest} ({dest.stat().st_size:,} bytes)")


def download_yolo(models_dir: Path, force: bool) -> None:
    _download(YOLO_ONNX_URL, models_dir / YOLO_FILENAME, force)


def download_insightface(models_dir: Path, force: bool) -> None:
    """Trigger insightface to download + unpack the model pack into models_dir.

    insightface caches packs under INSIGHTFACE_HOME/models/<pack>/. We point its
    home at the bind-mounted models dir so the pack lands next to yolov8n.onnx and
    survives container rebuilds.
    """
    pack_dir = models_dir / "insightface" / "models" / INSIGHTFACE_PACK
    if pack_dir.exists() and any(pack_dir.glob("*.onnx")) and not force:
        print(f"[skip] insightface pack '{INSIGHTFACE_PACK}' already present at {pack_dir}")
        return

    os.environ["INSIGHTFACE_HOME"] = str(models_dir / "insightface")
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        print(
            "[warn] insightface not installed in this environment; skipping the\n"
            "       buffalo_l download. It will be fetched automatically on first\n"
            "       use inside the container (INSIGHTFACE_HOME=/app/models/insightface),\n"
            "       or install insightface and re-run this script.",
            file=sys.stderr,
        )
        return

    print(f"[get ] insightface pack '{INSIGHTFACE_PACK}' -> {pack_dir}")
    # Instantiating FaceAnalysis downloads + unpacks the pack if missing.
    app = FaceAnalysis(name=INSIGHTFACE_PACK, root=str(models_dir / "insightface"))
    app.prepare(ctx_id=-1)  # ctx_id=-1: prepare on CPU, just to force the download
    print(f"[ok  ] insightface pack '{INSIGHTFACE_PACK}' ready at {pack_dir}")


def download_reid(models_dir: Path, force: bool) -> None:
    """Fetch (or export) the OSNet appearance Re-ID ONNX model.

    Order of preference:
      1. If the file already exists and not --force -> skip.
      2. If REID_ONNX_URL is set -> download it (like yolov8n.onnx).
      3. Otherwise -> run scripts/export_reid_onnx.py to export from torchreid
         (requires torch + torchreid; only needed offline on a dev box).
    """
    dest = models_dir / REID_FILENAME
    if dest.exists() and not force:
        print(f"[skip] {dest} already exists ({dest.stat().st_size:,} bytes)")
        return

    if REID_ONNX_URL:
        try:
            _download(REID_ONNX_URL, dest, force)
            return
        except Exception as exc:  # noqa: BLE001 - fall through to exporter
            print(f"[warn] OSNet download failed ({exc}); trying offline export…", file=sys.stderr)

    # Fallback: invoke the offline exporter if it (and torch) are available.
    exporter = Path(__file__).resolve().parent / "export_reid_onnx.py"
    if not exporter.exists():
        print(
            "[warn] No REID_ONNX_URL set and scripts/export_reid_onnx.py is\n"
            "       missing; skipping OSNet. Set REID_ONNX_URL to a release\n"
            "       asset or add the exporter, then re-run.",
            file=sys.stderr,
        )
        return
    try:
        import torch  # noqa: F401
        import torchreid  # noqa: F401
    except ImportError:
        print(
            "[warn] torch/torchreid not installed; cannot export OSNet here.\n"
            "       Run scripts/export_reid_onnx.py on a dev box with torch +\n"
            "       torchreid, or set REID_ONNX_URL to a pre-exported asset.",
            file=sys.stderr,
        )
        return

    import subprocess

    print(f"[get ] exporting OSNet via {exporter} -> {dest}")
    subprocess.check_call(
        [sys.executable, str(exporter), "--output", str(dest)]
    )
    print(f"[ok  ] OSNet ready at {dest}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-dir",
        default=DEFAULT_MODELS_DIR,
        help=f"target models directory (default: {DEFAULT_MODELS_DIR})",
    )
    parser.add_argument(
        "--force", action="store_true", help="re-download even if files exist"
    )
    parser.add_argument(
        "--only",
        choices=["yolo", "insightface", "reid"],
        help="download only one model group",
    )
    args = parser.parse_args()

    models_dir = Path(args.models_dir).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"Models dir: {models_dir}\n")

    if args.only in (None, "yolo"):
        download_yolo(models_dir, args.force)
    if args.only in (None, "insightface"):
        download_insightface(models_dir, args.force)
    if args.only in (None, "reid"):
        download_reid(models_dir, args.force)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
