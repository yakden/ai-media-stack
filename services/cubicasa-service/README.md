# cubicasa-service (CPU)
Our serving wrapper around **CubiCasa5k** (neural floor-plan parser) exposing `POST /parse` →
vector walls, room polygons, doors/windows, fixtures — plus original additions: **colour-based apartment
segmentation** for multi-unit floors and a medial-axis full wall network.

> **CubiCasa5k is CC BY-NC 4.0 (non-commercial).** Its code/weights are **not** included here.
> Clone https://github.com/CubiCasa/CubiCasa5k, fetch the weights, then add our `serve_cubi.py` and `docker compose up`.
Files here: `serve_cubi.py`, `infer_cubi.py`, `Dockerfile.cpu`, `docker-compose.yml`.
