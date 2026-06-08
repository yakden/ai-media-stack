# floorplan3d (CPU)
Our serving wrapper around the **FloorPlanTo3D / Matterport Mask R-CNN** detector, plus original additions:
Tesseract OCR (room names/areas/dimensions), and a **medial-axis wall vectorizer** that captures walls of
**any angle/curve/thickness** and classifies load-bearing vs partition, plus room segmentation.

> Weights & upstream model code are **not** included. Clone the upstream repo and download weights first:
> https://github.com/fadwabadwy/FloorPlanTo3D-API — then drop our `serve.py` in and `docker compose up`.
Files here: `serve.py` (our endpoints `/` `/ocr` `/vectorize` `/segment`), `Dockerfile.cpu`, `docker-compose.yml`.
