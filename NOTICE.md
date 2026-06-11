# Third-party notices

This repository contains the **original code** of ai-media-stack (MIT). It *integrates* the
third-party projects below by calling/wrapping them. Their code, models and weights are **NOT
redistributed here** — you download them from upstream, under their own licenses. Check each
license before any commercial use.

| Project | Used for | License | Note |
|---|---|---|---|
| [CubiCasa5k](https://github.com/CubiCasa/CubiCasa5k) | neural floor-plan parsing | **CC BY-NC 4.0** | **non-commercial**; weights not bundled |
| [DeepFloorplan / TF2DeepFloorplan](https://github.com/zcemycl/TF2DeepFloorplan) | floor-plan segmentation (alt) | GPL-3.0 | not bundled |
| [Coqui XTTS v2](https://github.com/coqui-ai/TTS) | voice cloning / TTS | Coqui Public Model License (non-commercial) | not bundled |
| [OpenAI Whisper](https://github.com/openai/whisper) | speech-to-text | MIT | |
| [MuseTalk](https://github.com/TMElyralab/MuseTalk) | lip-sync | check upstream | not bundled |
| [Matterport Mask R-CNN](https://github.com/matterport/Mask_RCNN) | wall/door/window detection | MIT | |
| [FloorPlanTo3D-API](https://github.com/fadwabadwy/FloorPlanTo3D-API) | base floor-plan detector | upstream | our `serve.py` wraps it |
| [Ollama](https://github.com/ollama/ollama) · Qwen · Llama · [EuroLLM](https://huggingface.co/utter-project/EuroLLM-9B-Instruct) · [TranslateGemma](https://huggingface.co/google/translategemma-12b-it) | on-box LLMs / translation | respective licenses (Gemma terms apply to TranslateGemma) | weights not bundled |
| [Three.js](https://github.com/mrdoob/three.js) | 3D viewer | MIT | via CDN |
| [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) | person detection (VMS) | **AGPL-3.0** | ⚠️ copyleft — review before commercial use; weights not bundled |
| [InsightFace](https://github.com/deepinsight/insightface) (SCRFD + ArcFace) | face detection/recognition (VMS) | MIT code / **non-commercial models** | weights not bundled |
| [FAISS](https://github.com/facebookresearch/faiss) | face-embedding similarity search (VMS) | MIT | |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) · [FFmpeg](https://ffmpeg.org/) | inference / video (VMS) | MIT / LGPL-GPL | |
| [IOPaint](https://github.com/Sanster/IOPaint) (LaMa + SAM2) | object removal | Apache-2.0 / check models | deployed, not vendored |
| [Wan](https://github.com/Wan-Video) · [ComfyUI](https://github.com/comfyanonymous/ComfyUI) | video generation | Apache-2.0 / GPL-3.0 | deployed, not vendored |

The `floorplan3d`, `cubicasa-service` and `vms` directories contain only **our wrapper/serving code**;
their READMEs explain how to clone the upstream model repos and fetch weights. The VMS service ships
**no camera data, no face embeddings, no recordings and no database** — those live only on the operator's
box; configure cameras via the API/UI after deployment.
