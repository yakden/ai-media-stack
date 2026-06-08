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
| [Ollama](https://github.com/ollama/ollama) · Qwen · Llama | on-box LLMs | respective licenses | |
| [Three.js](https://github.com/mrdoob/three.js) | 3D viewer | MIT | via CDN |

The `floorplan3d` and `cubicasa-service` directories contain only **our wrapper/serving code**; their
READMEs explain how to clone the upstream model repos and fetch weights.
