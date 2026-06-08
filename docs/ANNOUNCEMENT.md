# Announcement drafts

Ready-to-post copy for sharing the project. Edit freely.

---

## LinkedIn (EN)

🚀 I open-sourced **ai-media-stack** — a self-hosted, multi-tenant AI media platform that runs a whole
fleet of generative services on a **single 16 GB GPU**.

What's inside:
• 🏗️ Floor-plan → 3D (typed rooms, walls of any shape, apartments, true scale) + photoreal renders
• 🗣️ Live voice-to-voice translation **in your own cloned voice**
• 🎥 Video dubbing with lip-sync · 🖼️ talking avatars · 🤖 on-box LLMs (Qwen/Llama)

The interesting engineering:
• A **GPU broker** that keeps one model resident and swaps gracefully between jobs, with a live queue + ETA
• A **multi-tenant API gateway**: one key for everything, usage metering, monthly quotas, rate limits, billing & an admin dashboard
• Hybrid CV + neural floor-plan understanding (medial-axis wall vectorization, colour-based apartment detection, a multi-source scale solver)

MIT-licensed. Code + architecture + screenshots 👇
https://github.com/yakden/ai-media-stack

#AI #MachineLearning #ComputerVision #LLM #OpenSource #FastAPI #GPU

---

## Habr / Reddit (RU)

**Выложил в open-source: ai-media-stack — ИИ-медиаплатформа на одном GPU**

Self-hosted платформа, которая держит целый набор генеративных сервисов на одной NVIDIA T4 (16 ГБ):

- 🏗️ план помещения → 3D (типы комнат, стены любой формы, квартиры, реальный масштаб) + фотореалистичные рендеры
- 🗣️ потоковый перевод речи **твоим собственным голосом**
- 🎥 дубляж видео с липсинком, 🖼️ говорящие аватары, 🤖 локальные LLM (Qwen/Llama)

Самое интересное инженерно:
- **GPU-брокер**: одна тяжёлая модель в памяти, аккуратный своп между задачами (без kill -9, не трогая прод), очередь с ETA
- **Мультитенантный API-шлюз**: один ключ на всё, учёт потребления, месячные квоты, rate-limit, биллинг и админка
- Гибридное CV+нейро понимание планов: векторизация стен по медиальной оси, сегментация квартир по цвету, решатель масштаба с отбросом мисридов OCR

Лицензия MIT. Код, архитектура и скриншоты:
https://github.com/yakden/ai-media-stack

---

## X / Twitter

Open-sourced ai-media-stack 🚀 — a whole AI media platform on ONE 16GB GPU:
floor-plan→3D, live voice-to-voice in your own voice, dubbing+lip-sync, avatars, on-box LLMs —
behind a GPU model-swap broker + a multi-tenant API gateway (keys, quotas, billing).
MIT. https://github.com/yakden/ai-media-stack
