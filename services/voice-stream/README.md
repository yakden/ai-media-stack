# voice-stream
Live **voice-to-voice translation in your own voice**, streaming from the microphone (not file-based).
Browser VAD cuts utterances → STT (Whisper) → translate (Llama) → re-speak via XTTS clone, played back live.
Voice **library** (save/select your voices) + **58 built-in speakers** with language flags & gender + per-voice
preview. Single file: `app.py` (FastAPI, :8202). Needs the whisper-xtts backend + Ollama.
