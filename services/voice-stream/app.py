"""Live streaming voice translator: microphone speech -> STT -> translate ->
the SAME user's voice in another language, in (near) real time.

Not file-based: the browser captures the mic continuously, cuts it into utterances
by voice-activity (VAD), and streams each utterance to /chunk. Each utterance is
transcribed (whisper), translated (llama3.2), and re-spoken in the user's cloned
voice (XTTS) — played back immediately. Record a voice reference ONCE (/ref) so the
clone stays consistent.

Reuses the resident on-box backends (no GPU swap, so the broker/duty model is fine
and 1C is untouched): whisper-xtts STT+XTTS (:8000), Ollama llama3.2 translate (:11434).
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import threading
import urllib.parse

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

KEY = os.environ.get("TTS_API_KEY", "")
TTS = "http://127.0.0.1:8000"
OLLAMA = "http://127.0.0.1:11434"
TR_MODEL = "llama3.2:3b"                 # light, fast, low VRAM — kept warm during a session
REFDIR = "/tmp/voicestream"              # ephemeral per-session ref (fallback)
VOICES = "/opt/voice-stream/voices"      # PERSISTENT named voice library
os.makedirs(REFDIR, exist_ok=True)
os.makedirs(VOICES, exist_ok=True)
REF_SECONDS = 8                          # trim refs — shorter = faster XTTS speaker encoding


def _safe(name):
    return "".join(ch for ch in (name or "") if ch.isalnum() or ch in " _-")[:40].strip() or "voice"


def _voice_path(name):
    return os.path.join(VOICES, _safe(name) + ".wav")

# the T4 runs ONE whisper-xtts model — serialize backend calls so concurrent streamed
# utterances don't hit STT/XTTS at the same time (that 500s the backend).
_GPU_LOCK = threading.Lock()

# Built-in XTTS v2 studio speakers (preset voices) — synthesized via /v1/audio/speech
# (no reference needed, and faster: speaker latents are precomputed).
PRESETS = [
    "Claribel Dervla", "Daisy Studious", "Gracie Wise", "Tammie Ema", "Alison Dietlinde",
    "Ana Florence", "Annmarie Nele", "Asya Anara", "Brenda Stern", "Gitta Nikolina",
    "Henriette Usha", "Sofia Hellen", "Tammy Grit", "Tanja Adelina", "Vjollca Johnnie",
    "Andrew Chipper", "Badr Odhiambo", "Dionisio Schuyler", "Royston Min", "Viktor Eka",
    "Abrahan Mack", "Adde Michal", "Baldur Sanjin", "Craig Gutsy", "Damien Black",
    "Gilberto Mathias", "Ilkin Urbano", "Kazuhiko Atallah", "Ludvig Milivoj", "Suad Qasim",
    "Torcull Diarmuid", "Viktor Menelaos", "Zacharie Aimilios", "Nova Hogarth", "Maja Ruoho",
    "Uta Obando", "Lidiya Szekeres", "Chandra MacFarland", "Szofi Granger", "Camilla Holmström",
    "Lilya Stainthorpe", "Zofija Kendrick", "Narelle Moon", "Barbora MacLean", "Alexandra Hisakawa",
    "Alma María", "Rosemary Okafor", "Ige Behringer", "Filip Traverse", "Damjan Chapman",
    "Wulf Carlevaro", "Aaron Dreschner", "Kumar Dahl", "Eugenio Mataracı", "Ferran Simen",
    "Xavier Hayasaka", "Luis Moray", "Marcos Rudaski",
]

LANGS = {
    "en": "English", "ru": "Russian", "es": "Spanish", "fr": "French",
    "de": "German", "it": "Italian", "pt": "Portuguese", "pl": "Polish",
    "tr": "Turkish", "nl": "Dutch", "cs": "Czech", "ar": "Arabic",
    "zh-cn": "Chinese", "ja": "Japanese", "hu": "Hungarian", "ko": "Korean",
}

# short sample phrase per language for the preview button
SAMPLE = {
    "en": "Hello! This is a sample of this voice.",
    "ru": "Привет! Это образец этого голоса.",
    "es": "¡Hola! Esta es una muestra de esta voz.",
    "fr": "Bonjour ! Ceci est un échantillon de cette voix.",
    "de": "Hallo! Dies ist eine Probe dieser Stimme.",
    "it": "Ciao! Questo è un campione di questa voce.",
    "pt": "Olá! Esta é uma amostra desta voz.",
    "pl": "Cześć! To jest próbka tego głosu.",
    "tr": "Merhaba! Bu, bu sesin bir örneğidir.",
    "nl": "Hallo! Dit is een voorbeeld van deze stem.",
    "cs": "Ahoj! Toto je ukázka tohoto hlasu.",
    "ar": "مرحبا! هذه عينة من هذا الصوت.",
    "zh-cn": "你好！这是这个声音的示例。",
    "ja": "こんにちは！これはこの声のサンプルです。",
    "hu": "Helló! Ez egy minta ebből a hangból.",
    "ko": "안녕하세요! 이것은 이 목소리의 샘플입니다.",
}

# per-voice gender + origin flag (heuristic by name; flag 🌐 = origin outside the supported set)
VOICE_META = {
    "Claribel Dervla": ("ж", "🇬🇧"), "Daisy Studious": ("ж", "🇬🇧"), "Gracie Wise": ("ж", "🇬🇧"),
    "Tammie Ema": ("ж", "🇬🇧"), "Alison Dietlinde": ("ж", "🇩🇪"), "Ana Florence": ("ж", "🇵🇹"),
    "Annmarie Nele": ("ж", "🇩🇪"), "Asya Anara": ("ж", "🇷🇺"), "Brenda Stern": ("ж", "🇬🇧"),
    "Gitta Nikolina": ("ж", "🇩🇪"), "Henriette Usha": ("ж", "🇫🇷"), "Sofia Hellen": ("ж", "🇪🇸"),
    "Tammy Grit": ("ж", "🇬🇧"), "Tanja Adelina": ("ж", "🇩🇪"), "Vjollca Johnnie": ("ж", "🌐"),
    "Andrew Chipper": ("м", "🇬🇧"), "Badr Odhiambo": ("м", "🇸🇦"), "Dionisio Schuyler": ("м", "🇪🇸"),
    "Royston Min": ("м", "🇨🇳"), "Viktor Eka": ("м", "🇷🇺"), "Abrahan Mack": ("м", "🇪🇸"),
    "Adde Michal": ("м", "🇵🇱"), "Baldur Sanjin": ("м", "🇩🇪"), "Craig Gutsy": ("м", "🇬🇧"),
    "Damien Black": ("м", "🇬🇧"), "Gilberto Mathias": ("м", "🇵🇹"), "Ilkin Urbano": ("м", "🇹🇷"),
    "Kazuhiko Atallah": ("м", "🇯🇵"), "Ludvig Milivoj": ("м", "🇷🇺"), "Suad Qasim": ("м", "🇸🇦"),
    "Torcull Diarmuid": ("м", "🇬🇧"), "Viktor Menelaos": ("м", "🌐"), "Zacharie Aimilios": ("м", "🇫🇷"),
    "Nova Hogarth": ("ж", "🇬🇧"), "Maja Ruoho": ("ж", "🌐"), "Uta Obando": ("ж", "🇩🇪"),
    "Lidiya Szekeres": ("ж", "🇭🇺"), "Chandra MacFarland": ("ж", "🇬🇧"), "Szofi Granger": ("ж", "🇭🇺"),
    "Camilla Holmström": ("ж", "🌐"), "Lilya Stainthorpe": ("ж", "🇷🇺"), "Zofija Kendrick": ("ж", "🇵🇱"),
    "Narelle Moon": ("ж", "🇬🇧"), "Barbora MacLean": ("ж", "🇨🇿"), "Alexandra Hisakawa": ("ж", "🇯🇵"),
    "Alma María": ("ж", "🇪🇸"), "Rosemary Okafor": ("ж", "🇬🇧"), "Ige Behringer": ("м", "🇩🇪"),
    "Filip Traverse": ("м", "🇫🇷"), "Damjan Chapman": ("м", "🇵🇱"), "Wulf Carlevaro": ("м", "🇩🇪"),
    "Aaron Dreschner": ("м", "🇩🇪"), "Kumar Dahl": ("м", "🌐"), "Eugenio Mataracı": ("м", "🇹🇷"),
    "Ferran Simen": ("м", "🇪🇸"), "Xavier Hayasaka": ("м", "🇯🇵"), "Luis Moray": ("м", "🇪🇸"),
    "Marcos Rudaski": ("м", "🇪🇸"),
}

# recommended preset voices per language (by accent/origin — approximate; all are multilingual)
RECOMMEND = {
    "en": ["Claribel Dervla", "Andrew Chipper", "Gracie Wise", "Craig Gutsy", "Daisy Studious"],
    "ru": ["Asya Anara", "Lidiya Szekeres", "Tanja Adelina", "Ludvig Milivoj"],
    "es": ["Alma María", "Luis Moray", "Marcos Rudaski", "Ferran Simen"],
    "fr": ["Dionisio Schuyler", "Zacharie Aimilios", "Filip Traverse"],
    "de": ["Aaron Dreschner", "Baldur Sanjin", "Wulf Carlevaro", "Damien Black"],
    "it": ["Gilberto Mathias", "Alma María", "Luis Moray"],
    "pt": ["Gilberto Mathias", "Marcos Rudaski"],
    "pl": ["Zofija Kendrick", "Damjan Chapman", "Barbora MacLean"],
    "tr": ["Eugenio Mataracı", "Ilkin Urbano", "Suad Qasim"],
    "nl": ["Baldur Sanjin", "Adde Michal", "Wulf Carlevaro"],
    "cs": ["Barbora MacLean", "Ludvig Milivoj"],
    "ar": ["Suad Qasim", "Badr Odhiambo", "Ilkin Urbano"],
    "zh-cn": ["Royston Min", "Xavier Hayasaka"],
    "ja": ["Xavier Hayasaka", "Kazuhiko Atallah"],
    "hu": ["Lidiya Szekeres", "Szofi Granger"],
    "ko": ["Royston Min", "Xavier Hayasaka"],
}

app = FastAPI(title="voice-stream", version="0.1.0")


def _hdr():
    return {"Authorization": f"Bearer {KEY}"} if KEY else {}


def _to_wav(src, dst, max_s=None):
    cmd = ["ffmpeg", "-y", "-i", src, "-vn", "-ac", "1", "-ar", "16000"]
    if max_s:
        cmd += ["-t", str(max_s)]
    cmd += [dst]
    subprocess.run(cmd, capture_output=True)
    return os.path.exists(dst) and os.path.getsize(dst) > 800


@app.get("/", response_class=HTMLResponse)
def index():
    opts = "".join(f'<option value="{c}"{" selected" if c=="en" else ""}>{n}</option>'
                   for c, n in LANGS.items())
    return PAGE.replace("{{OPTS}}", opts)


@app.get("/voices")
def list_voices():
    """Preset (built-in XTTS) voices + saved user voices + per-language recommendations."""
    names = sorted(f[:-4] for f in os.listdir(VOICES) if f.endswith(".wav"))
    return JSONResponse({"voices": names, "presets": PRESETS, "recommend": RECOMMEND,
                         "meta": VOICE_META})


@app.post("/preview")
def preview(voice: str = Form(...), lang: str = Form("en")):
    """Short sample of a voice in the chosen language — to pick a voice by ear."""
    if lang not in LANGS:
        lang = "en"
    text = SAMPLE.get(lang, SAMPLE["en"])
    work = tempfile.mkdtemp(prefix="vsp_")
    out_wav = os.path.join(work, "preview.wav")
    with _GPU_LOCK, httpx.Client(timeout=120) as c:
        try:
            if voice in PRESETS:
                r = c.post(f"{TTS}/v1/audio/speech", headers=_hdr(),
                           json={"input": text, "voice": voice, "language": lang, "response_format": "wav"})
            else:
                ref = _voice_path(voice)
                if not os.path.exists(ref):
                    raise HTTPException(404, "voice not found")
                with open(ref, "rb") as fh:
                    r = c.post(f"{TTS}/tts/clone", headers=_hdr(),
                               data={"text": text, "language": lang},
                               files={"speaker_wav": ("ref.wav", fh, "audio/wav")})
            r.raise_for_status()
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(502, f"preview failed: {exc}")
        with open(out_wav, "wb") as out:
            out.write(r.content)
    return FileResponse(out_wav, media_type="audio/wav")


@app.post("/voices")
async def save_voice(audio: UploadFile = File(...), name: str = Form(...)):
    """Save/replace a named voice in the persistent library (trimmed for fast cloning)."""
    name = _safe(name)
    work = tempfile.mkdtemp(prefix="vsv_")
    raw = os.path.join(work, "raw")
    with open(raw, "wb") as f:
        f.write(await audio.read())
    if not _to_wav(raw, _voice_path(name), max_s=REF_SECONDS):
        raise HTTPException(400, "bad voice audio")
    return JSONResponse({"ok": True, "name": name})


@app.delete("/voices/{name}")
def delete_voice(name: str):
    p = _voice_path(name)
    if os.path.exists(p):
        os.remove(p)
    return JSONResponse({"ok": True})


@app.post("/ref")
async def ref(audio: UploadFile = File(...), sid: str = Form(...)):
    """Ephemeral per-session reference (used when no named voice is selected)."""
    sid = "".join(ch for ch in sid if ch.isalnum())[:32] or "anon"
    work = tempfile.mkdtemp(prefix="vsr_")
    raw = os.path.join(work, "raw")
    with open(raw, "wb") as f:
        f.write(await audio.read())
    if not _to_wav(raw, os.path.join(REFDIR, f"{sid}.wav"), max_s=REF_SECONDS):
        raise HTTPException(400, "bad reference audio")
    return JSONResponse({"ok": True, "sid": sid})


@app.post("/chunk")
async def chunk(audio: UploadFile = File(...), target_lang: str = Form("en"),
                sid: str = Form(""), source_lang: str = Form("ru"), voice: str = Form("")):
    """One utterance: STT -> translate -> clone in the chosen voice. Returns wav."""
    if target_lang not in LANGS:
        raise HTTPException(400, "unsupported language")
    sid = "".join(ch for ch in sid if ch.isalnum())[:32] or "anon"
    work = tempfile.mkdtemp(prefix="vsc_")
    raw = os.path.join(work, "raw")
    with open(raw, "wb") as f:
        f.write(await audio.read())
    wav = os.path.join(work, "u.wav")
    # too-short / empty segments are normal in a VAD stream — skip cleanly (200, no error/500)
    if not _to_wav(raw, wav) or os.path.getsize(wav) < 16000:   # ~0.5s @16k mono 16-bit
        return JSONResponse({"skip": "too_short"})
    # voice selection: a PRESET (built-in speaker) -> /v1/audio/speech (fast, no ref);
    # else a saved/session/utterance reference -> /tts/clone.
    is_preset = voice in PRESETS
    ref_wav = ""
    if not is_preset:
        ref_wav = _voice_path(voice) if voice else ""
        if not (ref_wav and os.path.exists(ref_wav)):
            ref_wav = os.path.join(REFDIR, f"{sid}.wav")
        if not os.path.exists(ref_wav):
            ref_wav = wav

    # serialize ALL backend work — the single T4 model can't do two inferences at once
    with _GPU_LOCK, httpx.Client(timeout=300) as c:
        try:
            with open(wav, "rb") as fh:
                r = c.post(f"{TTS}/v1/audio/transcriptions", headers=_hdr(),
                           files={"file": ("u.wav", fh, "audio/wav")},
                           data={"language": source_lang})
            r.raise_for_status()
            src_text = (r.json().get("text") or "").strip()
        except Exception:
            return JSONResponse({"skip": "stt_error"})
        if len(src_text) < 2:
            return JSONResponse({"skip": "silence"})   # empty segment — not an error

        try:
            prompt = (f"Translate the text to {LANGS[target_lang]}. Output ONLY the translation, "
                      f"no quotes, no notes:\n\n{src_text}")
            r = c.post(f"{OLLAMA}/api/generate",
                       json={"model": TR_MODEL, "prompt": prompt, "stream": False,
                             "keep_alive": "5m", "options": {"temperature": 0.2}})
            r.raise_for_status()
            tr = (r.json().get("response") or "").strip()
        except Exception:
            return JSONResponse({"skip": "translate_error"})
        if not tr:
            return JSONResponse({"skip": "no_translation"})

        out_wav = os.path.join(work, "out.wav")
        try:
            if is_preset:                       # built-in studio speaker (no reference, faster)
                r = c.post(f"{TTS}/v1/audio/speech", headers=_hdr(),
                           json={"input": tr, "voice": voice, "language": target_lang,
                                 "response_format": "wav"})
            else:                               # clone the user's own voice
                with open(ref_wav, "rb") as fh:
                    r = c.post(f"{TTS}/tts/clone", headers=_hdr(),
                               data={"text": tr, "language": target_lang},
                               files={"speaker_wav": ("ref.wav", fh, "audio/wav")})
            r.raise_for_status()
        except Exception:
            return JSONResponse({"skip": "tts_error", "source": src_text[:200], "translation": tr[:200]})
        with open(out_wav, "wb") as out:
            out.write(r.content)

    resp = FileResponse(out_wav, media_type="audio/wav")
    resp.headers["X-Source"] = urllib.parse.quote(src_text[:400])
    resp.headers["X-Translation"] = urllib.parse.quote(tr[:400])
    resp.headers["Access-Control-Expose-Headers"] = "X-Source, X-Translation"
    return resp


PAGE = r"""<!doctype html><html lang=ru><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>Поточный перевод голоса</title>
<style>
body{margin:0;background:#0e1116;color:#e6edf3;font:15px system-ui;padding:18px;max-width:820px;margin:auto}
h1{font-size:20px}.mut{color:#8b97a7;font-size:13px}
button{padding:12px 18px;border:0;border-radius:10px;font-size:15px;cursor:pointer;margin:6px 6px 0 0}
button:disabled{opacity:.5;cursor:default}
.rec{background:#e74c3c;color:#fff}.go{background:#2ecc71;color:#06210f}.stop{background:#f1c40f;color:#111}
select{padding:10px;border-radius:9px;background:#0b0e13;color:#e6edf3;border:1px solid #222a35;font-size:15px}
.card{background:#171c24;border:1px solid #222a35;border-radius:12px;padding:16px;margin-top:14px}
.lvl{height:10px;background:#0b0e13;border-radius:6px;overflow:hidden;margin:10px 0}
.lvl>i{display:block;height:100%;width:0;background:linear-gradient(90deg,#2ecc71,#4f8cff)}
.log{max-height:300px;overflow:auto;margin-top:10px}
.u{border-bottom:1px solid #222a35;padding:8px 0}.u .s{color:#8b97a7}.u .t{color:#e6edf3;font-weight:600}
#dot{display:inline-block;width:10px;height:10px;border-radius:50%;background:#555;margin-right:6px;vertical-align:middle}
.on{background:#2ecc71!important;box-shadow:0 0 8px #2ecc71}
</style></head><body>
<h1>🎙️ Поточный перевод голоса — твоим голосом</h1>
<p class=mut>Говоришь в микрофон → распознаётся → переводится → произносится <b>выбранным голосом</b> на другом
языке, прямо в потоке (по фразам). Сохрани свои голоса в библиотеку и переключайся между ними.</p>

<div class=card>
  <b>1. Библиотека голосов</b>
  <div style="margin-top:8px;display:flex;gap:8px;align-items:center;flex-wrap:wrap">
    Активный голос: <select id=voiceSel></select>
    <button class=go id=previewBtn>▶ Прослушать</button>
    <button id=delVoice class=stop>Удалить</button>
    <button id=refreshVoices>↻</button>
  </div>
  <div id=recommend class=mut style="margin-top:8px"></div>
  <div style="margin-top:10px" class=mut>Добавить новый голос: введи имя и запиши ≈8 сек речи.</div>
  <div style="margin-top:6px;display:flex;gap:8px;align-items:center;flex-wrap:wrap">
    <input id=voiceName placeholder="имя голоса (напр. Мой голос)" style="padding:10px;border-radius:9px;background:#0b0e13;color:#e6edf3;border:1px solid #222a35;font-size:15px">
    <button class=rec id=refBtn>● Записать и сохранить (8с)</button>
    <span id=refStatus class=mut></span>
  </div>
</div>

<div class=card>
  <b>2. Поток</b>
  <div style="margin-top:8px">
    Исходный: <select id=src><option value=ru selected>Русский</option><option value=en>English</option></select>
    → Перевод: <select id=lang>{{OPTS}}</select>
  </div>
  <div class=lvl><i id=lvl></i></div>
  <div style="margin-top:8px">
    <button class=go id=startBtn disabled>▶ Старт потока</button>
    <button class=stop id=stopBtn disabled>■ Стоп</button>
    <span class=mut><span id=dot></span><span id=state>остановлено</span></span>
  </div>
  <div class=mut style=margin-top:6px id=hint>Сначала запиши образец голоса.</div>
</div>

<div class=card>
  <b>Лента</b>
  <div class=log id=log></div>
</div>

<audio id=player></audio>
<script>
const $=id=>document.getElementById(id);
const SID=(Math.random().toString(36).slice(2)+Date.now()).replace(/[^a-z0-9]/g,'').slice(0,24);
let stream,ac,analyser,data,recording=false,seg,segChunks=[],playQ=[],playing=false,running=false;
let speaking=false,silenceMs=0,segMs=0,hadSpeech=false,lastT=0;

async function mic(){ if(stream)return stream;
  stream=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true}});
  ac=new (window.AudioContext||window.webkitAudioContext)();
  const s=ac.createMediaStreamSource(stream); analyser=ac.createAnalyser(); analyser.fftSize=1024;
  data=new Uint8Array(analyser.fftSize); s.connect(analyser); return stream; }

function rms(){ analyser.getByteTimeDomainData(data); let s=0;
  for(let i=0;i<data.length;i++){const v=(data[i]-128)/128; s+=v*v;} return Math.sqrt(s/data.length); }

// ---- voice library ----
let RECO={}, META={};
const gi=g=>g==='м'?'♂':(g==='ж'?'♀':'');
function vlabel(v){ const m=META[v]; return m?(v+' · '+m[1]+' '+gi(m[0])):v; }
async function loadVoices(sel){ try{ const d=await (await fetch('voices')).json();
  RECO=d.recommend||{}; META=d.meta||{};
  const cur=sel||$('voiceSel').value; $('voiceSel').innerHTML='';
  if((d.voices||[]).length){ const g=document.createElement('optgroup'); g.label='🎤 Мои голоса';
    d.voices.forEach(v=>{const o=document.createElement('option');o.value=v;o.textContent=v;g.appendChild(o);}); $('voiceSel').appendChild(g); }
  if((d.presets||[]).length){ const g=document.createElement('optgroup'); g.label='⭐ Предустановленные';
    d.presets.forEach(v=>{const o=document.createElement('option');o.value=v;o.textContent=vlabel(v);g.appendChild(o);}); $('voiceSel').appendChild(g); }
  const all=[...(d.voices||[]),...(d.presets||[])];
  if(cur&&all.includes(cur)) $('voiceSel').value=cur;
  $('startBtn').disabled=false;
  $('hint').textContent='Выбери голос (свой или предустановленный) и жми «Старт потока».';
  renderReco();
  }catch(e){} }
function renderReco(){ const lang=$('lang').value, list=RECO[lang]||[];
  if(!list.length){ $('recommend').innerHTML=''; return; }
  $('recommend').innerHTML='★ Лучшие для «'+$('lang').selectedOptions[0].textContent+'»: '+
    list.map(v=>'<a href="#" data-v="'+v+'" style="color:#4f8cff;text-decoration:none">'+vlabel(v)+'</a>').join(', ')+
    ' <span style="opacity:.7">(подбор по акценту; все голоса говорят на всех языках)</span>';
  $('recommend').querySelectorAll('a').forEach(a=>a.onclick=e=>{e.preventDefault();$('voiceSel').value=a.dataset.v;});
}
async function preview(){ const v=$('voiceSel').value; if(!v)return;
  $('previewBtn').disabled=true; const old=$('previewBtn').textContent; $('previewBtn').textContent='…';
  try{ const fd=new FormData(); fd.append('voice',v); fd.append('lang',$('lang').value);
    const r=await fetch('preview',{method:'POST',body:fd});
    if(r.ok){ const b=await r.blob(); const p=$('player'); p.src=URL.createObjectURL(b); p.play(); }
  }catch(e){} $('previewBtn').disabled=false; $('previewBtn').textContent=old;
}
$('previewBtn').onclick=preview;
$('lang').onchange=renderReco;
loadVoices();
$('refreshVoices').onclick=()=>loadVoices();
$('delVoice').onclick=async()=>{ const v=$('voiceSel').value; if(!v)return;
  await fetch('voices/'+encodeURIComponent(v),{method:'DELETE'}); loadVoices(); };

// ---- record + save a NAMED voice (8s) ----
$('refBtn').onclick=async()=>{
  const name=($('voiceName').value||'').trim(); if(!name){$('refStatus').textContent='введи имя голоса';return;}
  try{await mic();}catch(e){$('refStatus').textContent='нет микрофона: '+e;return;}
  $('refBtn').disabled=true; $('refStatus').textContent='запись… говори ≈8 сек';
  const mt=MediaRecorder.isTypeSupported('audio/webm;codecs=opus')?'audio/webm;codecs=opus':'audio/webm';
  const r=new MediaRecorder(stream,{mimeType:mt}); const ch=[];
  r.ondataavailable=e=>{if(e.data.size)ch.push(e.data)};
  r.onstop=async()=>{ const b=new Blob(ch,{type:'audio/webm'}); const fd=new FormData();
    fd.append('audio',b,'voice.webm'); fd.append('name',name);
    $('refStatus').textContent='сохраняю голос…';
    try{const x=await fetch('voices',{method:'POST',body:fd});
      if(x.ok){ $('refStatus').textContent='голос «'+name+'» сохранён ✓'; $('voiceName').value='';
        await loadVoices(name); }
      else $('refStatus').textContent='ошибка: '+x.status; }
    catch(e){$('refStatus').textContent='сбой: '+e;}
    $('refBtn').disabled=false; };
  r.start(); setTimeout(()=>r.stop(),8000);
};

// ---- streaming loop (VAD-segmented utterances) ----
function newSeg(){ const mt=MediaRecorder.isTypeSupported('audio/webm;codecs=opus')?'audio/webm;codecs=opus':'audio/webm';
  seg=new MediaRecorder(stream,{mimeType:mt}); segChunks=[];
  seg.ondataavailable=e=>{if(e.data.size)segChunks.push(e.data)};
  seg.onstop=()=>{ const b=new Blob(segChunks,{type:'audio/webm'});
    if(hadSpeech && b.size>9000) sendUtt(b);   // skip tiny/silent fragments client-side
    if(running) newSeg(); };
  seg.start(); segMs=0; hadSpeech=false; }

function loop(ts){ if(!running)return;
  const dt=lastT?ts-lastT:16; lastT=ts; segMs+=dt;
  const level=rms(); $('lvl').style.width=Math.min(100,level*300)+'%';
  const SPEAK=0.025, HANG=450, MAXSEG=8000;   // shorter silence wait = lower latency
  if(level>SPEAK){ speaking=true; silenceMs=0; hadSpeech=true; }
  else if(speaking){ silenceMs+=dt; if(silenceMs>HANG){ speaking=false; cut(); } }
  if(segMs>MAXSEG && hadSpeech) cut();
  requestAnimationFrame(loop);
}
function cut(){ if(seg&&seg.state==='recording') seg.stop(); }   // onstop sends + starts next

async function sendUtt(b){
  addLog('…', '', true);
  const fd=new FormData(); fd.append('audio',b,'u.webm'); fd.append('sid',SID);
  fd.append('target_lang',$('lang').value); fd.append('source_lang',$('src').value);
  fd.append('voice',$('voiceSel').value||'');
  try{
    const r=await fetch('chunk',{method:'POST',body:fd});
    const ct=r.headers.get('content-type')||'';
    if(!r.ok || ct.includes('application/json')){ dropPending(); return; }  // silence/skip — no error row
    const s=decodeURIComponent(r.headers.get('X-Source')||''), t=decodeURIComponent(r.headers.get('X-Translation')||'');
    updLog(s,t);
    const wav=await r.blob(); playQ.push(URL.createObjectURL(wav)); drain();
  }catch(e){ dropPending(); }
}
function drain(){ if(playing||!playQ.length)return; playing=true;
  const u=playQ.shift(); const p=$('player'); p.src=u; p.onended=()=>{playing=false;drain();};
  p.play().catch(()=>{playing=false;drain();}); }

let lastRow=null;
function addLog(s,t,pending){ const d=document.createElement('div'); d.className='u';
  d.innerHTML='<div class="s">'+(s||'')+'</div><div class="t">'+(t||'')+'</div>';
  $('log').prepend(d); if(pending) lastRow=d; }
function updLog(s,t){ const d=lastRow||null; if(d){ d.querySelector('.s').textContent=s; d.querySelector('.t').textContent=t; lastRow=null; }
  else addLog(s,t); }
function dropPending(){ if(lastRow){ lastRow.remove(); lastRow=null; } }   // silence/skip: no error row

$('startBtn').onclick=async()=>{ try{await mic();}catch(e){return;} if(ac.state==='suspended')ac.resume();
  running=true; $('startBtn').disabled=true; $('stopBtn').disabled=false; $('refBtn').disabled=true;
  $('dot').className='on'; $('state').textContent='слушаю…'; lastT=0; newSeg(); requestAnimationFrame(loop); };
$('stopBtn').onclick=()=>{ running=false; cut(); $('startBtn').disabled=false; $('stopBtn').disabled=false;
  $('refBtn').disabled=false; $('dot').className=''; $('state').textContent='остановлено'; };
</script></body></html>"""
