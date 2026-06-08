"""Webcam dub tool: capture audio+video -> translate -> voice-clone -> lip-sync.

GET  /        -> webcam capture page
POST /run     -> multipart (video blob + target_lang) -> dubbed mp4

Near-live (offline) on a single T4: record a short clip, get a dubbed,
lip-synced video back in your own voice in the chosen language.
Reuses on-box services: whisper-xtts STT+XTTS (:8000), Ollama translate (:11434),
avatar-muse MuseTalk lip-sync (:8100).
"""
from __future__ import annotations

import os
import subprocess
import tempfile

import httpx
from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

KEY = os.environ.get("TTS_API_KEY", "")
TTS = "http://127.0.0.1:8000"
OLLAMA = "http://127.0.0.1:11434"
AVATAR = "http://127.0.0.1:8100"

# XTTS-supported languages (code -> human name for the translation prompt)
LANGS = {
    "en": "English", "ru": "Russian", "es": "Spanish", "fr": "French",
    "de": "German", "it": "Italian", "pt": "Portuguese", "pl": "Polish",
    "tr": "Turkish", "nl": "Dutch", "cs": "Czech", "ar": "Arabic",
    "zh-cn": "Chinese", "ja": "Japanese", "hu": "Hungarian", "ko": "Korean",
}

app = FastAPI(title="dub-web", version="0.1.0")


@app.get("/", response_class=HTMLResponse)
def index():
    opts = "".join(f'<option value="{c}"{" selected" if c=="en" else ""}>{n}</option>'
                   for c, n in LANGS.items())
    return PAGE.replace("{{OPTS}}", opts)


def _hdr():
    return {"Authorization": f"Bearer {KEY}"} if KEY else {}


@app.post("/run")
async def run(video: UploadFile = File(...), target_lang: str = Form("en")):
    if target_lang not in LANGS:
        raise HTTPException(400, "unsupported language")
    work = tempfile.mkdtemp(prefix="dub_")
    src = os.path.join(work, "src")  # raw webcam blob (webm/mp4)
    with open(src, "wb") as f:
        f.write(await video.read())

    wav = os.path.join(work, "speech.wav")
    mp4 = os.path.join(work, "face.mp4")
    # extract 16k mono wav (STT + voice ref) and a clean H264 mp4 (for MuseTalk)
    subprocess.run(["ffmpeg", "-y", "-i", src, "-vn", "-ac", "1", "-ar", "16000", wav],
                   capture_output=True)
    subprocess.run(["ffmpeg", "-y", "-i", src, "-an", "-r", "25", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", mp4], capture_output=True)
    if not os.path.exists(wav) or os.path.getsize(wav) < 1000:
        raise HTTPException(400, "no audio captured")

    with httpx.Client(timeout=1200) as c:
        # 1) STT (ru)
        with open(wav, "rb") as fh:
            r = c.post(f"{TTS}/v1/audio/transcriptions", headers=_hdr(),
                       files={"file": ("s.wav", fh, "audio/wav")}, data={"language": "ru"})
        r.raise_for_status()
        ru = r.json().get("text", "").strip()
        if not ru:
            raise HTTPException(422, "no speech recognized")

        # 2) translate (Ollama, free VRAM after)
        prompt = (f"Translate the following text to {LANGS[target_lang]}. "
                  f"Output ONLY the translation, no notes:\n\n{ru}")
        r = c.post(f"{OLLAMA}/api/generate",
                   json={"model": "qwen2.5vl:7b", "prompt": prompt, "stream": False, "keep_alive": 0})
        r.raise_for_status()
        tr = r.json().get("response", "").strip()

        # 3) voice clone in target language (ref = the user's own captured voice)
        cloned = os.path.join(work, "cloned.wav")
        with open(wav, "rb") as fh:
            r = c.post(f"{TTS}/tts/clone", headers=_hdr(),
                       data={"text": tr, "language": target_lang},
                       files={"speaker_wav": ("ref.wav", fh, "audio/wav")})
        r.raise_for_status()
        with open(cloned, "wb") as out:
            out.write(r.content)

        # 4) lip-sync (MuseTalk) onto the captured face video with the dubbed audio
        with open(mp4, "rb") as vf, open(cloned, "rb") as af:
            r = c.post(f"{AVATAR}/avatar",
                       files={"video": ("face.mp4", vf, "video/mp4"),
                              "audio": ("dub.wav", af, "audio/wav")})
        r.raise_for_status()
        out_mp4 = os.path.join(work, "dubbed.mp4")
        with open(out_mp4, "wb") as out:
            out.write(r.content)

    resp = FileResponse(out_mp4, media_type="video/mp4", filename=f"dubbed_{target_lang}.mp4")
    # surface transcripts (URL-encoded) so the page can show them
    import urllib.parse
    resp.headers["X-Source-RU"] = urllib.parse.quote(ru[:500])
    resp.headers["X-Translation"] = urllib.parse.quote(tr[:500])
    resp.headers["Access-Control-Expose-Headers"] = "X-Source-RU, X-Translation"
    return resp


PAGE = """<!doctype html><html lang=ru><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>Веб-дубляж</title>
<style>
body{margin:0;background:#0e1116;color:#e6edf3;font:15px system-ui;padding:18px;max-width:900px;margin:auto}
h1{font-size:19px}.row{display:flex;gap:16px;flex-wrap:wrap}
video{width:100%;max-width:420px;background:#000;border-radius:12px;border:1px solid #222a35}
button{padding:11px 16px;border:0;border-radius:10px;font-size:15px;cursor:pointer;margin:6px 6px 6px 0}
.rec{background:#e74c3c;color:#fff}.go{background:#4f8cff;color:#fff}.stop{background:#f1c40f;color:#111}
select{padding:9px;border-radius:9px;background:#0b0e13;color:#e6edf3;border:1px solid #222a35;font-size:15px}
.card{background:#171c24;border:1px solid #222a35;border-radius:12px;padding:14px;margin-top:14px}
.mut{color:#8b97a7;font-size:13px}#status{margin-top:10px}
</style></head><body>
<h1>🎥 Веб-дубляж: камера → перевод → твой голос → липсинк</h1>
<p class=mut>Запиши короткий клип (говори по-русски, смотри в камеру). Получишь видео,
где ты говоришь на выбранном языке своим голосом с синхронными губами.
Обработка идёт несколько минут (полный липсинк на T4).</p>
<div class=row>
  <div>
    <div class=mut>Камера (запись)</div>
    <video id=preview autoplay muted playsinline></video>
  </div>
  <div>
    <div class=mut>Результат</div>
    <video id=result controls playsinline></video>
  </div>
</div>
<div class=card>
  Язык перевода: <select id=lang>{{OPTS}}</select>
  <div style="margin-top:10px">
    <button class=rec id=recBtn>● Запись</button>
    <button class=stop id=stopBtn disabled>■ Стоп</button>
    <button class=go id=goBtn disabled>Дублировать →</button>
  </div>
  <div id=status class=mut></div>
  <div id=texts class=card style="display:none">
    <div><b>Распознано (RU):</b> <span id=ru></span></div>
    <div style="margin-top:6px"><b>Перевод:</b> <span id=tr></span></div>
  </div>
</div>
<script>
let rec, chunks=[], blob=null, stream=null;
const $=id=>document.getElementById(id);
const st=m=>{$('status').textContent=m};
async function initCam(){
  stream=await navigator.mediaDevices.getUserMedia({video:{width:640,height:480},audio:true});
  $('preview').srcObject=stream;
}
initCam().catch(e=>st('Нет доступа к камере: '+e));
$('recBtn').onclick=()=>{
  chunks=[]; const mt=MediaRecorder.isTypeSupported('video/webm;codecs=vp8,opus')?'video/webm;codecs=vp8,opus':'video/webm';
  rec=new MediaRecorder(stream,{mimeType:mt});
  rec.ondataavailable=e=>{if(e.data.size)chunks.push(e.data)};
  rec.onstop=()=>{blob=new Blob(chunks,{type:'video/webm'}); $('goBtn').disabled=false; st('Записано '+(blob.size/1e6).toFixed(1)+' МБ. Жми «Дублировать».')};
  rec.start(); $('recBtn').disabled=true; $('stopBtn').disabled=false; $('goBtn').disabled=true; st('Идёт запись… говори по-русски.');
};
$('stopBtn').onclick=()=>{rec.stop(); $('recBtn').disabled=false; $('stopBtn').disabled=true;};
$('goBtn').onclick=async()=>{
  if(!blob)return; $('goBtn').disabled=true; st('Обработка (STT → перевод → голос → липсинк), это несколько минут…');
  const fd=new FormData(); fd.append('video',blob,'cam.webm'); fd.append('target_lang',$('lang').value);
  try{
    const r=await fetch('run',{method:'POST',body:fd});
    if(!r.ok){st('Ошибка: '+r.status+' '+await r.text());$('goBtn').disabled=false;return;}
    const ru=decodeURIComponent(r.headers.get('X-Source-RU')||''), tr=decodeURIComponent(r.headers.get('X-Translation')||'');
    if(ru||tr){$('texts').style.display='block';$('ru').textContent=ru;$('tr').textContent=tr;}
    const v=await r.blob(); $('result').src=URL.createObjectURL(v); $('result').play();
    st('Готово ✓'); $('goBtn').disabled=false;
  }catch(e){st('Сбой: '+e);$('goBtn').disabled=false;}
};
</script></body></html>"""
