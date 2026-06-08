"""Animate-from-text tool: text + photo -> talking video, with live progress + ETA.

Pipeline per job (single GPU, serialized):
  1) TTS: text -> voice (XTTS, :8000)
  2) free VRAM: stop whisper-xtts/avatar/ollama
  3) start wan-comfy (ComfyUI), submit Wan2.2-S2V prompt
  4) stream progress from ComfyUI WebSocket -> job state (step/total + ETA)
  5) fetch mp4, stop wan-comfy, restore services

Page polls /status/<job> for stage / step x/y / ETA / result.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import uuid
import wave

import httpx
import websocket  # websocket-client
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

KEY = os.environ.get("TTS_API_KEY", "")
TTS = "http://127.0.0.1:8000"
COMFY = "127.0.0.1:8188"
WAN_DIR = "/home/deploy/wan22-i2v"
WAN_COMPOSE = f"{WAN_DIR}/docker-compose.wan.yml"
RESULTS = "/opt/animate-web/results"
os.makedirs(RESULTS, exist_ok=True)

VOICES = ["Ana Florence", "Claribel Dervla", "Daisy Studious", "Sofia Hellen",
          "Damien Black", "Viktor Eka", "Craig Gutsy"]
PRESETS = {
    "fast":    {"w": 448, "h": 448, "steps": 8,  "label": "Быстро (~8 мин)"},
    "balanced":{"w": 512, "h": 512, "steps": 14, "label": "Сбалансировано (~18 мин)"},
    "quality": {"w": 576, "h": 576, "steps": 20, "label": "Качество (~30+ мин)"},
}

app = FastAPI(title="animate-web")
JOBS: dict = {}
LOCK = threading.Lock()


def _sh(cmd, timeout=300):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _set(job, **kw):
    JOBS[job].update(kw)


def _wait_comfy(timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if httpx.get(f"http://{COMFY}/system_stats", timeout=3).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


NEG = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
       "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
       "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景")


def build_prompt(img_name, audio_name, w, h, length, steps):
    return {
        "61": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "Wan2.2-S2V-14B-Q4_K_M.gguf"}},
        "54": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["61", 0], "shift": 8.0}},
        "62": {"class_type": "CLIPLoaderGGUF", "inputs": {"clip_name": "umt5-xxl-encoder-Q4_K_M.gguf", "type": "wan"}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["62", 0], "text": "a person speaking to the camera, natural subtle head movement, hair moves slightly, lively facial expression, photorealistic"}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["62", 0], "text": NEG}},
        "63": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
        "57": {"class_type": "AudioEncoderLoader", "inputs": {"audio_encoder_name": "wav2vec2_large_english_fp16.safetensors"}},
        "58": {"class_type": "LoadAudio", "inputs": {"audio": audio_name}},
        "56": {"class_type": "AudioEncoderEncode", "inputs": {"audio_encoder": ["57", 0], "audio": ["58", 0]}},
        "52": {"class_type": "LoadImage", "inputs": {"image": img_name}},
        "55": {"class_type": "WanSoundImageToVideo", "inputs": {"positive": ["6", 0], "negative": ["7", 0], "vae": ["63", 0], "width": w, "height": h, "length": length, "batch_size": 1, "audio_encoder_output": ["56", 0], "ref_image": ["52", 0]}},
        "3": {"class_type": "KSampler", "inputs": {"model": ["54", 0], "seed": 12345, "steps": steps, "cfg": 6.0, "sampler_name": "uni_pc", "scheduler": "simple", "positive": ["55", 0], "negative": ["55", 1], "latent_image": ["55", 2], "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["63", 0]}},
        "59": {"class_type": "CreateVideo", "inputs": {"images": ["8", 0], "fps": 16.0, "audio": ["58", 0]}},
        "60": {"class_type": "SaveVideo", "inputs": {"video": ["59", 0], "filename_prefix": "animate", "format": "mp4", "codec": "h264"}},
    }


def ws_listen(job, prompt_id):
    try:
        ws = websocket.create_connection(f"ws://{COMFY}/ws?clientId={job}", timeout=10)
    except Exception:
        return
    step_t0 = None
    while True:
        try:
            msg = ws.recv()
        except Exception:
            break
        if isinstance(msg, bytes):
            continue
        try:
            m = json.loads(msg)
        except Exception:
            continue
        t = m.get("type"); d = m.get("data", {})
        if t == "progress":
            v, mx = d.get("value", 0), d.get("max", 1)
            now = time.time()
            if step_t0 is None or v <= 1:
                step_t0 = now
            eta = None
            if v >= 1 and now > step_t0:
                eta = int((now - step_t0) / v * (mx - v))
            _set(job, stage="Генерация видео", step=v, total=mx, eta=eta)
        elif t == "executing" and d.get("node") is None and d.get("prompt_id") == prompt_id:
            break
    try:
        ws.close()
    except Exception:
        pass


def run_job(job, text, voice, photo_path, preset):
    p = PRESETS.get(preset, PRESETS["fast"])
    try:
        _set(job, stage="Озвучка текста (TTS)")
        hdr = {"Authorization": f"Bearer {KEY}"} if KEY else {}
        r = httpx.post(f"{TTS}/v1/audio/speech", headers=hdr,
                       json={"input": text, "voice": voice, "response_format": "wav"}, timeout=180)
        if r.status_code != 200:
            raise RuntimeError(f"TTS {r.status_code}: {r.text[:150]}")
        apath = os.path.join(WAN_DIR, "input", f"{job}.wav")
        with open(apath, "wb") as f:
            f.write(r.content)
        wv = wave.open(apath); dur = wv.getnframes() / wv.getframerate(); wv.close()
        length = max(25, min(int(dur * 16) // 4 * 4 + 1, 121))

        ipath = os.path.join(WAN_DIR, "input", f"{job}.png")
        _sh(["ffmpeg", "-y", "-i", photo_path, "-vf",
             f"scale={p['w']}:{p['h']}:force_original_aspect_ratio=increase,crop={p['w']}:{p['h']}", ipath])

        _set(job, stage="Освобождение GPU")
        _sh(["docker", "stop", "whisper-xtts-server", "avatar-muse"], timeout=120)
        _sh(["docker", "exec", "ollama", "ollama", "stop", "qwen2.5vl:7b"], timeout=30)

        _set(job, stage="Загрузка модели Wan2.2-S2V")
        _sh(["docker", "compose", "-f", WAN_COMPOSE, "up", "-d"], timeout=120)
        if not _wait_comfy():
            raise RuntimeError("ComfyUI не стартовал")

        prompt = build_prompt(f"{job}.png", f"{job}.wav", p["w"], p["h"], length, p["steps"])
        sub = httpx.post(f"http://{COMFY}/prompt", json={"prompt": prompt, "client_id": job}, timeout=30).json()
        pid = sub.get("prompt_id")
        if not pid:
            raise RuntimeError(f"submit failed: {sub}")
        _set(job, stage="Генерация видео", step=0, total=p["steps"], eta=None)
        threading.Thread(target=ws_listen, args=(job, pid), daemon=True).start()

        out = None; t0 = time.time()
        while time.time() - t0 < 3600:
            h = httpx.get(f"http://{COMFY}/history/{pid}", timeout=10).json()
            if h:
                k = list(h)[0]; stt = h[k]["status"]["status_str"]
                if stt == "error":
                    msgs = h[k]["status"].get("messages", [])
                    err = next((mm[1] for mm in msgs if mm[0] == "execution_error"), {})
                    raise RuntimeError(f"render: {err.get('exception_message','?')[:160]}")
                vids = h[k].get("outputs", {}).get("60", {}).get("images", [])
                if vids:
                    out = vids[0]["filename"]; break
            time.sleep(4)
        if not out:
            raise RuntimeError("нет результата")
        _set(job, stage="Финализация")
        _sh(["cp", os.path.join(WAN_DIR, "output", out), os.path.join(RESULTS, f"{job}.mp4")])
        _set(job, stage="Готово", status="done", step=p["steps"], total=p["steps"], eta=0)
    except Exception as e:
        _set(job, stage="Ошибка", status="error", error=str(e)[:400])
    finally:
        _sh(["docker", "compose", "-f", WAN_COMPOSE, "stop"], timeout=120)
        _sh(["docker", "start", "whisper-xtts-server", "avatar-muse"], timeout=120)
        for fn in (f"{job}.wav", f"{job}.png"):
            try: os.remove(os.path.join(WAN_DIR, "input", fn))
            except Exception: pass
        try: os.remove(photo_path)
        except Exception: pass
        if LOCK.locked():
            LOCK.release()


@app.post("/run")
async def run(text: str = Form(...), photo: UploadFile = File(...),
              voice: str = Form("Ana Florence"), preset: str = Form("fast")):
    if not LOCK.acquire(blocking=False):
        raise HTTPException(409, "GPU занят другой генерацией — подождите")
    job = uuid.uuid4().hex[:8]
    photo_path = f"/tmp/animate_{job}_src"
    with open(photo_path, "wb") as f:
        f.write(await photo.read())
    JOBS[job] = {"stage": "В очереди", "status": "running", "step": 0, "total": 0,
                 "eta": None, "error": None, "t0": time.time()}
    threading.Thread(target=run_job, args=(job, text, voice, photo_path, preset), daemon=True).start()
    return {"job_id": job}


@app.get("/status/{job}")
def status(job: str):
    if job not in JOBS:
        raise HTTPException(404, "unknown job")
    j = dict(JOBS[job]); j["elapsed"] = int(time.time() - j["t0"])
    return JSONResponse(j)


@app.get("/result/{job}")
def result(job: str):
    p = os.path.join(RESULTS, f"{job}.mp4")
    if not os.path.exists(p):
        raise HTTPException(404, "not ready")
    return FileResponse(p, media_type="video/mp4", filename=f"animate_{job}.mp4")


@app.get("/", response_class=HTMLResponse)
def index():
    vopts = "".join(f"<option>{v}</option>" for v in VOICES)
    popts = "".join(f'<option value="{k}"{" selected" if k=="fast" else ""}>{v["label"]}</option>' for k, v in PRESETS.items())
    return PAGE.replace("{{VOICES}}", vopts).replace("{{PRESETS}}", popts)


PAGE = """<!doctype html><html lang=ru><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>Текст+Фото -> Видео</title>
<style>
body{margin:0;background:#0e1116;color:#e6edf3;font:15px system-ui;padding:18px;max-width:820px;margin:auto}
h1{font-size:19px}textarea,input,select{width:100%;box-sizing:border-box;background:#0b0e13;color:#e6edf3;border:1px solid #222a35;border-radius:9px;padding:10px;font-size:15px;margin:6px 0}
textarea{height:90px}button{background:#4f8cff;color:#fff;border:0;border-radius:10px;padding:12px 18px;font-size:16px;cursor:pointer}
button:disabled{opacity:.5}.card{background:#171c24;border:1px solid #222a35;border-radius:12px;padding:16px;margin-top:14px}
.bar{height:14px;background:#0b0e13;border-radius:7px;overflow:hidden;margin:10px 0}.bar>i{display:block;height:100%;width:0;background:linear-gradient(90deg,#4f8cff,#7aa7ff);transition:width .4s}
.mut{color:#8b97a7;font-size:13px}video{width:100%;max-width:480px;border-radius:12px;background:#000;margin-top:10px}
.row{display:flex;gap:12px;flex-wrap:wrap}.row>div{flex:1;min-width:170px}
</style></head><body>
<h1>🖼️ -> 🎬 Текст + Фото -> говорящее видео</h1>
<p class=mut>Загрузи фото (фронтальное лицо) и впиши текст — персонаж оживёт и проговорит его.
Генерация несколько минут на T4; прогресс и оставшееся время — ниже.</p>
<div class=card>
  <textarea id=text placeholder="Текст, который произнесёт персонаж..."></textarea>
  <div class=row>
    <div><div class=mut>Фото</div><input id=photo type=file accept="image/*"></div>
    <div><div class=mut>Голос</div><select id=voice>{{VOICES}}</select></div>
    <div><div class=mut>Режим</div><select id=preset>{{PRESETS}}</select></div>
  </div>
  <button id=go>Сгенерировать видео -></button>
  <div id=prog style="display:none">
    <div class=bar><i id=fill></i></div>
    <div id=stage class=mut></div>
  </div>
  <video id=result controls style="display:none"></video>
</div>
<script>
const $=id=>document.getElementById(id);let timer=null;
function fmt(s){if(s==null)return '—';if(s<60)return s+' с';return Math.floor(s/60)+' мин '+(s%60)+' с';}
$('go').onclick=async()=>{
  const t=$('text').value.trim(); const f=$('photo').files[0];
  if(!t||!f){alert('Нужны текст и фото');return;}
  $('go').disabled=true; $('prog').style.display='block'; $('result').style.display='none';
  $('stage').textContent='Запуск...'; $('fill').style.width='3%';
  const fd=new FormData(); fd.append('text',t); fd.append('photo',f); fd.append('voice',$('voice').value); fd.append('preset',$('preset').value);
  let r=await fetch('run',{method:'POST',body:fd});
  if(!r.ok){$('stage').textContent='Ошибка: '+r.status+' '+await r.text();$('go').disabled=false;return;}
  const {job_id}=await r.json();
  timer=setInterval(async()=>{
    const s=await (await fetch('status/'+job_id)).json();
    let pct=3; if(s.total>0) pct=Math.max(3,Math.round(s.step/s.total*100)); if(s.status==='done')pct=100;
    $('fill').style.width=pct+'%';
    let line=s.stage; if(s.total>0&&s.stage==='Генерация видео')line+=` · шаг ${s.step}/${s.total} · осталось ~${fmt(s.eta)}`;
    line+=` · прошло ${fmt(s.elapsed)}`;
    $('stage').textContent=line;
    if(s.status==='done'){clearInterval(timer);$('go').disabled=false;$('result').src='result/'+job_id+'?t='+Date.now();$('result').style.display='block';$('result').play();}
    if(s.status==='error'){clearInterval(timer);$('go').disabled=false;$('stage').textContent='Ошибка: '+s.error;}
  },2000);
};
</script></body></html>"""
