"""Multi-tenant API gateway for the AI platform.

Fronts the GPU broker (and, over time, the other services) with:
  * API-KEY authentication (X-API-Key or Authorization: Bearer)
  * per-key USAGE accounting (requests + weighted units, by service)
  * QUEUE visibility in every job response (your position, how many ahead, ETA)
  * admin endpoints to issue / revoke keys and read usage

Design: this is the REST core. MCP consumers (via mcpo) call the SAME endpoints with
the SAME API key, so auth + metering live here once. It never touches 1C.
"""
from __future__ import annotations

import json
import os
import secrets
import threading
import time

import httpx
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

BROKER = "http://127.0.0.1:8092"
OLLAMA = "http://127.0.0.1:11434"
ADMIN_KEY = os.environ.get("API_ADMIN_KEY", "")
# LLMs exposed through the gateway (allowlist). llama3.2 = light/fast (translate, chat);
# qwen2.5vl = heavier multimodal — note it competes for VRAM with the broker's resident model.
LLM_MODELS = {"llama3.2:3b": "fast text LLM (translate/chat)",
              "qwen2.5vl:7b": "multimodal vision-language (heavier)"}
DEFAULT_LLM = "llama3.2:3b"
DATA = "/opt/api-gateway"
KEYS_F = os.path.join(DATA, "keys.json")
USAGE_F = os.path.join(DATA, "usage.json")
JOBMAP_F = os.path.join(DATA, "jobmap.json")
EVENTS_F = os.path.join(DATA, "events.jsonl")    # append-only billing audit log
_lock = threading.Lock()

VOICESTREAM = "http://127.0.0.1:8202"
DUB = "http://127.0.0.1:8200"
ANIMATE = "http://127.0.0.1:8201"

# weighted "units" per service call (the billing unit). Price set per unit below.
UNITS = {"project_3d": 10, "project_3d_render": 25, "render": 3, "furnish": 3,
         "interior": 12, "reference": 4, "voice_chunk": 1, "voice_translate": 1,
         "avatar": 20, "dub": 20, "llm": 1, "translate": 1}
PRICE_PER_UNIT = float(os.environ.get("API_PRICE_PER_UNIT", "1.0"))   # money per 1 unit (configurable)
CURRENCY = os.environ.get("API_CURRENCY", "₽")

app = FastAPI(title="ai-api-gateway", version="0.1.0")


def _load(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def _save(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=0)
    os.replace(tmp, path)


_RATE = {}   # in-memory: key -> [timestamps] for rate limiting


def _auth(x_api_key, authorization):
    """Resolve+validate the key, enforce per-key rate limit. Returns (key, rec)."""
    key = x_api_key
    if not key and authorization and authorization.lower().startswith("bearer "):
        key = authorization[7:].strip()
    if not key:
        raise HTTPException(401, "missing API key (X-API-Key or Bearer)")
    with _lock:
        rec = _load(KEYS_F, {}).get(key)
    if not rec or not rec.get("active", True):
        raise HTTPException(403, "invalid or revoked API key")
    rpm = rec.get("rate_per_min")
    if rpm:
        now = time.time()
        win = [t for t in _RATE.get(key, []) if now - t < 60]
        if len(win) >= int(rpm):
            raise HTTPException(429, f"rate limit: {rpm}/min exceeded")
        win.append(now)
        _RATE[key] = win
    return key


def _authq(x_api_key, authorization):
    """auth + MONTHLY QUOTA enforcement (for billable endpoints)."""
    key = _auth(x_api_key, authorization)
    with _lock:
        rec = _load(KEYS_F, {}).get(key, {})
        quota = rec.get("quota_units")
        u = _load(USAGE_F, {}).get(key, {})
        used = u.get("month_units", 0) if u.get("month") == _month() else 0
    if quota and used >= int(quota):
        raise HTTPException(402, f"monthly quota reached ({used}/{quota} units)")
    return key


def _month():
    return time.strftime("%Y-%m", time.gmtime())


def _meter(key, service, units=None, tokens=0, job_id=None):
    """Record one billable call: update per-key rollup (lifetime + current month) AND append
    an audit event (events.jsonl) so billing is fully reconstructable per period."""
    n_units = units if units is not None else UNITS.get(service, 1)
    ts = int(time.time())
    with _lock:
        usage = _load(USAGE_F, {})
        owner = _load(KEYS_F, {}).get(key, {}).get("owner", "")
        u = usage.setdefault(key, {"requests": 0, "units": 0, "tokens": 0, "by_service": {},
                                   "units_by_service": {}, "last_ts": 0, "month": _month(), "month_units": 0})
        if u.get("month") != _month():                 # new month -> reset the monthly counter
            u["month"] = _month()
            u["month_units"] = 0
        u["requests"] += 1
        u["units"] += n_units
        u["month_units"] = u.get("month_units", 0) + n_units
        u["tokens"] = u.get("tokens", 0) + int(tokens)
        u["by_service"][service] = u["by_service"].get(service, 0) + 1
        u.setdefault("units_by_service", {})[service] = u.get("units_by_service", {}).get(service, 0) + n_units
        u["last_ts"] = ts
        _save(USAGE_F, usage)
        try:
            with open(EVENTS_F, "a") as f:
                f.write(json.dumps({"ts": ts, "key": key, "owner": owner, "service": service,
                                    "units": n_units, "tokens": int(tokens), "job_id": job_id},
                                   ensure_ascii=False) + "\n")
        except Exception:
            pass


def _admin(authorization, x_admin_key):
    tok = x_admin_key or (authorization[7:].strip() if authorization and authorization.lower().startswith("bearer ") else "")
    if not ADMIN_KEY or tok != ADMIN_KEY:
        raise HTTPException(403, "admin only")


def _queue_of(jid):
    """Pull this job's queue position / ETA / status from the broker state."""
    try:
        st = httpx.get(f"{BROKER}/api/state", timeout=8).json()
    except Exception:
        return {}
    counts = st.get("counts", {})
    for j in st.get("jobs", []):
        if j.get("id") == jid:
            q = {"status": j.get("status"), "total_in_queue": counts.get("queued", 0)}
            if j.get("status") == "queued":
                q.update({"position": j.get("position"), "ahead": (j.get("position") or 1) - 1,
                          "eta_seconds": j.get("eta")})
            elif j.get("status") == "running":
                q.update({"position": 0, "ahead": 0, "elapsed_seconds": j.get("elapsed"),
                          "eta_seconds": max((j.get("est") or 0) - (j.get("elapsed") or 0), 0),
                          "step": j.get("step")})
            return q
    return {"status": "unknown"}


# --------------------------------------------------------------------------- #
# Public API (API-key auth)                                                   #
# --------------------------------------------------------------------------- #
@app.get("/v1/ping")
def ping(x_api_key: str = Header(None), authorization: str = Header(None)):
    key = _auth(x_api_key, authorization)
    with _lock:
        owner = _load(KEYS_F, {}).get(key, {}).get("owner", "")
    return {"ok": True, "owner": owner}


@app.post("/v1/3d/project")
async def submit_project(files: list[UploadFile] = File(...), description: str = Form(""),
                         render: bool = Form(False),
                         x_api_key: str = Header(None), authorization: str = Header(None)):
    """Submit a floor-plan → 3D project. Returns job id + your queue position/ETA."""
    key = _authq(x_api_key, authorization)
    data = {"description": description, "render": "true" if render else "false"}
    fs = [("files", (f.filename or "in", await f.read(), f.content_type or "application/octet-stream")) for f in files]
    try:
        r = httpx.post(f"{BROKER}/api/project", data=data, files=fs, timeout=60)
        r.raise_for_status()
        jid = r.json()["id"]
    except Exception as exc:
        raise HTTPException(502, f"broker error: {exc}")
    _meter(key, "project_3d_render" if render else "project_3d")
    with _lock:
        jm = _load(JOBMAP_F, {})
        jm[jid] = {"key": key, "ts": int(time.time()), "service": "project_3d"}
        _save(JOBMAP_F, jm)
    return {"job_id": jid, "queue": _queue_of(jid)}


@app.get("/v1/jobs/{jid}")
def job_status(jid: str, x_api_key: str = Header(None), authorization: str = Header(None)):
    key = _auth(x_api_key, authorization)
    with _lock:
        owner_key = _load(JOBMAP_F, {}).get(jid, {}).get("key")
    if owner_key and owner_key != key:
        raise HTTPException(403, "this job belongs to another key")
    q = _queue_of(jid)
    return {"job_id": jid, "queue": q,
            "result_url": f"/v1/jobs/{jid}/result" if q.get("status") == "done" else None}


@app.get("/v1/jobs/{jid}/result")
def job_result(jid: str, x_api_key: str = Header(None), authorization: str = Header(None)):
    key = _auth(x_api_key, authorization)
    with _lock:
        owner_key = _load(JOBMAP_F, {}).get(jid, {}).get("key")
    if owner_key and owner_key != key:
        raise HTTPException(403, "this job belongs to another key")
    try:
        r = httpx.get(f"{BROKER}/api/jobs/{jid}", timeout=15)
        return Response(content=r.content, media_type=r.headers.get("content-type", "application/json"),
                        status_code=r.status_code)
    except Exception as exc:
        raise HTTPException(502, f"broker error: {exc}")


@app.get("/v1/usage")
def my_usage(x_api_key: str = Header(None), authorization: str = Header(None)):
    key = _auth(x_api_key, authorization)
    with _lock:
        u = _load(USAGE_F, {}).get(key, {"requests": 0, "units": 0, "by_service": {}})
        owner = _load(KEYS_F, {}).get(key, {}).get("owner", "")
    return {"owner": owner, "usage": u}


# --------------------------------------------------------------------------- #
# LLM / translation (Ollama: Qwen, Llama)                                      #
# --------------------------------------------------------------------------- #
@app.get("/v1/models")
def models(x_api_key: str = Header(None), authorization: str = Header(None)):
    _auth(x_api_key, authorization)
    return {"models": [{"id": m, "description": d} for m, d in LLM_MODELS.items()],
            "default": DEFAULT_LLM}


@app.post("/v1/llm/chat")
async def llm_chat(payload: dict, x_api_key: str = Header(None), authorization: str = Header(None)):
    """Chat with an on-box LLM. Body: {model?, messages:[{role,content}] | prompt, temperature?}.
    Metered per request + by token count. Returns the model reply + token usage."""
    key = _authq(x_api_key, authorization)
    model = payload.get("model") or DEFAULT_LLM
    if model not in LLM_MODELS:
        raise HTTPException(400, f"model not allowed; use one of {list(LLM_MODELS)}")
    opts = {"temperature": float(payload.get("temperature", 0.3))}
    try:
        if payload.get("messages"):
            r = httpx.post(f"{OLLAMA}/api/chat",
                           json={"model": model, "messages": payload["messages"],
                                 "stream": False, "keep_alive": "5m", "options": opts}, timeout=300)
            r.raise_for_status()
            j = r.json()
            text = (j.get("message") or {}).get("content", "")
        else:
            r = httpx.post(f"{OLLAMA}/api/generate",
                           json={"model": model, "prompt": payload.get("prompt", ""),
                                 "stream": False, "keep_alive": "5m", "options": opts}, timeout=300)
            r.raise_for_status()
            j = r.json()
            text = j.get("response", "")
    except Exception as exc:
        raise HTTPException(502, f"llm error: {exc}")
    toks = int(j.get("prompt_eval_count", 0)) + int(j.get("eval_count", 0))
    _meter(key, "llm", units=max(1, toks // 100), tokens=toks)
    return {"model": model, "text": text,
            "tokens": {"prompt": j.get("prompt_eval_count", 0), "completion": j.get("eval_count", 0)}}


@app.post("/v1/translate")
def translate(payload: dict, x_api_key: str = Header(None), authorization: str = Header(None)):
    """Translate text. Body: {text, to (lang name/code), from?}. Uses the fast LLM."""
    key = _authq(x_api_key, authorization)
    text = (payload.get("text") or "").strip()
    to = (payload.get("to") or "English").strip()
    model = _pick_model(payload)
    if not text:
        raise HTTPException(400, "text required")
    want_detect = bool(payload.get("detect") or payload.get("skip_same"))
    detected, dtoks = (None, 0)
    if want_detect:
        try:
            detected, dtoks = _detect_lang(text)
        except Exception:
            detected = None
    # skip_same: if already in the target language, don't re-translate (saves cost)
    if payload.get("skip_same") and detected and detected.lower() in to.lower():
        _meter(key, "translate", units=1, tokens=dtoks)
        return {"translation": text, "to": to, "detected_source": detected, "skipped": True}
    try:
        r = httpx.post(f"{OLLAMA}/api/generate",
                       json={"model": model,
                             "prompt": f"Translate the following text to {to}. Output ONLY the translation, no notes:\n\n{text}",
                             "stream": False, "keep_alive": "5m", "options": {"temperature": 0.2}}, timeout=300)
        r.raise_for_status()
        j = r.json()
    except Exception as exc:
        raise HTTPException(502, f"translate error: {exc}")
    toks = int(j.get("prompt_eval_count", 0)) + int(j.get("eval_count", 0)) + dtoks
    _meter(key, "translate", units=max(1, toks // 100), tokens=toks)
    res = {"translation": (j.get("response") or "").strip(), "to": to, "model": model,
           "tokens": {"prompt": j.get("prompt_eval_count", 0), "completion": j.get("eval_count", 0)}}
    if want_detect:
        res["detected_source"] = detected
    return res


@app.post("/v1/translate/batch")
def translate_batch(payload: dict, x_api_key: str = Header(None), authorization: str = Header(None)):
    """Translate MANY strings in one call — for migrating data from another system.
    Body: {texts:[...], to, from?}. Returns {translations:[...]} (same order, "" for blanks).
    Metered per non-empty item + by total tokens."""
    key = _authq(x_api_key, authorization)
    texts = payload.get("texts")
    to = (payload.get("to") or "English").strip()
    model = _pick_model(payload)
    if not isinstance(texts, list) or not texts:
        raise HTTPException(400, "texts (non-empty list) required")
    if len(texts) > 200:
        raise HTTPException(400, "max 200 items per batch")
    out, total_toks, billable = [], 0, 0
    for t in texts:
        t = ("" if t is None else str(t)).strip()
        if not t:
            out.append("")
            continue
        try:
            prompt = f"Translate the following text to {to}. Output ONLY the translation, no notes:\n\n{t}"
            r = httpx.post(f"{OLLAMA}/api/generate",
                           json={"model": model, "prompt": prompt, "stream": False,
                                 "keep_alive": "5m", "options": {"temperature": 0.2}}, timeout=300)
            r.raise_for_status()
            j = r.json()
            out.append((j.get("response") or "").strip())
            total_toks += int(j.get("prompt_eval_count", 0)) + int(j.get("eval_count", 0))
            billable += 1
        except Exception:
            out.append(None)            # null = this item failed; others still returned
    if billable:
        _meter(key, "translate", units=max(billable, total_toks // 100), tokens=total_toks)
    return {"to": to, "count": len(out), "translated": billable,
            "translations": out, "tokens": total_toks}


def _detect_lang(text):
    """Detect the source language of a text via the fast LLM. Returns (name, tokens)."""
    prompt = ("Identify the language of the text below. Reply with ONLY the English name of the "
              "language as one word (e.g. Russian, English, German, Chinese). Text:\n\n" + text[:500])
    r = httpx.post(f"{OLLAMA}/api/generate",
                   json={"model": DEFAULT_LLM, "prompt": prompt, "stream": False,
                         "keep_alive": "5m", "options": {"temperature": 0, "num_predict": 8}}, timeout=120)
    r.raise_for_status()
    j = r.json()
    lang = (j.get("response") or "").strip().strip(".").split()[0:2]
    lang = " ".join(lang) if lang else ""
    return lang, int(j.get("prompt_eval_count", 0)) + int(j.get("eval_count", 0))


@app.post("/v1/detect")
def detect(payload: dict, x_api_key: str = Header(None), authorization: str = Header(None)):
    """Detect source language. Body: {text} -> {language}; or {texts:[...]} -> {languages:[...]}."""
    key = _authq(x_api_key, authorization)
    single = payload.get("text") is not None
    texts = [payload.get("text")] if single else payload.get("texts")
    if not isinstance(texts, list) or not texts:
        raise HTTPException(400, "`text` or `texts` required")
    if len(texts) > 200:
        raise HTTPException(400, "max 200 items")
    out, toks, n = [], 0, 0
    for t in texts:
        t = ("" if t is None else str(t)).strip()
        if not t:
            out.append("")
            continue
        try:
            lang, tk = _detect_lang(t)
            out.append(lang)
            toks += tk
            n += 1
        except Exception:
            out.append(None)
    if n:
        _meter(key, "translate", units=max(1, toks // 100), tokens=toks)
    return ({"language": out[0], "tokens": toks} if single
            else {"count": len(out), "languages": out, "tokens": toks})


def _pick_model(payload):
    m = payload.get("model") or DEFAULT_LLM
    if m not in LLM_MODELS:
        raise HTTPException(400, f"model not allowed; use one of {list(LLM_MODELS)}")
    return m


def _tr_one(text, to, model=DEFAULT_LLM):
    prompt = f"Translate the following text to {to}. Output ONLY the translation, no notes:\n\n{text}"
    r = httpx.post(f"{OLLAMA}/api/generate",
                   json={"model": model, "prompt": prompt, "stream": False,
                         "keep_alive": "5m", "options": {"temperature": 0.2}}, timeout=300)
    r.raise_for_status()
    j = r.json()
    return (j.get("response") or "").strip(), int(j.get("prompt_eval_count", 0)) + int(j.get("eval_count", 0))


@app.post("/v1/translate/multi")
def translate_multi(payload: dict, x_api_key: str = Header(None), authorization: str = Header(None)):
    """Translate to MANY target languages at once (multilingual catalogs / data).
    Body: {text:"..."  OR  texts:[...], to:["English","German","zh-cn", ...]}.
      • single `text`  -> {"translations": {lang: tr, ...}}
      • list `texts`   -> {"results":[{"text": orig, "translations": {lang: tr,...}}, ...]}
    Cap: texts × langs ≤ 200. Metered per produced translation + by total tokens."""
    key = _authq(x_api_key, authorization)
    model = _pick_model(payload)
    to = payload.get("to")
    if isinstance(to, str):
        to = [to]
    if not isinstance(to, list) or not to:
        raise HTTPException(400, "`to` (list of target languages) required")
    single = payload.get("text") is not None
    texts = [payload.get("text")] if single else payload.get("texts")
    if not isinstance(texts, list) or not texts:
        raise HTTPException(400, "`text` or `texts` required")
    if len(texts) * len(to) > 200:
        raise HTTPException(400, "texts × langs must be ≤ 200")
    total_toks, billable, results = 0, 0, []
    for t in texts:
        t = ("" if t is None else str(t)).strip()
        trs = {}
        for lang in to:
            if not t:
                trs[lang] = ""
                continue
            try:
                tr, toks = _tr_one(t, lang, model)
                trs[lang] = tr
                total_toks += toks
                billable += 1
            except Exception:
                trs[lang] = None
        results.append({"text": t, "translations": trs})
    if billable:
        _meter(key, "translate", units=max(billable, total_toks // 100), tokens=total_toks)
    if single:
        return {"to": to, "model": model, "translations": results[0]["translations"], "tokens": total_toks}
    return {"to": to, "model": model, "count": len(results), "translated": billable,
            "results": results, "tokens": total_toks}


# --------------------------------------------------------------------------- #
# Voice / media services                                                       #
# --------------------------------------------------------------------------- #
@app.post("/v1/voice/translate")
async def voice_translate(audio: UploadFile = File(...), target_lang: str = Form("en"),
                          voice: str = Form(""), source_lang: str = Form("ru"),
                          x_api_key: str = Header(None), authorization: str = Header(None)):
    """One utterance: speech in -> translated speech in the chosen voice (wav out)."""
    key = _authq(x_api_key, authorization)
    fs = {"audio": (audio.filename or "a.webm", await audio.read(), audio.content_type or "audio/webm")}
    data = {"target_lang": target_lang, "voice": voice, "source_lang": source_lang, "sid": "api_" + key[-6:]}
    try:
        r = httpx.post(f"{VOICESTREAM}/chunk", data=data, files=fs, timeout=300)
    except Exception as exc:
        raise HTTPException(502, f"voice error: {exc}")
    if r.status_code == 200 and "audio" in r.headers.get("content-type", ""):
        _meter(key, "voice_translate")
    hdrs = {k: v for k, v in r.headers.items() if k.lower() in ("x-source", "x-translation")}
    hdrs["Access-Control-Expose-Headers"] = "X-Source, X-Translation"
    return Response(content=r.content, media_type=r.headers.get("content-type", "audio/wav"),
                    status_code=r.status_code, headers=hdrs)


@app.get("/v1/voices")
def voices(x_api_key: str = Header(None), authorization: str = Header(None)):
    _auth(x_api_key, authorization)
    try:
        return httpx.get(f"{VOICESTREAM}/voices", timeout=8).json()
    except Exception as exc:
        raise HTTPException(502, f"voice error: {exc}")


@app.post("/v1/3d/render")
async def submit_render(image: UploadFile = File(...), type: str = Form("render"),
                        prompt: str = Form(""), style: str = Form("scandinavian, warm wood, natural daylight"),
                        x_api_key: str = Header(None), authorization: str = Header(None)):
    """Top-down render / furnish of a single plan image. Returns job id + queue."""
    key = _authq(x_api_key, authorization)
    if type not in ("render", "furnish", "interior", "reference"):
        raise HTTPException(400, "type must be render|furnish|interior|reference")
    fs = {"image": (image.filename or "img", await image.read(), image.content_type or "image/png")}
    try:
        r = httpx.post(f"{BROKER}/api/jobs", data={"type": type, "prompt": prompt, "style": style},
                       files=fs, timeout=60)
        r.raise_for_status()
        jid = r.json()["id"]
    except Exception as exc:
        raise HTTPException(502, f"broker error: {exc}")
    _meter(key, type if type in UNITS else "render", job_id=jid)
    with _lock:
        jm = _load(JOBMAP_F, {})
        jm[jid] = {"key": key, "ts": int(time.time()), "service": type}
        _save(JOBMAP_F, jm)
    return {"job_id": jid, "queue": _queue_of(jid)}


@app.post("/v1/avatar")
async def avatar(text: str = Form(...), photo: UploadFile = File(...),
                 voice: str = Form("Ana Florence"), preset: str = Form("fast"),
                 x_api_key: str = Header(None), authorization: str = Header(None)):
    """Talking-avatar from a photo + text (async job on animate-web)."""
    key = _authq(x_api_key, authorization)
    fs = {"photo": (photo.filename or "p.png", await photo.read(), photo.content_type or "image/png")}
    try:
        r = httpx.post(f"{ANIMATE}/run", data={"text": text, "voice": voice, "preset": preset},
                       files=fs, timeout=60)
        r.raise_for_status()
    except Exception as exc:
        raise HTTPException(502, f"avatar error: {exc}")
    _meter(key, "avatar")
    return JSONResponse(r.json(), status_code=r.status_code)


@app.get("/v1/avatar/{job}/status")
def avatar_status(job: str, x_api_key: str = Header(None), authorization: str = Header(None)):
    _auth(x_api_key, authorization)
    try:
        return JSONResponse(httpx.get(f"{ANIMATE}/status/{job}", timeout=10).json())
    except Exception as exc:
        raise HTTPException(502, f"avatar error: {exc}")


@app.get("/v1/avatar/{job}/result")
def avatar_result(job: str, x_api_key: str = Header(None), authorization: str = Header(None)):
    _auth(x_api_key, authorization)
    try:
        r = httpx.get(f"{ANIMATE}/result/{job}", timeout=60)
        return Response(content=r.content, media_type=r.headers.get("content-type", "video/mp4"),
                        status_code=r.status_code)
    except Exception as exc:
        raise HTTPException(502, f"avatar error: {exc}")


@app.post("/v1/dub")
async def dub(video: UploadFile = File(...), target_lang: str = Form("en"),
              x_api_key: str = Header(None), authorization: str = Header(None)):
    """Dub a short video clip into another language in the user's voice + lip-sync (mp4 out)."""
    key = _authq(x_api_key, authorization)
    fs = {"video": (video.filename or "v.webm", await video.read(), video.content_type or "video/webm")}
    try:
        r = httpx.post(f"{DUB}/run", data={"target_lang": target_lang}, files=fs, timeout=1800)
    except Exception as exc:
        raise HTTPException(502, f"dub error: {exc}")
    if r.status_code == 200:
        _meter(key, "dub")
    return Response(content=r.content, media_type=r.headers.get("content-type", "video/mp4"),
                    status_code=r.status_code)


# --------------------------------------------------------------------------- #
# Billing                                                                      #
# --------------------------------------------------------------------------- #
def _bill_row(key, u):
    units = u.get("units", 0)
    return {"owner": _load(KEYS_F, {}).get(key, {}).get("owner", ""),
            "requests": u.get("requests", 0), "units": units, "tokens": u.get("tokens", 0),
            "by_service": u.get("by_service", {}), "units_by_service": u.get("units_by_service", {}),
            "cost": round(units * PRICE_PER_UNIT, 2), "currency": CURRENCY,
            "price_per_unit": PRICE_PER_UNIT}


@app.get("/v1/billing")
def my_billing(x_api_key: str = Header(None), authorization: str = Header(None)):
    """Caller's own bill: usage + cost."""
    key = _auth(x_api_key, authorization)
    with _lock:
        u = _load(USAGE_F, {}).get(key, {})
        return _bill_row(key, u)


# --------------------------------------------------------------------------- #
# Admin (admin-key auth)                                                       #
# --------------------------------------------------------------------------- #
@app.get("/admin/billing")
def admin_billing(since: int = 0, authorization: str = Header(None), x_admin_key: str = Header(None)):
    """Billing for ALL keys. ?since=<unix ts> aggregates the period from the event log."""
    _admin(authorization, x_admin_key)
    with _lock:
        keys = _load(KEYS_F, {})
        if since <= 0:
            usage = _load(USAGE_F, {})
            rows = [{"key": k[:10] + "…", **_bill_row(k, usage.get(k, {}))} for k in keys]
        else:
            agg = {}
            try:
                with open(EVENTS_F) as f:
                    for line in f:
                        e = json.loads(line)
                        if e["ts"] >= since:
                            a = agg.setdefault(e["key"], {"requests": 0, "units": 0, "tokens": 0,
                                                          "by_service": {}, "units_by_service": {}})
                            a["requests"] += 1
                            a["units"] += e["units"]
                            a["tokens"] += e.get("tokens", 0)
                            a["by_service"][e["service"]] = a["by_service"].get(e["service"], 0) + 1
                            a["units_by_service"][e["service"]] = a["units_by_service"].get(e["service"], 0) + e["units"]
            except FileNotFoundError:
                pass
            rows = [{"key": k[:10] + "…", **_bill_row(k, agg.get(k, {}))} for k in keys]
    total = round(sum(r["cost"] for r in rows), 2)
    return {"since": since, "currency": CURRENCY, "total_cost": total, "rows": rows}


@app.post("/admin/keys")
def create_key(owner: str = Form(...), quota_units: str = Form(""), rate_per_min: str = Form(""),
               authorization: str = Header(None), x_admin_key: str = Header(None)):
    _admin(authorization, x_admin_key)
    new = "sk_" + secrets.token_hex(20)
    with _lock:
        keys = _load(KEYS_F, {})
        keys[new] = {"owner": owner, "active": True, "created": int(time.time()),
                     "quota_units": int(quota_units) if quota_units.strip().isdigit() else None,
                     "rate_per_min": int(rate_per_min) if rate_per_min.strip().isdigit() else None}
        _save(KEYS_F, keys)
    return {"api_key": new, "owner": owner}


@app.post("/admin/keys/{key}/limits")
def set_limits(key: str, quota_units: str = Form(""), rate_per_min: str = Form(""),
               active: str = Form(""), authorization: str = Header(None), x_admin_key: str = Header(None)):
    """Set/clear monthly unit quota, per-minute rate limit, and active flag for a key.
    Empty quota/rate field = unlimited (cleared)."""
    _admin(authorization, x_admin_key)
    with _lock:
        keys = _load(KEYS_F, {})
        if key not in keys:
            raise HTTPException(404, "no such key")
        keys[key]["quota_units"] = int(quota_units) if quota_units.strip().isdigit() else None
        keys[key]["rate_per_min"] = int(rate_per_min) if rate_per_min.strip().isdigit() else None
        if active in ("true", "false"):
            keys[key]["active"] = (active == "true")
        _save(KEYS_F, keys)
    return {"ok": True, "key": key, "limits": keys[key]}


@app.get("/admin/keys")
def list_keys(authorization: str = Header(None), x_admin_key: str = Header(None)):
    _admin(authorization, x_admin_key)
    with _lock:
        keys = _load(KEYS_F, {})
        usage = _load(USAGE_F, {})
    out = []
    for k, rec in keys.items():
        u = usage.get(k, {})
        mu = u.get("month_units", 0) if u.get("month") == _month() else 0
        out.append({"key": k[:10] + "…", "full_key": k, "owner": rec.get("owner"),
                    "active": rec.get("active", True), "requests": u.get("requests", 0),
                    "units": u.get("units", 0), "month_units": mu, "tokens": u.get("tokens", 0),
                    "quota_units": rec.get("quota_units"), "rate_per_min": rec.get("rate_per_min"),
                    "by_service": u.get("by_service", {}),
                    "cost": round(u.get("units", 0) * PRICE_PER_UNIT, 2),
                    "month_cost": round(mu * PRICE_PER_UNIT, 2), "last_ts": u.get("last_ts", 0)})
    return {"keys": out, "currency": CURRENCY, "price_per_unit": PRICE_PER_UNIT, "month": _month()}


@app.post("/admin/keys/{key}/revoke")
def revoke_key(key: str, authorization: str = Header(None), x_admin_key: str = Header(None)):
    _admin(authorization, x_admin_key)
    with _lock:
        keys = _load(KEYS_F, {})
        if key in keys:
            keys[key]["active"] = False
            _save(KEYS_F, keys)
    return {"ok": True}


@app.get("/admin/broker-state")
def broker_state(authorization: str = Header(None), x_admin_key: str = Header(None)):
    """Live platform load + queue (proxied from the broker) for the admin dashboard."""
    _admin(authorization, x_admin_key)
    try:
        return JSONResponse(httpx.get(f"{BROKER}/api/state", timeout=8).json())
    except Exception as exc:
        raise HTTPException(502, f"broker error: {exc}")


@app.get("/admin/usage")
def all_usage(authorization: str = Header(None), x_admin_key: str = Header(None)):
    _admin(authorization, x_admin_key)
    with _lock:
        return {"usage": _load(USAGE_F, {}), "keys": _load(KEYS_F, {})}


@app.get("/admin/ui")
def admin_ui():
    return Response(content=ADMIN_PAGE, media_type="text/html")


@app.get("/health")
def health():
    return {"status": "ok"}


ADMIN_PAGE = r"""<!doctype html><html lang=ru><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>API · Админка</title>
<style>
:root{--bg:#0c0f14;--card:#161b22;--card2:#1c232d;--line:#2a323d;--mut:#8b97a7;--acc:#4f8cff;--ok:#2ecc71;--warn:#f1c40f;--bad:#e15b64;--txt:#e6edf3}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--txt);font:15px/1.5 -apple-system,Segoe UI,Roboto,system-ui;max-width:880px;margin:auto;padding:14px 14px 60px}
h1{font-size:20px;margin:6px 0}h3{margin:0;font-size:16px}
.mut{color:var(--mut);font-size:13px}.sml{font-size:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:16px;margin-top:14px}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
input,select{background:#0b0e13;border:1px solid var(--line);color:var(--txt);padding:11px 12px;border-radius:11px;font-size:15px;outline:none;width:100%}
input:focus,select:focus{border-color:var(--acc)}
label.fld{flex:1;min-width:130px}label.fld>span{display:block;color:var(--mut);font-size:11px;margin:0 0 3px 3px}
button{background:var(--card2);border:1px solid var(--line);color:var(--txt);padding:11px 15px;border-radius:11px;cursor:pointer;font-size:14px;font-weight:600;transition:.12s}
button:active{transform:scale(.97)}button.p{background:var(--acc);border-color:var(--acc);color:#fff}
button.d{background:#2a1a1d;border-color:#5a2a32;color:#ff9a9a}button.ghost{background:transparent}
.hd{display:flex;align-items:center;gap:10px;justify-content:space-between}
.badge{display:inline-flex;align-items:center;gap:6px;background:#0b0e13;border:1px solid var(--line);border-radius:999px;padding:4px 11px;font-size:12px}
.dot{width:9px;height:9px;border-radius:50%;background:#5b6673}.dot.on{background:var(--ok);box-shadow:0 0 8px var(--ok);animation:pulse 1.6s infinite}
@keyframes pulse{50%{opacity:.45}}
.bar{height:8px;background:#0b0e13;border-radius:6px;overflow:hidden;margin-top:5px}.bar>i{display:block;height:100%;background:linear-gradient(90deg,var(--acc),#7aa7ff);transition:width .4s}
.bar.q>i{background:linear-gradient(90deg,var(--ok),#7be0a0)}.bar.q.hot>i{background:linear-gradient(90deg,var(--warn),var(--bad))}
.kcard{background:var(--card2);border:1px solid var(--line);border-radius:14px;padding:13px;margin-top:10px}
.kcard.off{opacity:.55}
.mono{font-family:ui-monospace,Menlo,monospace;font-size:12px}
.sw{position:relative;width:46px;height:26px;flex:none}.sw input{opacity:0;width:0;height:0;position:absolute}
.sw label{position:absolute;inset:0;background:#3a434f;border-radius:999px;cursor:pointer;transition:.15s}
.sw label:before{content:"";position:absolute;width:20px;height:20px;left:3px;top:3px;background:#fff;border-radius:50%;transition:.15s}
.sw input:checked+label{background:var(--ok)}.sw input:checked+label:before{transform:translateX(20px)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.stat{background:#0b0e13;border:1px solid var(--line);border-radius:10px;padding:8px 10px}.stat b{font-size:16px}.stat .mut{font-size:11px}
.acts{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap}.acts button{flex:1;min-width:90px}
#toast{position:fixed;left:50%;bottom:22px;transform:translateX(-50%) translateY(80px);background:#1c232d;border:1px solid var(--line);padding:11px 18px;border-radius:12px;opacity:0;transition:.25s;z-index:9;max-width:90vw}
#toast.show{opacity:1;transform:translateX(-50%) translateY(0)}
.keyout{background:#0b1a10;border:1px solid #2a5a3a;border-radius:11px;padding:10px;margin-top:8px;display:none}
a{color:var(--acc)}
</style></head><body>
<div class=hd><h1>🛠️ API-шлюз · Админ</h1><span id=conn class=badge><span class=dot></span>не подключено</span></div>
<div class=mut>Ключи · лимиты · тариф · живая загрузка и очередь · биллинг</div>

<div class=card id=loginCard>
  <h3>Admin-ключ</h3>
  <div class="row" style=margin-top:8px>
    <input id=adm type=password placeholder="admin key" style=flex:1>
    <button class=p onclick=saveAdm()>Войти</button>
  </div>
  <div id=admst class="mut sml" style=margin-top:6px></div>
</div>

<div class=card>
  <div class=hd><h3>⚡ Загрузка</h3><span id=clock class="mut sml"></span></div>
  <div id=gpu class=row style=margin-top:10px></div>
  <div id=now style=margin-top:10px></div>
  <h3 style=margin-top:14px>📋 Очередь <span id=qn class="mut sml"></span></h3>
  <div id=queue style=margin-top:6px></div>
</div>

<div class=card>
  <div class=hd><h3>🔑 Ключи и лимиты</h3><span id=bill class="mut sml"></span></div>
  <div class=kcard style=margin-top:10px>
    <div class=mut style=margin-bottom:6px>Выдать новый ключ</div>
    <div class=row>
      <label class=fld><span>владелец</span><input id=newOwner placeholder="имя клиента"></label>
      <label class=fld><span>квота units/мес (пусто=∞)</span><input id=newQuota type=number placeholder="∞"></label>
      <label class=fld><span>лимит req/мин (пусто=∞)</span><input id=newRate type=number placeholder="∞"></label>
    </div>
    <div class=acts><button class=p style=flex:1 onclick=createKey()>+ Выдать ключ</button></div>
    <div id=keyout class=keyout></div>
  </div>
  <div id=keys></div>
</div>

<div id=toast></div>
<script>
const $=s=>document.querySelector(s);
const BASE=location.pathname.replace(/\/admin\/ui\/?$/,'');
let ADM=localStorage.getItem('gw_adm')||'';
const esc=s=>(s||'').toString().replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/"/g,'&quot;');
function toast(m){const t=$('#toast');t.textContent=m;t.classList.add('show');clearTimeout(t._h);t._h=setTimeout(()=>t.classList.remove('show'),2600);}
function setConn(ok){$('#conn').innerHTML='<span class="dot'+(ok?' on':'')+'"></span>'+(ok?'подключено':'не подключено');if(ok)$('#loginCard').style.display='none';}
$('#adm').value=ADM;
function saveAdm(){ADM=$('#adm').value.trim();localStorage.setItem('gw_adm',ADM);refresh(true);}
async function api(path,opts={}){opts.headers=Object.assign({'X-Admin-Key':ADM},opts.headers||{});
  const r=await fetch(BASE+path,opts);
  if(r.status===403){setConn(false);$('#loginCard').style.display='';$('#admst').textContent='403 — неверный admin-ключ';throw 0;}
  return r;}
function ago(ts){if(!ts)return '—';let d=Math.floor(Date.now()/1000-ts);if(d<60)return d+'с назад';if(d<3600)return Math.floor(d/60)+'м назад';return Math.floor(d/3600)+'ч назад';}
function copy(t){navigator.clipboard.writeText(t).then(()=>toast('Скопировано'));}

async function loadLoad(){try{const s=await (await api('/admin/broker-state')).json();setConn(true);
  const g=s.gpu||{},pct=g.vram_total?Math.round(g.vram_used/g.vram_total*100):0;
  $('#gpu').innerHTML='<span class=badge>🧠 '+esc(s.model||'—')+'</span><span class=badge>util '+(g.util||0)+'%</span>'+
    '<div style=flex:1;min-width:160px><div class=mut style=font-size:11px>VRAM '+(g.vram_used||0)+' / '+(g.vram_total||0)+' MiB</div><div class=bar><i style=width:'+pct+'%></i></div></div>'+
    (s.swapping?'<span class=badge style=color:var(--warn)>↻ '+esc(s.swapping)+'</span>':'');
  const jobs=s.jobs||[],run=jobs.find(j=>j.status==='running');
  if(run){const p=run.est?Math.min(100,Math.round(run.elapsed/run.est*100)):0;
    $('#now').innerHTML='<div class=kcard><div class=hd><b><span class="dot on" style=display:inline-block;margin-right:6px></span>'+esc(run.type)+'</b><span class=mut>'+(run.elapsed||0)+'с / ~'+(run.est||0)+'с</span></div><div class=mut style=margin-top:4px>'+esc(run.step||'')+'</div><div class=bar><i style=width:'+p+'%></i></div></div>';}
  else $('#now').innerHTML='<div class=mut>сейчас ничего не выполняется</div>';
  const q=jobs.filter(j=>j.status==='queued');$('#qn').textContent=q.length?('· '+q.length+' в ожидании'):'';
  $('#queue').innerHTML=q.length?q.map(j=>'<div class=kcard style=padding:9px_12px><div class=hd><span><span class=badge>#'+j.position+'</span> '+esc(j.type)+'</span><span class=mut>~'+j.eta+'с</span></div></div>').join(''):'<div class=mut>очередь пуста</div>';
}catch(e){}}

async function loadKeys(){try{const d=await (await api('/admin/keys')).json();setConn(true);
  $('#bill').textContent='тариф '+d.price_per_unit+' '+d.currency+'/unit · '+d.month;
  let total=0,mtotal=0;
  $('#keys').innerHTML=d.keys.map(k=>{total+=k.cost;mtotal+=k.month_cost;
    const used=k.quota_units?Math.min(100,Math.round(k.month_units/k.quota_units*100)):0,hot=used>=85?' hot':'';
    const svc=Object.entries(k.by_service||{}).map(([s,n])=>'<span class=badge style=font-size:10px>'+esc(s)+' '+n+'</span>').join(' ');
    return '<div class="kcard'+(k.active?'':' off')+'" id="c_'+k.full_key+'">'+
      '<div class=hd><div><b>'+esc(k.owner)+'</b><div class="mono mut" onclick="copy(\''+k.full_key+'\')" style=cursor:pointer title="копировать">'+esc(k.key)+' ⧉</div></div>'+
        '<div class=sw><input type=checkbox id="a_'+k.full_key+'" '+(k.active?'checked':'')+'><label for="a_'+k.full_key+'"></label></div></div>'+
      '<div class=grid2 style=margin-top:10px>'+
        '<div class=stat><div class=mut>За месяц</div><b>'+k.month_units+(k.quota_units?' / '+k.quota_units:'')+'</b> units'+(k.quota_units?'<div class="bar q'+hot+'"><i style=width:'+used+'%></i></div>':'')+'</div>'+
        '<div class=stat><div class=mut>Счёт (всего)</div><b>'+k.cost+' '+d.currency+'</b><div class=mut>'+k.units+' units · '+ago(k.last_ts)+'</div></div>'+
      '</div>'+
      (svc?'<div style=margin-top:8px>'+svc+'</div>':'')+
      '<div class=row style=margin-top:10px>'+
        '<label class=fld><span>квота units/мес</span><input type=number class=q_quota value="'+(k.quota_units??'')+'" placeholder=∞></label>'+
        '<label class=fld><span>лимит req/мин</span><input type=number class=q_rate value="'+(k.rate_per_min??'')+'" placeholder=∞></label></div>'+
      '<div class=acts><button class=p onclick="saveLim(\''+k.full_key+'\')">💾 Сохранить</button>'+
        '<button class=d onclick="revoke(\''+k.full_key+'\')">⨯ Отозвать</button></div>'+
    '</div>';}).join('')+
    '<div class=mut style=margin-top:12px>Итог: всего <b>'+total.toFixed(2)+' '+d.currency+'</b> · за месяц <b>'+mtotal.toFixed(2)+' '+d.currency+'</b></div>';
  // wire active toggles -> instant apply
  d.keys.forEach(k=>{const el=$('#a_'+CSS.escape(k.full_key));if(el)el.onchange=()=>saveLim(k.full_key,true);});
}catch(e){}}

async function saveLim(key,silent){const c=$('#c_'+CSS.escape(key));
  const q=c.querySelector('.q_quota').value,r=c.querySelector('.q_rate').value,a=$('#a_'+CSS.escape(key)).checked?'true':'false';
  const fd=new FormData();fd.append('quota_units',q);fd.append('rate_per_min',r);fd.append('active',a);
  await api('/admin/keys/'+encodeURIComponent(key)+'/limits',{method:'POST',body:fd});
  toast(silent?'Статус применён':'Лимиты применены ✓');loadKeys();}
async function revoke(key){if(!confirm('Отозвать ключ? Доступ прекратится сразу.'))return;
  await api('/admin/keys/'+encodeURIComponent(key)+'/revoke',{method:'POST'});toast('Ключ отозван');loadKeys();}
async function createKey(){const o=$('#newOwner').value.trim();if(!o){toast('Введи владельца');return;}
  const fd=new FormData();fd.append('owner',o);fd.append('quota_units',$('#newQuota').value);fd.append('rate_per_min',$('#newRate').value);
  const d=await (await api('/admin/keys',{method:'POST',body:fd})).json();
  const box=$('#keyout');box.style.display='block';
  box.innerHTML='Ключ для <b>'+esc(o)+'</b> (показывается один раз):<div class="mono" style=margin-top:6px;word-break:break-all>'+d.api_key+'</div><button class=p style=margin-top:8px onclick="copy(\''+d.api_key+'\')">⧉ Скопировать</button>';
  $('#newOwner').value='';$('#newQuota').value='';$('#newRate').value='';toast('Ключ выдан ✓');loadKeys();}

function refresh(login){if(!ADM){setConn(false);return;}loadLoad();loadKeys();if(login)toast('Подключено');}
setInterval(()=>{if(ADM)loadLoad();},2500);
setInterval(()=>{if(ADM)loadKeys();},9000);
setInterval(()=>$('#clock').textContent=new Date().toLocaleTimeString(),1000);
refresh();
</script></body></html>"""
