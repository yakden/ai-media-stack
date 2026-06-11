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
import subprocess
import threading
import time

import httpx
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

BROKER = "http://127.0.0.1:8092"
OLLAMA = "http://127.0.0.1:11434"
ADMIN_KEY = os.environ.get("API_ADMIN_KEY", "")
# LLMs exposed through the gateway (allowlist). llama3.2 = light/fast (translate, chat);
# qwen2.5vl = heavier multimodal; eurollm/translategemma = translation-specialised (heavier, Q6_K) —
# all the 7B+ models compete for VRAM with the broker's resident model (see VRAM note in README).
LLM_MODELS = {"llama3.2:3b": "fast text LLM (translate/chat)",
              "qwen2.5vl:7b": "multimodal vision-language (heavier)",
              "eurollm:9b": "EuroLLM-9B — translation-tuned for 35 European languages (Q6_K)",
              "translategemma:12b": "TranslateGemma-12B — Google translation model, highest quality (Q6_K)"}
DEFAULT_LLM = "eurollm:9b"
# models best suited for translation, surfaced first in the UI picker
TRANSLATE_MODELS = ["eurollm:9b", "translategemma:12b", "qwen2.5vl:7b", "llama3.2:3b"]
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

app = FastAPI(title="ai-api-gateway", version="1.0.0")


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


def _tail_lines(path, n):
    """Read only the last `n` non-empty lines of a file by seeking from the end —
    O(n) regardless of total file size, so the live feed never loads a huge log."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            pos = f.tell()
            data = b""
            block = 4096
            while pos > 0 and data.count(b"\n") <= n:
                step = min(block, pos)
                pos -= step
                f.seek(pos)
                data = f.read(step) + data
        lines = [ln for ln in data.split(b"\n") if ln.strip()]
        return [ln.decode("utf-8", "replace") for ln in lines[-n:]]
    except Exception:
        return []


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
# Live activity tracking — what every key is doing RIGHT NOW (in-flight) so the #
# admin panel shows real-time work (translations/LLM/voice never hit the broker #
# queue, so we track them here at the gateway).                                 #
# --------------------------------------------------------------------------- #
import itertools as _it

_INFLIGHT = {}                 # rid -> {owner, service, path, started}
_inflight_seq = _it.count(1)
_inflight_lock = threading.Lock()
# light owner cache so the per-request middleware doesn't hammer keys.json
_OWNER_CACHE = {"ts": 0.0, "map": {}}


def _service_label(path):
    if path.startswith("/v1/translate") or path == "/v1/detect":
        return "translate"
    if path.startswith("/v1/llm"):
        return "llm"
    if path.startswith("/v1/3d/render"):
        return "render"
    if path.startswith("/v1/3d"):
        return "3d-проект"
    if path.startswith("/v1/voice"):
        return "голос"
    if path.startswith("/v1/avatar"):
        return "аватар"
    if path.startswith("/v1/dub"):
        return "дубляж"
    return path.replace("/v1/", "")


def _owner_for(req):
    key = req.headers.get("x-api-key")
    if not key:
        a = req.headers.get("authorization", "")
        if a.lower().startswith("bearer "):
            key = a[7:].strip()
    if not key:
        return "—"
    now = time.time()
    if now - _OWNER_CACHE["ts"] > 3:          # refresh map every few seconds
        _OWNER_CACHE["map"] = {k: v.get("owner", "") for k, v in _load(KEYS_F, {}).items()}
        _OWNER_CACHE["ts"] = now
    return _OWNER_CACHE["map"].get(key) or "неизв. ключ"


_NO_TRACK = {"/v1/ping", "/v1/models", "/v1/voices", "/v1/usage", "/v1/billing"}


@app.middleware("http")
async def _track_inflight(request, call_next):
    path = request.url.path
    rid = None
    if path.startswith("/v1/") and path not in _NO_TRACK and request.method in ("POST", "PUT"):
        rid = next(_inflight_seq)
        with _inflight_lock:
            _INFLIGHT[rid] = {"owner": _owner_for(request), "service": _service_label(path),
                              "path": path, "started": time.time()}
    try:
        return await call_next(request)
    finally:
        if rid is not None:
            with _inflight_lock:
                _INFLIGHT.pop(rid, None)


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
    allowed = _allowed_models()
    order = [m for m in TRANSLATE_MODELS if m in allowed] + [m for m in allowed if m not in TRANSLATE_MODELS]
    return {"models": [{"id": m, "description": allowed[m],
                        "good_for_translation": m in TRANSLATE_MODELS[:3]} for m in order],
            "default": DEFAULT_LLM, "translate_recommended": [m for m in TRANSLATE_MODELS[:3] if m in allowed]}


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
        j = _gen(model, f"Translate the following text to {to}. Output ONLY the translation, no notes:\n\n{text}")
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
            j = _gen(model, prompt)
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
    allowed = _allowed_models()
    if m not in allowed:
        raise HTTPException(400, f"model not allowed; use one of {list(allowed)}")
    return m


# Heavy models that get the WHOLE GPU via the broker (it swaps whisper/avatar/render/vlm
# off first, so the model loads fully on-GPU = fast, instead of CPU-offloading next to whisper).
BROKER_LLM = {"translategemma:12b"}


def _gen(model, prompt, options=None, timeout=200):
    """Run an Ollama generate. Heavy models route through the broker (/api/llm) so they
    get the full GPU; light models hit Ollama directly. Returns the Ollama JSON dict.
    For the broker route we BOUND everything: the broker fails fast with 503 when the GPU
    slot is busy, and we retry a few times with backoff instead of blocking forever — this
    is what stops concurrent bursts from piling up into hung sessions (2026-06-11 incident)."""
    # keep_alive=-1 pins the model in VRAM indefinitely (box is dedicated to translation now,
    # so we never want the ~68s cold reload after an idle gap).
    body = {"model": model, "prompt": prompt, "stream": False, "keep_alive": "5m",
            "options": options or {"temperature": 0.2}}
    if model not in BROKER_LLM:
        r = httpx.post(f"{OLLAMA}/api/generate", json=body, timeout=timeout)
        r.raise_for_status()
        return r.json()
    # broker route: retry on 503 (GPU busy) with backoff, bounded total wait
    last = None
    for attempt in range(5):
        try:
            r = httpx.post(f"{BROKER}/api/llm", json=body, timeout=timeout)
        except httpx.HTTPError as exc:
            last = exc
            time.sleep(2)
            continue
        if r.status_code == 503:          # GPU busy with another translation — back off
            last = "broker busy (503)"
            time.sleep(2 + attempt * 2)
            continue
        r.raise_for_status()
        return r.json()
    raise HTTPException(503, f"translation model busy, retry shortly ({last})")


def _tr_one(text, to, model=DEFAULT_LLM):
    prompt = f"Translate the following text to {to}. Output ONLY the translation, no notes:\n\n{text}"
    j = _gen(model, prompt)
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


@app.get("/admin/activity")
def admin_activity(limit: int = 20, authorization: str = Header(None), x_admin_key: str = Header(None)):
    """Unified LIVE view for the dashboard:
      gpu/model/swapping, broker running+queued GPU jobs, in-flight gateway requests
      (translations/LLM/voice happening right now, with owner), and a recent-completed feed.
    `limit` bounds the recent feed (default 20, max 200) — only the file TAIL is read, so the
    events log can grow arbitrarily large without slowing the dashboard."""
    _admin(authorization, x_admin_key)
    limit = max(1, min(int(limit or 20), 200))
    out = {"now": int(time.time())}
    broker_up = False
    try:
        st = httpx.get(f"{BROKER}/api/state", timeout=5).json()
        broker_up = True
    except Exception:
        st = {}
    out["broker_up"] = broker_up
    # GPU: prefer the broker's reading; if the broker is off (translation-only mode), read nvidia-smi directly
    out["gpu"] = st.get("gpu") or _gpu_direct()
    loaded = _ollama_loaded()
    out["loaded"] = loaded            # the model actually resident in Ollama right now
    out["model"] = st.get("model") or (loaded.get("name") if loaded else None)
    out["swapping"] = st.get("swapping")
    out["system"] = _sysinfo()
    jobs = st.get("jobs", [])
    out["running"] = [j for j in jobs if j.get("status") == "running"]
    out["queued"] = sorted((j for j in jobs if j.get("status") == "queued"),
                           key=lambda j: j.get("position") or 0)
    now = time.time()
    with _inflight_lock:
        # prune zombies: any request older than the max request timeout (~200s) can't still
        # be running — drop it so the dashboard never shows phantom "hung" sessions.
        for rid in [k for k, v in _INFLIGHT.items() if now - v["started"] > 240]:
            _INFLIGHT.pop(rid, None)
        out["inflight"] = sorted(
            ({"owner": v["owner"], "service": v["service"], "elapsed": round(now - v["started"], 1)}
             for v in _INFLIGHT.values()), key=lambda x: -x["elapsed"])
    recent = []
    for ln in reversed(_tail_lines(EVENTS_F, limit)):
        try:
            e = json.loads(ln)
            recent.append({"ts": e["ts"], "owner": e.get("owner") or "—",
                           "service": e["service"], "units": e.get("units", 0),
                           "tokens": e.get("tokens", 0)})
        except Exception:
            pass
    out["recent"] = recent
    return out


def _sysinfo():
    """Host disk + RAM for the dashboard (disk is the critical 1C-safety metric)."""
    import shutil
    info = {}
    try:
        du = shutil.disk_usage("/")
        info["disk_free_gb"] = round(du.free / 1e9, 1)
        info["disk_total_gb"] = round(du.total / 1e9, 1)
    except Exception:
        pass
    try:
        mem = {}
        with open("/proc/meminfo") as f:
            for ln in f:
                p = ln.split()
                if p and p[0].rstrip(":") in ("MemTotal", "MemAvailable"):
                    mem[p[0].rstrip(":")] = int(p[1]) // 1024
        info["ram_avail_mb"] = mem.get("MemAvailable", 0)
        info["ram_total_mb"] = mem.get("MemTotal", 0)
    except Exception:
        pass
    return info


def _gpu_direct():
    """GPU VRAM/util straight from nvidia-smi — used when the broker is off (translation-only mode)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=10).stdout.strip().split(",")
        return {"vram_used": int(out[0]), "vram_total": int(out[1]), "util": int(out[2])}
    except Exception:
        return {}


def _ollama_loaded():
    """The model currently resident in Ollama (name + VRAM) — what's actually serving translations."""
    try:
        ms = httpx.get(f"{OLLAMA}/api/ps", timeout=5).json().get("models", [])
        if not ms:
            return None
        m = ms[0]
        return {"name": m.get("name"), "vram_mib": round(int(m.get("size_vram", 0)) / 1048576),
                "expires": m.get("expires_at")}
    except Exception:
        return None


# Services manageable from the admin panel. kind=docker -> docker start/stop/restart <container>;
# kind=systemd -> systemctl <action> <unit>. Grouped for the UI.
SERVICES = [
    {"name": "ollama", "title": "Ollama — перевод / LLM", "group": "🌐 Перевод",
     "kind": "docker", "container": "ollama", "health": "http://127.0.0.1:11434/api/version"},
    {"name": "api-gateway", "title": "API-шлюз (этот сервис)", "group": "🧩 Инфраструктура",
     "kind": "systemd", "unit": "api-gateway", "self": True},
    {"name": "gpu-broker", "title": "GPU-брокер (3D / своп моделей)", "group": "🎮 GPU-оркестрация",
     "kind": "systemd", "unit": "gpu-broker"},
    {"name": "whisper", "title": "Whisper STT / TTS (голос)", "group": "🎙️ Голос",
     "kind": "docker", "container": "whisper-xtts-server", "health": "http://127.0.0.1:8000/health"},
    {"name": "avatar", "title": "Аватар — MuseTalk", "group": "🎙️ Голос",
     "kind": "docker", "container": "avatar-muse"},
    {"name": "voice-stream", "title": "Поточный перевод голоса", "group": "🎙️ Голос",
     "kind": "systemd", "unit": "voice-stream"},
    {"name": "floorplan3d", "title": "Floorplan 3D (Mask R-CNN, CPU)", "group": "🏗️ 3D / планы",
     "kind": "docker", "container": "floorplan3d", "health": "http://127.0.0.1:8204/health"},
    {"name": "cubicasa", "title": "CubiCasa парсер (CPU)", "group": "🏗️ 3D / планы",
     "kind": "docker", "container": "cubicasa", "health": "http://127.0.0.1:8205/health"},
    {"name": "interior-render", "title": "Рендер SD + ControlNet (GPU)", "group": "🏗️ 3D / планы",
     "kind": "docker", "container": "interior-render"},
    {"name": "control-plane", "title": "Control Plane (общая панель)", "group": "🧩 Инфраструктура",
     "kind": "systemd", "unit": "control-plane"},
    {"name": "open-webui", "title": "Open WebUI (чат)", "group": "🧩 Инфраструктура",
     "kind": "docker", "container": "open-webui", "health": "http://127.0.0.1:8088/health"},
]
SVC_BY_NAME = {s["name"]: s for s in SERVICES}


def _svc_status(s):
    if s["kind"] == "docker":
        rc = subprocess.run(["sudo", "docker", "inspect", "-f", "{{.State.Status}}", s["container"]],
                            capture_output=True, text=True, timeout=15)
        status = rc.stdout.strip() if rc.returncode == 0 else "absent"
        running = status == "running"
    else:
        rc = subprocess.run(["systemctl", "is-active", s["unit"]], capture_output=True, text=True, timeout=10)
        status = rc.stdout.strip() or "unknown"
        running = status == "active"
    health = None
    if running and s.get("health"):
        try:
            health = httpx.get(s["health"], timeout=3).status_code < 500
        except Exception:
            health = False
    return {"name": s["name"], "title": s["title"], "group": s["group"], "kind": s["kind"],
            "status": status, "running": running, "health": health, "self": s.get("self", False)}


@app.get("/admin/services")
def list_services(authorization: str = Header(None), x_admin_key: str = Header(None)):
    """Status of every manageable service (docker + systemd), grouped, for the control panel."""
    _admin(authorization, x_admin_key)
    return {"services": [_svc_status(s) for s in SERVICES]}


@app.post("/admin/services/{name}/{action}")
def control_service(name: str, action: str,
                    authorization: str = Header(None), x_admin_key: str = Header(None)):
    """start / stop / restart a service from the panel — applies immediately on the box."""
    _admin(authorization, x_admin_key)
    s = SVC_BY_NAME.get(name)
    if not s:
        raise HTTPException(404, "unknown service")
    if action not in ("start", "stop", "restart"):
        raise HTTPException(400, "action must be start|stop|restart")
    if s.get("self") and action in ("stop", "restart"):
        raise HTTPException(400, "нельзя останавливать сам шлюз из его же интерфейса")
    if s["kind"] == "docker":
        cmd = ["sudo", "docker", action, s["container"]]
    else:
        cmd = ["sudo", "systemctl", action, s["unit"]]
    rc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    return {"ok": rc.returncode == 0, "name": name, "action": action,
            "log": ((rc.stdout or "") + (rc.stderr or "")).strip()[-400:]}


# ----- Launchpad (USING the services) + connect info + model management -----
LAUNCH_BUILTIN = [
    {"title": "🎙️ Поточный перевод голоса", "url": "/voicestream/", "desc": "Микрофон → перевод → твой голос"},
    {"title": "🎥 Веб-дубляж", "url": "/dub/", "desc": "Видео → перевод → липсинк"},
    {"title": "🖼️ Аватар из фото", "url": "/animate/", "desc": "Фото + текст → говорящее видео"},
    {"title": "🏠 Интерьер 3D", "url": "/interior3d/", "desc": "План → 3D-сцена, фотореализм"},
    {"title": "🧮 GPU Очередь", "url": "/broker/", "desc": "Очередь задач, своп моделей, ETA"},
    {"title": "💬 Open WebUI", "url": "https://webui.1c-rus.ru", "desc": "Чат с моделями"},
    {"title": "🎨 IOPaint", "url": "https://paint.1c-rus.ru", "desc": "Удаление объектов с фото"},
    {"title": "📹 VMS", "url": "https://vms.1c-rus.ru", "desc": "Камеры, распознавание лиц"},
    {"title": "🐳 Portainer", "url": "https://portainer.1c-rus.ru", "desc": "Docker — логи, консоль"},
    {"title": "🛰️ Control Plane", "url": "/", "desc": "Общая панель платформы"},
]
LAUNCH_F = os.path.join(DATA, "launch.json")     # пользовательские ссылки [{title,url,desc}]
MODELS_F = os.path.join(DATA, "models.json")     # пользовательские модели {name: description}
_PULLS = {}                                      # name -> {status, msg}
_pull_lock = threading.Lock()


def _allowed_models():
    """Builtin allowlist + user-added models (persisted) — the effective set for translate/chat."""
    m = dict(LLM_MODELS)
    m.update(_load(MODELS_F, {}))
    return m


def _do_pull(name, desc):
    with _pull_lock:
        _PULLS[name] = {"status": "pulling", "msg": "загрузка…"}
    try:
        rc = subprocess.run(["sudo", "docker", "exec", "ollama", "ollama", "pull", name],
                            capture_output=True, text=True, timeout=3600)
        ok = rc.returncode == 0
        msg = ((rc.stdout or "") + (rc.stderr or "")).strip()[-300:]
    except Exception as exc:
        ok, msg = False, str(exc)
    if ok:
        ex = _load(MODELS_F, {})
        ex[name] = desc or "добавлена через админку"
        _save(MODELS_F, ex)
    with _pull_lock:
        _PULLS[name] = {"status": "done" if ok else "error", "msg": msg}


@app.get("/admin/launch")
def admin_launch(authorization: str = Header(None), x_admin_key: str = Header(None)):
    _admin(authorization, x_admin_key)
    return {"builtin": LAUNCH_BUILTIN, "custom": _load(LAUNCH_F, []),
            "api": {"base": "https://ai.1c-rus.ru/gw", "model": DEFAULT_LLM,
                    "docs": "https://github.com/yakden/ai-media-stack/blob/main/docs/API.md"}}


@app.post("/admin/launch")
def add_launch(payload: dict, authorization: str = Header(None), x_admin_key: str = Header(None)):
    _admin(authorization, x_admin_key)
    t, u = (payload.get("title") or "").strip(), (payload.get("url") or "").strip()
    if not t or not u:
        raise HTTPException(400, "нужны title и url")
    c = _load(LAUNCH_F, [])
    c.append({"title": t, "url": u, "desc": (payload.get("desc") or "").strip()})
    _save(LAUNCH_F, c)
    return {"ok": True}


@app.post("/admin/launch/delete")
def del_launch(payload: dict, authorization: str = Header(None), x_admin_key: str = Header(None)):
    _admin(authorization, x_admin_key)
    i = int(payload.get("index", -1))
    c = _load(LAUNCH_F, [])
    if 0 <= i < len(c):
        c.pop(i)
        _save(LAUNCH_F, c)
    return {"ok": True}


@app.get("/admin/models")
def admin_models(authorization: str = Header(None), x_admin_key: str = Header(None)):
    """Installed Ollama models + which are allowed for the API + any running pulls."""
    _admin(authorization, x_admin_key)
    try:
        tags = httpx.get(f"{OLLAMA}/api/tags", timeout=8).json().get("models", [])
    except Exception:
        tags = []
    allowed = _allowed_models()
    installed = [{"name": t.get("name"), "size_gb": round(int(t.get("size", 0)) / 1e9, 1),
                  "allowed": t.get("name") in allowed} for t in tags]
    with _pull_lock:
        pulls = dict(_PULLS)
    return {"installed": installed, "allowed": allowed, "builtin": list(LLM_MODELS),
            "default": DEFAULT_LLM, "pulls": pulls}


@app.post("/admin/models/pull")
def models_pull(payload: dict, authorization: str = Header(None), x_admin_key: str = Header(None)):
    """Download a new model into Ollama (background) — e.g. 'hf.co/<repo>:<quant>' or 'qwen2.5:7b'.
    On success it's auto-added to the allowlist so it becomes selectable via the API."""
    _admin(authorization, x_admin_key)
    name = (payload.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "укажи имя модели")
    threading.Thread(target=_do_pull, args=(name, (payload.get("description") or "").strip()), daemon=True).start()
    return {"ok": True, "started": name}


@app.post("/admin/models/allow")
def models_allow(payload: dict, authorization: str = Header(None), x_admin_key: str = Header(None)):
    """Toggle whether an installed model is selectable via the API (allowlist)."""
    _admin(authorization, x_admin_key)
    name = (payload.get("name") or "").strip()
    allow = bool(payload.get("allow", True))
    ex = _load(MODELS_F, {})
    if allow:
        ex[name] = (payload.get("description") or "").strip() or "пользовательская модель"
    else:
        ex.pop(name, None)
    _save(MODELS_F, ex)
    return {"ok": True, "allowed": name in _allowed_models()}


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
<meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>AI · Админ</title>
<style>
:root{--bg:#0c0f14;--card:#161b22;--card2:#1c232d;--line:#2a323d;--mut:#8b97a7;--acc:#4f8cff;--ok:#2ecc71;--warn:#f1c40f;--bad:#e15b64;--txt:#e6edf3}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--txt);font:15px/1.5 -apple-system,Segoe UI,Roboto,system-ui}
.hdr{position:sticky;top:0;z-index:20;display:flex;align-items:center;gap:10px;justify-content:space-between;padding:11px 16px;background:rgba(12,15,20,.92);backdrop-filter:blur(8px);border-bottom:1px solid var(--line)}
.brand{font-weight:700;font-size:16px}
main{max-width:940px;margin:0 auto;padding:14px}
h3{margin:0;font-size:15px}.mut{color:var(--mut);font-size:13px}.sml{font-size:12px}
nav#nav{position:sticky;top:48px;z-index:19;display:flex;gap:6px;max-width:940px;margin:0 auto;padding:8px 14px;background:var(--bg);border-bottom:1px solid var(--line)}
nav#nav button{flex:1;background:transparent;border:1px solid transparent;color:var(--mut);padding:9px 6px;border-radius:11px;font-size:13px;font-weight:600;cursor:pointer;display:flex;align-items:center;justify-content:center;gap:7px;transition:.12s}
nav#nav button.on{background:var(--card2);color:var(--txt);border-color:var(--line)}
nav#nav .ic{font-size:16px}
.tab[hidden]{display:none}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:15px;margin-top:13px}
.hd{display:flex;align-items:center;gap:10px;justify-content:space-between}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
.badge{display:inline-flex;align-items:center;gap:6px;background:#0b0e13;border:1px solid var(--line);border-radius:999px;padding:4px 11px;font-size:12px;white-space:nowrap}
.dot{width:9px;height:9px;border-radius:50%;background:#5b6673;flex:none}.dot.on{background:var(--ok);box-shadow:0 0 8px var(--ok)}.dot.run{background:var(--ok);box-shadow:0 0 8px var(--ok);animation:pulse 1.6s infinite}.dot.bad{background:var(--bad)}.dot.warn{background:var(--warn)}
@keyframes pulse{50%{opacity:.4}}
.bar{height:8px;background:#0b0e13;border-radius:6px;overflow:hidden;margin-top:6px}.bar>i{display:block;height:100%;background:linear-gradient(90deg,var(--acc),#7aa7ff);transition:width .4s}
.bar.hot>i{background:linear-gradient(90deg,var(--warn),var(--bad))}
button{background:var(--card2);border:1px solid var(--line);color:var(--txt);padding:10px 14px;border-radius:11px;cursor:pointer;font-size:14px;font-weight:600;transition:.12s}
button:active{transform:scale(.97)}button.p{background:var(--acc);border-color:var(--acc);color:#fff}
button.d{background:#2a1a1d;border-color:#5a2a32;color:#ff9a9a}button.ghost{background:transparent}button.sm{padding:7px 11px;font-size:13px}
input{background:#0b0e13;border:1px solid var(--line);color:var(--txt);padding:11px 12px;border-radius:11px;font-size:15px;outline:none;width:100%}
input:focus{border-color:var(--acc)}
label.fld{flex:1;min-width:130px}label.fld>span{display:block;color:var(--mut);font-size:11px;margin:0 0 3px 3px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:9px}
.stat{background:#0b0e13;border:1px solid var(--line);border-radius:12px;padding:11px 13px}.stat b{font-size:19px}.stat .mut{font-size:11px}
.kcard{background:var(--card2);border:1px solid var(--line);border-radius:14px;padding:13px;margin-top:10px}.kcard.off{opacity:.55}
.mono{font-family:ui-monospace,Menlo,monospace;font-size:12px}
.sw{position:relative;width:46px;height:26px;flex:none}.sw input{opacity:0;width:0;height:0;position:absolute}
.sw label{position:absolute;inset:0;background:#3a434f;border-radius:999px;cursor:pointer;transition:.15s}
.sw label:before{content:"";position:absolute;width:20px;height:20px;left:3px;top:3px;background:#fff;border-radius:50%;transition:.15s}
.sw input:checked+label{background:var(--ok)}.sw input:checked+label:before{transform:translateX(20px)}
.svc{display:flex;align-items:center;gap:10px;padding:11px 0;border-bottom:1px solid var(--line);flex-wrap:wrap}.svc:last-child{border:0}
.svc .nm{font-weight:600;flex:1;min-width:140px}
.svc .acts{display:flex;gap:6px}
.grp{margin-top:16px;color:var(--mut);font-size:13px;font-weight:600}
.acts{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap}.acts>button{flex:1;min-width:88px}
.rrow{display:flex;align-items:center;gap:8px;padding:7px 2px;border-bottom:1px solid var(--line);font-size:13px}.rrow:last-child{border:0}
.rico{width:20px;text-align:center;flex:none}.rown{font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:34%}
.rmeta{color:var(--mut);font-size:12px;margin-left:auto;white-space:nowrap}.rago{color:var(--mut);font-size:11px;width:44px;text-align:right;flex:none}
#toast{position:fixed;left:50%;bottom:84px;transform:translateX(-50%) translateY(80px);background:#1c232d;border:1px solid var(--line);padding:11px 18px;border-radius:12px;opacity:0;transition:.25s;z-index:30;max-width:90vw;text-align:center}
#toast.show{opacity:1;transform:translateX(-50%) translateY(0)}
.keyout{background:#0b1a10;border:1px solid #2a5a3a;border-radius:11px;padding:10px;margin-top:8px;display:none}
canvas{width:100%;height:46px;display:block;margin-top:10px}
.lgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}
.tile{display:block;text-decoration:none;color:inherit;background:var(--card2);border:1px solid var(--line);border-radius:14px;padding:13px;transition:.12s;position:relative}
.tile:hover{border-color:var(--acc);transform:translateY(-2px)}
.tile b{display:block;font-size:14px}.tile .d{color:var(--mut);font-size:12px;margin-top:4px}.tile .go{color:var(--acc);font-size:12px;margin-top:8px}
pre{background:#0b0e13;border:1px solid var(--line);border-radius:10px;padding:10px;overflow:auto;font-size:12px}
a{color:var(--acc)}
@media(max-width:680px){
  nav#nav{position:fixed;top:auto;bottom:0;left:0;right:0;max-width:none;border-top:1px solid var(--line);border-bottom:0;background:rgba(12,15,20,.96);padding:6px 8px calc(6px + env(safe-area-inset-bottom))}
  nav#nav button{flex-direction:column;gap:2px;font-size:10px;padding:5px 2px}
  nav#nav .ic{font-size:19px}
  main{padding-bottom:84px}
}
</style></head><body>
<div class=hdr><div class=brand>🛠️ AI-платформа · Админ</div><span id=conn class=badge><span class=dot></span>…</span></div>
<nav id=nav>
  <button data-tab=overview class=on><span class=ic>📊</span><span>Обзор</span></button>
  <button data-tab=launch><span class=ic>🚀</span><span>Запуск</span></button>
  <button data-tab=services><span class=ic>🧩</span><span>Сервисы</span></button>
  <button data-tab=keys><span class=ic>🔑</span><span>Ключи</span></button>
  <button data-tab=activity><span class=ic>🕒</span><span>Активность</span></button>
</nav>
<main>

<section id=overview class=tab>
  <div class=card>
    <div class=hd><h3>⚡ Нагрузка GPU</h3><span id=clock class="mut sml"></span></div>
    <div id=gpu class=row style=margin-top:10px></div>
    <canvas id=spark></canvas>
  </div>
  <div class=card>
    <h3>▶️ Выполняется сейчас <span id=ln class="mut sml"></span></h3>
    <div id=live style=margin-top:8px></div>
    <h3 style=margin-top:16px>📋 В очереди <span id=qn class="mut sml"></span></h3>
    <div id=queue style=margin-top:6px></div>
  </div>
  <div class=card>
    <h3>🖥️ Система</h3>
    <div id=sys class=grid style=margin-top:10px></div>
  </div>
</section>

<section id=launch class=tab hidden>
  <div class=card>
    <h3>🚀 Открыть инструменты</h3>
    <div id=launchpad class=lgrid style=margin-top:11px></div>
  </div>
  <div class=card>
    <h3>🔌 Быстрое подключение (API)</h3>
    <div id=connect style=margin-top:9px></div>
  </div>
  <div class=card>
    <div class=hd><h3>🧠 Модели</h3><button class="ghost sm" onclick=loadModels()>↻ Обновить</button></div>
    <div class=kcard style=margin-top:10px>
      <div class=mut sml style=margin-bottom:7px>Добавить модель в Ollama. Имя: <span class=mono>hf.co/&lt;repo&gt;:&lt;quant&gt;</span> или из библиотеки Ollama (<span class=mono>qwen2.5:7b</span>). После загрузки станет доступна в API.</div>
      <div class=row>
        <label class=fld><span>имя модели</span><input id=mdlName placeholder="hf.co/...:Q4_K_M"></label>
        <label class=fld><span>описание (необяз.)</span><input id=mdlDesc placeholder="для чего"></label>
      </div>
      <div class=acts><button class=p style=flex:1 onclick=pullModel()>⬇ Скачать и добавить</button></div>
      <div id=pulls style=margin-top:8px></div>
    </div>
    <div id=models-list style=margin-top:6px></div>
  </div>
  <div class=card>
    <h3>🔗 Свои ссылки на сервисы</h3>
    <div class=mut sml style=margin-top:4px>Добавь ссылку на любой свой сервис — появится выше в «Открыть инструменты».</div>
    <div class=row style=margin-top:9px>
      <label class=fld><span>название</span><input id=lkTitle placeholder="🔧 Мой сервис"></label>
      <label class=fld><span>URL</span><input id=lkUrl placeholder="https://..."></label>
    </div>
    <div class=acts><button class=p style=flex:1 onclick=addLink()>+ Добавить ссылку</button></div>
  </div>
</section>

<section id=services class=tab hidden>
  <div class=card>
    <div class=hd><h3>🧩 Управление сервисами</h3><button class="ghost sm" onclick=loadServices()>↻ Обновить</button></div>
    <div class=mut sml style=margin-top:4px>Старт/стоп/рестарт применяются сразу на сервере.</div>
    <div id=services-list style=margin-top:6px></div>
  </div>
</section>

<section id=keys class=tab hidden>
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
    <div id=keys-list></div>
  </div>
</section>

<section id=activity class=tab hidden>
  <div class=card>
    <div class=hd><h3>🕒 Последние операции</h3><span class="mut sml">кто · что · сколько</span></div>
    <div id=recent style=margin-top:8px></div>
  </div>
</section>

</main>
<div id=toast></div>
<script>
const $=s=>document.querySelector(s);
const BASE=location.pathname.replace(/\/admin\/ui\/?$/,'');
let ADM=localStorage.getItem('gw_adm')||'';
let TAB='overview';let RLIMIT=20;let UTILH=[];
const esc=s=>(s||'').toString().replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/"/g,'&quot;');
function toast(m){const t=$('#toast');t.textContent=m;t.classList.add('show');clearTimeout(t._h);t._h=setTimeout(()=>t.classList.remove('show'),2800);}
function setConn(ok){$('#conn').innerHTML='<span class="dot'+(ok?' on':'')+'"></span>'+(ok?'на связи':'нет связи');}
function copy(t){navigator.clipboard.writeText(t).then(()=>toast('Скопировано'));}
function ago(ts){if(!ts)return '—';let d=Math.floor(Date.now()/1000-ts);if(d<0)d=0;if(d<60)return d+'с';if(d<3600)return Math.floor(d/60)+'м';if(d<86400)return Math.floor(d/3600)+'ч';return Math.floor(d/86400)+'д';}
async function api(path,opts={}){opts.headers=Object.assign({'X-Admin-Key':ADM},opts.headers||{});
  const r=await fetch(BASE+path,opts);
  if(r.status===403){setConn(false);throw 0;}
  return r;}

// ---- navigation ----
document.querySelectorAll('#nav button').forEach(b=>b.onclick=()=>setTab(b.dataset.tab));
function setTab(name){TAB=name;
  document.querySelectorAll('#nav button').forEach(b=>b.classList.toggle('on',b.dataset.tab===name));
  document.querySelectorAll('.tab').forEach(s=>s.hidden=(s.id!==name));
  load();}

// ---- icons ----
const SVC_ICON={translate:'🌐',llm:'🤖','3d-проект':'🏗️',render:'🎨','голос':'🎙️',voice_translate:'🎙️',voice_chunk:'🎙️','аватар':'🖼️',avatar:'🖼️','дубляж':'🎥',dub:'🎥',project_3d:'🏗️',reference:'🖼️',interior:'🏠',furnish:'🛋️'};
const svcIco=s=>SVC_ICON[s]||'•';

// ---- sparkline ----
function drawSpark(){const c=$('#spark');if(!c||c.offsetParent===null)return;const w=c.clientWidth||300,h=46;c.width=w;c.height=h;const x=c.getContext('2d');x.clearRect(0,0,w,h);
  if(UTILH.length<2)return;const n=UTILH.length,step=w/Math.max(59,n-1);
  x.beginPath();x.moveTo(0,h-(UTILH[0]/100*(h-3))-1);for(let i=1;i<n;i++)x.lineTo(i*step,h-(UTILH[i]/100*(h-3))-1);
  x.lineTo((n-1)*step,h);x.lineTo(0,h);x.closePath();const g=x.createLinearGradient(0,0,0,h);g.addColorStop(0,'rgba(79,140,255,.35)');g.addColorStop(1,'rgba(79,140,255,.02)');x.fillStyle=g;x.fill();
  x.beginPath();x.moveTo(0,h-(UTILH[0]/100*(h-3))-1);for(let i=1;i<n;i++)x.lineTo(i*step,h-(UTILH[i]/100*(h-3))-1);x.strokeStyle='#4f8cff';x.lineWidth=2;x.stroke();}

// ---- OVERVIEW ----
async function loadOverview(){let s;try{s=await(await api('/admin/activity?limit=8')).json();setConn(true);}catch(e){return;}
  const g=s.gpu||{},pct=g.vram_total?Math.round(g.vram_used/g.vram_total*100):0,util=g.util||0;
  UTILH.push(util);if(UTILH.length>60)UTILH.shift();drawSpark();
  const brk=s.broker_up?'':'<span class=badge>брокер выкл</span>';
  $('#gpu').innerHTML='<span class=badge>🧠 '+esc(s.model||'—')+'</span><span class=badge>util '+util+'%</span>'+brk+
    '<div style=flex:1;min-width:170px><div class=mut style=font-size:11px>VRAM '+(g.vram_used||0)+' / '+(g.vram_total||0)+' MiB ('+pct+'%)</div><div class="bar'+(pct>=90?' hot':'')+'"><i style=width:'+pct+'%></i></div></div>';
  const run=s.running||[],fly=s.inflight||[];let html='';
  run.forEach(j=>{const p=j.est?Math.min(100,Math.round(j.elapsed/j.est*100)):0;
    html+='<div class=kcard><div class=hd><b><span class="dot run"></span> '+svcIco(j.type)+' '+esc(j.type)+'</b><span class=mut>'+(j.elapsed||0)+'с / ~'+(j.est||0)+'с</span></div>'+(j.step?'<div class=mut style=margin-top:4px>'+esc(j.step)+'</div>':'')+'<div class=bar><i style=width:'+p+'%></i></div></div>';});
  fly.forEach(f=>{html+='<div class=kcard><div class=hd><b><span class="dot run"></span> '+svcIco(f.service)+' '+esc(f.service)+'</b><span class=mut>'+f.elapsed+'с</span></div><div class=mut style=margin-top:4px>👤 '+esc(f.owner)+' · обрабатывается…</div></div>';});
  $('#live').innerHTML=html||'<div class=mut>сейчас ничего не выполняется</div>';
  $('#ln').textContent=(run.length+fly.length)?('· '+(run.length+fly.length)):'';
  const q=s.queued||[];$('#qn').textContent=q.length?('· '+q.length):'';
  $('#queue').innerHTML=q.length?q.map(j=>'<div class=kcard style=padding:9px_12px><div class=hd><span><span class=badge>#'+j.position+'</span> '+svcIco(j.type)+' '+esc(j.type)+'</span><span class=mut>~'+(j.eta||0)+'с</span></div></div>').join(''):'<div class=mut>очередь пуста</div>';
  const sy=s.system||{};const dlow=(sy.disk_free_gb||99)<30;
  $('#sys').innerHTML=
    '<div class=stat><div class=mut>Диск свободно</div><b style="color:'+(dlow?'var(--bad)':'inherit')+'">'+(sy.disk_free_gb??'—')+' ГБ</b><div class=mut>из '+(sy.disk_total_gb??'—')+' ГБ</div></div>'+
    '<div class=stat><div class=mut>RAM свободно</div><b>'+(sy.ram_avail_mb?Math.round(sy.ram_avail_mb/1024):'—')+' ГБ</b><div class=mut>из '+(sy.ram_total_mb?Math.round(sy.ram_total_mb/1024):'—')+' ГБ</div></div>'+
    '<div class=stat><div class=mut>Модель в VRAM</div><b style=font-size:14px>'+esc((s.loaded&&s.loaded.name)||'—')+'</b><div class=mut>'+((s.loaded&&s.loaded.vram_mib)||0)+' MiB</div></div>'+
    '<div class=stat><div class=mut>GPU util</div><b>'+util+'%</b></div>';
}

// ---- SERVICES ----
async function loadServices(){let d;try{d=await(await api('/admin/services')).json();setConn(true);}catch(e){return;}
  const groups={};d.services.forEach(s=>{(groups[s.group]=groups[s.group]||[]).push(s);});
  let html='';
  for(const g in groups){html+='<div class=grp>'+esc(g)+'</div>';
    groups[g].forEach(s=>{
      const run=s.running,dotc=run?(s.health===false?'warn':'run'):(s.status==='absent'?'bad':'');
      const stt=run?(s.health===false?'нет ответа':'работает'):(s.status==='exited'||s.status==='inactive'?'остановлен':esc(s.status));
      html+='<div class=svc><span class="dot '+dotc+'"></span><span class=nm>'+esc(s.title)+'<div class="mut sml">'+esc(s.name)+' · '+stt+'</div></span>'+
        '<span class=acts>'+
        (run?'':'<button class="p sm" onclick="svc(\''+s.name+'\',\'start\')">Старт</button>')+
        (run?'<button class="sm" onclick="svc(\''+s.name+'\',\'restart\')">Рестарт</button>':'')+
        (run&&!s.self?'<button class="d sm" onclick="svc(\''+s.name+'\',\'stop\')">Стоп</button>':'')+
        '</span></div>';
    });}
  $('#services-list').innerHTML=html;
}
async function svc(name,action){if(action==='stop'&&!confirm('Остановить '+name+'?'))return;
  toast(name+': '+action+'…');
  try{const r=await(await api('/admin/services/'+name+'/'+action,{method:'POST'})).json();
    toast(name+' '+action+(r.ok?' ✓':' ✗'));}catch(e){toast('ошибка');}
  setTimeout(loadServices,1500);}

// ---- KEYS ----
async function loadKeys(){let d;try{d=await(await api('/admin/keys')).json();setConn(true);}catch(e){return;}
  $('#bill').textContent='тариф '+d.price_per_unit+' '+d.currency+'/unit · '+d.month;
  let total=0,mtotal=0;
  $('#keys-list').innerHTML=d.keys.map(k=>{total+=k.cost;mtotal+=k.month_cost;
    const used=k.quota_units?Math.min(100,Math.round(k.month_units/k.quota_units*100)):0,hot=used>=85?' hot':'';
    const svc=Object.entries(k.by_service||{}).map(([s,n])=>'<span class=badge style=font-size:10px>'+esc(s)+' '+n+'</span>').join(' ');
    return '<div class="kcard'+(k.active?'':' off')+'" id="c_'+k.full_key+'">'+
      '<div class=hd><div><b>'+esc(k.owner)+'</b><div class="mono mut" onclick="copy(\''+k.full_key+'\')" style=cursor:pointer title="копировать">'+esc(k.key)+' ⧉</div></div>'+
        '<div class=sw><input type=checkbox id="a_'+k.full_key+'" '+(k.active?'checked':'')+'><label for="a_'+k.full_key+'"></label></div></div>'+
      '<div class=grid style=margin-top:10px>'+
        '<div class=stat><div class=mut>За месяц</div><b>'+k.month_units+(k.quota_units?' / '+k.quota_units:'')+'</b> units'+(k.quota_units?'<div class="bar'+hot+'"><i style=width:'+used+'%></i></div>':'')+'</div>'+
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
  d.keys.forEach(k=>{const el=$('#a_'+CSS.escape(k.full_key));if(el)el.onchange=()=>saveLim(k.full_key,true);});
}
async function saveLim(key,silent){const c=$('#c_'+CSS.escape(key));
  const q=c.querySelector('.q_quota').value,r=c.querySelector('.q_rate').value,a=$('#a_'+CSS.escape(key)).checked?'true':'false';
  const fd=new FormData();fd.append('quota_units',q);fd.append('rate_per_min',r);fd.append('active',a);
  await api('/admin/keys/'+encodeURIComponent(key)+'/limits',{method:'POST',body:fd});
  toast(silent?'Статус применён':'Лимиты применены ✓');loadKeys();}
async function revoke(key){if(!confirm('Отозвать ключ? Доступ прекратится сразу.'))return;
  await api('/admin/keys/'+encodeURIComponent(key)+'/revoke',{method:'POST'});toast('Ключ отозван');loadKeys();}
async function createKey(){const o=$('#newOwner').value.trim();if(!o){toast('Введи владельца');return;}
  const fd=new FormData();fd.append('owner',o);fd.append('quota_units',$('#newQuota').value);fd.append('rate_per_min',$('#newRate').value);
  const d=await(await api('/admin/keys',{method:'POST',body:fd})).json();
  const box=$('#keyout');box.style.display='block';
  box.innerHTML='Ключ для <b>'+esc(o)+'</b> (показывается один раз):<div class="mono" style=margin-top:6px;word-break:break-all>'+d.api_key+'</div><button class=p style=margin-top:8px onclick="copy(\''+d.api_key+'\')">⧉ Скопировать</button>';
  $('#newOwner').value='';$('#newQuota').value='';$('#newRate').value='';toast('Ключ выдан ✓');loadKeys();}

// ---- ACTIVITY ----
async function loadActivity(){let s;try{s=await(await api('/admin/activity?limit='+RLIMIT)).json();setConn(true);}catch(e){return;}
  const r=s.recent||[];
  $('#recent').innerHTML=(r.length?r.map(e=>'<div class=rrow><span class=rico>'+svcIco(e.service)+'</span><span class=rown>'+esc(e.owner)+'</span><span class=badge style=font-size:10px>'+esc(e.service)+'</span><span class=rmeta>'+e.units+' u'+(e.tokens?' · '+e.tokens+' tok':'')+'</span><span class=rago>'+ago(e.ts)+'</span></div>').join(''):'<div class=mut>пока пусто</div>')+
    (r.length>=RLIMIT?'<div class=acts style=margin-top:10px><button class=ghost style=flex:1 onclick=moreRecent()>↓ Показать больше</button></div>':'');
}
function moreRecent(){RLIMIT=Math.min(200,RLIMIT+20);loadActivity();}

// ---- LAUNCH (использование / подключение / модели) ----
async function loadLaunch(){let d;try{d=await(await api('/admin/launch')).json();setConn(true);}catch(e){return;}
  const tools=[...d.builtin.map(t=>({...t})), ...d.custom.map((c,i)=>({...c,ci:i}))];
  $('#launchpad').innerHTML=tools.map(t=>'<a class=tile href="'+esc(t.url)+'" target=_blank rel=noopener><b>'+esc(t.title)+'</b><div class=d>'+esc(t.desc||'')+'</div><div class=go>Открыть ↗'+(t.ci!==undefined?' · <span style=color:var(--bad) onclick="event.preventDefault();event.stopPropagation();delLink('+t.ci+')">убрать</span>':'')+'</div></a>').join('');
  const a=d.api;const ex='curl -X POST '+a.base+'/v1/translate \\\n  -H "X-API-Key: <ВАШ_КЛЮЧ>" -H "Content-Type: application/json" \\\n  -d \'{"text":"Привет","to":"English","model":"'+a.model+'"}\'';
  $('#connect').innerHTML=
    '<div class=stat><div class=mut>Base URL · авторизация</div><b class=mono style=font-size:13px>'+esc(a.base)+'</b><div class=mut>заголовок: X-API-Key</div></div>'+
    '<pre id=curlex>'+esc(ex)+'</pre>'+
    '<div class=acts><button class=sm onclick=copyCurl()>⧉ Копировать пример</button><a href="'+esc(a.docs)+'" target=_blank rel=noopener style=flex:1><button class="sm ghost" style=width:100%>📖 Документация API</button></a></div>'+
    '<div class=mut sml style=margin-top:8px>Ключ выдаётся во вкладке 🔑 Ключи. Один ключ — все сервисы. Модель по умолчанию: <b>'+esc(a.model)+'</b>.</div>';
  loadModels();
}
function copyCurl(){copy($('#curlex').textContent);}
async function loadModels(){let d;try{d=await(await api('/admin/models')).json();}catch(e){return;}
  const pulls=Object.entries(d.pulls||{}).map(([n,p])=>{const ic=p.status==='pulling'?'⏳':(p.status==='done'?'✓':'✗');return '<div class="mut sml">'+ic+' '+esc(n)+' — '+esc(p.status)+(p.status==='error'?' ('+esc(p.msg)+')':'')+'</div>';}).join('');
  $('#pulls').innerHTML=pulls;
  $('#models-list').innerHTML=d.installed.map(m=>{
    const bi=(d.builtin||[]).includes(m.name);
    const ctrl=bi?'<span class="badge sml">✓ встроена</span>'
      :'<button class="sm'+(m.allowed?' p':'')+'" onclick="toggleModel(\''+m.name.replace(/'/g,"")+'\','+(!m.allowed)+')">'+(m.allowed?'✓ в API':'включить')+'</button>';
    return '<div class=svc><span class=nm>'+esc(m.name)+'<div class="mut sml">'+m.size_gb+' ГБ'+(m.name===d.default?' · по умолчанию':'')+'</div></span><span class=acts>'+ctrl+'</span></div>';
  }).join('');
}
async function pullModel(){const n=$('#mdlName').value.trim();if(!n){toast('Укажи имя модели');return;}
  try{await api('/admin/models/pull',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:n,description:$('#mdlDesc').value.trim()})});
  toast('Загрузка началась: '+n);$('#mdlName').value='';$('#mdlDesc').value='';setTimeout(loadModels,800);}catch(e){toast('ошибка');}}
async function toggleModel(name,allow){try{await api('/admin/models/allow',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:name,allow:allow})});
  toast(allow?'Модель включена в API':'Модель исключена');loadModels();}catch(e){toast('ошибка');}}
async function addLink(){const t=$('#lkTitle').value.trim(),u=$('#lkUrl').value.trim();if(!t||!u){toast('Нужны название и URL');return;}
  try{await api('/admin/launch',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({title:t,url:u})});
  $('#lkTitle').value='';$('#lkUrl').value='';toast('Ссылка добавлена');loadLaunch();}catch(e){toast('ошибка');}}
async function delLink(i){try{await api('/admin/launch/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({index:i})});toast('Убрано');loadLaunch();}catch(e){}}

// ---- dispatcher ----
function load(){if(TAB==='overview')loadOverview();else if(TAB==='launch')loadLaunch();else if(TAB==='services')loadServices();else if(TAB==='keys')loadKeys();else if(TAB==='activity')loadActivity();}
setInterval(()=>{if(TAB==='overview')loadOverview();else if(TAB==='services')loadServices();else if(TAB==='activity')loadActivity();else if(TAB==='launch')loadModels();},3000);
setInterval(()=>{if(TAB==='keys')loadKeys();},9000);
setInterval(()=>$('#clock').textContent=new Date().toLocaleTimeString(),1000);
load();
</script></body></html>"""
