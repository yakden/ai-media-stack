# Using the API

Base URL: `https://ai.1c-rus.ru/gw`  · Auth: header `X-API-Key: <your key>` (one key works for every service).

Every job response includes your **queue position + ETA**; every billable call is **metered** (see `/v1/billing`).

```bash
KEY=YOUR_API_KEY
B=https://ai.1c-rus.ru/gw

# who am I
curl $B/v1/ping -H "X-API-Key: $KEY"

# translate text (optionally pick a model for higher quality, auto-detect the source)
curl -X POST $B/v1/translate -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
     -d '{"text":"Привет, мир","to":"German","model":"qwen2.5vl:7b","detect":true}'
# -> {"translation":"Hallo, Welt","to":"German","model":"qwen2.5vl:7b","detected_source":"Russian","tokens":{...}}

# detect language only
curl -X POST $B/v1/detect -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
     -d '{"text":"Wie geht es dir?"}'   # -> {"language":"German"}

# LLM chat (models: llama3.2:3b, qwen2.5vl:7b)
curl -X POST $B/v1/llm/chat -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
     -d '{"model":"llama3.2:3b","prompt":"Say hello in 3 languages"}'

# floor-plan -> 3D project (async). Returns {job_id, queue:{position,ahead,eta_seconds,total_in_queue}}
curl -X POST $B/v1/3d/project -H "X-API-Key: $KEY" -F "files=@plan.png" -F "description=my flat"

# poll a job's status / queue
curl $B/v1/jobs/JOB_ID -H "X-API-Key: $KEY"

# voices (presets + saved) and live voice translation (audio in -> translated audio out)
curl $B/v1/voices -H "X-API-Key: $KEY"
curl -X POST $B/v1/voice/translate -H "X-API-Key: $KEY" \
     -F "audio=@clip.wav" -F "target_lang=en" -F "voice=Claribel Dervla" -o out.wav

# YOUR usage + cost (billing)
curl $B/v1/billing -H "X-API-Key: $KEY"
```


## Batch translation (for migrating data from another system)

Translate many fields/rows in one call:

```bash
curl -X POST $B/v1/translate/batch -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"to":"English","texts":["Заказ №123","Статус: оплачен","Адрес: Москва"]}'
# -> {"to":"English","count":3,"translated":3,"translations":["Order #123","Status: paid","Address: Moscow"],"tokens":...}
```
Up to 200 items per call · blanks return `""` · a failed item returns `null` (others still come back) · metered per non-empty item.


## Multi-language translation (one call → many languages)

```bash
# single text -> several languages
curl -X POST $B/v1/translate/multi -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"text":"Стол дубовый","to":["English","German","French","zh-cn"]}'
# -> {"translations":{"English":"Oak table","German":"Eichentisch","French":"...","zh-cn":"..."}}

# many texts × many languages (matrix)
curl -X POST $B/v1/translate/multi -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"texts":["Статус: оплачен","Доставка завтра"],"to":["English","Spanish"]}'
# -> {"results":[{"text":"...","translations":{"English":"...","Spanish":"..."}}, ...]}
```
Cap: texts × langs ≤ 200 · metered per produced translation.

## Choosing the translation model & auto-detect

All translation endpoints (`/v1/translate`, `/v1/translate/batch`, `/v1/translate/multi`) accept an optional
`"model"` field. Pick the quality/speed trade-off you need:

| Model | Use for |
|---|---|
| `llama3.2:3b` *(default)* | fast, light, good for short UI strings / bulk catalogs |
| `qwen2.5vl:7b` | higher quality, better with long/legal/technical text |
| `eurollm:9b` | **translation-tuned** for 35 European languages (Q6_K); fits fully on GPU — good quality/speed balance |
| `translategemma:12b` | Google **TranslateGemma** (Q6_K) — highest translation quality; heaviest, slower under concurrent GPU load |

`GET /v1/models` returns `translate_recommended` (best-for-translation first) so a client can default sensibly.

An unknown model returns **HTTP 400** with the allowed list (see `/v1/models`).

Auto-detect the source language with `"detect":true` (adds `"detected_source"` to the response).
Add `"skip_same":true` to return the original untouched when it's already in the target language —
handy when migrating mixed-language data so you don't waste tokens re-translating.

```bash
curl -X POST $B/v1/translate -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"text":"Already English","to":"English","skip_same":true}'
# -> {"translation":"Already English","skipped":true,"detected_source":"English","to":"English"}
```

**Limits:** keys can have a monthly unit **quota** (→ HTTP 402 when reached) and a **rate limit**
(→ HTTP 429). **Units** are weighted per service (3D project = 10, render = 3, voice = 1, avatar/dub = 20,
LLM = ~tokens/100). Billing = `units × price_per_unit`.
