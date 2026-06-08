# Using the API

Base URL: `https://ai.1c-rus.ru/gw`  · Auth: header `X-API-Key: <your key>` (one key works for every service).

Every job response includes your **queue position + ETA**; every billable call is **metered** (see `/v1/billing`).

```bash
KEY=YOUR_API_KEY
B=https://ai.1c-rus.ru/gw

# who am I
curl $B/v1/ping -H "X-API-Key: $KEY"

# translate text
curl -X POST $B/v1/translate -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
     -d '{"text":"Привет, мир","to":"German"}'

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

**Limits:** keys can have a monthly unit **quota** (→ HTTP 402 when reached) and a **rate limit**
(→ HTTP 429). **Units** are weighted per service (3D project = 10, render = 3, voice = 1, avatar/dub = 20,
LLM = ~tokens/100). Billing = `units × price_per_unit`.
