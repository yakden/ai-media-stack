# api-gateway
Multi-tenant REST gateway: **API-key auth**, per-key **usage metering** (requests + weighted units + LLM tokens),
**monthly quotas** (402) and **rate limits** (429), **billing** (append-only event log + price table) and an
**admin dashboard**. One key works across every service (`/v1/3d/*`, `/v1/voice/*`, `/v1/avatar`, `/v1/dub`,
`/v1/llm/*`, `/v1/translate`). Admin: issue/revoke keys, set limits, see live load + queue, read billing.
Single file: `app.py` (FastAPI, :8190). Copy `.env.example`→`.env`.
