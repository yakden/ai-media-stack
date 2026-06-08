# Contributing

Thanks for your interest! This is an opinionated reference platform — issues, ideas and PRs are welcome.

- **Bugs / ideas:** open an issue with what you expected vs. what happened.
- **PRs:** keep each service a small, readable single-file FastAPI app where possible; match the existing
  style and docstrings. Don't commit secrets (`.env`, keys) or model weights (see `.gitignore` / `NOTICE.md`).
- **Third-party models** stay referenced, not vendored — add new ones to `NOTICE.md` with their license.
- **Run locally:** each `services/<name>` has a README + `.env.example`. Most need only
  `pip install fastapi uvicorn httpx` and `uvicorn app:app`.

Be kind, be concise. 🙌
