"""Authentication dependency.

Two accepted credentials, in priority order:

1. Trusted SSO header (default ``X-Email``) injected by the nginx cookie-SSO
   gateway after it validates the session against the SSO service. We trust the
   header because the app only ever binds to 127.0.0.1 and is reached solely
   through nginx; the presence of the header means the request is authenticated.
2. An optional ``Authorization: Bearer <API_KEY>`` for SSH-tunnel / CLI use that
   bypasses nginx (e.g. hitting 127.0.0.1:8120 directly over a tunnel).

When ``auth_required`` is false (local dev with neither nginx nor a key) the
dependency degrades to an anonymous principal so the API is usable out of the box.
"""

from __future__ import annotations

import hmac
from dataclasses import dataclass

from fastapi import Depends, HTTPException, Request, status

from .config import Settings, get_settings


@dataclass(frozen=True)
class Principal:
    """The authenticated caller. ``email`` is None for the anonymous principal."""

    email: str | None
    user: str | None
    via: str  # "sso" | "api_key" | "anonymous"

    @property
    def is_authenticated(self) -> bool:
        return self.via != "anonymous"


def _bearer_token(request: Request) -> str | None:
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return None


def require_user(
    request: Request, settings: Settings = Depends(get_settings)
) -> Principal:
    """FastAPI dependency yielding the authenticated :class:`Principal`.

    Raises 401 only when ``auth_required`` is set and no credential is present.
    """
    email = request.headers.get(settings.sso_email_header)
    user = request.headers.get(settings.sso_user_header)
    if email:
        return Principal(email=email, user=user or email, via="sso")

    token = _bearer_token(request)
    # Constant-time compare to avoid leaking the key via response timing.
    if settings.api_key and token and hmac.compare_digest(token, settings.api_key):
        return Principal(email=None, user="api-key", via="api_key")

    if not settings.auth_required:
        return Principal(email=None, user=None, via="anonymous")

    # Distinguish a bad key from missing creds for clearer client errors.
    detail = "Invalid API key." if token else "Authentication required."
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


# Alias: some routers depend on ``require_auth`` (e.g. used purely as a gate,
# ``_=Depends(require_auth)``), others on ``require_user`` (to read the Principal).
# Both resolve to the same dependency.
require_auth = require_user
