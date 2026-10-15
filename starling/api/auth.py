from __future__ import annotations

from fastapi import HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from starling.config import Settings

_bearer = HTTPBearer(auto_error=False)

# Paths that remain public even when API_KEY is set
PUBLIC_PATHS = {
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/metrics",
    "/",
}

PUBLIC_PREFIXES = (
    "/static/",
    "/ui/",
)


def make_auth_dependency(settings: Settings):
    async def require_api_key(
        request: Request,
        creds: HTTPAuthorizationCredentials | None = Security(_bearer),
    ) -> None:
        if not settings.api_key:
            return
        path = request.url.path
        if path in PUBLIC_PATHS:
            return
        if any(path.startswith(p) for p in PUBLIC_PREFIXES):
            return
        if creds is None or creds.scheme.lower() != "bearer" or creds.credentials != settings.api_key:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    return require_api_key
