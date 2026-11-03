import secrets

from fastapi import Header, HTTPException

from .config import settings


async def verify_token(
    x_service_token: str | None = Header(default=None, alias="X-Service-Token"),
):
    if not x_service_token or not secrets.compare_digest(x_service_token, settings.service_token):
        raise HTTPException(status_code=401, detail="invalid service token")
    return True
