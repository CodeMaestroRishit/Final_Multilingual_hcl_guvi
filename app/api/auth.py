import os
from fastapi import Header, HTTPException, status
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY", "demo_key_123")

async def verify_api_key(x_api_key: str | None = Header(None, alias="x-api-key")):
    # Make the header optional at the validation layer so missing keys yield a clear 401,
    # not a FastAPI validation error (422) that gets rewrapped by the global handler.
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key",
        )
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid X-API-Key",
        )
    return x_api_key
