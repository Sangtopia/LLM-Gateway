from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from gateway.config import settings
from gateway.proxy import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # One pooled client for the whole app, so connections to Anthropic are reused.
    app.state.http = httpx.AsyncClient(base_url=settings.anthropic_base_url, timeout=httpx.Timeout(600.0, connect=10.0))
    yield
    await app.state.http.aclose()


app = FastAPI(title="LLM Gateway", lifespan=lifespan)
app.include_router(router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "upstream_key_configured": bool(settings.anthropic_api_key)}
