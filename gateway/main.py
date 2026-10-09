from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import create_async_engine

from gateway.config import settings
from gateway.db import RequestLogger
from gateway.proxy import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # One pooled client for the whole app, so connections to Anthropic are reused.
    app.state.http = httpx.AsyncClient(base_url=settings.anthropic_base_url, timeout=httpx.Timeout(600.0, connect=10.0))
    engine = create_async_engine(settings.database_url)
    app.state.request_log = RequestLogger(engine)
    await app.state.request_log.start()
    yield
    await app.state.http.aclose()
    await app.state.request_log.stop()
    await engine.dispose()


app = FastAPI(title="LLM Gateway", lifespan=lifespan)
app.include_router(router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "upstream_key_configured": bool(settings.anthropic_api_key)}
