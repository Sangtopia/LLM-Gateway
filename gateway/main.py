from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from redis.asyncio import BlockingConnectionPool, Redis
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff
from sqlalchemy.ext.asyncio import create_async_engine

from gateway.config import settings
from gateway.db import RequestLogger, init_db, spend_since
from gateway.keys import KeyStore
from gateway.proxy import router
from gateway.ratelimit import RateLimiter


@asynccontextmanager
async def lifespan(app: FastAPI):
    # One pooled client for the whole app, so connections to Anthropic are reused.
    app.state.http = httpx.AsyncClient(base_url=settings.anthropic_base_url, timeout=httpx.Timeout(600.0, connect=10.0))
    engine = create_async_engine(settings.database_url)
    await init_db(engine)
    app.state.keys = KeyStore(engine)
    app.state.request_log = RequestLogger(engine)
    await app.state.request_log.start()
    # Short timeouts and no automatic retries (redis-py's default retries add up
    # to ~5s): if Redis is slow or down, the limiter fails open quickly instead of
    # stalling requests. Pool size stays under Redis Cloud free tier's 30.
    redis = Redis(
        connection_pool=BlockingConnectionPool.from_url(
            settings.redis_url,
            max_connections=20,
            timeout=1.0,
            socket_timeout=1.0,
            socket_connect_timeout=1.0,
            retry=Retry(NoBackoff(), 0),
        )
    )
    app.state.limiter = RateLimiter(redis, load_spend=lambda key_id, since: spend_since(engine, key_id, since))
    yield
    await app.state.http.aclose()
    await app.state.limiter.drain()
    await redis.aclose()
    await app.state.request_log.stop()
    await engine.dispose()


app = FastAPI(title="LLM Gateway", lifespan=lifespan)
app.include_router(router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "upstream_key_configured": bool(settings.anthropic_api_key)}
