from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Response
from redis.asyncio import BlockingConnectionPool, Redis
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff
from sqlalchemy.ext.asyncio import create_async_engine

from gateway.circuit import CircuitBreaker
from gateway import metrics
from gateway.admin import router as admin_router
from gateway.config import settings
from gateway.db import RequestLogger, init_db, spend_since
from gateway.keys import KeyStore
from gateway.proxy import router
from gateway.ratelimit import RateLimiter
from gateway.responsecache import ResponseCache
from gateway.semanticcache import Embedder, SemanticCache
from gateway.retry import RetryPolicy
from gateway.routing import LatencyTracker, Router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # One pooled client for the whole app, so connections to Anthropic are reused.
    app.state.http = httpx.AsyncClient(
        base_url=settings.anthropic_base_url,
        timeout=httpx.Timeout(settings.upstream_read_timeout_s, connect=settings.upstream_connect_timeout_s),
        limits=httpx.Limits(
            max_connections=settings.upstream_max_connections,
            max_keepalive_connections=settings.upstream_max_keepalive,
        ),
    )
    app.state.retry_policy = RetryPolicy(max_retries=settings.upstream_max_retries)
    app.state.fallback_chains = settings.fallback_chains
    app.state.refusal_fallbacks = settings.refusal_fallbacks
    app.state.breaker = CircuitBreaker(settings.circuit_failure_threshold, settings.circuit_cooldown_s)
    app.state.router = Router(settings.route_aliases, LatencyTracker())
    engine = create_async_engine(settings.database_url)
    await init_db(engine)
    app.state.engine = engine
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
    app.state.response_cache = ResponseCache(redis)
    app.state.semantic_cache = SemanticCache(redis, Embedder()) if settings.semantic_cache else None
    app.state.limiter = RateLimiter(redis, load_spend=lambda key_id, since: spend_since(engine, key_id, since))
    yield
    await app.state.http.aclose()
    await app.state.limiter.drain()
    await app.state.response_cache.drain()
    if app.state.semantic_cache:
        await app.state.semantic_cache.drain()
    await redis.aclose()
    await app.state.request_log.stop()
    await engine.dispose()


app = FastAPI(title="LLM Gateway", lifespan=lifespan)
app.include_router(router)
app.include_router(admin_router)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "upstream_key_configured": bool(settings.anthropic_api_key),
        "circuits": app.state.breaker.snapshot(),  # Models currently skipped or probing.
        "ttft_p50_ms": app.state.router.latency.snapshot(),  # Used by the "fastest" strategy.
    }


@app.get("/metrics")
async def prometheus_metrics() -> Response:
    return Response(metrics.render(app.state.breaker), media_type="text/plain; version=0.0.4")
