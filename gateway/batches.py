"""Message Batches through the gateway (/v1/messages/batches).

Anthropic runs batches asynchronously at 50% off every token: a good fit for evals
and other work that doesn't need an answer right away. On top of passing requests
through, the gateway:

  - records which gateway key created each batch. Batch ids are organization-wide
    at Anthropic, so without this any key could read or cancel any batch. A batch
    owned by another key answers 404, as if it didn't exist;
  - checks the key's rate limit and budget when a batch is created;
  - charges the key when results are fetched, at the batch price, by adding up the
    usage of each succeeded request as the results stream past. A batch is charged
    once, however many times its results are read;
  - rewrites `results_url` to point at the gateway. The Anthropic SDK fetches results
    from that URL with the client's key, which Anthropic wouldn't accept.
"""

import json
import time
import uuid
from datetime import UTC, datetime

import httpx
from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse
from sqlalchemy import insert, select, update

from gateway.db import RequestLogger, batches_table
from gateway.keys import ApiKey, KeyStore, extract_key
from gateway.models import cost_usd
from gateway.proxy import client_headers, elapsed_ms, error_response, fail, parse_json_object, upstream_headers, write_log
from gateway.ratelimit import RateLimiter

router = APIRouter()
BASE = "/v1/messages/batches"


def _public_base(request: Request) -> str:
    return str(request.base_url).rstrip("/")


def _rewrite(batch: dict, request: Request) -> dict:
    """Point results_url at the gateway, so the SDK fetches results through it."""
    if batch.get("results_url") and batch.get("id"):
        batch["results_url"] = f"{_public_base(request)}{BASE}/{batch['id']}/results"
    return batch


def _record(endpoint: str, key: ApiKey | None) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "created_at": datetime.now(UTC),
        "endpoint": endpoint,
        "api_key_id": key.id if key else None,
        "provider": "anthropic",
        "stream": False,
        "cache_hit": False,
        "fallback_used": False,
    }


async def _authenticate(request: Request) -> ApiKey | None:
    keys: KeyStore = request.app.state.keys
    return await keys.authenticate(extract_key(request.headers))


async def _owner(request: Request, batch_id: str) -> str | None:
    async with request.app.state.engine.connect() as conn:
        row = await conn.execute(select(batches_table.c.api_key_id).where(batches_table.c.id == batch_id))
        return row.scalar_one_or_none()


def _not_found(batch_id: str) -> Response:
    return error_response(404, "not_found_error", f"No batch {batch_id} for this gateway key.")


def _unauthorized() -> Response:
    return error_response(401, "authentication_error", "Invalid or missing gateway API key.")


async def _forward(request: Request, method: str, path: str, **kwargs) -> httpx.Response:
    http: httpx.AsyncClient = request.app.state.http
    return await http.request(method, path, headers=upstream_headers(request), timeout=60.0, **kwargs)


def _json_response(upstream: httpx.Response, payload: dict | list | None = None) -> Response:
    content = json.dumps(payload).encode() if payload is not None else upstream.content
    headers = {k: v for k, v in client_headers(upstream).items() if k.lower() != "content-length"}
    return Response(content=content, status_code=upstream.status_code, headers=headers, media_type="application/json")


@router.post(BASE)
async def create_batch(request: Request) -> Response:
    started = time.perf_counter()
    request_log: RequestLogger = request.app.state.request_log
    key = await _authenticate(request)
    record = _record("batch_create", key)
    if key is None:
        return fail(record, request_log, started, 401, "authentication_error", "Invalid or missing gateway API key.")
    limiter: RateLimiter = request.app.state.limiter
    decision = await limiter.admit(key)
    if not decision.allowed:
        record.update(status_code=429, outcome="error", latency_ms=elapsed_ms(started),
                      error_type="gateway_budget_exceeded" if decision.limit == "monthly budget" else "gateway_rate_limited")
        write_log(request_log, record)
        response = error_response(429, "rate_limit_error", decision.message or "Gateway limit exceeded.")
        response.headers["retry-after"] = str(decision.retry_after_s)
        return response

    body = await request.body()
    try:
        upstream = await _forward(request, "POST", BASE, content=body)
    except httpx.HTTPError as exc:
        return fail(record, request_log, started, 502, "upstream_unreachable", f"Could not reach Anthropic: {type(exc).__name__}")
    batch = parse_json_object(upstream.content)
    if upstream.status_code == 200 and batch.get("id"):
        requests = parse_json_object(body).get("requests") or []
        models = sorted({(r.get("params") or {}).get("model") for r in requests if isinstance(r, dict)} - {None})
        record["model"] = ",".join(models) or None
        async with request.app.state.engine.begin() as conn:
            await conn.execute(insert(batches_table).values(
                id=batch["id"], api_key_id=key.id, created_at=datetime.now(UTC), request_count=len(requests), charged=False,
            ))
        record.update(outcome="ok", upstream_request_id=batch["id"])
    else:
        record.update(outcome="error", error_type=(batch.get("error") or {}).get("type"))
    record.update(status_code=upstream.status_code, latency_ms=elapsed_ms(started))
    write_log(request_log, record)
    return _json_response(upstream, _rewrite(batch, request) if upstream.status_code == 200 else None)


@router.get(BASE)
async def list_batches(request: Request) -> Response:
    """Anthropic's list, filtered to this key's batches. Pages can come back short as a result."""
    key = await _authenticate(request)
    if key is None:
        return _unauthorized()
    upstream = await _forward(request, "GET", BASE, params=dict(request.query_params))
    page = parse_json_object(upstream.content)
    if upstream.status_code != 200:
        return _json_response(upstream)
    async with request.app.state.engine.connect() as conn:
        owned = {r for r in (await conn.execute(
            select(batches_table.c.id).where(batches_table.c.api_key_id == key.id))).scalars()}
    page["data"] = [_rewrite(b, request) for b in page.get("data", []) if b.get("id") in owned]
    return _json_response(upstream, page)


@router.get(BASE + "/{batch_id}")
async def retrieve_batch(request: Request, batch_id: str) -> Response:
    key = await _authenticate(request)
    if key is None:
        return _unauthorized()
    if await _owner(request, batch_id) != key.id:
        return _not_found(batch_id)
    upstream = await _forward(request, "GET", f"{BASE}/{batch_id}")
    return _json_response(upstream, _rewrite(parse_json_object(upstream.content), request) if upstream.status_code == 200 else None)


@router.post(BASE + "/{batch_id}/cancel")
async def cancel_batch(request: Request, batch_id: str) -> Response:
    key = await _authenticate(request)
    if key is None:
        return _unauthorized()
    if await _owner(request, batch_id) != key.id:
        return _not_found(batch_id)
    upstream = await _forward(request, "POST", f"{BASE}/{batch_id}/cancel")
    return _json_response(upstream, _rewrite(parse_json_object(upstream.content), request) if upstream.status_code == 200 else None)


@router.delete(BASE + "/{batch_id}")
async def delete_batch(request: Request, batch_id: str) -> Response:
    key = await _authenticate(request)
    if key is None:
        return _unauthorized()
    if await _owner(request, batch_id) != key.id:
        return _not_found(batch_id)
    upstream = await _forward(request, "DELETE", f"{BASE}/{batch_id}")
    return _json_response(upstream)


@router.get(BASE + "/{batch_id}/results")
async def batch_results(request: Request, batch_id: str) -> Response:
    """Stream the results (JSONL) and charge the batch to the key once, at the batch price."""
    started = time.perf_counter()
    key = await _authenticate(request)
    if key is None:
        return _unauthorized()
    if await _owner(request, batch_id) != key.id:
        return _not_found(batch_id)
    http: httpx.AsyncClient = request.app.state.http
    upstream = await http.send(
        http.build_request("GET", f"{BASE}/{batch_id}/results", headers=upstream_headers(request), timeout=300.0),
        stream=True,
    )
    if upstream.status_code != 200:
        await upstream.aread()
        await upstream.aclose()
        return _json_response(upstream)

    async def relay():
        totals = {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0}
        cost, succeeded, models = 0.0, 0, set()
        buffer = b""
        complete = False
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
                buffer += chunk
                *lines, buffer = buffer.split(b"\n")
                for line in lines:
                    result = parse_json_object(line)
                    message = (result.get("result") or {}).get("message") or {}
                    if (result.get("result") or {}).get("type") != "succeeded":
                        continue
                    usage = message.get("usage") or {}
                    succeeded += 1
                    models.add(message.get("model"))
                    cost += cost_usd(message.get("model"), usage, batch=True) or 0.0
                    totals["input_tokens"] += usage.get("input_tokens") or 0
                    totals["output_tokens"] += usage.get("output_tokens") or 0
                    totals["cache_read_tokens"] += usage.get("cache_read_input_tokens") or 0
                    totals["cache_write_tokens"] += usage.get("cache_creation_input_tokens") or 0
            complete = True
        finally:
            await upstream.aclose()
            if complete:
                await _charge_once(request, key, batch_id, cost, totals, succeeded, models, started)

    headers = {k: v for k, v in client_headers(upstream).items() if k.lower() != "content-length"}
    return StreamingResponse(relay(), status_code=200, headers=headers, media_type="application/binary")


async def _charge_once(request, key, batch_id, cost, totals, succeeded, models, started) -> None:
    """Charge the key for this batch, unless an earlier read of its results already did."""
    async with request.app.state.engine.begin() as conn:
        claimed = await conn.execute(
            update(batches_table)
            .where(batches_table.c.id == batch_id, batches_table.c.charged.is_(False))
            .values(charged=True, cost_usd=cost, succeeded=succeeded, **totals)
        )
    if claimed.rowcount != 1:
        return
    limiter: RateLimiter = request.app.state.limiter
    limiter.record_usage(key, 0, cost)
    record = _record("batch_results", key)
    record.update(
        status_code=200, outcome="ok", upstream_request_id=batch_id, cost_usd=cost,
        model=",".join(sorted(m for m in models if m)) or None, served_model=None,
        latency_ms=elapsed_ms(started), **totals,
    )
    write_log(request.app.state.request_log, record)
