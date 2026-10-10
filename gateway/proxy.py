import json
import time
import uuid
from datetime import UTC, datetime

import httpx
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from gateway.config import settings
from gateway.db import RequestLogger
from gateway.keys import KeyStore, extract_key
from gateway.models import cost_usd
from gateway.ratelimit import RateLimiter, tokens_for_rate_limit
from gateway.retry import Attempt, RetryPolicy, send_with_retries, should_retry
from gateway.sse import StreamTracker

router = APIRouter()

# Client headers passed through to Anthropic. Everything else (notably the
# client's x-api-key, which will be a gateway key) is dropped.
FORWARD_REQUEST_HEADERS = ("anthropic-version", "anthropic-beta", "content-type")

# Upstream headers passed back to the client.
FORWARD_RESPONSE_PREFIXES = ("anthropic-", "retry-after", "request-id", "content-type")


def upstream_headers(request: Request) -> dict[str, str]:
    headers = {k: v for k, v in request.headers.items() if k.lower() in FORWARD_REQUEST_HEADERS}
    headers.setdefault("anthropic-version", "2023-06-01")
    headers.setdefault("content-type", "application/json")
    headers["x-api-key"] = settings.anthropic_api_key
    return headers


def client_headers(upstream: httpx.Response) -> dict[str, str]:
    return {k: v for k, v in upstream.headers.items() if k.lower().startswith(FORWARD_RESPONSE_PREFIXES)}


def parse_json_object(raw: bytes) -> dict:
    try:
        parsed = json.loads(raw)
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def elapsed_ms(since: float) -> float:
    return (time.perf_counter() - since) * 1000


def add_usage(record: dict, model: str | None, usage: dict) -> None:
    cache_creation = usage.get("cache_creation") or {}
    cache_write = usage.get("cache_creation_input_tokens")
    if cache_write is None and cache_creation:
        cache_write = sum(v for v in cache_creation.values() if isinstance(v, int))
    record.update(
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        cache_read_tokens=usage.get("cache_read_input_tokens"),
        cache_write_tokens=cache_write,
        cost_usd=cost_usd(model, usage) if model and usage else None,
    )


def error_response(status: int, error_type: str, message: str) -> JSONResponse:
    # Same shape as Anthropic's own errors, so SDKs parse it normally.
    return JSONResponse({"type": "error", "error": {"type": error_type, "message": message}}, status_code=status)


def write_log(request_log: RequestLogger, record: dict) -> None:
    if isinstance(record.get("attempts"), list):
        record["attempts"] = [a.as_dict() if isinstance(a, Attempt) else a for a in record["attempts"]]
    request_log.write(record)


def fail(record: dict, request_log: RequestLogger, started: float, status: int, error_type: str, message: str) -> JSONResponse:
    """Log a request that failed inside the gateway and return an Anthropic-shaped error."""
    record.update(status_code=status, outcome="error", error_type=error_type, latency_ms=elapsed_ms(started))
    write_log(request_log, record)
    api_error = "timeout_error" if status == 504 else "api_error"
    return error_response(status, api_error, message)


@router.post("/v1/messages")
async def messages(request: Request) -> Response:
    started = time.perf_counter()
    http: httpx.AsyncClient = request.app.state.http
    request_log: RequestLogger = request.app.state.request_log
    keys: KeyStore = request.app.state.keys
    limiter: RateLimiter = request.app.state.limiter

    body = await request.body()
    parsed = parse_json_object(body)
    stream = parsed.get("stream") is True
    model = parsed.get("model") if isinstance(parsed.get("model"), str) else None
    record = {
        "id": str(uuid.uuid4()),
        "created_at": datetime.now(UTC),
        "model": model,
        "provider": "anthropic",
        "stream": stream,
        "cache_hit": False,
        "fallback_used": False,
    }

    api_key = await keys.authenticate(extract_key(request.headers))
    if api_key is None:
        record.update(status_code=401, outcome="error", error_type="authentication_error", latency_ms=elapsed_ms(started))
        request_log.write(record)
        return error_response(401, "authentication_error", "Invalid or missing gateway API key.")
    record["api_key_id"] = api_key.id

    decision = await limiter.admit(api_key)
    if not decision.allowed:
        error_type = "gateway_budget_exceeded" if decision.limit == "monthly budget" else "gateway_rate_limited"
        record.update(status_code=429, outcome="error", error_type=error_type, latency_ms=elapsed_ms(started))
        request_log.write(record)
        response = error_response(429, "rate_limit_error", decision.message or "Gateway limit exceeded.")
        response.headers["retry-after"] = str(decision.retry_after_s)
        return response

    headers = upstream_headers(request)
    timeout = None
    if stream:
        # Compression would make the upstream buffer events into batches, and
        # we'd have to re-encode for the client anyway. Ask for plain bytes.
        headers["accept-encoding"] = "identity"
        # For streams the read timeout is the longest allowed gap between events.
        timeout = httpx.Timeout(settings.upstream_stream_idle_timeout_s, connect=settings.upstream_connect_timeout_s)

    def send_to(target_body: bytes):
        async def send() -> httpx.Response:
            upstream_req = http.build_request(
                "POST", "/v1/messages", content=target_body, headers=headers, timeout=timeout
            )
            return await http.send(upstream_req, stream=stream)

        return send

    policy: RetryPolicy = request.app.state.retry_policy
    attempts: list[Attempt] = []
    record["attempts"] = attempts

    chain = [model]
    if model and request.headers.get("x-gateway-fallback", "").lower() != "off":
        chain += request.app.state.fallback_chains.get(model, [])

    upstream = None
    primary_failure: httpx.Response | None = None  # The requested model's last error, if it failed.
    for position, target in enumerate(chain):
        last = position == len(chain) - 1
        target_body = body if target == model else json.dumps({**parsed, "model": target}).encode()
        try:
            upstream = await send_with_retries(send_to(target_body), policy, attempts, target)
        except httpx.ReadTimeout:
            return fail(record, request_log, started, 504, "upstream_timeout", "Anthropic did not respond in time.")
        except httpx.TransportError as exc:
            if not last:
                continue
            message = f"Could not reach Anthropic: {type(exc).__name__}"
            return fail(record, request_log, started, 502, "upstream_unreachable", message)

        if upstream.status_code < 400:
            break
        if position > 0 and not should_retry(upstream):
            # The fallback model rejected the request outright (often a parameter
            # it doesn't support). Report the requested model's failure instead of
            # an error about a model the app never asked for.
            await upstream.aclose()
            if primary_failure is None:
                return fail(record, request_log, started, 502, "upstream_unreachable", "Could not reach Anthropic.")
            upstream = primary_failure
            break
        if last or not should_retry(upstream):
            break
        # This model is failing; fall back to the next one.
        await upstream.aread()
        await upstream.aclose()
        if position == 0:
            primary_failure = upstream

    served = attempts[-1].model if upstream is not primary_failure else model
    upstream_ms = elapsed_ms(started)  # Including retries and fallbacks. For streams: until headers arrive.
    record.update(
        status_code=upstream.status_code,
        upstream_ms=upstream_ms,
        upstream_request_id=upstream.headers.get("request-id"),
        served_model=served if upstream.status_code < 400 else None,
        fallback_used=upstream.status_code < 400 and served != model,
    )
    response_headers = client_headers(upstream)
    response_headers["x-gateway-upstream-ms"] = f"{upstream_ms:.1f}"
    response_headers["x-gateway-attempts"] = str(len(attempts))
    if upstream.status_code < 400 and served:
        response_headers["x-gateway-model"] = served

    # Errors arrive as a plain JSON body even when streaming was requested,
    # so read them fully and return them like a normal response.
    if not stream or upstream.status_code != 200:
        if stream and not upstream.is_stream_consumed:
            await upstream.aread()
            await upstream.aclose()
        payload = parse_json_object(upstream.content)
        if upstream.status_code == 200:
            record.update(outcome="ok", stop_reason=payload.get("stop_reason"))
            usage = payload.get("usage") or {}
            add_usage(record, payload.get("model") or model, usage)
            limiter.record_usage(api_key, tokens_for_rate_limit(usage), record["cost_usd"])
        else:
            error_type = (payload.get("error") or {}).get("type") or f"http_{upstream.status_code}"
            record.update(outcome="error", error_type=error_type)
        record["latency_ms"] = elapsed_ms(started)
        write_log(request_log, record)
        return Response(content=upstream.content, status_code=upstream.status_code, headers=response_headers)

    async def relay():
        # Pass each chunk on as soon as it arrives, reading usage as it goes by.
        # If the client disconnects, this generator is cancelled and the finally
        # block closes the upstream connection, so Anthropic stops generating
        # (and billing) tokens nobody will read.
        tracker = StreamTracker()
        try:
            async for chunk in upstream.aiter_bytes():  # decoded, in case upstream compresses anyway
                tracker.feed(chunk)
                if tracker.saw_content and record.get("ttft_ms") is None:
                    record["ttft_ms"] = elapsed_ms(started)
                yield chunk
        finally:
            await upstream.aclose()
            if tracker.error_type:
                record.update(outcome="error", error_type=tracker.error_type)
            else:
                record["outcome"] = "ok" if tracker.completed else "incomplete"
            record.update(stop_reason=tracker.stop_reason, latency_ms=elapsed_ms(started))
            add_usage(record, tracker.model or model, tracker.usage)
            limiter.record_usage(api_key, tokens_for_rate_limit(tracker.usage), record["cost_usd"])
            write_log(request_log, record)

    return StreamingResponse(relay(), status_code=upstream.status_code, headers=response_headers)
