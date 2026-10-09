import json
import time
import uuid
from datetime import UTC, datetime

import httpx
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from gateway.config import settings
from gateway.db import RequestLogger
from gateway.models import cost_usd
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


@router.post("/v1/messages")
async def messages(request: Request) -> Response:
    started = time.perf_counter()
    http: httpx.AsyncClient = request.app.state.http
    request_log: RequestLogger = request.app.state.request_log

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

    headers = upstream_headers(request)
    if stream:
        # Compression would make the upstream buffer events into batches, and
        # we'd have to re-encode for the client anyway. Ask for plain bytes.
        headers["accept-encoding"] = "identity"
    upstream_req = http.build_request("POST", "/v1/messages", content=body, headers=headers)

    try:
        upstream = await http.send(upstream_req, stream=stream)
    except httpx.TransportError as exc:
        record.update(status_code=502, outcome="error", error_type="upstream_unreachable", latency_ms=elapsed_ms(started))
        request_log.write(record)
        return error_response(502, "api_error", f"Could not reach Anthropic: {type(exc).__name__}")

    upstream_ms = elapsed_ms(started)  # For streams: time until headers arrive.
    record.update(
        status_code=upstream.status_code,
        upstream_ms=upstream_ms,
        upstream_request_id=upstream.headers.get("request-id"),
    )
    response_headers = client_headers(upstream)
    response_headers["x-gateway-upstream-ms"] = f"{upstream_ms:.1f}"

    # Errors arrive as a plain JSON body even when streaming was requested,
    # so read them fully and return them like a normal response.
    if not stream or upstream.status_code != 200:
        if stream:
            await upstream.aread()
            await upstream.aclose()
        payload = parse_json_object(upstream.content)
        if upstream.status_code == 200:
            record.update(outcome="ok", stop_reason=payload.get("stop_reason"))
            add_usage(record, payload.get("model") or model, payload.get("usage") or {})
        else:
            error_type = (payload.get("error") or {}).get("type") or f"http_{upstream.status_code}"
            record.update(outcome="error", error_type=error_type)
        record["latency_ms"] = elapsed_ms(started)
        request_log.write(record)
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
            request_log.write(record)

    return StreamingResponse(relay(), status_code=upstream.status_code, headers=response_headers)
