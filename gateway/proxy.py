import json
import time

import httpx
from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse

from gateway.config import settings

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


def wants_stream(body: bytes) -> bool:
    try:
        parsed = json.loads(body)
    except ValueError:
        return False  # Let Anthropic reject the malformed body.
    return isinstance(parsed, dict) and parsed.get("stream") is True


@router.post("/v1/messages")
async def messages(request: Request) -> Response:
    http: httpx.AsyncClient = request.app.state.http
    body = await request.body()
    stream = wants_stream(body)
    headers = upstream_headers(request)
    if stream:
        # Compression would make the upstream buffer events into batches, and
        # we'd have to re-encode for the client anyway. Ask for plain bytes.
        headers["accept-encoding"] = "identity"
    upstream_req = http.build_request("POST", "/v1/messages", content=body, headers=headers)

    start = time.perf_counter()
    upstream = await http.send(upstream_req, stream=stream)
    upstream_ms = (time.perf_counter() - start) * 1000  # For streams: time until headers arrive.

    headers = client_headers(upstream)
    headers["x-gateway-upstream-ms"] = f"{upstream_ms:.1f}"

    # Errors arrive as a plain JSON body even when streaming was requested,
    # so read them fully and return them like a normal response.
    if not stream or upstream.status_code != 200:
        if stream:
            await upstream.aread()
            await upstream.aclose()
        return Response(content=upstream.content, status_code=upstream.status_code, headers=headers)

    async def relay():
        # Pass each chunk on as soon as it arrives. If the client disconnects,
        # this generator is cancelled and the finally block closes the upstream
        # connection, so Anthropic stops generating (and billing) tokens.
        try:
            async for chunk in upstream.aiter_bytes():  # decoded, in case upstream compresses anyway
                yield chunk
        finally:
            await upstream.aclose()

    return StreamingResponse(relay(), status_code=upstream.status_code, headers=headers)
