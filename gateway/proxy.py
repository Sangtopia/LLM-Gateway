import time

import httpx
from fastapi import APIRouter, Request, Response

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


@router.post("/v1/messages")
async def messages(request: Request) -> Response:
    http: httpx.AsyncClient = request.app.state.http
    body = await request.body()

    start = time.perf_counter()
    upstream = await http.post("/v1/messages", content=body, headers=upstream_headers(request))
    upstream_ms = (time.perf_counter() - start) * 1000

    headers = client_headers(upstream)
    headers["x-gateway-upstream-ms"] = f"{upstream_ms:.1f}"
    return Response(content=upstream.content, status_code=upstream.status_code, headers=headers)
