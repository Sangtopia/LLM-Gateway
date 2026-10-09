import json

import httpx
from fastapi.testclient import TestClient

from gateway.config import settings
from gateway.main import app


def make_client(handler) -> TestClient:
    # Not used as a context manager, so the lifespan doesn't replace our mock client.
    app.state.http = httpx.AsyncClient(base_url="https://upstream.test", transport=httpx.MockTransport(handler))
    return TestClient(app)


def test_forwards_request_and_swaps_key():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["path"] = req.url.path
        seen["headers"] = req.headers
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"type": "message"}, headers={"request-id": "req_1"})

    resp = make_client(handler).post(
        "/v1/messages",
        json={"model": "claude-haiku-5-5", "max_tokens": 10, "messages": []},
        headers={"x-api-key": "gw-client-key", "anthropic-beta": "some-beta"},
    )

    assert resp.status_code == 200
    assert resp.json() == {"type": "message"}
    assert resp.headers["request-id"] == "req_1"
    assert "x-gateway-upstream-ms" in resp.headers
    assert seen["path"] == "/v1/messages"
    assert seen["headers"]["x-api-key"] == settings.anthropic_api_key
    assert seen["headers"]["anthropic-beta"] == "some-beta"
    assert seen["headers"]["anthropic-version"] == "2023-06-01"
    assert seen["body"]["model"] == "claude-haiku-5-5"


def test_passes_through_upstream_errors():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"type": "error"}, headers={"retry-after": "7"})

    resp = make_client(handler).post("/v1/messages", json={})

    assert resp.status_code == 429
    assert resp.headers["retry-after"] == "7"


SSE_EVENTS = [
    b'event: message_start\ndata: {"type":"message_start"}\n\n',
    b'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hi"}}\n\n',
    b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
]


def test_streams_sse_through_unchanged():
    async def events():
        for event in SSE_EVENTS:
            yield event

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=events(), headers={"content-type": "text/event-stream"})

    with make_client(handler).stream("POST", "/v1/messages", json={"stream": True}) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert resp.read() == b"".join(SSE_EVENTS)


def test_stream_request_that_errors_returns_json_error():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(529, json={"type": "error", "error": {"type": "overloaded_error"}})

    resp = make_client(handler).post("/v1/messages", json={"stream": True})

    assert resp.status_code == 529
    assert resp.json()["error"]["type"] == "overloaded_error"


def test_stream_asks_upstream_not_to_compress_and_decodes_if_it_does():
    import gzip

    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["accept-encoding"] = req.headers.get("accept-encoding")
        return httpx.Response(
            200,
            content=gzip.compress(b"".join(SSE_EVENTS)),
            headers={"content-type": "text/event-stream", "content-encoding": "gzip"},
        )

    with make_client(handler).stream("POST", "/v1/messages", json={"stream": True}) as resp:
        assert resp.read() == b"".join(SSE_EVENTS)
    assert seen["accept-encoding"] == "identity"
