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
