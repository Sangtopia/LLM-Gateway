import gzip
import json

import httpx
from fastapi.testclient import TestClient

from gateway.config import settings
from gateway.main import app


class FakeRequestLog:
    def __init__(self):
        self.records: list[dict] = []

    def write(self, record: dict) -> None:
        self.records.append(record)


def make_client(handler) -> tuple[TestClient, FakeRequestLog]:
    # Not used as a context manager, so the lifespan doesn't replace our fakes.
    app.state.http = httpx.AsyncClient(base_url="https://upstream.test", transport=httpx.MockTransport(handler))
    app.state.request_log = FakeRequestLog()
    return TestClient(app), app.state.request_log


USAGE = {"input_tokens": 1_000, "output_tokens": 100, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}


def test_forwards_request_and_swaps_key():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["path"] = req.url.path
        seen["headers"] = req.headers
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"type": "message"}, headers={"request-id": "req_1"})

    client, _ = make_client(handler)
    resp = client.post(
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


def test_logs_successful_request_with_usage_and_cost():
    def handler(req: httpx.Request) -> httpx.Response:
        message = {"type": "message", "model": "claude-sonnet-5-5", "stop_reason": "end_turn", "usage": USAGE}
        return httpx.Response(200, json=message, headers={"request-id": "req_1"})

    client, log = make_client(handler)
    client.post("/v1/messages", json={"model": "claude-sonnet-5-5", "messages": []})

    [record] = log.records
    assert record["model"] == "claude-sonnet-5-5"
    assert record["stream"] is False
    assert record["status_code"] == 200
    assert record["outcome"] == "ok"
    assert record["stop_reason"] == "end_turn"
    assert record["input_tokens"] == 1_000
    assert record["output_tokens"] == 100
    assert record["cost_usd"] == (1_000 * 2 + 100 * 10) / 1e6
    assert record["upstream_request_id"] == "req_1"
    assert record["latency_ms"] >= record["upstream_ms"] >= 0
    assert record.get("ttft_ms") is None


def test_passes_through_and_logs_upstream_errors():
    def handler(req: httpx.Request) -> httpx.Response:
        error = {"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}}
        return httpx.Response(429, json=error, headers={"retry-after": "7"})

    client, log = make_client(handler)
    resp = client.post("/v1/messages", json={"model": "claude-haiku-5-5"})

    assert resp.status_code == 429
    assert resp.headers["retry-after"] == "7"
    [record] = log.records
    assert record["outcome"] == "error"
    assert record["error_type"] == "rate_limit_error"
    assert record["status_code"] == 429


def test_unreachable_upstream_returns_502_and_logs():
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client, log = make_client(handler)
    resp = client.post("/v1/messages", json={"model": "claude-haiku-5-5"})

    assert resp.status_code == 502
    assert resp.json()["type"] == "error"
    [record] = log.records
    assert record["error_type"] == "upstream_unreachable"


def sse(event: dict) -> bytes:
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()


SSE_EVENTS = [
    sse({"type": "message_start", "message": {"model": "claude-sonnet-5-5", "usage": {**USAGE, "output_tokens": 1}}}),
    sse({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}}),
    sse({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 100}}),
    sse({"type": "message_stop"}),
]


def stream_handler(events: list[bytes]):
    async def body():
        for event in events:
            yield event

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body(), headers={"content-type": "text/event-stream"})

    return handler


def test_streams_sse_through_unchanged_and_logs_usage():
    client, log = make_client(stream_handler(SSE_EVENTS))
    with client.stream("POST", "/v1/messages", json={"model": "claude-sonnet-5-5", "stream": True}) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert resp.read() == b"".join(SSE_EVENTS)

    [record] = log.records
    assert record["stream"] is True
    assert record["outcome"] == "ok"
    assert record["stop_reason"] == "end_turn"
    assert record["input_tokens"] == 1_000
    assert record["output_tokens"] == 100  # From message_delta, not the 1 in message_start.
    assert record["cost_usd"] == (1_000 * 2 + 100 * 10) / 1e6
    assert record["latency_ms"] >= record["ttft_ms"] >= record["upstream_ms"]


def test_stream_that_ends_without_message_stop_is_incomplete():
    client, log = make_client(stream_handler(SSE_EVENTS[:2]))
    with client.stream("POST", "/v1/messages", json={"model": "claude-sonnet-5-5", "stream": True}) as resp:
        resp.read()

    [record] = log.records
    assert record["outcome"] == "incomplete"


def test_stream_with_error_event_is_logged_as_error():
    overloaded = sse({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})
    client, log = make_client(stream_handler([*SSE_EVENTS[:2], overloaded]))
    with client.stream("POST", "/v1/messages", json={"model": "claude-sonnet-5-5", "stream": True}) as resp:
        resp.read()

    [record] = log.records
    assert record["status_code"] == 200  # Headers were already sent when the error arrived.
    assert record["outcome"] == "error"
    assert record["error_type"] == "overloaded_error"


def test_stream_request_that_errors_returns_json_error():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(529, json={"type": "error", "error": {"type": "overloaded_error"}})

    client, log = make_client(handler)
    resp = client.post("/v1/messages", json={"stream": True})

    assert resp.status_code == 529
    assert resp.json()["error"]["type"] == "overloaded_error"
    assert log.records[0]["error_type"] == "overloaded_error"


def test_stream_asks_upstream_not_to_compress_and_decodes_if_it_does():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["accept-encoding"] = req.headers.get("accept-encoding")
        return httpx.Response(
            200,
            content=gzip.compress(b"".join(SSE_EVENTS)),
            headers={"content-type": "text/event-stream", "content-encoding": "gzip"},
        )

    client, _ = make_client(handler)
    with client.stream("POST", "/v1/messages", json={"stream": True}) as resp:
        assert resp.read() == b"".join(SSE_EVENTS)
    assert seen["accept-encoding"] == "identity"
