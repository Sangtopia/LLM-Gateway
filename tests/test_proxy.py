import gzip
import json

import httpx
from fastapi.testclient import TestClient

from gateway.circuit import CircuitBreaker
from gateway.config import settings
from gateway.keys import ApiKey
from gateway.main import app
from gateway.ratelimit import ALLOW, Decision
from gateway.retry import RetryPolicy

TEST_KEY = "gw-test-key"


class FakeRequestLog:
    def __init__(self):
        self.records: list[dict] = []

    def write(self, record: dict) -> None:
        self.records.append(record)


class FakeKeyStore:
    async def authenticate(self, plaintext: str | None) -> ApiKey | None:
        return ApiKey(id="key_test", name="test") if plaintext == TEST_KEY else None


class FakeLimiter:
    def __init__(self, decision: Decision = ALLOW):
        self.decision = decision
        self.charged: list[tuple[str, int, float | None]] = []

    async def admit(self, key: ApiKey) -> Decision:
        return self.decision

    def record_usage(self, key: ApiKey, tokens: int, cost_usd: float | None) -> None:
        self.charged.append((key.id, tokens, cost_usd))


def make_client(handler) -> tuple[TestClient, FakeRequestLog]:
    # Not used as a context manager, so the lifespan doesn't replace our fakes.
    app.state.http = httpx.AsyncClient(base_url="https://upstream.test", transport=httpx.MockTransport(handler))
    app.state.request_log = FakeRequestLog()
    app.state.keys = FakeKeyStore()
    app.state.limiter = FakeLimiter()
    app.state.retry_policy = RetryPolicy(max_retries=2, base_delay_s=0, max_delay_s=0)
    app.state.fallback_chains = {"claude-sonnet-5-5": ["claude-sonnet-5", "claude-haiku-5-5"]}
    app.state.breaker = CircuitBreaker(failure_threshold=1_000)  # Effectively off; tested on its own.
    return TestClient(app, headers={"x-api-key": TEST_KEY}), app.state.request_log


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
        headers={"anthropic-beta": "some-beta"},
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
    assert record["api_key_id"] == "key_test"
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


def test_rejects_missing_or_unknown_key_without_calling_upstream():
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        return httpx.Response(200, json={})

    client, log = make_client(handler)
    missing = client.post("/v1/messages", json={"model": "claude-haiku-5-5"}, headers={"x-api-key": ""})
    wrong = client.post("/v1/messages", json={"model": "claude-haiku-5-5"}, headers={"x-api-key": "gw-wrong"})

    for resp in (missing, wrong):
        assert resp.status_code == 401
        assert resp.json()["error"]["type"] == "authentication_error"
    assert calls == []
    assert [r["error_type"] for r in log.records] == ["authentication_error"] * 2
    assert all(r.get("api_key_id") is None for r in log.records)


def test_accepts_key_as_bearer_token():
    client, _ = make_client(lambda req: httpx.Response(200, json={}))
    resp = client.post(
        "/v1/messages", json={}, headers={"x-api-key": "", "authorization": f"Bearer {TEST_KEY}"}
    )
    assert resp.status_code == 200


def test_rate_limited_key_gets_429_with_retry_after_and_upstream_is_not_called():
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        return httpx.Response(200, json={})

    client, log = make_client(handler)
    app.state.limiter = FakeLimiter(Decision(False, retry_after_s=12, limit="requests", message="too fast"))
    resp = client.post("/v1/messages", json={"model": "claude-haiku-5-5"})

    assert resp.status_code == 429
    assert resp.headers["retry-after"] == "12"
    assert resp.json()["error"] == {"type": "rate_limit_error", "message": "too fast"}
    assert calls == []
    [record] = log.records
    assert record["error_type"] == "gateway_rate_limited"
    assert record["api_key_id"] == "key_test"


def test_budget_exceeded_is_logged_separately_from_rate_limits():
    client, log = make_client(lambda req: httpx.Response(200, json={}))
    app.state.limiter = FakeLimiter(Decision(False, retry_after_s=86_400, limit="monthly budget", message="broke"))
    resp = client.post("/v1/messages", json={"model": "claude-haiku-5-5"})

    assert resp.status_code == 429
    assert resp.headers["retry-after"] == "86400"
    assert log.records[0]["error_type"] == "gateway_budget_exceeded"


def test_charges_tokens_after_non_streaming_response():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model": "claude-sonnet-5-5", "usage": USAGE})

    client, _ = make_client(handler)
    client.post("/v1/messages", json={"model": "claude-sonnet-5-5"})
    assert app.state.limiter.charged == [("key_test", 1_100, (1_000 * 2 + 100 * 10) / 1e6)]


def test_charges_tokens_after_stream_ends():
    client, _ = make_client(stream_handler(SSE_EVENTS))
    with client.stream("POST", "/v1/messages", json={"model": "claude-sonnet-5-5", "stream": True}) as resp:
        resp.read()
    assert app.state.limiter.charged == [("key_test", 1_100, (1_000 * 2 + 100 * 10) / 1e6)]


def test_retries_overloaded_upstream_and_logs_each_attempt():
    statuses = iter([529, 529, 200])

    def handler(req: httpx.Request) -> httpx.Response:
        status = next(statuses)
        if status == 200:
            return httpx.Response(200, json={"model": "claude-sonnet-5-5", "usage": USAGE})
        return httpx.Response(status, json={"type": "error", "error": {"type": "overloaded_error"}})

    client, log = make_client(handler)
    resp = client.post("/v1/messages", json={"model": "claude-sonnet-5-5"})

    assert resp.status_code == 200
    assert resp.headers["x-gateway-attempts"] == "3"
    [record] = log.records
    assert record["outcome"] == "ok"
    assert [a["status"] for a in record["attempts"]] == [529, 529, 200]
    assert all(a["model"] == "claude-sonnet-5-5" for a in record["attempts"])


def test_read_timeout_returns_504_without_retrying():
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        raise httpx.ReadTimeout("slow", request=req)

    client, log = make_client(handler)
    resp = client.post("/v1/messages", json={"model": "claude-haiku-5-5"})

    assert resp.status_code == 504
    assert resp.json()["error"]["type"] == "timeout_error"
    assert len(calls) == 1
    assert log.records[0]["error_type"] == "upstream_timeout"


def by_model(behavior: dict):
    """Upstream whose response depends on the requested model. Records models called."""
    seen: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        payload = json.loads(req.content)
        model = payload["model"]
        seen.append(model)
        status = behavior.get(model, 200)
        if status != 200:
            error_type = "invalid_request_error" if status == 400 else "overloaded_error"
            return httpx.Response(status, json={"type": "error", "error": {"type": error_type, "message": model}})
        if payload.get("stream"):
            events = [e.replace(b"claude-sonnet-5-5", model.encode()) for e in SSE_EVENTS]
            return stream_handler(events)(req)
        return httpx.Response(200, json={"model": model, "usage": USAGE, "stop_reason": "end_turn"})

    return handler, seen


def test_falls_back_to_next_model_after_retries_run_out():
    handler, seen = by_model({"claude-sonnet-5-5": 529})
    client, log = make_client(handler)
    resp = client.post("/v1/messages", json={"model": "claude-sonnet-5-5", "max_tokens": 10})

    assert resp.status_code == 200
    assert resp.json()["model"] == "claude-sonnet-5"
    assert resp.headers["x-gateway-model"] == "claude-sonnet-5"
    assert seen == ["claude-sonnet-5-5"] * 3 + ["claude-sonnet-5"]
    [record] = log.records
    assert record["model"] == "claude-sonnet-5-5"
    assert record["served_model"] == "claude-sonnet-5"
    assert record["fallback_used"] is True
    assert record["cost_usd"] == (1_000 * 2 + 100 * 10) / 1e6  # Priced as the model that answered.
    assert [a["model"] for a in record["attempts"]] == seen


def test_walks_the_whole_chain():
    handler, seen = by_model({"claude-sonnet-5-5": 529, "claude-sonnet-5": 503})
    client, log = make_client(handler)
    resp = client.post("/v1/messages", json={"model": "claude-sonnet-5-5"})

    assert resp.json()["model"] == "claude-haiku-5-5"
    assert seen[-1] == "claude-haiku-5-5" and len(seen) == 7
    assert log.records[0]["served_model"] == "claude-haiku-5-5"


def test_stream_falls_back_before_streaming_starts():
    handler, seen = by_model({"claude-sonnet-5-5": 529})
    client, log = make_client(handler)
    with client.stream("POST", "/v1/messages", json={"model": "claude-sonnet-5-5", "stream": True}) as resp:
        body = resp.read()

    assert resp.status_code == 200
    assert b'"model": "claude-sonnet-5"' in body
    assert log.records[0]["fallback_used"] is True
    assert log.records[0]["outcome"] == "ok"


def test_whole_chain_failing_returns_last_error():
    handler, seen = by_model({m: 529 for m in ("claude-sonnet-5-5", "claude-sonnet-5", "claude-haiku-5-5")})
    client, log = make_client(handler)
    resp = client.post("/v1/messages", json={"model": "claude-sonnet-5-5"})

    assert resp.status_code == 529
    assert len(seen) == 9
    assert log.records[0]["fallback_used"] is False
    assert log.records[0]["served_model"] is None


def test_no_fallback_on_client_errors():
    handler, seen = by_model({"claude-sonnet-5-5": 400})
    client, _ = make_client(handler)
    resp = client.post("/v1/messages", json={"model": "claude-sonnet-5-5"})

    assert resp.status_code == 400
    assert seen == ["claude-sonnet-5-5"]


def test_fallback_model_rejecting_request_returns_primary_error():
    handler, seen = by_model({"claude-sonnet-5-5": 529, "claude-sonnet-5": 400})
    client, _ = make_client(handler)
    resp = client.post("/v1/messages", json={"model": "claude-sonnet-5-5"})

    assert resp.status_code == 529
    assert resp.json()["error"]["message"] == "claude-sonnet-5-5"  # Not the fallback's 400.
    assert seen == ["claude-sonnet-5-5"] * 3 + ["claude-sonnet-5"]


def test_app_can_opt_out_of_fallback():
    handler, seen = by_model({"claude-sonnet-5-5": 529})
    client, _ = make_client(handler)
    resp = client.post("/v1/messages", json={"model": "claude-sonnet-5-5"}, headers={"x-gateway-fallback": "off"})

    assert resp.status_code == 529
    assert set(seen) == {"claude-sonnet-5-5"}


def test_falls_back_when_primary_is_unreachable():
    def handler(req: httpx.Request) -> httpx.Response:
        if json.loads(req.content)["model"] == "claude-sonnet-5-5":
            raise httpx.ConnectError("refused")
        return httpx.Response(200, json={"model": "claude-sonnet-5", "usage": USAGE})

    client, log = make_client(handler)
    resp = client.post("/v1/messages", json={"model": "claude-sonnet-5-5"})

    assert resp.status_code == 200
    assert log.records[0]["served_model"] == "claude-sonnet-5"
    assert [a.get("error") for a in log.records[0]["attempts"]] == ["ConnectError"] * 3 + [None]


def test_open_circuit_skips_model_and_goes_straight_to_fallback():
    handler, seen = by_model({"claude-sonnet-5-5": 529})
    client, log = make_client(handler)
    app.state.breaker = CircuitBreaker(failure_threshold=3)

    first = client.post("/v1/messages", json={"model": "claude-sonnet-5-5"})  # 3 failures open the circuit.
    seen.clear()
    second = client.post("/v1/messages", json={"model": "claude-sonnet-5-5"})

    assert first.headers["x-gateway-model"] == second.headers["x-gateway-model"] == "claude-sonnet-5"
    assert seen == ["claude-sonnet-5"]  # Sonnet 5.5 wasn't even tried.
    assert log.records[1]["attempts"][0] == {"model": "claude-sonnet-5-5", "error": "circuit_open", "ms": 0.0}
    assert log.records[1]["fallback_used"] is True


def test_all_circuits_open_fails_fast_with_503():
    handler, seen = by_model({})
    client, log = make_client(handler)
    breaker = CircuitBreaker(failure_threshold=1, cooldown_s=30)
    for m in ("claude-sonnet-5-5", "claude-sonnet-5", "claude-haiku-5-5"):
        breaker.record(m, False)
    app.state.breaker = breaker

    resp = client.post("/v1/messages", json={"model": "claude-sonnet-5-5"})

    assert resp.status_code == 503
    assert resp.json()["error"]["type"] == "overloaded_error"
    assert 1 <= int(resp.headers["retry-after"]) <= 30
    assert seen == []
    assert log.records[0]["error_type"] == "circuit_open"


def test_client_errors_do_not_trip_the_breaker():
    handler, _ = by_model({"claude-sonnet-5-5": 400})
    client, _ = make_client(handler)
    app.state.breaker = CircuitBreaker(failure_threshold=2)
    for _ in range(5):
        client.post("/v1/messages", json={"model": "claude-sonnet-5-5"})
    assert app.state.breaker.state("claude-sonnet-5-5") == "closed"


# --- Streams that fail partway through ---------------------------------------

OVERLOADED_EVENT = sse({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})


def scripted_streams(*scripts):
    """Upstream that plays one script per call. A script is a list of SSE chunks;
    an exception in it is raised at that point, like a dropped connection."""
    calls: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        model = json.loads(req.content)["model"]
        script = scripts[len(calls)]
        calls.append(model)

        async def body():
            for item in script:
                if isinstance(item, Exception):
                    raise item
                yield item.replace(b"claude-sonnet-5-5", model.encode())

        return httpx.Response(200, content=body(), headers={"content-type": "text/event-stream"})

    return handler, calls


def stream_request(client, model="claude-sonnet-5-5"):
    with client.stream("POST", "/v1/messages", json={"model": model, "stream": True}) as resp:
        return resp.status_code, resp.read()


def events_in(body: bytes) -> list[dict]:
    return [json.loads(line[6:]) for line in body.decode().splitlines() if line.startswith("data: ")]


def test_error_before_content_is_recovered_on_the_next_model_without_the_app_noticing():
    handler, calls = scripted_streams([SSE_EVENTS[0], OVERLOADED_EVENT], SSE_EVENTS)
    client, log = make_client(handler)
    status, body = stream_request(client)

    assert status == 200
    assert [e["type"] for e in events_in(body)] == ["message_start", "content_block_delta", "message_delta", "message_stop"]
    assert events_in(body)[0]["message"]["model"] == "claude-sonnet-5"  # Only the second stream reached the app.
    assert calls == ["claude-sonnet-5-5", "claude-sonnet-5"]
    [record] = log.records
    assert record["outcome"] == "ok"
    assert record["served_model"] == "claude-sonnet-5" and record["fallback_used"] is True
    assert record["attempts"][1] == {"model": "claude-sonnet-5-5", "error": "stream:overloaded_error", "ms": 0.0}


def test_connection_lost_before_content_is_recovered():
    handler, calls = scripted_streams([SSE_EVENTS[0], httpx.ReadError("reset")], SSE_EVENTS)
    client, log = make_client(handler)
    status, body = stream_request(client)

    assert [e["type"] for e in events_in(body)][-1] == "message_stop"
    assert log.records[0]["outcome"] == "ok"


def test_model_without_fallback_is_retried_after_a_failed_stream():
    handler, calls = scripted_streams([SSE_EVENTS[0], OVERLOADED_EVENT], SSE_EVENTS)
    client, log = make_client(handler)
    status, body = stream_request(client, model="claude-haiku-5-5")

    assert calls == ["claude-haiku-5-5", "claude-haiku-5-5"]
    assert log.records[0]["outcome"] == "ok"


def test_gives_up_after_max_recoveries_and_passes_upstream_error_on():
    failing = [SSE_EVENTS[0], OVERLOADED_EVENT]
    handler, calls = scripted_streams(failing, failing, failing)
    client, log = make_client(handler)
    status, body = stream_request(client)

    assert status == 200
    assert [e["type"] for e in events_in(body)] == ["message_start", "error"]
    assert events_in(body)[-1]["error"]["type"] == "overloaded_error"
    assert len(calls) == 3  # The original stream plus 2 recoveries.
    assert log.records[0]["outcome"] == "error"
    assert log.records[0]["error_type"] == "overloaded_error"


def test_connection_lost_after_content_ends_stream_with_error_event():
    handler, calls = scripted_streams([SSE_EVENTS[0], SSE_EVENTS[1], httpx.ReadError("reset")])
    client, log = make_client(handler)
    app.state.breaker = CircuitBreaker(failure_threshold=1)
    status, body = stream_request(client)

    events = events_in(body)
    assert [e["type"] for e in events] == ["message_start", "content_block_delta", "error"]
    assert events[-1]["error"]["type"] == "api_error"
    assert calls == ["claude-sonnet-5-5"]  # No restart once content went out.
    [record] = log.records
    assert record["outcome"] == "error"
    assert record["error_type"] == "upstream_stream_interrupted"
    assert record["ttft_ms"] is not None
    assert app.state.breaker.state("claude-sonnet-5-5") == "open"  # Counted against the model.


def test_stall_after_content_ends_stream_with_timeout_event():
    handler, _ = scripted_streams([SSE_EVENTS[0], SSE_EVENTS[1], httpx.ReadTimeout("idle")])
    client, log = make_client(handler)
    status, body = stream_request(client)

    assert events_in(body)[-1]["error"]["type"] == "timeout_error"
    assert log.records[0]["error_type"] == "upstream_timeout"


def test_opening_events_are_held_until_first_content():
    handler, _ = scripted_streams(SSE_EVENTS)
    client, log = make_client(handler)
    status, body = stream_request(client)

    assert body == b"".join(SSE_EVENTS)  # Nothing lost or reordered by holding.
