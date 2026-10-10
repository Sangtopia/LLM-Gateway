import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

import httpx
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from gateway.circuit import CircuitBreaker, attempt_succeeded
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

# How many times a stream that fails before sending any content is transparently restarted.
MAX_STREAM_RECOVERIES = 2


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


def error_body(error_type: str, message: str) -> dict:
    # Same shape as Anthropic's own errors, so SDKs parse it normally.
    return {"type": "error", "error": {"type": error_type, "message": message}}


def error_response(status: int, error_type: str, message: str) -> JSONResponse:
    return JSONResponse(error_body(error_type, message), status_code=status)


def sse_error(error_type: str, message: str) -> bytes:
    """An `error` event, the way Anthropic reports failures inside a stream."""
    return f"event: error\ndata: {json.dumps(error_body(error_type, message))}\n\n".encode()


def write_log(request_log: RequestLogger, record: dict) -> None:
    if isinstance(record.get("attempts"), list):
        record["attempts"] = [a.as_dict() if isinstance(a, Attempt) else a for a in record["attempts"]]
    request_log.write(record)


API_ERROR_TYPES = {401: "authentication_error", 503: "overloaded_error", 504: "timeout_error"}


def fail(record: dict, request_log: RequestLogger, started: float, status: int, error_type: str, message: str) -> JSONResponse:
    """Log a request that failed inside the gateway and return an Anthropic-shaped error."""
    record.update(status_code=status, outcome="error", error_type=error_type, latency_ms=elapsed_ms(started))
    write_log(request_log, record)
    return error_response(status, API_ERROR_TYPES.get(status, "api_error"), message)


@dataclass
class StreamFailure:
    log_type: str  # error_type for the request log
    api_type: str = "api_error"  # error type shown to the app
    message: str = "Upstream stream failed."
    from_event: bool = False  # Upstream sent an error event (vs. the connection failing).


@dataclass
class Upstream:
    """Result of trying the model chain: a response, or why there isn't one."""

    response: httpx.Response | None = None
    served: str | None = None  # The model that produced `response`, if it succeeded.
    status: int = 502
    error_type: str = "upstream_unreachable"
    message: str = "Could not reach Anthropic."
    retry_after_s: int | None = None


@dataclass
class UpstreamCall:
    """Everything needed to (re)send one app request through the model chain."""

    http: httpx.AsyncClient
    headers: dict[str, str]
    timeout: httpx.Timeout | None
    stream: bool
    body: bytes
    parsed: dict
    model: str | None
    chain: list[str | None]
    policy: RetryPolicy
    breaker: CircuitBreaker
    attempts: list[Attempt] = field(default_factory=list)

    def _sender(self, target: str | None):
        target_body = self.body if target == self.model else json.dumps({**self.parsed, "model": target}).encode()

        async def send() -> httpx.Response:
            req = self.http.build_request("POST", "/v1/messages", content=target_body, headers=self.headers, timeout=self.timeout)
            return await self.http.send(req, stream=self.stream)

        return send

    async def open(self, skip: frozenset[str | None] = frozenset()) -> Upstream:
        """Try each model in the chain (with retries) until one answers.

        Skips models whose circuit is open, and models in `skip` (ones whose
        stream just failed) unless that would leave nothing to try. Falls back
        only on 429/529/5xx or unreachable models; any other error is the
        request's fault and is returned.
        """
        chain = [m for m in self.chain if m not in skip] or self.chain
        first_failure: httpx.Response | None = None  # The requested model's error, if it failed.
        last_failure: httpx.Response | None = None
        tried_any = False
        for position, target in enumerate(chain):
            if target and not self.breaker.allow(target):
                self.attempts.append(Attempt(model=target, error="circuit_open"))
                continue
            tried_any = True
            tried_from = len(self.attempts)
            try:
                response = await send_with_retries(self._sender(target), self.policy, self.attempts, target)
            except httpx.ReadTimeout:
                return Upstream(status=504, error_type="upstream_timeout", message="Anthropic did not respond in time.")
            except httpx.TransportError:
                continue
            finally:
                if target:
                    for attempt in self.attempts[tried_from:]:
                        self.breaker.record(target, attempt_succeeded(attempt.status))

            if response.status_code < 400:
                return Upstream(response=response, served=target)
            if not should_retry(response):
                if position == 0:
                    return Upstream(response=response)  # The request itself is bad; nothing to fall back from.
                # The fallback model rejected the request outright (often a parameter
                # it doesn't support). Report the requested model's failure instead
                # of an error about a model the app never asked for.
                await response.aclose()
                break
            # This model is failing; fall back to the next one.
            await response.aread()
            await response.aclose()
            last_failure = response
            if position == 0:
                first_failure = response

        if failure := first_failure or last_failure:
            return Upstream(response=failure)
        if not tried_any:
            retry_after = max(1, round(self.breaker.retry_after_s([m for m in chain if m])))
            return Upstream(
                status=503,
                error_type="circuit_open",
                message=f"{self.model} and its fallbacks are unavailable.",
                retry_after_s=retry_after,
            )
        return Upstream(message=f"Could not reach Anthropic ({self.attempts[-1].error}).")


@router.post("/v1/messages")
async def messages(request: Request) -> Response:
    started = time.perf_counter()
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
        return fail(record, request_log, started, 401, "authentication_error", "Invalid or missing gateway API key.")
    record["api_key_id"] = api_key.id

    decision = await limiter.admit(api_key)
    if not decision.allowed:
        error_type = "gateway_budget_exceeded" if decision.limit == "monthly budget" else "gateway_rate_limited"
        record.update(status_code=429, outcome="error", error_type=error_type, latency_ms=elapsed_ms(started))
        write_log(request_log, record)
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

    chain = [model]
    if model and request.headers.get("x-gateway-fallback", "").lower() != "off":
        chain += request.app.state.fallback_chains.get(model, [])
    call = UpstreamCall(
        http=request.app.state.http,
        headers=headers,
        timeout=timeout,
        stream=stream,
        body=body,
        parsed=parsed,
        model=model,
        chain=chain,
        policy=request.app.state.retry_policy,
        breaker=request.app.state.breaker,
    )
    record["attempts"] = call.attempts

    result = await call.open()
    if result.response is None:
        response = fail(record, request_log, started, result.status, result.error_type, result.message)
        if result.retry_after_s is not None:
            response.headers["retry-after"] = str(result.retry_after_s)
        return response

    upstream, served = result.response, result.served
    upstream_ms = elapsed_ms(started)  # Including retries and fallbacks. For streams: until headers arrive.
    record.update(
        status_code=upstream.status_code,
        upstream_ms=upstream_ms,
        upstream_request_id=upstream.headers.get("request-id"),
        served_model=served,
        fallback_used=served is not None and served != model,
    )
    response_headers = client_headers(upstream)
    response_headers["x-gateway-upstream-ms"] = f"{upstream_ms:.1f}"
    response_headers["x-gateway-attempts"] = str(len(call.attempts))
    if served:
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
        """Relay the stream, recovering from failures that happen before any content.

        Opening events (message_start, pings) are held back until the first
        content delta. If the stream fails before then, the app has seen nothing
        but a 200, so the gateway restarts it (retrying or falling back) and the
        app never knows. After content has gone out, switching models would
        splice two different answers together, so a failure ends the stream with
        an Anthropic-style `error` event instead of a cut connection.

        If the app disconnects, this generator is cancelled and the finally
        blocks close the upstream connection, so Anthropic stops generating (and
        billing) tokens nobody will read.
        """
        nonlocal upstream, served
        recoveries = 0
        failed_models: set[str | None] = set()
        tracker = StreamTracker()
        try:
            while True:
                tracker = StreamTracker()
                held: list[bytes] = []
                released = False
                failure: StreamFailure | None = None
                try:
                    async for chunk in upstream.aiter_bytes():  # decoded, in case upstream compresses anyway
                        tracker.feed(chunk)
                        if released:
                            yield chunk
                            continue
                        held.append(chunk)
                        if tracker.saw_content or tracker.completed:
                            record["ttft_ms"] = elapsed_ms(started)
                            released = True
                            yield b"".join(held)
                            held = []
                        elif tracker.error_type:
                            failure = StreamFailure(tracker.error_type, from_event=True)
                            break
                except httpx.ReadTimeout:
                    failure = StreamFailure("upstream_timeout", "timeout_error", "Anthropic stopped sending events.")
                except httpx.TransportError as exc:
                    message = f"Connection to Anthropic lost: {type(exc).__name__}"
                    failure = StreamFailure("upstream_stream_interrupted", "api_error", message)
                finally:
                    await upstream.aclose()

                if failure is None:
                    if held:  # Ended without content or an error; pass on whatever came.
                        yield b"".join(held)
                    return

                if served:
                    call.breaker.record(served, False)
                failed_models.add(served)
                call.attempts.append(Attempt(model=served, error=f"stream:{failure.log_type}"))
                tracker.error_type = failure.log_type

                if released or recoveries >= MAX_STREAM_RECOVERIES:
                    # Content already went out, or we've restarted enough: end cleanly.
                    if failure.from_event and not released:
                        yield b"".join(held)  # Upstream's own error event, unchanged.
                    else:
                        yield sse_error(failure.api_type, failure.message)
                    return

                # Nothing has reached the app yet: start over, retrying or falling back.
                recoveries += 1
                retry = await call.open(skip=frozenset(failed_models))
                if retry.response is None:
                    tracker.error_type = retry.error_type
                    yield sse_error(API_ERROR_TYPES.get(retry.status, "api_error"), retry.message)
                    return
                if retry.response.status_code != 200:
                    await retry.response.aread()
                    await retry.response.aclose()
                    error = parse_json_object(retry.response.content).get("error") or {}
                    tracker.error_type = error.get("type") or f"http_{retry.response.status_code}"
                    yield sse_error(error.get("type", "api_error"), error.get("message", "Upstream request failed."))
                    return
                upstream, served = retry.response, retry.served
                record.update(
                    served_model=served,
                    fallback_used=served != model,
                    upstream_request_id=upstream.headers.get("request-id"),
                )
        finally:
            if tracker.error_type:
                record.update(outcome="error", error_type=tracker.error_type)
            else:
                record["outcome"] = "ok" if tracker.completed else "incomplete"
            record.update(stop_reason=tracker.stop_reason, latency_ms=elapsed_ms(started))
            add_usage(record, tracker.model or model, tracker.usage)
            limiter.record_usage(api_key, tokens_for_rate_limit(tracker.usage), record["cost_usd"])
            write_log(request_log, record)

    return StreamingResponse(relay(), status_code=upstream.status_code, headers=response_headers)
