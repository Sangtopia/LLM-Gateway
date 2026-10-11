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
from gateway import metrics
from gateway.keys import KeyStore, extract_key
from gateway.models import cache_savings_usd, cost_usd, format_usd
from gateway.promptcache import apply_prompt_caching
from gateway.responsecache import ResponseCache, cache_key, cache_mode, cache_ttl
from gateway.semanticcache import SemanticCache, semantic_scope, semantic_threshold
from gateway.ratelimit import RateLimiter, next_month_start, tokens_for_rate_limit
from gateway.retry import Attempt, RetryPolicy, send_with_retries, should_retry
from gateway.routing import CHARS_PER_TOKEN, Router
from gateway.sse import StreamTracker
from gateway.upstreamlimits import UpstreamLimits

router = APIRouter()

# Client headers passed through to Anthropic. Everything else (notably the
# client's x-api-key, which will be a gateway key) is dropped.
FORWARD_REQUEST_HEADERS = ("anthropic-version", "anthropic-beta", "content-type")

# Upstream headers passed back to the client.
FORWARD_RESPONSE_PREFIXES = ("anthropic-", "retry-after", "request-id", "content-type")

# How many times a stream that fails before sending any content is transparently restarted.
MAX_STREAM_RECOVERIES = 2

# Fields /v1/messages accepts that /v1/messages/count_tokens rejects ("Extra inputs are not
# permitted"). They shape generation, not the prompt, so they don't change the count.
GENERATION_ONLY_FIELDS = {
    "max_tokens", "stream", "metadata", "stop_sequences", "temperature", "top_p", "top_k",
    "service_tier", "fallbacks", "speed", "inference_geo", "container",
}


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
        prompt_cache_savings_usd=cache_savings_usd(model, usage) if model and usage else None,
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
    metrics.observe(record)


API_ERROR_TYPES = {401: "authentication_error", 429: "rate_limit_error", 503: "overloaded_error", 504: "timeout_error"}


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
    timeout: httpx.Timeout | httpx._client.UseClientDefault
    stream: bool
    body: bytes
    parsed: dict
    model: str | None
    chain: list[str | None]
    policy: RetryPolicy
    breaker: CircuitBreaker
    limits: UpstreamLimits | None = None  # Anthropic's rate-limit headers, for pacing.
    attempts: list[Attempt] = field(default_factory=list)

    def _sender(self, target: str | None):
        target_body = self.body if target == self.model else json.dumps({**self.parsed, "model": target}).encode()

        async def send() -> httpx.Response:
            req = self.http.build_request("POST", "/v1/messages", content=target_body, headers=self.headers, timeout=self.timeout)
            response = await self.http.send(req, stream=self.stream)
            if self.limits is not None:
                self.limits.observe(target, response.status_code, response.headers)
            return response

        return send

    async def open(self, skip: frozenset[str | None] = frozenset()) -> Upstream:
        """Try each model in the chain (with retries) until one answers.

        Skips models whose circuit is open, models in `skip` (ones whose stream
        just failed) unless that would leave nothing to try, and models whose
        Anthropic rate limit is nearly used up unless every usable model is.
        Falls back only on 429/529/5xx or unreachable models; any other error is
        the request's fault and is returned.
        """
        chain = [m for m in self.chain if m not in skip] or self.chain
        paced: dict[str | None, str] = {}
        if self.limits is not None:
            estimated_input = len(self.body) // CHARS_PER_TOKEN
            usable = [m for m in chain if not m or self.breaker.state(m) != "open"]
            paced = {m: why for m in usable if (why := self.limits.paced(m, estimated_input))}
            if len(paced) == len(usable):
                paced = {}  # Everything is near its limit: send anyway and let Anthropic decide.
        first_failure: httpx.Response | None = None  # The requested model's error, if it failed.
        last_failure: httpx.Response | None = None
        tried_any = False
        for position, target in enumerate(chain):
            if target in paced:
                self.attempts.append(Attempt(model=target, error=f"paced:{paced[target]}"))
                continue
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


async def count_input_tokens(http: httpx.AsyncClient, headers: dict[str, str], request: dict, model: str) -> int | None:
    """Exact input tokens for `request` on `model` from Anthropic's count_tokens, or None if it fails."""
    body = {k: v for k, v in request.items() if k not in GENERATION_ONLY_FIELDS} | {"model": model}
    count_headers = {k: v for k, v in headers.items() if k.lower() != "accept-encoding"}
    try:
        resp = await http.post("/v1/messages/count_tokens", json=body, headers=count_headers, timeout=10.0)
    except httpx.HTTPError:
        return None
    tokens = parse_json_object(resp.content).get("input_tokens") if resp.status_code == 200 else None
    return tokens if isinstance(tokens, int) else None


async def input_exceeds_budget(
    http: httpx.AsyncClient, headers: dict[str, str], request: dict, body_size: int, model: str | None, remaining: float
) -> float | None:
    """The input cost of this request if it alone exceeds the key's remaining budget, else None.

    A size-based estimate (about 4 characters per token) is free and settles almost every
    request. Only when it comes within 2x of the remaining budget does the gateway ask
    count_tokens for the exact figure. The check covers input only: output length isn't
    known until the model writes it.
    """
    if not model:
        return None
    estimate = cost_usd(model, {"input_tokens": body_size // CHARS_PER_TOKEN})
    if estimate is None or estimate * 2 < remaining:
        return None
    exact_tokens = await count_input_tokens(http, headers, request, model)
    cost = cost_usd(model, {"input_tokens": exact_tokens}) if exact_tokens is not None else estimate
    return cost if cost > remaining else None


@router.post("/v1/messages/count_tokens")
async def count_tokens(request: Request) -> Response:
    """Pass through to Anthropic's token counter. Free upstream, so it doesn't touch budgets."""
    started = time.perf_counter()
    request_log: RequestLogger = request.app.state.request_log
    body = await request.body()
    parsed = parse_json_object(body)
    record = {
        "id": str(uuid.uuid4()),
        "created_at": datetime.now(UTC),
        "endpoint": "count_tokens",
        "model": parsed.get("model") if isinstance(parsed.get("model"), str) else None,
        "provider": "anthropic",
        "stream": False,
        "cache_hit": False,
        "fallback_used": False,
    }
    api_key = await request.app.state.keys.authenticate(extract_key(request.headers))
    if api_key is None:
        return fail(record, request_log, started, 401, "authentication_error", "Invalid or missing gateway API key.")
    record["api_key_id"] = api_key.id
    http: httpx.AsyncClient = request.app.state.http
    try:
        upstream = await http.post("/v1/messages/count_tokens", content=body, headers=upstream_headers(request), timeout=30.0)
    except httpx.HTTPError as exc:
        return fail(record, request_log, started, 502, "upstream_unreachable", f"Could not reach Anthropic: {type(exc).__name__}")
    payload = parse_json_object(upstream.content)
    record.update(
        status_code=upstream.status_code,
        outcome="ok" if upstream.status_code == 200 else "error",
        error_type=None if upstream.status_code == 200 else (payload.get("error") or {}).get("type"),
        input_tokens=payload.get("input_tokens"),
        cost_usd=0.0,
        latency_ms=elapsed_ms(started),
        upstream_request_id=upstream.headers.get("request-id"),
    )
    write_log(request_log, record)
    return Response(content=upstream.content, status_code=upstream.status_code, headers=client_headers(upstream))


@router.post("/v1/messages")
async def messages(request: Request) -> Response:
    started = time.perf_counter()
    request_log: RequestLogger = request.app.state.request_log
    keys: KeyStore = request.app.state.keys
    limiter: RateLimiter = request.app.state.limiter

    body = await request.body()
    parsed = parse_json_object(body)
    prompt_cache_added = False
    if (
        settings.auto_prompt_cache
        and request.headers.get("x-gateway-prompt-cache", "").lower() != "off"
        and (with_cache := apply_prompt_caching(parsed)) is not None
    ):
        parsed, body, prompt_cache_added = with_cache, json.dumps(with_cache).encode(), True
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
        "prompt_cache_added": prompt_cache_added,
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

    response_cache: ResponseCache = request.app.state.response_cache
    semantic_cache: SemanticCache | None = request.app.state.semantic_cache
    mode = cache_mode(request.headers)
    use_cache = mode is not None
    key_for_cache = cache_key(parsed, request.headers, api_key.id) if use_cache else None
    hit, similarity = (await response_cache.get(key_for_cache) if use_cache else None), None
    semantic = None  # (scope, vector) when the semantic cache applies to this request.
    if hit is None and mode == "semantic" and semantic_cache is not None:
        if scoped := semantic_scope(parsed, request.headers, api_key.id):
            scope, text = scoped
            vector = await semantic_cache.embedder.embed(text)
            semantic = (scope, vector)
            if found := await semantic_cache.lookup(scope, vector, semantic_threshold(request.headers)):
                hit, similarity = found
    if hit:
        record.update(
            status_code=200,
            outcome="ok",
            cache_hit=True,
            cache_kind="semantic" if similarity is not None else "exact",
            cache_similarity=round(similarity, 4) if similarity is not None else None,
            served_model=hit["model"],
            cost_usd=0.0,
            response_cache_saved_usd=hit["cost_usd"],
            latency_ms=elapsed_ms(started),
        )
        write_log(request_log, record)
        hit_headers = {"x-gateway-cache": f"hit-{record['cache_kind']}",
                       **({"x-gateway-model": hit["model"]} if hit["model"] else {})}
        if similarity is not None:
            hit_headers["x-gateway-cache-similarity"] = f"{similarity:.4f}"
        body_bytes = hit["body"].encode("utf-8")
        if stream:
            return StreamingResponse(iter([body_bytes]), media_type=hit["content_type"], headers=hit_headers)
        return Response(content=body_bytes, media_type=hit["content_type"], headers=hit_headers)

    headers = upstream_headers(request)
    # Not None: to httpx, timeout=None means no timeout at all, not the client's default.
    timeout = httpx.USE_CLIENT_DEFAULT
    if stream:
        # Compression would make the upstream buffer events into batches, and
        # we'd have to re-encode for the client anyway. Ask for plain bytes.
        headers["accept-encoding"] = "identity"
        # For streams the read timeout is the longest allowed gap between events.
        timeout = httpx.Timeout(settings.upstream_stream_idle_timeout_s, connect=settings.upstream_connect_timeout_s)

    fallback_allowed = request.headers.get("x-gateway-fallback", "").lower() != "off"
    router: Router = request.app.state.router
    route = router.resolve(model, parsed, len(body))
    if route is not None:
        # An alias: the strategy's ranking is the chain, best first.
        chain: list[str | None] = route.models if fallback_allowed else route.models[:1]
        record["route"] = f"{route.alias}:{route.strategy}"
    else:
        chain = [model]
        if model and fallback_allowed:
            chain += request.app.state.fallback_chains.get(model, [])
    primary = chain[0]  # The model a fallback is measured against.

    if decision.remaining_budget_usd is not None:
        too_expensive = await input_exceeds_budget(
            request.app.state.http, headers, parsed, len(body), primary, decision.remaining_budget_usd
        )
        if too_expensive is not None:
            message = (
                f"This request's input alone would cost about {format_usd(too_expensive)}, more than the "
                f"{format_usd(decision.remaining_budget_usd)} left in gateway key {api_key.id}'s monthly budget."
            )
            response = fail(record, request_log, started, 429, "gateway_budget_exceeded", message)
            # The budget only grows back at the start of next month.
            now = datetime.now(UTC)
            response.headers["retry-after"] = str(max(1, int((next_month_start(now) - now).total_seconds())))
            return response
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
        limits=request.app.state.upstream_limits,
    )
    record["attempts"] = call.attempts

    upstream_started = time.perf_counter()
    result = await call.open()
    if result.response is None:
        response = fail(record, request_log, started, result.status, result.error_type, result.message)
        if result.retry_after_s is not None:
            response.headers["retry-after"] = str(result.retry_after_s)
        return response

    upstream, served = result.response, result.served

    # A declined non-streaming request from a model without Anthropic's server-side
    # fallback: retry it on the configured model. The declined call still costs money.
    refused_cost = 0.0
    refusal_chain = request.app.state.refusal_fallbacks.get(served or "", []) if fallback_allowed else []
    if not stream and upstream.status_code == 200 and refusal_chain:
        refused = parse_json_object(upstream.content)
        if refused.get("stop_reason") == "refusal":
            refused_cost = cost_usd(served, refused.get("usage") or {}) or 0.0
            call.attempts[-1].error = "refusal"
            call.chain = refusal_chain
            retry = await call.open()
            if retry.response is not None and retry.response.status_code == 200:
                upstream, served = retry.response, retry.served
                record["refusal_fallback"] = True

    upstream_ms = elapsed_ms(started)  # Including retries and fallbacks. For streams: until headers arrive.
    record.update(
        status_code=upstream.status_code,
        upstream_ms=upstream_ms,
        upstream_request_id=upstream.headers.get("request-id"),
        served_model=served,
        fallback_used=served is not None and served != primary,
    )
    response_headers = client_headers(upstream)
    response_headers["x-gateway-upstream-ms"] = f"{upstream_ms:.1f}"
    response_headers["x-gateway-attempts"] = str(len(call.attempts))
    if served:
        response_headers["x-gateway-model"] = served
    if route is not None:
        response_headers["x-gateway-route"] = record["route"]
    if use_cache:
        response_headers["x-gateway-cache"] = "miss"

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
            if refused_cost:
                record["cost_usd"] = (record["cost_usd"] or 0.0) + refused_cost
            limiter.record_usage(api_key, tokens_for_rate_limit(usage), record["cost_usd"])
        else:
            error_type = (payload.get("error") or {}).get("type") or f"http_{upstream.status_code}"
            record.update(outcome="error", error_type=error_type)
        record["latency_ms"] = elapsed_ms(started)
        write_log(request_log, record)
        if use_cache and record["outcome"] == "ok":
            content_type = upstream.headers.get("content-type", "application/json")
            ttl = cache_ttl(request.headers)
            response_cache.put_later(key_for_cache, upstream.content, content_type, served, record.get("cost_usd"), ttl)
            if semantic:
                semantic_cache.store_later(*semantic, upstream.content, content_type, served, record.get("cost_usd"), ttl)
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
        sent: list[bytes] = []  # Everything the app received, for the response cache.
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
                            if use_cache:
                                sent.append(chunk)
                            yield chunk
                            continue
                        held.append(chunk)
                        if tracker.saw_content or tracker.completed:
                            record["ttft_ms"] = elapsed_ms(started)
                            if use_cache:
                                sent.extend(held)
                            if served and len(call.attempts) == 1:
                                # A clean sample (no retries or fallbacks) of this model's
                                # speed, timed from the upstream call, for "fastest" routing.
                                router.latency.record(served, elapsed_ms(upstream_started))
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
                    fallback_used=served != primary,
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
            if use_cache and record["outcome"] == "ok":
                ttl = cache_ttl(request.headers)
                body = b"".join(sent)
                response_cache.put_later(key_for_cache, body, "text/event-stream", served, record.get("cost_usd"), ttl)
                if semantic:
                    semantic_cache.store_later(*semantic, body, "text/event-stream", served, record.get("cost_usd"), ttl)

    return StreamingResponse(relay(), status_code=upstream.status_code, headers=response_headers)
