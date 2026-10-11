"""Prometheus metrics, served at /metrics.

Every finished request goes through `observe(record)` with the same record that
is written to the request log, so the metrics and the log never disagree.
Label values are kept low-cardinality: models, status codes, outcomes, routes and
gateway key ids (a handful per deployment), never request ids or prompts.
"""

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from gateway.circuit import CircuitBreaker

REGISTRY = CollectorRegistry()

LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 15, 30, 60, 120, 300)

REQUESTS = Counter(
    "gateway_requests",
    "Requests handled, by requested and serving model and result.",
    ["model", "served_model", "status", "outcome", "route", "cache_hit", "fallback"],
    registry=REGISTRY,
)
LATENCY = Histogram(
    "gateway_request_latency_seconds",
    "Request arrival to last byte sent.",
    ["served_model", "stream"],
    buckets=LATENCY_BUCKETS,
    registry=REGISTRY,
)
TTFT = Histogram(
    "gateway_time_to_first_token_seconds",
    "Request arrival to the first content delta (streams).",
    ["served_model"],
    buckets=LATENCY_BUCKETS,
    registry=REGISTRY,
)
TOKENS = Counter(
    "gateway_tokens",
    "Tokens processed, by kind.",
    ["served_model", "kind"],
    registry=REGISTRY,
)
COST = Counter(
    "gateway_cost_usd",
    "Spend in US dollars.",
    ["api_key_id", "served_model"],
    registry=REGISTRY,
)
SAVINGS = Counter(
    "gateway_cache_savings_usd",
    "Dollars saved by caching (prompt: Anthropic prompt caching; response: gateway response cache).",
    ["kind"],
    registry=REGISTRY,
)
ATTEMPTS = Counter(
    "gateway_upstream_attempts",
    "Upstream tries, including retries, fallbacks and circuit-open skips.",
    ["model", "result"],
    registry=REGISTRY,
)
CIRCUIT_OPEN = Gauge(
    "gateway_circuit_open",
    "1 if the model's circuit is open or half-open (being skipped or probed).",
    ["model"],
    registry=REGISTRY,
)

TOKEN_FIELDS = {
    "input_tokens": "input",
    "output_tokens": "output",
    "cache_read_tokens": "cache_read",
    "cache_write_tokens": "cache_write",
}


def observe(record: dict) -> None:
    served = record.get("served_model") or "none"
    REQUESTS.labels(
        model=record.get("model") or "none",
        served_model=served,
        status=str(record.get("status_code", "")),
        outcome=record.get("outcome", ""),
        route=record.get("route") or "none",
        cache_hit=str(bool(record.get("cache_hit"))).lower(),
        fallback=str(bool(record.get("fallback_used"))).lower(),
    ).inc()
    if record.get("latency_ms") is not None:
        LATENCY.labels(served_model=served, stream=str(bool(record.get("stream"))).lower()).observe(record["latency_ms"] / 1000)
    if record.get("ttft_ms") is not None:
        TTFT.labels(served_model=served).observe(record["ttft_ms"] / 1000)
    for field, kind in TOKEN_FIELDS.items():
        if record.get(field):
            TOKENS.labels(served_model=served, kind=kind).inc(record[field])
    if record.get("cost_usd"):
        COST.labels(api_key_id=record.get("api_key_id") or "none", served_model=served).inc(record["cost_usd"])
    # Counters can't go down, so net-negative prompt caching (unread writes) shows in the request log only.
    if (saved := record.get("prompt_cache_savings_usd")) and saved > 0:
        SAVINGS.labels(kind="prompt").inc(saved)
    if saved := record.get("response_cache_saved_usd"):
        SAVINGS.labels(kind="response").inc(saved)
    for attempt in record.get("attempts") or []:
        result = attempt.get("error") or str(attempt.get("status", "unknown"))
        ATTEMPTS.labels(model=attempt.get("model") or "none", result=result).inc()


def render(breaker: CircuitBreaker) -> bytes:
    """The /metrics page. Circuit state is read at scrape time."""
    open_now = breaker.snapshot()
    for labels in list(CIRCUIT_OPEN._metrics):  # Reset models whose circuits have closed.
        if labels[0] not in open_now:
            CIRCUIT_OPEN.labels(model=labels[0]).set(0)
    for model in open_now:
        CIRCUIT_OPEN.labels(model=model).set(1)
    return generate_latest(REGISTRY)
