from gateway import metrics
from gateway.circuit import CircuitBreaker


def sample(name: str, **labels) -> float:
    return metrics.REGISTRY.get_sample_value(name, labels) or 0.0


def test_observe_counts_requests_tokens_cost_attempts_and_savings():
    record = {
        "model": "fast", "served_model": "claude-haiku-5-5", "status_code": 200, "outcome": "ok",
        "route": "fast:fastest", "cache_hit": False, "fallback_used": True, "stream": True,
        "latency_ms": 1500.0, "ttft_ms": 400.0, "input_tokens": 100, "output_tokens": 20,
        "cache_read_tokens": 1_000, "cost_usd": 0.002, "api_key_id": "key_m",
        "prompt_cache_savings_usd": 0.01, "response_cache_saved_usd": None,
        "attempts": [{"model": "claude-sonnet-5-5", "error": "circuit_open"}, {"model": "claude-haiku-5-5", "status": 200}],
    }
    before = sample("gateway_cost_usd_total", api_key_id="key_m", served_model="claude-haiku-5-5")
    metrics.observe(record)

    assert sample("gateway_requests_total", model="fast", served_model="claude-haiku-5-5", status="200", outcome="ok",
                  route="fast:fastest", cache_hit="false", fallback="true") >= 1
    assert sample("gateway_cost_usd_total", api_key_id="key_m", served_model="claude-haiku-5-5") - before == 0.002
    assert sample("gateway_tokens_total", served_model="claude-haiku-5-5", kind="cache_read") >= 1_000
    assert sample("gateway_upstream_attempts_total", model="claude-sonnet-5-5", result="circuit_open") >= 1
    assert sample("gateway_time_to_first_token_seconds_count", served_model="claude-haiku-5-5") >= 1
    assert sample("gateway_cache_savings_usd_total", kind="prompt") >= 0.01


def test_render_reports_open_circuits_and_clears_closed_ones():
    breaker = CircuitBreaker(failure_threshold=1)
    breaker.record("claude-opus-5-5", False)
    text = metrics.render(breaker).decode()
    assert 'gateway_circuit_open{model="claude-opus-5-5"} 1.0' in text
    breaker.record("claude-opus-5-5", True)
    text = metrics.render(breaker).decode()
    assert 'gateway_circuit_open{model="claude-opus-5-5"} 0.0' in text
