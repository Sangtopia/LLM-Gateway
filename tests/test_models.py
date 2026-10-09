import pytest

from gateway.models import MODELS, cost_usd, get_model


def test_unknown_model_has_no_cost():
    assert get_model("not-a-model") is None
    assert cost_usd("not-a-model", {"input_tokens": 100}) is None


def test_plain_input_and_output():
    # Sonnet 5.5: $2 in, $10 out per 1M tokens.
    assert cost_usd("claude-sonnet-5-5", {"input_tokens": 1_000_000, "output_tokens": 1_000_000}) == pytest.approx(12.0)


def test_cache_reads_and_writes():
    usage = {
        "input_tokens": 1_000,
        "output_tokens": 0,
        "cache_read_input_tokens": 1_000_000,
        "cache_creation": {"ephemeral_5m_input_tokens": 1_000_000, "ephemeral_1h_input_tokens": 1_000_000},
    }
    # Opus 5.5: input $4, cache read $0.20, 5m write 1.25x, 1h write 2x.
    expected = 0.004 + 0.20 + 4 * 1.25 + 4 * 2.0
    assert cost_usd("claude-opus-5-5", usage) == pytest.approx(expected)


def test_cache_creation_total_without_ttl_breakdown_counts_as_5m():
    usage = {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 1_000_000}
    assert cost_usd("claude-sonnet-5-5", usage) == pytest.approx(2.0 * 1.25)


def test_haiku_5_5_long_prompt_pricing():
    short = {"input_tokens": 100_000, "output_tokens": 1_000_000}
    long = {"input_tokens": 100_001, "output_tokens": 1_000_000}
    assert cost_usd("claude-haiku-5-5", short) == pytest.approx(0.01 + 0.50)
    assert cost_usd("claude-haiku-5-5", long) == pytest.approx(100_001 * 0.50 / 1e6 + 2.50)


def test_long_prompt_threshold_counts_cached_tokens_and_raises_cache_read_price():
    usage = {"input_tokens": 10, "output_tokens": 0, "cache_read_input_tokens": 200_000}
    assert cost_usd("claude-haiku-5-5", usage) == pytest.approx((10 * 0.50 + 200_000 * 0.05) / 1e6)


def test_sonnet_5_5_cache_reads_are_0_05x_input():
    assert cost_usd("claude-sonnet-5-5", {"cache_read_input_tokens": 1_000_000}) == pytest.approx(0.10)


def test_batch_halves_everything():
    usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000, "cache_read_input_tokens": 1_000_000}
    assert cost_usd("claude-sonnet-5-5", usage, batch=True) == pytest.approx((2 + 10 + 0.1) / 2)


def test_us_only_inference_costs_10_percent_more_on_newer_models():
    usage = {"input_tokens": 1_000_000, "output_tokens": 0, "inference_geo": "us"}
    assert cost_usd("claude-sonnet-5-5", usage) == pytest.approx(2.2)
    assert cost_usd("claude-sonnet-5-5", {**usage, "inference_geo": "global"}) == pytest.approx(2.0)
    assert cost_usd("claude-haiku-4-5", usage) == pytest.approx(1.0)


def test_registry_ids_match_keys():
    for key, model in MODELS.items():
        assert key == model.id
