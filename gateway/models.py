"""Claude model registry: pricing and limits, used for cost tracking and routing.

Prices are USD per 1M tokens, Anthropic first-party API rates, checked against
https://platform.claude.com/docs/en/about-claude/pricing on 2026-10-09.
"""

from dataclasses import dataclass

# Cache write prices are multiples of the base input price, by cache lifetime.
CACHE_WRITE_5M_MULTIPLIER = 1.25
CACHE_WRITE_1H_MULTIPLIER = 2.0
BATCH_DISCOUNT = 0.5  # Applies to every token type, including cache reads and writes.
US_INFERENCE_MULTIPLIER = 1.1  # inference_geo="us" on Claude 4.6 and later.


@dataclass(frozen=True)
class ModelInfo:
    id: str
    tier: str  # "fable", "opus", "sonnet" or "haiku"
    input_price: float
    output_price: float
    cache_read_price: float
    context_window: int
    # Some models charge more once the prompt (including cached tokens) passes a size threshold.
    long_prompt_threshold: int | None = None
    long_input_price: float | None = None
    long_output_price: float | None = None
    long_cache_read_price: float | None = None
    us_inference_surcharge: bool = True


MODELS: dict[str, ModelInfo] = {
    m.id: m
    for m in [
        # Current generation
        ModelInfo("claude-fable-5-1", "fable", 10.00, 50.00, 0.25, 1_000_000),
        ModelInfo("claude-opus-5-5", "opus", 4.00, 20.00, 0.20, 1_000_000),
        ModelInfo("claude-sonnet-5-5", "sonnet", 2.00, 10.00, 0.10, 1_000_000),
        ModelInfo(
            "claude-haiku-5-5", "haiku", 0.10, 0.50, 0.01, 1_000_000,
            long_prompt_threshold=100_000, long_input_price=0.50, long_output_price=2.50, long_cache_read_price=0.05,
        ),
        # Previous generations, still served
        ModelInfo("claude-fable-5", "fable", 10.00, 50.00, 1.00, 1_000_000),
        ModelInfo("claude-opus-5", "opus", 5.00, 25.00, 0.50, 1_000_000),
        ModelInfo("claude-opus-4-8", "opus", 5.00, 25.00, 0.50, 1_000_000),
        ModelInfo("claude-opus-4-7", "opus", 5.00, 25.00, 0.50, 1_000_000),
        ModelInfo("claude-opus-4-6", "opus", 5.00, 25.00, 0.50, 1_000_000),
        ModelInfo("claude-sonnet-5", "sonnet", 2.00, 10.00, 0.20, 1_000_000),
        ModelInfo("claude-sonnet-4-6", "sonnet", 3.00, 15.00, 0.30, 1_000_000),
        ModelInfo("claude-haiku-4-5", "haiku", 1.00, 5.00, 0.10, 200_000, us_inference_surcharge=False),
    ]
}


# Models to try, in order, when a model keeps failing (429/529/5xx after retries).
# Each chain tries the previous version of the same model first: similar quality,
# and Anthropic tracks rate limits per model. There's no fallback from Haiku,
# since stepping up to a pricier model should be a deliberate choice.
# All models here share the Claude 5 request surface (adaptive thinking, effort).
DEFAULT_FALLBACKS: dict[str, list[str]] = {
    "claude-fable-5-1": ["claude-fable-5", "claude-opus-5-5"],
    "claude-opus-5-5": ["claude-opus-5", "claude-sonnet-5-5"],
    "claude-sonnet-5-5": ["claude-sonnet-5", "claude-haiku-5-5"],
}


# Where to retry a request a model declined (stop_reason "refusal"). Anthropic's
# server-side fallback (the `fallbacks` parameter) covers Opus and Sonnet but not
# Claude Haiku 5.5, so the gateway provides one. This steps up in price, but only
# for requests the cheaper model refused.
DEFAULT_REFUSAL_FALLBACKS: dict[str, list[str]] = {
    "claude-haiku-5-5": ["claude-sonnet-5-5"],
}


def format_usd(amount: float) -> str:
    """Dollars for people: cents normally, more digits for sub-cent amounts."""
    if amount == 0 or abs(amount) >= 0.01:
        return f"${amount:,.2f}"
    return f"${amount:.8f}".rstrip("0")


def get_model(model_id: str) -> ModelInfo | None:
    return MODELS.get(model_id)


def cost_usd(model_id: str, usage: dict, batch: bool = False) -> float | None:
    """Dollar cost of one request from its `usage` object. None for unknown models."""
    model = get_model(model_id)
    if model is None:
        return None

    input_tokens = usage.get("input_tokens") or 0
    output_tokens = usage.get("output_tokens") or 0
    cache_read = usage.get("cache_read_input_tokens") or 0
    cache_creation = usage.get("cache_creation") or {}
    write_5m = cache_creation.get("ephemeral_5m_input_tokens")
    write_1h = cache_creation.get("ephemeral_1h_input_tokens") or 0
    if write_5m is None:
        # Older responses only report the total; treat it all as the default 5-minute TTL.
        write_5m = (usage.get("cache_creation_input_tokens") or 0) - write_1h

    input_price, output_price, cache_read_price = model.input_price, model.output_price, model.cache_read_price
    prompt_tokens = input_tokens + cache_read + write_5m + write_1h
    if model.long_prompt_threshold is not None and prompt_tokens > model.long_prompt_threshold:
        input_price, output_price = model.long_input_price, model.long_output_price
        cache_read_price = model.long_cache_read_price

    total = (
        input_tokens * input_price
        + write_5m * input_price * CACHE_WRITE_5M_MULTIPLIER
        + write_1h * input_price * CACHE_WRITE_1H_MULTIPLIER
        + cache_read * cache_read_price
        + output_tokens * output_price
    ) / 1_000_000
    if usage.get("inference_geo") == "us" and model.us_inference_surcharge:
        total *= US_INFERENCE_MULTIPLIER
    return total * BATCH_DISCOUNT if batch else total


def cache_savings_usd(model_id: str, usage: dict) -> float | None:
    """What prompt caching saved on one request: its cost if every cached token had been
    plain input, minus what it actually cost. Negative when cache writes weren't read back."""
    cached = (usage.get("cache_read_input_tokens") or 0) + (usage.get("cache_creation_input_tokens") or 0)
    actual = cost_usd(model_id, usage)
    if actual is None or not cached:
        return None if actual is None else 0.0
    uncached_usage = {
        "input_tokens": (usage.get("input_tokens") or 0) + cached,
        "output_tokens": usage.get("output_tokens") or 0,
        "inference_geo": usage.get("inference_geo"),
    }
    return cost_usd(model_id, uncached_usage) - actual
