from gateway.routing import DEFAULT_ALIASES, Alias, LatencyTracker, Router, estimate_cost


class Clock:
    def __init__(self):
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


def tracker(**kwargs) -> tuple[LatencyTracker, Clock]:
    clock = Clock()
    return LatencyTracker(clock=clock, **kwargs), clock


def feed(latency: LatencyTracker, model: str, *samples: float) -> None:
    for ms in samples:
        latency.record(model, ms)


def test_non_alias_is_not_routed():
    router = Router(DEFAULT_ALIASES, LatencyTracker())
    assert router.resolve("claude-sonnet-5-5", {}, 100) is None
    assert router.resolve(None, {}, 100) is None


def test_fixed_keeps_configured_order():
    route = Router(DEFAULT_ALIASES, LatencyTracker()).resolve("smart", {}, 100)
    assert route.models == ["claude-opus-5-5", "claude-sonnet-5-5"]
    assert route.strategy == "fixed"


def test_cheapest_orders_by_estimated_cost():
    aliases = {"cheap": Alias("cheapest", ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"])}
    route = Router(aliases, LatencyTracker()).resolve("cheap", {"max_tokens": 500}, 4_000)
    assert route.models == ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"]


def test_cost_estimate_accounts_for_prompt_size_and_max_tokens():
    small = estimate_cost("claude-haiku-5-5", 4_000, 100)  # ~1K input tokens
    huge = estimate_cost("claude-haiku-5-5", 800_000, 100)  # ~200K input tokens: long-prompt prices
    assert small == (1_000 * 0.10 + 100 * 0.50) / 1e6
    assert huge == (200_000 * 0.50 + 100 * 2.50) / 1e6
    assert estimate_cost("not-a-model", 100, 100) == float("inf")


def test_fastest_uses_configured_order_until_measured():
    latency, _ = tracker(min_samples=3)
    router = Router({"fast": Alias("fastest", ["a", "b", "c"])}, latency)
    assert router.resolve("fast", {}, 0).models == ["a", "b", "c"]


def test_fastest_ranks_by_p50_once_measured():
    latency, _ = tracker(min_samples=3)
    feed(latency, "a", 900, 1000, 1100)
    feed(latency, "b", 300, 5000, 350)  # One slow outlier; median is 350.
    feed(latency, "c", 600, 600, 600)
    router = Router({"fast": Alias("fastest", ["a", "b", "c"])}, latency)
    assert router.resolve("fast", {}, 0).models == ["b", "c", "a"]


def test_fastest_explores_unmeasured_models_first():
    latency, _ = tracker(min_samples=3)
    feed(latency, "a", 100, 100, 100)
    router = Router({"fast": Alias("fastest", ["a", "b"])}, latency)
    # "b" has no data; without trying it first it could never be measured.
    assert router.resolve("fast", {}, 0).models == ["b", "a"]


def test_old_samples_expire():
    latency, clock = tracker(min_samples=2, max_age_s=60)
    feed(latency, "a", 100, 100)
    assert latency.p50("a") == 100
    clock.now += 61
    assert latency.p50("a") is None


def test_window_keeps_only_recent_samples():
    latency, _ = tracker(min_samples=1, window=3)
    feed(latency, "a", 1, 1, 1, 900, 900, 900)
    assert latency.p50("a") == 900
    assert latency.snapshot() == {"a": 900}


def test_default_alias_candidates_are_known_models():
    from gateway.models import MODELS

    for alias in DEFAULT_ALIASES.values():
        assert all(m in MODELS for m in alias.models)


def test_fastest_treats_near_ties_as_ties_and_prefers_configured_order():
    latency, _ = tracker(min_samples=1)
    feed(latency, "haiku", 500)
    feed(latency, "sonnet", 470)  # 6% faster: noise, not worth 20x the price.
    feed(latency, "slow", 900)
    router = Router({"fast": Alias("fastest", ["haiku", "sonnet", "slow"])}, latency)
    assert router.resolve("fast", {}, 0).models == ["haiku", "sonnet", "slow"]


def test_fastest_switches_when_a_model_is_meaningfully_faster():
    latency, _ = tracker(min_samples=1)
    feed(latency, "haiku", 1_200)  # Haiku having a slow hour.
    feed(latency, "sonnet", 470)
    router = Router({"fast": Alias("fastest", ["haiku", "sonnet"])}, latency)
    assert router.resolve("fast", {}, 0).models == ["sonnet", "haiku"]
