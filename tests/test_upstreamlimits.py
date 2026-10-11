from gateway.upstreamlimits import UpstreamLimits

M = "claude-sonnet-5-5"


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def headers(requests=(1000, 900), input_tokens=(2_000_000, 1_500_000), output_tokens=(400_000, 300_000), **extra):
    h = {}
    for kind, (limit, remaining) in (("requests", requests), ("input-tokens", input_tokens), ("output-tokens", output_tokens)):
        h[f"anthropic-ratelimit-{kind}-limit"] = str(limit)
        h[f"anthropic-ratelimit-{kind}-remaining"] = str(remaining)
    return {**h, **extra}


def test_plenty_of_quota_is_not_paced():
    limits = UpstreamLimits(Clock())
    limits.observe(M, 200, headers())
    assert limits.paced(M, estimated_input_tokens=5_000) is None
    assert limits.paced("never-seen") is None


def test_nearly_empty_request_bucket_is_paced_until_it_refills():
    clock = Clock()
    limits = UpstreamLimits(clock)
    limits.observe(M, 200, headers(requests=(1000, 3)))
    assert limits.paced(M) == "requests"  # Under 1% (10) left.
    clock.now += 1  # 1000/min refills ~16.7 per second.
    assert limits.paced(M) is None


def test_input_bucket_too_small_for_this_request():
    limits = UpstreamLimits(Clock())
    limits.observe(M, 200, headers(input_tokens=(2_000_000, 30_000)))
    assert limits.paced(M, estimated_input_tokens=10_000) is None
    assert limits.paced(M, estimated_input_tokens=50_000) == "input-tokens"


def test_429_blocks_for_retry_after():
    clock = Clock()
    limits = UpstreamLimits(clock)
    limits.observe(M, 429, headers(**{"retry-after": "12"}))
    assert limits.paced(M) == "retry-after"
    clock.now += 12.1
    assert limits.paced(M) is None


def test_missing_or_garbled_headers_are_ignored():
    limits = UpstreamLimits(Clock())
    limits.observe(M, 200, {"anthropic-ratelimit-requests-limit": "abc", "anthropic-ratelimit-requests-remaining": "1"})
    limits.observe(None, 200, headers(requests=(1000, 0)))
    assert limits.paced(M) is None and limits.snapshot() == {}


def test_snapshot_reports_refilled_estimates():
    clock = Clock()
    limits = UpstreamLimits(clock)
    limits.observe(M, 200, headers(requests=(600, 0)))
    clock.now += 30  # Half a minute: half the bucket back.
    assert limits.snapshot()[M]["requests"] == 300
