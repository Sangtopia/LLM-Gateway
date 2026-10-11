"""Pacing from Anthropic's rate-limit headers.

Every Anthropic response says how much of the organization's per-model quota is
left (anthropic-ratelimit-{requests,input-tokens,output-tokens}-{limit,remaining,reset}).
The gateway remembers the latest numbers per model and treats a model as "paced"
when sending this request would likely get a 429:

  - its request bucket is nearly empty (under 1% of the limit);
  - its input-token bucket can't fit this request's estimated input;
  - its output-token bucket is nearly empty;
  - a recent 429 said to wait (retry-after).

Anthropic refills buckets continuously, so between responses the remaining amount
is estimated as the last reading plus the refill since then.

A paced model is skipped in favor of the next model in the chain, like an open
circuit, so the request avoids the 429 instead of paying for a retry. Unlike the
circuit breaker, if every model in the chain is paced the request is sent anyway:
these are estimates, and Anthropic has the final say.

Kept per process, like the circuit breaker: no extra round trip per request.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

KINDS = ("requests", "input-tokens", "output-tokens")
RESERVE_FRACTION = 0.01  # Leave 1% of a bucket for requests already in flight.
WINDOW_S = 60.0  # Anthropic's limits are per minute.


@dataclass
class Bucket:
    limit: float
    remaining: float
    seen_at: float  # monotonic time of the reading

    def now_remaining(self, now: float) -> float:
        refill = (now - self.seen_at) * self.limit / WINDOW_S
        return min(self.limit, self.remaining + refill)


def _parse_reset(value: str | None) -> float | None:
    """Seconds from now until an RFC 3339 reset time."""
    if not value:
        return None
    try:
        reset = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (reset - datetime.now(reset.tzinfo)).total_seconds()


class UpstreamLimits:
    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._buckets: dict[tuple[str, str], Bucket] = {}
        self._blocked_until: dict[str, float] = {}

    def observe(self, model: str | None, status: int, headers) -> None:
        """Record what a response from `model` says about its remaining quota."""
        if not model:
            return
        now = self._clock()
        for kind in KINDS:
            try:
                limit = float(headers[f"anthropic-ratelimit-{kind}-limit"])
                remaining = float(headers[f"anthropic-ratelimit-{kind}-remaining"])
            except (KeyError, TypeError, ValueError):
                continue
            if limit > 0:
                self._buckets[(model, kind)] = Bucket(limit, remaining, now)
        if status == 429:
            try:
                wait = float(headers.get("retry-after", ""))
            except ValueError:
                wait = None
            if wait is None:  # No retry-after: wait for the soonest bucket reset Anthropic reported.
                resets = [_parse_reset(headers.get(f"anthropic-ratelimit-{k}-reset")) for k in KINDS]
                wait = min((r for r in resets if r is not None and r > 0), default=1.0)
            self._blocked_until[model] = now + max(0.0, wait)

    def paced(self, model: str | None, estimated_input_tokens: int = 0) -> str | None:
        """Why `model` should be skipped right now, or None if it's fine to send."""
        if not model:
            return None
        now = self._clock()
        if now < self._blocked_until.get(model, 0.0):
            return "retry-after"
        requests = self._buckets.get((model, "requests"))
        if requests and requests.now_remaining(now) < max(1.0, requests.limit * RESERVE_FRACTION):
            return "requests"
        input_tokens = self._buckets.get((model, "input-tokens"))
        if input_tokens and input_tokens.now_remaining(now) < max(estimated_input_tokens, input_tokens.limit * RESERVE_FRACTION):
            return "input-tokens"
        output_tokens = self._buckets.get((model, "output-tokens"))
        if output_tokens and output_tokens.now_remaining(now) < output_tokens.limit * RESERVE_FRACTION:
            return "output-tokens"
        return None

    def snapshot(self) -> dict[str, dict[str, int]]:
        """Estimated remaining quota per model, for /health and metrics."""
        now = self._clock()
        out: dict[str, dict[str, int]] = {}
        for (model, kind), bucket in self._buckets.items():
            out.setdefault(model, {})[kind] = int(bucket.now_remaining(now))
        return out
